"""
Kunci GPU bersama antara backend PRODUKSI dan DEV.

KENAPA PERLU
Kartu grafis mesin ini 8 GB, sedangkan satu kali jalan pipeline memakai sekitar
4 GB. Dua pemrosesan bersamaan TIDAK muat, dan yang kedua akan mati di tengah
jalan karena kehabisan memori. Itu sudah terjadi nyata saat dua pengukuran
dijalankan bersamaan.

DIAM ADA DUA MACAM, dan ini pernah salah ditulis di sini.

1. Diam SEBELUM analisis pertama: benar benar murah. Terukur GPU nol MiB dan
   RAM sekitar 730 MB, karena model baru dimuat ketika ada video diproses.
2. Diam SESUDAH pernah menganalisis: TIDAK murah. Terukur backend produksi
   masih memegang 6150 MiB, 14 jam setelah analisis terakhirnya selesai.
   `_release_gpu_cache` memanggil `torch.cuda.empty_cache()` yang hanya
   mengembalikan kolam cadangan milik torch; log mencatat "reserved 1.94 ->
   0.45 GB". Sisanya konteks CUDA beserta mesin TensorRT dan ONNX, yang berada
   di luar pengurus memori torch dan tidak bisa dilepas tanpa mematikan
   prosesnya. Komentarnya sendiri menulis "NOT a full unload".

AKIBATNYA, DAN INI PENTING
Pengunci ini menjaga PEMROSESAN, bukan tempat yang dipakai saat diam. Kartunya
8 GB. Kalau satu sisi pernah menganalisis lalu diam sambil memegang 6 GB, sisi
lain bisa kehabisan memori saat mulai menganalisis WALAU kuncinya bebas.

Maka: kalau mau menjalankan analisis di satu sisi sementara sisi lain pernah
menganalisis, NYALAKAN ULANG sisi yang menganggur itu lebih dulu supaya
memorinya benar benar kembali.

ATURAN
1. Hanya SATU sisi yang boleh memproses video pada satu waktu.
2. PRODUKSI didahulukan. Kalau dev sedang memproses lalu produksi butuh GPU,
   dev diminta berhenti baik baik, dan produksi menunggu sebentar sampai dev
   benar benar melepas.
3. Dev yang mencoba memproses saat produksi sedang jalan akan DITOLAK dengan
   keterangan yang jelas, bukan dibiarkan mati karena kehabisan memori.

KENAPA REDIS
Redis sudah hidup dan sudah DIPAKAI BERSAMA oleh kedua backend, jadi tidak
perlu menambah layanan baru. Kuncinya memakai SET NX yang bersifat atomik, dan
TTL membuat kunci terlepas sendiri kalau prosesnya mati mendadak, sehingga
tidak ada kunci yatim yang memblokir selamanya.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

KUNCI       = "smartvision:gpu_owner"
PREEMPT     = "smartvision:gpu_preempt"

# TTL kunci. Harus lebih panjang dari jarak denyut supaya kunci tidak kedaluwarsa
# di tengah pemrosesan yang masih sehat, tapi cukup pendek supaya kunci yatim
# dari proses yang mati tidak memblokir lama.
TTL_DETIK   = int(os.getenv("GPU_LOCK_TTL_S", "60"))
# Jeda pemeriksaan. Sengaja PENDEK: inilah yang menentukan seberapa cepat dev
# berhenti saat produksi meminta GPU. Biayanya cuma satu perintah Redis.
DENYUT_S    = int(os.getenv("GPU_LOCK_HEARTBEAT_S", "3"))
# Berapa lama produksi menunggu dev melepas sebelum mengambil paksa.
TUNGGU_S    = int(os.getenv("GPU_PREEMPT_WAIT_S", "30"))

# ── Kunci WAJIB berada di satu tempat yang sama untuk prod dan dev ──────────
# CACAT NYATA yang pernah terjadi: prod memakai REDIS_URL=redis://localhost:6379
# (basis data 0) sedangkan dev memakai .../1. Keduanya BENAR BENAR mengambil
# kunci, tapi di basis data yang BERBEDA, jadi tidak pernah saling melihat dan
# penguncinya tidak berguna sama sekali. Uji saya saat itu lolos karena kedua
# sisi diuji di dalam satu basis data yang sama.
#
# Maka modul ini TIDAK memakai sambungan Redis milik pemanggil. Ia membuka
# sambungannya sendiri ke basis data tetap, sehingga prod dan dev selalu
# bertemu di ruangan yang sama berapa pun REDIS_URL masing masing.
GPU_LOCK_URL = os.getenv(
    "GPU_LOCK_REDIS_URL",
    (os.getenv("REDIS_URL", "redis://localhost:6379").rsplit("/", 1)[0]
     if os.getenv("REDIS_URL", "").count("/") > 2
     else os.getenv("REDIS_URL", "redis://localhost:6379")).rstrip("/") + "/15",
)

_koneksi = None


async def _redis_kunci(cadangan=None):
    """Sambungan khusus kunci, selalu ke basis data yang sama.

    Kalau gagal, jatuh ke sambungan milik pemanggil supaya tidak memblokir
    pekerjaan; tapi itu dicatat keras karena saat itu penguncinya kembali
    rapuh terhadap perbedaan basis data.
    """
    global _koneksi
    if _koneksi is not None:
        return _koneksi
    try:
        import redis.asyncio as _ar
        _koneksi = _ar.from_url(GPU_LOCK_URL, decode_responses=True)
        await _koneksi.ping()
        logger.info("Kunci GPU memakai %s", GPU_LOCK_URL)
        return _koneksi
    except Exception as e:
        logger.warning("Sambungan khusus kunci GPU gagal (%s). "
                       "Memakai Redis pemanggil; kunci bisa TIDAK terlihat "
                       "oleh sisi lain kalau basis datanya berbeda.", e)
        return cadangan


def sisi() -> str:
    """"prod" atau "dev". Diambil dari env, kalau tidak ada ditebak dari port.

    Port 8000 dipakai produksi dan 8001 dipakai dev, sesuai .env masing masing.
    """
    s = os.getenv("DEPLOY_SIDE", "").strip().lower()
    if s in ("prod", "dev"):
        return s
    return "dev" if os.getenv("API_PORT", "8000").strip() == "8001" else "prod"


def _punyaku(data: dict) -> bool:
    return bool(data) and data.get("pid") == os.getpid()


async def keadaan(redis) -> dict:
    """Siapa yang sedang memegang GPU. Dipakai endpoint status dan dev UI."""
    redis = await _redis_kunci(redis)
    if redis is None:
        return {"terkunci": False, "alasan": "redis tidak tersedia"}
    try:
        mentah = await redis.get(KUNCI)
    except Exception as e:                               # pragma: no cover
        return {"terkunci": False, "alasan": f"redis galat: {e}"}
    if not mentah:
        return {"terkunci": False}
    try:
        d = json.loads(mentah)
    except Exception:
        return {"terkunci": True, "alasan": "isi kunci tidak terbaca"}
    d["terkunci"] = True
    d["lama_detik"] = round(time.time() - d.get("mulai", time.time()), 1)
    return d


async def ambil(redis, match_id: str):
    """Coba ambil kunci GPU. Mengembalikan (berhasil, keterangan).

    Dev DITOLAK kalau produksi sedang memegang. Produksi memasang bendera minta
    dahulu lalu menunggu dev melepas, dan setelah TUNGGU_S tetap mengambil
    karena produksi memang didahulukan.
    """
    redis = await _redis_kunci(redis)
    if redis is None:
        # Tanpa Redis tidak ada koordinasi. Jangan memblokir pekerjaan; cukup
        # beri tahu supaya ketahuan kalau dua sisi berebut.
        logger.warning("Kunci GPU dilewati: Redis tidak tersedia")
        return True, {"terkunci": False, "alasan": "redis tidak tersedia"}

    aku = sisi()
    nilai = json.dumps({"sisi": aku, "pid": os.getpid(),
                        "match_id": match_id, "mulai": time.time()})

    if await redis.set(KUNCI, nilai, nx=True, ex=TTL_DETIK):
        await redis.delete(PREEMPT)
        logger.info("Kunci GPU diambil oleh %s (match %s)", aku, match_id)
        return True, {"terkunci": False}

    pemegang = await keadaan(redis)

    # Pemegangnya proses ini sendiri, misalnya kuarter berikutnya dari video
    # yang sama. Perpanjang saja, jangan menolak diri sendiri.
    if _punyaku(pemegang):
        await redis.set(KUNCI, nilai, ex=TTL_DETIK)
        return True, pemegang

    if aku == "dev":
        logger.warning("Dev ditolak: GPU dipegang %s", pemegang.get("sisi"))
        return False, pemegang

    # Produksi didahulukan: minta dev berhenti, lalu tunggu.
    await redis.set(PREEMPT, "1", ex=TTL_DETIK)
    logger.info("Produksi meminta GPU, menunggu %s melepas (maks %ss)",
                pemegang.get("sisi"), TUNGGU_S)
    batas = time.time() + TUNGGU_S
    while time.time() < batas:
        import asyncio
        await asyncio.sleep(1)
        if await redis.set(KUNCI, nilai, nx=True, ex=TTL_DETIK):
            await redis.delete(PREEMPT)
            logger.info("Dev sudah melepas, produksi mengambil GPU")
            return True, {"terkunci": False}

    # Dev tidak melepas dalam batas waktu. Produksi tetap mengambil karena
    # didahulukan, TAPI dicatat keras karena ini berisiko kehabisan memori.
    await redis.set(KUNCI, nilai, ex=TTL_DETIK)
    await redis.delete(PREEMPT)
    logger.warning("Dev TIDAK melepas dalam %ss. Produksi mengambil paksa. "
                   "Risiko kehabisan memori GPU kalau dev masih benar benar jalan.",
                   TUNGGU_S)
    return True, pemegang


async def denyut(redis) -> bool:
    """Perpanjang kunci. True kalau pemegang diminta BERHENTI.

    Dipanggil berkala selama pemrosesan. Dev yang melihat bendera minta dahulu
    dari produksi harus berhenti baik baik, bukan dimatikan paksa, supaya hasil
    sampai frame terakhir tetap tersimpan.
    """
    redis = await _redis_kunci(redis)
    if redis is None:
        return False
    try:
        d = await keadaan(redis)
        if not _punyaku(d):
            return False
        nilai = json.dumps({"sisi": sisi(), "pid": os.getpid(),
                            "match_id": d.get("match_id", ""),
                            "mulai": d.get("mulai", time.time())})
        await redis.set(KUNCI, nilai, ex=TTL_DETIK)
        if sisi() == "dev" and await redis.get(PREEMPT):
            logger.warning("Produksi meminta GPU. Dev berhenti baik baik.")
            return True
    except Exception as e:                               # pragma: no cover
        logger.warning("Denyut kunci GPU gagal: %s", e)
    return False


async def lepas(redis) -> None:
    """Lepas kunci, HANYA kalau memang milik proses ini."""
    redis = await _redis_kunci(redis)
    if redis is None:
        return
    try:
        if _punyaku(await keadaan(redis)):
            await redis.delete(KUNCI)
            logger.info("Kunci GPU dilepas oleh %s", sisi())
        if sisi() == "dev":
            await redis.delete(PREEMPT)
    except Exception as e:                               # pragma: no cover
        logger.warning("Melepas kunci GPU gagal: %s", e)

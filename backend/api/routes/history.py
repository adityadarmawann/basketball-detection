"""
Riwayat analisis: daftar pertandingan yang pernah dianalisis, dan penghapusnya.

KENAPA ADA
Semua hasil analisis sebenarnya SUDAH bisa dibuka lagi lewat `/view/{match_id}`
tanpa render ulang. Yang tidak ada selama ini cuma DAFTARNYA, sehingga pengguna
harus tahu match_id-nya sendiri, dan itu mustahil. Berkas ini menutup lubang itu.

KENAPA ADA DUA TINGKAT PENGHAPUSAN
Yang memakan penyimpanan itu video hasil render dan berkas frame, bukan
angkanya. Ringkasan skor, statistik pemain, dan MPI ukurannya kecil dan
disimpan di Mongo. Jadi:
  - hapus "berkas" : video dan frame dibuang, ANGKANYA TETAP bisa dilihat.
  - hapus "semua"  : seluruh catatan analisis itu hilang.
Pengguna bisa membebaskan disk tanpa kehilangan hasil analisisnya.

YANG TIDAK PERNAH DISENTUH
`sponsor_library` dan `roster_library` tidak terikat match_id dan TIDAK ikut
terhapus. Daftar nama pemain pernah ikut terhapus sekali karena dokumennya
memakai match_id; sejak itu salinannya disimpan di roster_library supaya aman
dari pembersihan apa pun.
"""

import glob
import logging
import os

from fastapi import APIRouter, HTTPException, Query

from db.mongo import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_PATH = os.getenv("UPLOAD_PATH", "./uploads")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "./output")


def _ukuran(path: str | None) -> int:
    """Ukuran berkas dalam byte, 0 kalau tidak ada. Tidak pernah melempar."""
    try:
        return os.path.getsize(path) if path and os.path.exists(path) else 0
    except OSError:
        return 0


def _berkas_match(match_id: str) -> list:
    """Semua berkas di disk yang menempel ke satu match.

    Video unggahan disimpan sebagai {UPLOAD_PATH}/{waktu}_{match_id}_{nama},
    jadi dicari dengan pola. Keluaran render ada di OUTPUT_PATH.
    """
    hasil = []
    for pola in (os.path.join(UPLOAD_PATH, f"*{match_id}*"),
                 os.path.join(OUTPUT_PATH, f"*{match_id}*"),
                 os.path.join(OUTPUT_PATH, match_id, "*")):
        hasil.extend(p for p in glob.glob(pola) if os.path.isfile(p))
    return sorted(set(hasil))


def _nama_tim(t) -> str:
    """team_a/team_b kadang string, kadang objek {name, color}."""
    if isinstance(t, dict):
        return t.get("name") or "?"
    return str(t) if t else "?"


def _warna_tim(t) -> str:
    return t.get("color", "") if isinstance(t, dict) else ""


@router.get("/history")
async def daftar_riwayat(limit: int = Query(100, ge=1, le=500)):
    """Daftar analisis, terbaru dulu.

    Tiap baris memuat penanda apakah video dan ringkasannya MASIH ADA, supaya
    pengguna tidak mengklik lalu menemukan halaman kosong.
    """
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB tidak tersedia")

    baris = []
    for m in db.matches.find().sort("_id", -1).limit(limit):
        mid = m.get("match_id")
        if not mid:
            continue
        berkas = _berkas_match(mid)
        total_byte = sum(_ukuran(p) for p in berkas)
        ringkasan = m.get("ringkasan") or {}
        # Skor diambil dari kuarter mana pun yang tersimpan, yang terbaru menang.
        skor = None
        for q in sorted(ringkasan.keys()):
            s = (ringkasan[q] or {}).get("skor")
            if s:
                skor = s
        baris.append({
            "match_id":      mid,
            "tim_a":         _nama_tim(m.get("team_a")),
            "tim_b":         _nama_tim(m.get("team_b")),
            "warna_a":       _warna_tim(m.get("team_a")),
            "warna_b":       _warna_tim(m.get("team_b")),
            "status":        m.get("status", "?"),
            "dibuat":        m.get("created_at"),
            "diproses":      m.get("processed_at"),
            "kuarter":       sorted(ringkasan.keys()),
            "skor":          skor,
            # Jumlah kejadian adalah satu satunya penanda isi yang BISA
            # DIPERCAYA lintas sisi. Ringkasan hanya ditulis oleh dev, dan
            # berkas dicari di folder milik sisi ini saja (uploads_dev lawan
            # uploads), sehingga analisis yang dijalankan sisi LAIN akan
            # terbaca seolah kosong padahal datanya ada. Kejadian disimpan di
            # MongoDB yang dipakai bersama, jadi selalu terlihat.
            "kejadian":      db.events.count_documents({"match_id": mid}),
            # Angka tetap bisa dilihat selama ringkasan ada, walau berkas hilang.
            "ada_ringkasan": bool(ringkasan),
            "ada_berkas":    bool(berkas),
            "jumlah_berkas": len(berkas),
            "ukuran_byte":   total_byte,
            "ukuran_mb":     round(total_byte / 1048576, 1),
        })
    return {"jumlah": len(baris), "riwayat": baris}


@router.delete("/history/{match_id}")
async def hapus_riwayat(
    match_id: str,
    mode: str = Query("berkas", pattern="^(berkas|semua)$"),
):
    """Hapus riwayat.

    mode=berkas (bawaan) — buang video dan frame untuk membebaskan disk, TAPI
                           skor, statistik, dan MPI tetap bisa dilihat.
    mode=semua           — buang seluruh catatan analisis ini.

    Bawaannya sengaja "berkas", bukan "semua": penghapusan yang tidak bisa
    dibatalkan tidak pantas jadi perilaku bawaan.
    """
    db = get_db()
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB tidak tersedia")
    if not db.matches.find_one({"match_id": match_id}):
        raise HTTPException(status_code=404, detail=f"match {match_id} tidak ada")

    # ── PENGAMAN: jangan hapus analisis yang SEDANG BERJALAN ────────────────
    # Pernah terjadi: tombol hapus ditekan pada match yang pipeline-nya masih
    # menulis, sehingga catatan dan 226 kejadiannya hilang sementara prosesnya
    # jalan terus dan mulai menulis dari nol. Hasilnya kacau dan tidak bisa
    # dibatalkan. Pemegang kunci GPU adalah penanda paling bisa dipercaya untuk
    # "sedang diproses", karena dipegang selama pipeline hidup dan terlepas
    # sendiri kalau prosesnya mati.
    try:
        from db.redis_client import get_redis
        from db import gpu_lock as _gpu
        _r = await get_redis()
        _pemegang = await _gpu.keadaan(_r)
        if _pemegang.get("terkunci") and _pemegang.get("match_id") == match_id:
            raise HTTPException(
                status_code=409,
                detail=(f"Match {match_id} SEDANG dianalisis "
                        f"({_pemegang.get('lama_detik', 0):.0f} detik berjalan). "
                        f"Hentikan analisisnya dulu lewat tombol stop, atau "
                        f"tunggu sampai selesai, baru bisa dihapus."),
            )
    except HTTPException:
        raise
    except Exception as e:                                # pragma: no cover
        # Pengaman tidak boleh MEMBLOKIR penghapusan kalau Redis bermasalah,
        # tapi kegagalannya harus terlihat, bukan ditelan diam diam.
        logger.warning("Pengaman hapus tidak bisa memeriksa kunci GPU: %s", e)

    berkas = _berkas_match(match_id)
    dibebaskan = sum(_ukuran(p) for p in berkas)
    terhapus = 0
    for p in berkas:
        try:
            os.remove(p)
            terhapus += 1
        except OSError as e:
            logger.warning("gagal menghapus %s: %s", p, e)

    hasil = {
        "match_id": match_id,
        "mode": mode,
        "berkas_dihapus": terhapus,
        "disk_dibebaskan_mb": round(dibebaskan / 1048576, 1),
    }

    if mode == "semua":
        hasil["kejadian_dihapus"] = db.events.delete_many(
            {"match_id": match_id}).deleted_count
        hasil["pemain_dihapus"] = db.players.delete_many(
            {"match_id": match_id}).deleted_count
        hasil["sponsor_dihapus"] = db.sponsors.delete_many(
            {"match_id": match_id}).deleted_count
        db.matches.delete_one({"match_id": match_id})
        hasil["catatan_dihapus"] = True
        # roster_library dan sponsor_library TIDAK disentuh: keduanya tidak
        # terikat match_id dan memang dimaksudkan bertahan lintas pertandingan.
    else:
        # Jalur berkas tidak boleh menghapus jalur yang sudah tidak sahih,
        # supaya daftar tidak lagi menjanjikan video yang sebenarnya hilang.
        db.matches.update_one(
            {"match_id": match_id},
            {"$unset": {"frames_json_path": "", "output_csv_path": "",
                        "output_video_path": ""}},
        )
        hasil["catatan_dihapus"] = False
        hasil["ringkasan_tetap_ada"] = True

    logger.info("Riwayat %s dihapus mode=%s, %s berkas, %.1f MB",
                match_id, mode, terhapus, dibebaskan / 1048576)
    return hasil

@router.get("/video/{match_id}")
async def video_apa_saja(match_id: str):
    """Video mana pun yang MASIH ADA untuk satu match.

    Halaman hasil dulu hanya meminta video MENTAH hasil unggahan. Begitu berkas
    itu dibersihkan demi menghemat disk, pemutarnya kosong padahal video hasil
    ANALISIS kadang masih ada, atau sebaliknya. Endpoint ini mencari keduanya
    dan mengembalikan yang pertama ketemu, dengan mengutamakan video hasil
    analisis karena itu yang bergambar kotak pemain dan berguna untuk ditonton.

    404 hanya kalau benar benar tidak ada satu pun.
    """
    from fastapi.responses import FileResponse

    kandidat = []
    # 1. Video MENTAH hasil unggahan — DIUTAMAKAN, dan ini penting.
    #    Pemutar di halaman hasil menggambar kotak pemain sendiri dari frameData
    #    lewat CourtOverlay. Kalau yang diputar video hasil analisis yang kotaknya
    #    sudah TERCETAK, kotaknya tergambar DUA KALI dan tampilannya kacau.
    #    Video hasil analisis tetap bisa diunduh lewat /api/output/{id}/video.
    kandidat += sorted(glob.glob(os.path.join(UPLOAD_PATH, f"*{match_id}*")))
    # 2. Baru video hasil analisis, sebagai cadangan kalau yang mentah sudah
    #    dihapus demi menghemat disk. Lebih baik ada kotak dua kali daripada
    #    pemutar kosong sama sekali.
    out_dir = os.path.join(OUTPUT_PATH, match_id)
    kandidat += sorted(glob.glob(os.path.join(out_dir, "*.mp4")))
    kandidat += sorted(glob.glob(os.path.join(OUTPUT_PATH, f"*{match_id}*.mp4")))

    for p in kandidat:
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            jenis = "mentah" if UPLOAD_PATH in p else "analisis"
            return FileResponse(p, media_type="video/mp4",
                                filename=os.path.basename(p),
                                headers={"X-Sumber-Video": jenis})

    raise HTTPException(
        status_code=404,
        detail=("Video untuk match ini sudah tidak ada di disk. "
                "Statistik dan MPI tetap bisa dilihat."),
    )

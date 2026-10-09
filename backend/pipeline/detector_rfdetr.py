"""
Detektor RF-DETR 10 kelas, cara Roboflow.

BELUM AKTIF. Berkas ini disiapkan supaya begitu model RF-DETR selesai dilatih,
cukup menaruh bobotnya lalu menyalakan satu saklar, tanpa menyentuh pipeline.

Cara menyalakan nanti:
    1. taruh bobot di backend/models/rfdetr-basket.pth
    2. di .env isi:
           DETECTOR_METHOD=rfdetr
           RFDETR_WEIGHTS=rfdetr-basket.pth
           RFDETR_VARIANT=base          (nano|small|base|medium|large)
    3. pip install rfdetr
    4. untuk ikut mematikan VideoMAE:   ACTION_SOURCE=detector
    5. untuk memakai ShotEventTracker:  SHOT_METHOD=roboflow

Kenapa dibuat sebagai adaptor dan bukan mengganti detector.py:
Keluarannya DISAMAKAN BENTUKNYA dengan BasketballDetector.detect(), yaitu dict
dengan kunci players, ball, referees, hoops, backboards. Jadi pelacak,
homografi, pose, dan statistik sama sekali tidak perlu diubah. Kunci TAMBAHAN
dari 10 kelas (numbers, balls_in_basket, rims, dan aksi per pemain) ditaruh di
samping, dan yang belum siap memakainya cukup mengabaikannya.

Pemetaan 10 kelas ke keluaran lama:
    0 ball                  -> ball                (ambil yang keyakinan tertinggi)
    1 ball-in-basket        -> balls_in_basket     (BARU, pemicu ShotEventTracker)
    2 number                -> numbers             (BARU, dicocokkan ke pemain
                                                    dengan Mask IoS, bukan IoU)
    3 player                -> players, aksi None
    4 player-in-possession  -> players, aksi "possession"
    5 player-jump-shot      -> players, aksi "Shoot"
    6 player-layup-dunk     -> players, aksi "layup_dunk"
    7 player-shot-block     -> players, aksi "block"
    8 referee               -> referees
    9 rim                   -> hoops DAN rims

Catatan soal backboard: RF-DETR 10 kelas TIDAK punya kelas backboard, karena
recall backboard di model lama 0 persen dan Roboflow sendiri membuangnya.
Kunci "backboards" tetap ada tapi selalu kosong, supaya kode lama yang
membacanya tidak pecah.

Catatan soal rim: kelas rim diisikan ke "hoops" juga, karena mesin kejadian
yang sekarang membaca "hoops". Jadi tidak ada yang perlu diubah di sana.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = os.getenv(
    "MODELS_PATH", os.path.join(os.path.dirname(__file__), "..", "models")
)

# Urutan ini WAJIB sama dengan urutan kelas saat training. Kalau diubah, semua
# deteksi akan salah label tanpa pesan kesalahan apa pun. Resep dan alasan
# urutannya ada di basketball-roboflow/catatan-model-baru-dan-training.txt.
RFDETR_CLASS_NAMES = [
    "ball",                  # 0
    "ball-in-basket",        # 1
    "number",                # 2
    "player",                # 3
    "player-in-possession",  # 4
    "player-jump-shot",      # 5
    "player-layup-dunk",     # 6
    "player-shot-block",     # 7
    "referee",               # 8
    "rim",                   # 9
]

# Kelas pemain yang sekaligus menyandikan aksi. Nilainya dipakai action.py
# kalau ACTION_SOURCE=detector. "Shoot" sengaja memakai nama yang sudah ada di
# ACTION_NAMES supaya tidak perlu menambah kelas baru di sisi statistik.
PLAYER_CLASS_TO_ACTION = {
    "player":               None,
    "player-in-possession": "possession",
    "player-jump-shot":     "Shoot",
    "player-layup-dunk":    "layup_dunk",
    "player-shot-block":    "block",
}

# Varian hanya dipakai sebagai CADANGAN. Jalur utamanya
# RFDETR.from_checkpoint(), yang menentukan kelas model langsung dari isi
# checkpoint, jadi tidak mungkin salah pasang. Perhatikan "base" sudah USANG
# sejak rfdetr v1.7.0 dan akan dihapus di v2.0.0, karena itu bawaannya "medium".
VARIANT    = os.getenv("RFDETR_VARIANT", "medium").strip().lower()
# torch.load pada berkas .pth menjalankan pickle, jadi rfdetr memasang gerbang
# trust_checkpoint. Bawaannya di sini True karena yang dimuat adalah checkpoint
# hasil training sendiri di mesin ini. Kalau suatu saat memuat bobot dari pihak
# lain, setel RFDETR_TRUST=0 dulu dan periksa berkasnya.
TRUST      = os.getenv("RFDETR_TRUST", "1") == "1"
WEIGHTS    = os.getenv("RFDETR_WEIGHTS", "best-object-basket-detect-v1-DETR.pt")
# RF-DETR tidak memakai NMS IoU seperti YOLO; ambangnya hanya keyakinan.
CONF       = float(os.getenv("RFDETR_CONF", "0.35"))

# ── Ukuran masukan. WAJIB dikirim, bukan pilihan ────────────────────────────
# Checkpoint ini dilatih pada resolution=560, tapi rfdetr 1.11.2 MENOLAK
# menjalankannya tanpa shape eksplisit:
#     ValueError: Model's default resolution (560) is not divisible by
#     block_size=32 (patch_size=16 * num_windows=2).
# Jadi kalau shape tidak dikirim, inferensi gagal di SETIAP frame. Sudah diuji:
#     544x544 -> 44 deteksi di 4 frame
#     576x576 -> 48 deteksi di 4 frame   <- dipakai, paling dekat ke 560 dari atas
#     560x560 -> ditolak pustaka
RFDETR_SHAPE = int(os.getenv("RFDETR_SHAPE", "576"))

# Percepatan. rfdetr memperingatkan "Model is not optimized for inference"
# kalau ini tidak dipanggil. model.inference() mengompilasi dan memakai fp16,
# dan menurut pustakanya bisa sekitar 8x di GPU dengan Tensor Core.
# Dimatikan otomatis saat di CPU karena fp16 di CPU justru lebih lambat.
RFDETR_OPTIMIZE = os.getenv("RFDETR_OPTIMIZE", "1") == "1"
# Pencocokan kotak nomor ke pemain. IoS = Intersection over Smaller, bukan IoU,
# karena kotak nomor selalu jauh lebih kecil dari kotak pemain sehingga IoU-nya
# selalu kecil walau nomor itu memang milik pemain tersebut.
NUMBER_IOS = float(os.getenv("RFDETR_NUMBER_IOS", "0.60"))

# ── Offset indeks kelas. BACA INI, salah satu di sini merusak semuanya ───────
# Kenapa berbahaya: kalau offset-nya meleset satu langkah, SEMUA kelas bergeser.
# ball terbaca ball-in-basket, player terbaca player-in-possession, dan
# seterusnya. Model tetap termuat, inferensi tetap jalan, tidak ada pesan
# kesalahan, tapi seluruh statistik salah. Karena itu offset tidak ditebak di
# dalam kode, melainkan disebut eksplisit DAN diperiksa sendiri saat jalan.
#
# BAWAANNYA 0, DIUKUR PADA MODEL YANG SEBENARNYA 9 Oktober 2026.
# best-object-basket-detect-v1-DETR.pt dijalankan pada 4 frame
# test-Quarter-1_UNESA-UBAYA.mp4, id mentah yang keluar: 0, 2, 3, 4, 8, 9.
#   - id 0 muncul, dan dengan offset 1 itu akan jatuh di luar jangkauan.
#     Jadi offsetnya tidak mungkin 1.
#   - Tafsiran offset 0 masuk akal untuk frame basket: 33 player, 6 number,
#     5 referee, 1 ball, 1 rim, 2 player-in-possession.
#   - Tafsiran offset 1 berbunyi 33 "number" dan 5 "player-shot-block",
#     jelas keliru.
#
# CATATAN KOREKSI: sebelumnya bawaannya 1, disimpulkan dari bobot starter COCO
# rf-detr-nano yang mengembalikan person=1 dan sports ball=37. Kesimpulan itu
# TIDAK berlaku untuk checkpoint hasil latihan sendiri, dan sudah dicabut.
#
# Jalur paling aman tetap dipakai lebih dulu: kalau checkpoint membawa daftar
# class_names sendiri (checkpoint ini membawanya), daftar itulah yang dipakai,
# bukan RFDETR_CLASS_NAMES di atas. Lihat load_model().
CLASS_OFFSET = int(os.getenv("RFDETR_CLASS_OFFSET", "0"))


def _bbox_center(x1: float, y1: float, x2: float, y2: float) -> list:
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def ios(kecil: list, besar: list) -> float:
    """Intersection over Smaller: luas tumpang tindih dibagi luas kotak terkecil.

    Dipakai mencocokkan kotak nomor ke kotak pemain. Dengan IoU, kotak nomor
    yang benar benar berada di dada pemain tetap memberi nilai kecil karena
    penyebutnya ikut menghitung seluruh badan pemain.
    """
    ax1, ay1, ax2, ay2 = kecil
    bx1, by1, bx2, by2 = besar
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    tumpang = (ix2 - ix1) * (iy2 - iy1)
    luas_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    luas_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    return float(tumpang / min(luas_a, luas_b))


class RFDetrDetector:
    """Pembungkus RF-DETR yang keluarannya sebentuk dengan BasketballDetector."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = CONF,
        variant: str = VARIANT,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or os.path.join(MODELS_DIR, WEIGHTS)
        self.conf_threshold = conf_threshold
        self.variant = variant
        self.device = device
        self.model = None
        self._class_names = list(RFDETR_CLASS_NAMES)
        self.class_offset = CLASS_OFFSET
        # Keadaan pemeriksaan diri untuk offset indeks kelas.
        self._periksa_sisa: int = int(os.getenv("RFDETR_PERIKSA_FRAME", "20"))
        self._idx_lo = None
        self._idx_hi = None
        self._tak_dikenal_total: int = 0

    # ── daur hidup ────────────────────────────────────────────────────────────

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> None:
        """Muat bobot RF-DETR. Memberi pesan yang jelas kalau belum ada, karena
        sampai modelnya dilatih keadaan normalnya memang belum ada."""
        try:
            import rfdetr
        except ImportError as e:
            raise ImportError(
                "Paket rfdetr belum terpasang. Jalankan: pip install rfdetr"
            ) from e

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Bobot RF-DETR tidak ada di {self.model_path}. "
                "Model 10 kelas belum dilatih. Resepnya ada di "
                "basketball-roboflow/catatan-model-baru-dan-training.txt bagian DUA. "
                "Sementara ini pakai DETECTOR_METHOD=yolo."
            )

        # ── Jalur utama: biarkan rfdetr menebak kelasnya dari checkpoint ──
        # from_checkpoint membaca kunci model_name yang ditulis RFDETR.train(),
        # jadi varian yang dipakai saat inferensi DIJAMIN sama dengan varian
        # saat training. Kalau varian dipaksa dari env dan ternyata beda, model
        # bisa termuat tapi hasilnya ngawur tanpa pesan kesalahan apa pun.
        if hasattr(rfdetr.RFDETR, "from_checkpoint"):
            try:
                kwargs = {"trust_checkpoint": TRUST}
                if self.device:
                    kwargs["device"] = self.device
                self.model = rfdetr.RFDETR.from_checkpoint(
                    self.model_path, **kwargs)
                logger.info(
                    "RF-DETR dimuat lewat from_checkpoint dari %s (varian %s)",
                    self.model_path, type(self.model).__name__)
                self._ambil_nama_kelas_dari_model()
                self._percepat()
                return
            except Exception as e:
                logger.warning(
                    "from_checkpoint gagal (%s); coba varian dari "
                    "RFDETR_VARIANT=%s", e, self.variant)

        # ── Cadangan: varian ditentukan dari env ──
        kelas = {
            "nano":   "RFDETRNano",
            "small":  "RFDETRSmall",
            "medium": "RFDETRMedium",
            "large":  "RFDETRLarge",
            "base":   "RFDETRBase",      # USANG, masih diterima
        }.get(self.variant)
        if kelas is None:
            raise ValueError(
                f"RFDETR_VARIANT={self.variant!r} tidak dikenal. "
                "Pilih nano, small, medium, atau large."
            )
        if self.variant == "base":
            logger.warning(
                "RFDETR_VARIANT=base memakai RFDETRBase yang sudah usang sejak "
                "rfdetr v1.7.0 dan akan dihapus di v2.0.0. Pakai medium.")
        Pembuat = getattr(rfdetr, kelas)
        # num_classes WAJIB disebut di jalur ini, karena bawaan rfdetr 90 (COCO)
        # sementara model kita 10 kelas. from_checkpoint membacanya sendiri dari
        # checkpoint, jadi hanya jalur cadangan yang perlu ini.
        self.model = Pembuat(
            pretrain_weights=self.model_path,
            num_classes=len(RFDETR_CLASS_NAMES),
        )
        logger.info("RF-DETR %s dimuat dari %s", self.variant, self.model_path)
        self._ambil_nama_kelas_dari_model()
        self._percepat()

    def _ambil_nama_kelas_dari_model(self) -> None:
        """Pakai daftar kelas yang DIBAWA checkpoint, bukan daftar di berkas ini.

        Checkpoint RF-DETR menyimpan class_names hasil training. Memakainya
        menghilangkan satu kelas kesalahan sepenuhnya: kalau urutan kelas di
        dataset berubah suatu saat, daftar di modul ini akan basi dan semua
        label bergeser tanpa pesan apa pun. Daftar bawaan RFDETR_CLASS_NAMES
        hanya dipakai kalau checkpoint tidak membawanya.
        """
        nama = None
        for sumber in (self.model, getattr(self.model, "model", None)):
            kandidat = getattr(sumber, "class_names", None)
            if kandidat:
                nama = list(kandidat)
                break
        if not nama:
            logger.warning(
                "Checkpoint tidak membawa class_names; memakai daftar bawaan "
                "RFDETR_CLASS_NAMES. Pastikan urutannya sama dengan saat latih.")
            return

        if nama != self._class_names:
            logger.warning(
                "Daftar kelas dari checkpoint BERBEDA dari RFDETR_CLASS_NAMES "
                "di modul ini.\n  checkpoint: %s\n  modul     : %s\n"
                "Yang dipakai adalah milik checkpoint.", nama, self._class_names)
        self._class_names = nama
        logger.info("RF-DETR memakai %d kelas dari checkpoint: %s",
                    len(nama), nama)

    def _percepat(self) -> None:
        """Nyalakan jalur inferensi cepat rfdetr kalau sedang di GPU.

        Tanpa ini rfdetr mencatat "Model is not optimized for inference".
        Hanya di CUDA: fp16 di CPU justru lebih lambat, dan kompilasi memakan
        waktu di awal tanpa imbalan.
        """
        if not RFDETR_OPTIMIZE or self.model is None:
            return
        try:
            import torch
            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if not str(target).startswith("cuda"):
                logger.info("RF-DETR di %s, percepatan fp16 dilewati "
                            "(hanya berguna di CUDA)", target)
                return

            # inference() memakai model.resolution, dan nilainya 560 yang TIDAK
            # habis dibagi 32, sehingga optimasi gagal dengan
            #   "Backbone requires input shape to be divisible by 32"
            # Disamakan dulu dengan RFDETR_SHAPE yang memang dipakai saat
            # predict, supaya model yang dikompilasi dan model yang dipanggil
            # memakai ukuran yang sama persis.
            dalam = getattr(self.model, "model", None)
            if dalam is not None and getattr(dalam, "resolution", None) != RFDETR_SHAPE:
                logger.info("RF-DETR resolution %s -> %d supaya bisa "
                            "dioptimalkan dan sama dengan RFDETR_SHAPE",
                            getattr(dalam, "resolution", None), RFDETR_SHAPE)
                dalam.resolution = RFDETR_SHAPE
                self.model._optimized_resolution = None

            self.model.inference(compile=True, batch_size=1, dtype=torch.float16)
            # Terukur di RTX 3060 Ti, 10 frame test-Quarter-1_UNESA-UBAYA:
            # 21,5 ms -> 9,6 ms per frame (2,23x), deteksi 151 -> 150.
            logger.info("RF-DETR dioptimalkan untuk inferensi (fp16 + compile)")
        except Exception as e:
            # Gagal percepatan TIDAK boleh menggagalkan pemuatan. Model tetap
            # bisa dipakai, hanya lebih lambat.
            logger.warning("Percepatan RF-DETR gagal (%s); jalan tanpa itu", e)

    def warmup(self, height: int = 720, width: int = 1280) -> None:
        if self.model is None:
            return
        try:
            self.detect(np.zeros((height, width, 3), dtype=np.uint8))
        except Exception as e:
            logger.debug("Pemanasan RF-DETR dilewati (%s)", e)

    # ── inferensi ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> dict:
        """Satu frame BGR masuk, dict sebentuk BasketballDetector.detect() keluar."""
        out = self._empty_result()
        out.pop("raw_results", None)

        if frame is None or frame.size == 0 or self.model is None:
            return out

        # rfdetr menunggu RGB. .copy() wajib: irisan langkah negatif tidak bisa
        # dijadikan tensor torch.
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        try:
            # shape WAJIB: resolusi latih 560 tidak habis dibagi 32, dan rfdetr
            # 1.11.2 menolak menjalankannya tanpa shape eksplisit. Tanpa baris
            # ini setiap frame gagal. Lihat RFDETR_SHAPE di atas.
            det = self.model.predict(
                rgb, threshold=self.conf_threshold,
                shape=(RFDETR_SHAPE, RFDETR_SHAPE),
            )
        except Exception as e:
            # Dicatat sekali di awal lalu jarang, supaya kegagalan menetap tidak
            # membanjiri log 60 kali sedetik tapi tetap terlihat.
            self._gagal_total = getattr(self, "_gagal_total", 0) + 1
            if self._gagal_total <= 3 or self._gagal_total % 500 == 0:
                logger.warning("Inferensi RF-DETR gagal (%s) [kejadian ke-%d]",
                               e, self._gagal_total)
            return out

        kotak, keyakinan, kelas = self._bongkar(det)
        if kotak is None:
            return out

        self._periksa_indeks(kelas)
        tak_dikenal = 0

        for (x1, y1, x2, y2), conf, ci in zip(kotak, keyakinan, kelas):
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            conf = float(conf)
            idx = int(ci) - self.class_offset
            nama = (self._class_names[idx]
                    if 0 <= idx < len(self._class_names)
                    else f"class_{ci}")
            if not (0 <= idx < len(self._class_names)):
                tak_dikenal += 1

            if nama in PLAYER_CLASS_TO_ACTION:
                out["players"].append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class": "player",          # tetap "player" agar pelacak tidak berubah
                    "rf_class": nama,           # kelas aslinya, untuk yang butuh
                    "action": PLAYER_CLASS_TO_ACTION[nama],
                })
            elif nama == "ball":
                kandidat = {"bbox": bbox, "confidence": conf,
                            "center": _bbox_center(*bbox)}
                if out["ball"] is None or conf > out["ball"]["confidence"]:
                    out["ball"] = kandidat
            elif nama == "ball-in-basket":
                out["balls_in_basket"].append({
                    "bbox": bbox, "confidence": conf,
                    "center": _bbox_center(*bbox)})
            elif nama == "number":
                out["numbers"].append({"bbox": bbox, "confidence": conf})
            elif nama == "referee":
                out["referees"].append({"bbox": bbox, "confidence": conf,
                                        "class": "referee"})
            elif nama == "rim":
                pos = {"bbox": bbox, "confidence": conf,
                       "center": _bbox_center(*bbox)}
                out["rims"].append(pos)
                out["hoops"].append(pos)    # mesin kejadian lama membaca "hoops"

        if tak_dikenal:
            # Kelas di luar daftar DIBUANG, dan itu harus terlihat. Tanpa
            # peringatan ini, offset yang salah cuma tampak seperti "modelnya
            # kurang bagus", bukan seperti kesalahan konfigurasi.
            self._tak_dikenal_total += tak_dikenal
            if self._tak_dikenal_total <= 3 or self._tak_dikenal_total % 500 == 0:
                logger.warning(
                    "RF-DETR: %d kotak dibuang karena indeks kelasnya di luar "
                    "0..%d setelah offset %d (total %d). Periksa "
                    "RFDETR_CLASS_OFFSET.",
                    tak_dikenal, len(self._class_names) - 1, self.class_offset,
                    self._tak_dikenal_total)

        self._cocokkan_nomor(out)
        return out

    def _periksa_indeks(self, kelas) -> None:
        """Periksa sendiri apakah offset-nya masuk akal, sekali saja di awal.

        Dipanggil tiap frame tapi hanya bekerja sampai PERIKSA_FRAME frame
        pertama, supaya tidak membebani. Yang dicari: indeks mentah yang
        konsisten di luar rentang yang diharapkan, karena itu tanda offset
        salah, bukan tanda model jelek.
        """
        if self._periksa_sisa <= 0:
            return
        self._periksa_sisa -= 1
        lo, hi = int(np.min(kelas)), int(np.max(kelas))
        self._idx_lo = lo if self._idx_lo is None else min(self._idx_lo, lo)
        self._idx_hi = hi if self._idx_hi is None else max(self._idx_hi, hi)
        if self._periksa_sisa > 0:
            return

        n = len(self._class_names)
        harap_lo, harap_hi = self.class_offset, self.class_offset + n - 1
        logger.info(
            "RF-DETR pemeriksaan indeks kelas: terlihat %d sampai %d, "
            "diharapkan %d sampai %d (offset %d, %d kelas)",
            self._idx_lo, self._idx_hi, harap_lo, harap_hi, self.class_offset, n)
        if self._idx_lo < harap_lo or self._idx_hi > harap_hi:
            logger.error(
                "RF-DETR: indeks kelas DI LUAR rentang yang diharapkan. "
                "Terlihat %d sampai %d, diharapkan %d sampai %d. "
                "Hampir pasti RFDETR_CLASS_OFFSET salah (sekarang %d). "
                "Kalau indeks terkecil yang terlihat %d, coba "
                "RFDETR_CLASS_OFFSET=%d. JANGAN pakai hasilnya sebelum ini "
                "beres, karena semua kelas bisa bergeser satu langkah.",
                self._idx_lo, self._idx_hi, harap_lo, harap_hi,
                self.class_offset, self._idx_lo, self._idx_lo)

    @staticmethod
    def _bongkar(det):
        """Ambil (kotak, keyakinan, kelas) dari keluaran rfdetr.

        rfdetr mengembalikan supervision.Detections. Dibuat longgar supaya tidak
        pecah kalau bentuk keluarannya berubah di versi berikutnya.
        """
        try:
            kotak = np.asarray(det.xyxy, dtype=np.float64)
            conf = np.asarray(
                det.confidence if det.confidence is not None
                else np.ones(len(kotak)), dtype=np.float64)
            kelas = np.asarray(
                det.class_id if det.class_id is not None
                else np.zeros(len(kotak)), dtype=np.int64)
        except Exception as e:
            logger.warning("Bentuk keluaran RF-DETR tidak dikenali (%s)", e)
            return None, None, None
        if len(kotak) == 0:
            return None, None, None
        return kotak, conf, kelas

    @staticmethod
    def _cocokkan_nomor(out: dict) -> None:
        """Pasangkan tiap kotak nomor ke pemain yang paling menampungnya.

        Satu pemain hanya boleh dapat satu nomor, yaitu yang IoS-nya tertinggi.
        Pemain yang nomornya ketemu diberi kunci "number_bbox", jadi jersey_ocr
        bisa memotong tepat di kotak itu daripada menebak posisi dada.
        """
        pemain = out.get("players") or []
        nomor = out.get("numbers") or []
        if not pemain or not nomor:
            return
        for n in nomor:
            terbaik, skor_terbaik = None, 0.0
            for p in pemain:
                s = ios(n["bbox"], p["bbox"])
                if s > skor_terbaik:
                    terbaik, skor_terbaik = p, s
            if terbaik is not None and skor_terbaik >= NUMBER_IOS:
                lama = terbaik.get("_number_ios", 0.0)
                if skor_terbaik > lama:
                    terbaik["number_bbox"] = n["bbox"]
                    terbaik["_number_ios"] = skor_terbaik
                    n["track_match"] = True

    @staticmethod
    def _empty_result() -> dict:
        return {
            "players": [],
            "ball": None,
            "referees": [],
            "hoops": [],
            "backboards": [],      # RF-DETR tidak punya kelas ini, selalu kosong
            "rims": [],            # BARU
            "numbers": [],         # BARU
            "balls_in_basket": [], # BARU
            "raw_results": None,
        }

    @staticmethod
    def get_ball_center(ball_bbox: list) -> list:
        return _bbox_center(*ball_bbox)


def create_rfdetr_detector(
    models_path: Optional[str] = None,
    conf_threshold: float = CONF,
) -> RFDetrDetector:
    """Pabrik, sejajar dengan create_detector di detector.py."""
    det = RFDetrDetector(
        model_path=os.path.join(models_path or MODELS_DIR, WEIGHTS),
        conf_threshold=conf_threshold,
    )
    det.load_model()
    return det


# ═══════════════════════════════════════════════════════════════════
# Penyambung deteksi ke hasil pelacakan
# ═══════════════════════════════════════════════════════════════════
#
# Pelacak membangun dict pemain yang baru dan TIDAK membawa kunci tambahan dari
# deteksi (action, rf_class, number_bbox). ByteTrack juga menyaring dan
# mengurutkan ulang deteksi, jadi tidak ada indeks yang bisa dipakai. Satu
# satunya tautan yang tersisa adalah kotaknya sendiri, karena itu dicocokkan
# lewat IoU.
#
# Fungsi ini aman dipanggil walau detektornya YOLO: kalau tidak ada kunci
# tambahan sama sekali, dia langsung kembali tanpa melakukan apa pun.

ATTACH_IOU = float(os.getenv("RFDETR_ATTACH_IOU", "0.70"))


def _iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    t = (ix2 - ix1) * (iy2 - iy1)
    la = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    lb = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return float(t / (la + lb - t))


def attach_detection_classes(detections: dict, tracked_players: list) -> int:
    """Pindahkan aksi dan kotak nomor dari deteksi ke pemain hasil pelacakan.

    Mengembalikan jumlah pemain yang berhasil disambungkan. Nol berarti
    detektornya YOLO (tidak ada kunci tambahan) atau tidak ada yang cocok.
    """
    if not detections or not tracked_players:
        return 0
    sumber = [p for p in (detections.get("players") or [])
              if "action" in p or "number_bbox" in p]
    if not sumber:
        return 0                                  # jalur YOLO, tidak ada kerja

    n = 0
    terpakai = set()
    for tp in tracked_players:
        tb = tp.get("bbox")
        if not tb:
            continue
        terbaik, skor = None, 0.0
        for i, d in enumerate(sumber):
            if i in terpakai:
                continue
            s = _iou(tb, d["bbox"])
            if s > skor:
                terbaik, skor, idx = d, s, i
        if terbaik is not None and skor >= ATTACH_IOU:
            terpakai.add(idx)
            tp["action"]   = terbaik.get("action")
            tp["rf_class"] = terbaik.get("rf_class")
            if terbaik.get("number_bbox"):
                tp["number_bbox"] = terbaik["number_bbox"]
            n += 1
    return n

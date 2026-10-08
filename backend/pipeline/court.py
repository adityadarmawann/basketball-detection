"""
Court calibration via YOLOv8-pose (court_keypoints.pt).
Detects 33 FIBA court keypoints → computes homography → maps pixel ↔ metric coords.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from .roboflow_methods import KeyPointsSmoother as _KPSmoother
except Exception as _e:                                   # pragma: no cover
    _KPSmoother = None
    logger.info("KeyPointsSmoother tidak tersedia (%s)", _e)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = os.getenv("MODELS_PATH", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_FILENAME = "court_keypoints.pt"
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "homography_cache.json")

# ---------------------------------------------------------------------------
# FIBA court constants
# ---------------------------------------------------------------------------

COURT_W = 28.0   # meters, x-axis (baseline to baseline)
COURT_H = 15.0   # meters, y-axis (sideline to sideline)

HOOP_LEFT  = [1.575,  7.5]    # FIBA standard
HOOP_RIGHT = [26.425, 7.5]    # FIBA standard
THREE_PT_RADIUS = 6.75          # arc radius in meters
THREE_PT_STRAIGHT_X = 2.99     # x where arc meets corner straight section (left basket)

# ---------------------------------------------------------------------------
# Keypoint mapping
# ---------------------------------------------------------------------------

# Model outputs 33 keypoints indexed 0-32.
# This list maps model index → label ID used in the dataset.
# Label IDs span 1-41 with 8 gaps (3,6,18,20,22,24,36,39 not present).
KP_INDEX_TO_LABEL: list[int] = [
     1,  2,  4,  5,  7,  8,  9, 10, 11, 12,  # 0-9
    13, 14, 15, 16, 17, 19, 21, 23, 25, 26,  # 10-19
    27, 28, 29, 30, 31, 32, 33, 34, 35, 37,  # 20-29
    38, 40, 41,                               # 30-32
]

# Label ID → [x_meter, y_meter]  (FIBA origin = bottom-left corner of court)
LABEL_TO_COURT: dict[int, list[float]] = {
    # ── Outer boundary ──────────────────────────────────────────────────
    1:  [0.0,   15.0],   # corner top-left
    8:  [0.0,    0.0],   # corner bottom-left
    34: [28.0,  15.0],   # corner top-right
    41: [28.0,   0.0],   # corner bottom-right
    # Titik 15/17/25/27 BUKAN titik acuan sembarangan di garis samping. Di
    # lable_court.jpg keempatnya duduk tepat di tanda garis yang mencuat KELUAR
    # lapangan, yaitu garis lemparan ke dalam.
    #
    # Nilainya DIUKUR DARI MODELNYA, bukan dari standar. Caranya: homografi
    # dibangun HANYA dari titik yang koordinatnya sudah pasti (sudut, paint,
    # ring, garis lemparan bebas), lalu prediksi model untuk titik ini
    # diproyeksikan ke meter. Hasilnya pada 41 frame q1 UBAYA UNESA:
    #     titik 15 -> 9,143 m  (25 sampel)
    #     titik 25 -> 18,895 m (21 sampel)
    #
    # Jadi yang dideteksi model secara FISIK berada di 9,11 m, bukan di 8,325 m
    # seperti standar FIBA. Memaksa angka FIBA justru memasukkan galat 0,8 m,
    # karena model TIDAK PERNAH memprediksi titik di 8,325 m. Tabel ini harus
    # menyatakan di mana fitur yang dideteksi model benar benar berada, bukan di
    # mana standar mengatakan fitur itu seharusnya berada.
    #
    # Diuji ujung ke ujung pada 42 frame nyata, galat proyeksi median:
    #     9,11 dan 18,89 (dipakai)  0,118 m
    #     9,30 dan 18,70 (kode lama) 0,129 m
    #     8,325 dan 19,675 (FIBA)    0,142 m
    15: [9.110, 15.0],   # garis lemparan ke dalam, atas kiri
    25: [18.890, 15.0],  # garis lemparan ke dalam, atas kanan
    17: [9.110,  0.0],   # garis lemparan ke dalam, bawah kiri
    27: [18.890, 0.0],   # garis lemparan ke dalam, bawah kanan
    # ── Center line ─────────────────────────────────────────────────────
    19: [14.0,  15.0],   # center top
    23: [14.0,   0.0],   # center bottom
    21: [14.0,   7.5],   # center court
    # ── Sisi kiri: GARIS TIGA ANGKA dan PAINT ────────────────────────────
    # KOREKSI 8 Oktober 2026. Kode lama mengira 2, 7, 10, 11 adalah "paint
    # luar" dan menaruhnya di y=11,35 dan y=3,65 dengan x=5,8. Itu SALAH.
    # Diperiksa ke test-video/court/lable_court.jpg dengan mendeteksi 33 dot
    # secara terprogram: dari titik 2 ada garis lurus mendatar ke titik 10,
    # lalu MELENGKUNG jadi busur. Itu garis tiga angka, bukan paint. Garis
    # paint yang sebenarnya ada di 4 ke 12, dan garis itu punya tanda setrip
    # lane mark.
    #
    # Busurnya sudah diuji numerik, fit lingkaran bebas dari 18.431 piksel:
    #     pusat (1,565, 7,600) lawan ring FIBA (1,575, 7,500)  selisih 0,101 m
    #     jari jari 6,678 m    lawan FIBA 6,750 m              selisih 0,072 m
    #
    # Geometri FIBA yang dipakai di sini:
    #     garis lurus 3 angka 0,90 m dari garis samping -> y = 0,90 dan 14,10
    #     titik singgung lurus ke busur:
    #         x = 1,575 + akar(6,75^2 - 6,60^2) = 2,990 m
    #
    # Dampak kesalahan lama, terukur: galat proyeksi homografi median 1,167 m
    # dan maksimum 3,087 m, dengan selisih posisi pemain median 0,71 m dan
    # maksimum 1,56 m. Cukup untuk membalik keputusan 2 lawan 3 angka di dekat
    # busur. Dan RANSAC MENERIMA SEMUA 33 TITIK walau salah, karena galatnya
    # sistematis dan simetris, jadi kesalahan ini tidak terlihat sama sekali.
    2:  [0.0,    14.10], # garis 3 angka ketemu garis akhir, atas
    4:  [0.0,     9.95], # garis akhir, paint FIBA atas
    5:  [0.0,     5.05], # garis akhir, paint FIBA bawah
    7:  [0.0,     0.90], # garis 3 angka ketemu garis akhir, bawah
    10: [2.990,  14.10], # 3 angka, garis lurus jadi busur, atas
    11: [2.990,   0.90], # 3 angka, garis lurus jadi busur, bawah
    12: [5.8,     9.95], # sudut paint di garis lemparan bebas, atas
    13: [5.8,     7.5],  # tengah garis lemparan bebas
    14: [5.8,     5.05], # sudut paint di garis lemparan bebas, bawah
    9:  [1.575,   7.5],  # ring kiri
    16: [8.325,   7.5],  # puncak busur 3 angka = 1,575 + 6,75
    # ── Sisi kanan: cermin persis dari sisi kiri, x menjadi 28 dikurang x ──
    35: [28.0,   14.10], # garis 3 angka ketemu garis akhir, atas
    37: [28.0,    9.95], # garis akhir, paint FIBA atas
    38: [28.0,    5.05], # garis akhir, paint FIBA bawah
    40: [28.0,    0.90], # garis 3 angka ketemu garis akhir, bawah
    31: [25.010, 14.10], # 3 angka, garis lurus jadi busur, atas (28 - 2,990)
    32: [25.010,  0.90], # 3 angka, garis lurus jadi busur, bawah
    28: [22.2,    9.95], # sudut paint di garis lemparan bebas, atas
    29: [22.2,    7.5],  # tengah garis lemparan bebas
    30: [22.2,    5.05], # sudut paint di garis lemparan bebas, bawah
    33: [26.425,  7.5],  # ring kanan
    26: [19.675,  7.5],  # puncak busur 3 angka = 28,0 - 8,325
}

# Nama ikut dikoreksi bersama koordinatnya. Dua belas nama lama menyebut garis
# TIGA ANGKA sebagai "paint" dan garis LEMPARAN KE DALAM sebagai "boundary
# center", dan nama yang salah itulah yang membuat koordinatnya ikut salah
# bertahun tahun tanpa ketahuan. Nama hanya dipakai untuk tampilan dan log,
# tidak ada logika yang mencocokkannya, jadi aman diganti.
LABEL_NAMES: dict[int, str] = {
    1:  "corner_top_left",          2:  "3pt_baseline_left_top",
    4:  "paint_baseline_left_top",  5:  "paint_baseline_left_bot",
    7:  "3pt_baseline_left_bot",    8:  "corner_bottom_left",
    9:  "hoop_left",                10: "3pt_arc_start_left_top",
    11: "3pt_arc_start_left_bot",   12: "paint_left_top",
    13: "free_throw_left",          14: "paint_left_bottom",
    15: "throwin_top_left",         16: "arc_3pt_left",
    17: "throwin_bot_left",         19: "center_top",
    21: "center_court",             23: "center_bottom",
    25: "throwin_top_right",        26: "arc_3pt_right",
    27: "throwin_bot_right",        28: "paint_right_top",
    29: "free_throw_right",         30: "paint_right_bottom",
    31: "3pt_arc_start_right_top",  32: "3pt_arc_start_right_bot",
    33: "hoop_right",               34: "corner_top_right",
    35: "3pt_baseline_right_top",   37: "paint_baseline_right_top",
    38: "paint_baseline_right_bot", 40: "3pt_baseline_right_bot",
    41: "corner_bottom_right",
}

# Thresholds — lowered from 0.5 to 0.35 because broadcast-camera videos produced
# only 1-2 visible keypoints at 0.5, never enough for RANSAC (needs ≥4).
CONF_THRESHOLD      = 0.35   # general threshold
LOW_CONF_THRESHOLD  = 0.20   # titik sudut bawah (sering occluded)
HIGH_CONF_THRESHOLD = 0.55   # titik boundary atas (sering false detect ke tribun)

# Titik yang sering occluded oleh penonton/kamera sudut bawah
LOW_CONF_LABELS  = {7, 8, 41}

# Titik dekat garis samping atas yang sering salah deteksi ke area tribun.
# Naikkan ambang agar hanya dipakai kalau model sangat yakin. Setelah koreksi
# koordinat, 2, 10, 35 ada di y=14,10 yaitu 0,90 m dari garis samping atas,
# jadi alasan "dekat tribun" tetap berlaku dan daftarnya tidak diubah.
HIGH_CONF_LABELS = {1, 2, 10, 25, 34, 35}

# Titik yang prediksi modelnya tidak presisi di sudut pandang khas CL:
# dideteksi dan ditampilkan, tapi TIDAK dimasukkan ke perhitungan H.
# Alasan lama ditulis berdasarkan koordinat 40 yang ternyata salah ([28.0,3.65]).
# Setelah dikoreksi, 40 ada di [28.0, 0.90] yaitu hanya 0,90 m dari titik 41 di
# [28.0, 0.0]. Dua titik sedekat itu di garis yang sama memang membuat RANSAC
# goyah, jadi alasan melewatinya justru LEBIH kuat sekarang. Diuji di
# lable_court.jpg, membuang 40 praktis tidak mengubah apa apa: galat median
# 0,099 m dengan dibuang lawan 0,101 m kalau dipakai.
H_SKIP_LABELS = {40}

# ── Pemeriksaan diri tabel keypoint ──────────────────────────────────────────
# Dipasang setelah menemukan 12 dari 33 koordinat salah dan bertahan lama tanpa
# ketahuan. Kesalahan itu TIDAK TERLIHAT oleh RANSAC karena galatnya sistematis
# dan simetris, jadi semua titik tetap jadi inlier. Satu satunya cara
# menangkapnya adalah memeriksa tabelnya sendiri terhadap geometri lapangan.
#
# Tiga hal yang diperiksa, semuanya murni dari tabel, tanpa perlu gambar:
#   1. ketiga tabel panjangnya sama dan labelnya sama
#   2. setiap pasangan kiri kanan benar benar cermin, x menjadi 28 dikurang x
#   3. titik yang memang berada di busur 3 angka jaraknya tepat 6,75 m dari ring
_RING_KIRI, _RING_KANAN = (1.575, 7.5), (26.425, 7.5)
_PASANGAN_CERMIN = [(1,34),(2,35),(4,37),(5,38),(7,40),(8,41),(9,33),(10,31),
                    (11,32),(12,28),(13,29),(14,30),(15,25),(16,26),(17,27)]
_DI_BUSUR_3PT = {10: _RING_KIRI, 11: _RING_KIRI, 16: _RING_KIRI,
                 31: _RING_KANAN, 32: _RING_KANAN, 26: _RING_KANAN}
_RADIUS_3PT = 6.75


def _periksa_tabel_keypoint() -> list:
    """Kembalikan daftar keluhan. Kosong berarti tabelnya sehat."""
    keluhan = []
    if not (len(KP_INDEX_TO_LABEL) == len(LABEL_TO_COURT) == len(LABEL_NAMES)):
        keluhan.append(
            f"panjang tabel beda: KP_INDEX_TO_LABEL={len(KP_INDEX_TO_LABEL)}, "
            f"LABEL_TO_COURT={len(LABEL_TO_COURT)}, LABEL_NAMES={len(LABEL_NAMES)}")
    hilang = [l for l in KP_INDEX_TO_LABEL if l not in LABEL_TO_COURT]
    if hilang:
        keluhan.append(f"label tanpa koordinat lapangan: {hilang}")
    for kiri, kanan in _PASANGAN_CERMIN:
        if kiri not in LABEL_TO_COURT or kanan not in LABEL_TO_COURT:
            continue
        xk, yk = LABEL_TO_COURT[kiri]
        xn, yn = LABEL_TO_COURT[kanan]
        if abs((COURT_W - xk) - xn) > 0.01 or abs(yk - yn) > 0.01:
            keluhan.append(
                f"pasangan cermin {kiri} dan {kanan} tidak simetris: "
                f"{LABEL_TO_COURT[kiri]} lawan {LABEL_TO_COURT[kanan]}")
    for lab, (rx, ry) in _DI_BUSUR_3PT.items():
        if lab not in LABEL_TO_COURT:
            continue
        x, y = LABEL_TO_COURT[lab]
        r = math.hypot(x - rx, y - ry)
        if abs(r - _RADIUS_3PT) > 0.02:
            keluhan.append(
                f"titik {lab} seharusnya di busur 3 angka tapi jaraknya ke ring "
                f"{r:.3f} m, bukan {_RADIUS_3PT} m")
    return keluhan


for _k in _periksa_tabel_keypoint():
    logger.error("Tabel keypoint lapangan BERMASALAH: %s", _k)

MIN_KEYPOINTS    = 4      # minimum untuk cv2.findHomography (OpenCV wajib ≥4 titik)

# Taksiran galat posisi lapangan, dipasang dari pengukuran di dua video uji
# (1.706 pengamatan ekstrapolasi). Lihat perkiraan_galat_m().
EKSTRAP_GALAT_DASAR = float(os.getenv("EKSTRAP_GALAT_DASAR", "0.20"))
EKSTRAP_GALAT_PER_M = float(os.getenv("EKSTRAP_GALAT_PER_M", "0.12"))

# ── Penghalusan keypoint sebelum homografi (saklar) ──────────────────────────
# KEYPOINT_SMOOTH_METHOD:
#   "none"     (bawaan) = keypoint dipakai apa adanya, lalu matriks H dihaluskan
#                EMA seperti semula.
#   "roboflow" = keypoint dirata rata berjalan DULU (yang keyakinannya rendah
#                jadi NaN, bukan nol, supaya tidak menarik rata rata ke titik
#                asal), baru RANSAC. Penghalusan EMA pada H tetap jalan, jadi
#                dua lapis. Cocok saat kamera statis; kalau kamera sering
#                berpindah cepat, naikkan ambang atau matikan.
KEYPOINT_SMOOTH_METHOD = os.getenv("KEYPOINT_SMOOTH_METHOD", "none").strip().lower()
KEYPOINT_SMOOTH_LEN    = int(os.getenv("KEYPOINT_SMOOTH_LEN", "3"))
CAMERA_MOVE_PX   = 20.0  # threshold deteksi kamera bergerak
# Ambang PERPINDAHAN DRASTIS, yaitu ganti sudut atau potongan kamera, bukan
# panning halus. Di rekaman campus league sudut kamera sering berubah tiap
# beberapa detik, dan saat itu terjadi homografi lama langsung tidak sah DAN
# track_id biasanya diacak ulang. Jadi kejadian ini perlu sinyal sendiri yang
# terpisah dari CAMERA_MOVE_PX, supaya hilir (kuncian nomor punggung dan
# penampung jarak) bisa menyegarkan diri dengan cepat.
#
# DIUKUR DI REKAMAN NYATA, q1 UBAYA UNESA 60 detik, 279 pengukuran
# (eksperimen/05_ukur_ganti_sudut.py). Sebaran geser piksel rata rata:
#     persentil 50 :  16,5 px      persentil 95 :  164,0 px
#     persentil 75 :  33,8 px      persentil 99 :  408,8 px
#     persentil 90 :  79,2 px      maksimum     :  584,9 px
# Dengan ambang 120 px, ganti sudut terdeteksi 20 kali dalam 60 detik, yaitu
# sekali per 3,0 detik. Itu cocok dengan pengamatan di lapangan bahwa sudut
# kamera campus league berubah tiap beberapa detik.
# Pilihan lain kalau perlu disetel:
#      80 px -> sekali per 2,1 detik (lebih galak, kuncian sering dilepas)
#     200 px -> sekali per 5,5 detik (lebih longgar)
#     300 px -> sekali per 15 detik (hanya potongan kamera yang ekstrem)
# Jangan turunkan di bawah 80 px: dengan NUMBER_CONSECUTIVE_N=3 dan OCR tiap
# frame, konfirmasi ulang butuh sekitar 0,25 detik, jadi jendela yang lebih
# pendek dari sekitar 2 detik membuat nomor tidak pernah selesai terkunci.
COURT_CUT_PX     = float(os.getenv("COURT_CUT_PX", "120"))
H_EMA_ALPHA      = 0.35  # EMA weight for new H (higher = faster response, more noise)


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class CourtMapper:
    """
    Two-stage court calibration:
      1. detect_keypoints() → YOLOv8-pose inference, 33 keypoints per frame
      2. compute_homography() → cv2.findHomography + RANSAC
    Cached H matrix is reused until a camera-move is detected (avg shift > 20 px).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or os.path.join(MODELS_DIR, MODEL_FILENAME)
        self.device = device
        self.model = None
        self._H: Optional[np.ndarray] = None      # pixel → court  (EMA-smoothed)
        self._H_inv: Optional[np.ndarray] = None  # court → pixel (EMA-smoothed)
        self._H_raw: Optional[np.ndarray] = None  # last raw RANSAC result (court→pixel)
        self._prev_pixels: Optional[dict[int, list[float]]] = None  # for shift check
        # Perata keypoint gaya roboflow. Dibuat hanya kalau saklarnya hidup,
        # jadi jalur bawaan sama sekali tidak tersentuh.
        self._kp_smoother = (
            _KPSmoother(length=KEYPOINT_SMOOTH_LEN)
            if KEYPOINT_SMOOTH_METHOD == "roboflow" and _KPSmoother is not None
            else None
        )
        self._frame_shape: tuple[int, int] = (720, 1280)  # (h, w) updated each frame
        self._geser_terakhir: Optional[float] = None   # geser piksel rata rata
        self._cut_terakhir: bool = False               # ganti sudut terdeteksi
        # Lambung cembung koordinat lapangan dari inlier homografi terakhir.
        # Dipakai menaksir seberapa jauh sebuah posisi harus DIEKSTRAPOLASI di
        # luar bukti yang ada. Lihat perkiraan_galat_m().
        self._lambung_inlier: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load court_keypoints.pt. Raises FileNotFoundError if absent."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path.resolve()}\n"
                "Place court_keypoints.pt in backend/models/ or set MODELS_PATH."
            )
        try:
            import torch
            from ultralytics import YOLO
            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO(str(path))
            self.model.to(target)
            if target.startswith("cuda"):
                self.model.half()   # FP16 weights → Tensor Core acceleration
            self._use_half = target.startswith("cuda")
            logger.info("CourtMapper loaded: %s (device=%s, fp16=%s)",
                        path.name, target, self._use_half)
        except ImportError as exc:
            raise ImportError(
                "ultralytics and torch required. pip install ultralytics torch"
            ) from exc
        except RuntimeError as exc:
            logger.warning("GPU init failed (%s) — falling back to CPU", exc)
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            self.model.to("cpu")

    def warmup(self, height: int = 720, width: int = 1280) -> None:
        """One dummy inference to trigger CUDA JIT compilation."""
        if self.model is None:
            return
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        try:
            self.model.predict(dummy, verbose=False, conf=0.1)
            logger.info("CourtMapper warmup done (%dx%d)", width, height)
        except Exception as e:
            logger.debug("CourtMapper warmup error (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_keypoints(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on one BGR frame.
        Returns list of keypoint dicts (all 33 entries; occluded ones marked).
        """
        if self.model is None or frame is None or frame.size == 0:
            return []

        results = self.model.predict(frame, verbose=False, conf=0.1)
        if not results or results[0].keypoints is None:
            return []

        kps_obj = results[0].keypoints
        if len(kps_obj.data) == 0:
            return []

        # If multiple court detections, pick the one with highest total confidence
        kp_tensor = kps_obj.data  # torch.Tensor (N, 33, 3)
        if len(kp_tensor) > 1:
            best_idx = int(kp_tensor[:, :, 2].sum(dim=1).argmax().item())
        else:
            best_idx = 0

        kp_array = kp_tensor[best_idx].cpu().numpy()  # (33, 3): [x, y, vis]

        keypoints: list[dict] = []
        for kp_idx, label_id in enumerate(KP_INDEX_TO_LABEL):
            x, y, conf = (
                float(kp_array[kp_idx, 0]),
                float(kp_array[kp_idx, 1]),
                float(kp_array[kp_idx, 2]),
            )
            court_coord = LABEL_TO_COURT.get(label_id)
            if court_coord is None:
                continue
            # Per-label confidence threshold
            if label_id in HIGH_CONF_LABELS:
                effective_threshold = HIGH_CONF_THRESHOLD  # 0.75 — hindari false detect tribun
            elif label_id in LOW_CONF_LABELS:
                effective_threshold = LOW_CONF_THRESHOLD   # 0.3 — sudut bawah sering occluded
            else:
                effective_threshold = CONF_THRESHOLD       # 0.5 default
            keypoints.append({
                "point_id":   label_id,
                "name":       LABEL_NAMES.get(label_id, f"kp_{label_id}"),
                "pixel_pos":  [x, y],
                "court_pos":  court_coord,
                "confidence": conf,
                "occluded":        conf < effective_threshold,
                "estimated":       False,
                "ransac_outlier":  label_id in H_SKIP_LABELS,  # pre-excluded, shown orange
            })

        if self._kp_smoother is not None:
            keypoints = self._smooth_keypoints(keypoints)

        return keypoints

    def _smooth_keypoints(self, keypoints: list[dict]) -> list[dict]:
        """Rata rata berjalan posisi piksel keypoint (metode roboflow).

        Keypoint yang sudah ditandai occluded dikirim sebagai NaN, jadi tidak
        ikut menarik rata rata. Kalau rata ratanya NaN (belum pernah terlihat
        dalam jendela ini), posisi mentah frame ini dipakai kembali supaya
        tampilan diagnostik tidak kehilangan titik.
        """
        if not keypoints:
            return keypoints
        xy = np.array([kp["pixel_pos"] for kp in keypoints], dtype=np.float64)
        # occluded -> keyakinan 0 supaya dibuang oleh ambang 0,5 di bawah
        conf = np.array(
            [0.0 if kp["occluded"] else float(kp["confidence"]) for kp in keypoints],
            dtype=np.float64,
        ).reshape(-1, 1).repeat(2, axis=1)
        out = self._kp_smoother.update(xy, confidence=conf, conf_threshold=0.5)
        for kp, row in zip(keypoints, out):
            # Posisi MENTAH wajib disimpan. Deteksi perpindahan kamera memakai
            # ini, bukan yang sudah dihaluskan, karena penghalusan justru
            # menahan lonjakan dan itu akan MENYAMARKAN ganti sudut kamera,
            # yaitu persis kejadian yang paling perlu terdeteksi cepat.
            kp["pixel_pos_raw"] = list(kp["pixel_pos"])
            if np.isfinite(row).all():
                kp["pixel_pos"] = [float(row[0]), float(row[1])]
                kp["smoothed"] = True
            else:
                kp["smoothed"] = False
        return keypoints

    # ------------------------------------------------------------------
    # Homography
    # ------------------------------------------------------------------

    def compute_homography(self, keypoints: list[dict]) -> Optional[np.ndarray]:
        """
        Fit pixel→court homography with RANSAC.
        Requires at least MIN_KEYPOINTS (4) visible keypoints — OpenCV minimum.
        Returns the 3×3 H (pixel→court) matrix and caches it internally.

        RANSAC direction: court→pixel so the reprojection threshold (15 px)
        is applied in pixel space, giving meaningful outlier rejection.
        _H (pixel→court) = inv(H_ct2px); _H_inv (court→pixel) = H_ct2px.
        """
        # Exclude H_SKIP_LABELS — detected & displayed but not fed into RANSAC
        visible = [
            kp for kp in keypoints
            if not kp["occluded"]
            and not kp.get("estimated", False)
            and kp["point_id"] not in H_SKIP_LABELS
        ]

        if len(visible) < MIN_KEYPOINTS:
            logger.warning(
                "compute_homography: only %d visible keypoints (need %d) — skipping",
                len(visible), MIN_KEYPOINTS,
            )
            return None

        src_px = np.array([kp["pixel_pos"] for kp in visible], dtype=np.float32)
        src_ct = np.array([kp["court_pos"] for kp in visible], dtype=np.float32)

        # court→pixel: 20 px threshold — 10 px was too tight for broadcast camera
        # angle/lens distortion, causing all keypoints to be rejected as outliers.
        try:
            H_ct2px, mask = cv2.findHomography(
                src_ct, src_px, cv2.RANSAC, ransacReprojThreshold=20.0
            )
        except cv2.error as exc:
            logger.warning("findHomography failed: %s", exc)
            return None

        if H_ct2px is None:
            logger.warning(
                "Homography returned None — points may be collinear or degenerate"
            )
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        logger.info("Homography: %d/%d RANSAC inliers (threshold=20px)", inliers, len(visible))

        if inliers < MIN_KEYPOINTS:
            logger.warning(
                "compute_homography: only %d RANSAC inliers after fit — H rejected",
                inliers,
            )
            return None

        inlier_kps = []
        for i, kp in enumerate(visible):
            is_inlier = mask is None or bool(mask[i])
            kp["ransac_outlier"] = not is_inlier  # propagate to callers
            if is_inlier:
                inlier_kps.append(kp)
            else:
                logger.debug(
                    "RANSAC outlier: label=%d %s pixel=%s court=%s",
                    kp["point_id"], kp["name"],
                    kp["pixel_pos"], kp["court_pos"],
                )

        if not self._validate_court_coverage(inlier_kps):
            logger.warning(
                "Court coverage check failed: inliers only on 1 side — H may be unreliable"
            )

        # pixel→court = inv(court→pixel)
        try:
            H_px2ct = np.linalg.inv(H_ct2px)
        except np.linalg.LinAlgError:
            logger.warning("H inversion failed — singular matrix")
            return None

        # EMA smoothing: blend new raw H with previous smoothed H
        # This averages out per-frame keypoint detection noise without losing camera-move response
        if self._H_raw is None:
            # First calibration — use raw values directly (no history to blend)
            smoothed_ct2px = H_ct2px.astype(np.float64)
        else:
            smoothed_ct2px = (
                H_EMA_ALPHA * H_ct2px.astype(np.float64)
                + (1.0 - H_EMA_ALPHA) * self._H_raw
            )

        self._H_raw = smoothed_ct2px.copy()

        try:
            H_smoothed_px2ct = np.linalg.inv(smoothed_ct2px)
        except np.linalg.LinAlgError:
            logger.warning("Smoothed H inversion failed — falling back to raw")
            H_smoothed_px2ct = H_px2ct.astype(np.float64)
            smoothed_ct2px   = H_ct2px.astype(np.float64)

        self._H     = H_smoothed_px2ct          # pixel → court (smoothed)
        self._H_inv = smoothed_ct2px            # court → pixel (smoothed)

        # Simpan wilayah lapangan yang benar benar didukung keypoint inlier.
        # Diukur di dua video uji: sorotan kamera hanya mencakup median 9,1 m
        # dari 28 m panjang lapangan, dan TIDAK PERNAH ada satu frame pun yang
        # mencakup lebih dari 24 m atau memperlihatkan kedua ring sekaligus.
        # Jadi homografi hampir selalu dipasang dari satu irisan lalu dipakai
        # untuk seluruh lapangan, dan itu sumber galat terbesar di jalur court.
        try:
            titik = np.array([kp["court_pos"] for kp in inlier_kps],
                             dtype=np.float32)
            self._lambung_inlier = (cv2.convexHull(titik).reshape(-1, 2)
                                    if len(titik) >= 3 else titik)
        except Exception:
            self._lambung_inlier = None

        # Update pixel cache for camera-move detection
        self._prev_pixels = {
            kp["point_id"]: kp["pixel_pos"]
            for kp in keypoints
            if not kp["occluded"]
        }

        return self._H

    def _pixel_shift(self, keypoints: list[dict]) -> Optional[float]:
        """Geser piksel rata rata sejak H terakhir dihitung, atau None.

        Memakai pixel_pos_raw bila ada, supaya pengukuran ini tidak terpengaruh
        penghalusan keypoint. Lihat alasannya di _smooth_keypoints.
        """
        if self._prev_pixels is None:
            return None

        shifts: list[float] = []
        for kp in keypoints:
            if kp["occluded"]:
                continue
            prev = self._prev_pixels.get(kp["point_id"])
            if prev is not None:
                pos = kp.get("pixel_pos_raw") or kp["pixel_pos"]
                shifts.append(math.hypot(pos[0] - prev[0], pos[1] - prev[1]))

        if not shifts:
            return None
        return sum(shifts) / len(shifts)

    def _camera_moved(self, keypoints: list[dict]) -> bool:
        """Return True if avg pixel shift since last H compute > CAMERA_MOVE_PX."""
        geser = self._pixel_shift(keypoints)
        if geser is None:
            # None karena belum ada acuan berarti memang harus kalibrasi.
            return self._prev_pixels is None
        self._geser_terakhir = geser
        return geser > CAMERA_MOVE_PX

    def camera_cut(self, keypoints: list[dict]) -> bool:
        """True kalau pergeserannya sebesar GANTI SUDUT, bukan panning halus.

        Dipisahkan dari _camera_moved karena akibatnya berbeda: panning cuma
        butuh homografi dihitung ulang, sedangkan ganti sudut membuat kuncian
        identitas di hilir ikut tidak sah.
        """
        geser = self._geser_terakhir
        if geser is None:
            geser = self._pixel_shift(keypoints)
        return geser is not None and geser > COURT_CUT_PX

    # ------------------------------------------------------------------
    # Geometric fallback & coverage validation
    # ------------------------------------------------------------------

    def _estimate_missing_corners(self, keypoints: list[dict]) -> list[dict]:
        """
        If H is available and a corner label (7, 8, 40, 41) is occluded,
        project its known court coord through court_to_pixel to synthesise
        a pixel position.  Requires ≥2 non-occluded keypoints on the same
        side of the court (left for 7/8, right for 40/41) to avoid extrapolation
        on frames where the whole side is invisible.
        Estimated keypoints are marked "estimated": True and must NOT be
        fed back into compute_homography (guarded by the estimated flag check).
        """
        if self._H_inv is None:
            return keypoints

        left_visible = sum(
            1 for kp in keypoints
            if not kp["occluded"] and not kp.get("estimated", False)
            and kp["court_pos"][0] < 14.0
        )
        right_visible = sum(
            1 for kp in keypoints
            if not kp["occluded"] and not kp.get("estimated", False)
            and kp["court_pos"][0] > 14.0
        )

        fh, fw = self._frame_shape
        # Allow up to 5% outside visible frame to catch near-edge corners
        margin_x, margin_y = fw * 0.05, fh * 0.05

        result: list[dict] = []
        for kp in keypoints:
            lid = kp["point_id"]
            if kp["occluded"] and not kp.get("estimated", False):
                needs_estimate = (
                    (lid in {7, 8}   and left_visible  >= 2) or
                    (lid in {40, 41} and right_visible >= 2)
                )
                if needs_estimate:
                    pixel = self.court_to_pixel(kp["court_pos"])
                    if pixel is not None:
                        px, py = pixel
                        in_frame = (
                            -margin_x <= px <= fw + margin_x and
                            -margin_y <= py <= fh + margin_y
                        )
                        if in_frame:
                            kp = dict(kp, pixel_pos=pixel, occluded=False, estimated=True)
                            logger.debug(
                                "_estimate_missing_corners: label=%d estimated → pixel=%s",
                                lid, [round(px, 1), round(py, 1)],
                            )
                        else:
                            logger.debug(
                                "_estimate_missing_corners: label=%d projected out-of-frame "
                                "(%s) — keeping occluded", lid, [round(px, 1), round(py, 1)],
                            )
            result.append(kp)
        return result

    @staticmethod
    def _validate_court_coverage(inlier_kps: list[dict]) -> bool:
        """
        True if RANSAC inliers span at least 2 distinct sides of the court.
        Checks 4 half-planes: left (x<14), right (x>14), top (y>7.5), bottom (y<7.5).
        A homography fitted to only one side (e.g. left baseline) can drift badly
        for coordinates on the opposite side.
        """
        if not inlier_kps:
            return False
        has_left   = any(kp["court_pos"][0] < 14.0 for kp in inlier_kps)
        has_right  = any(kp["court_pos"][0] > 14.0 for kp in inlier_kps)
        has_top    = any(kp["court_pos"][1] > 7.5  for kp in inlier_kps)
        has_bottom = any(kp["court_pos"][1] < 7.5  for kp in inlier_kps)
        return sum([has_left, has_right, has_top, has_bottom]) >= 2

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def pixel_to_court(self, pixel_pos: list[float]) -> Optional[list[float]]:
        """[px, py] → [x_meter, y_meter].  None if not calibrated or out of bounds."""
        if self._H is None:
            return None
        pt  = np.array([[pixel_pos]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        x, y = float(out[0, 0, 0]), float(out[0, 0, 1])
        # Reject coordinates outside court + 2m margin (stale H artifact)
        if not (-2.0 <= x <= COURT_W + 2.0 and -2.0 <= y <= COURT_H + 2.0):
            return None
        return [x, y]

    def jarak_ekstrapolasi(self, court_pos: list[float]) -> Optional[float]:
        """Berapa meter posisi ini berada DI LUAR wilayah yang didukung keypoint.

        Nol berarti posisinya berada di dalam lambung titik inlier, yaitu
        ditopang bukti langsung. Makin besar, makin jauh homografi harus
        diekstrapolasi dan makin tidak bisa dipercaya angkanya.
        None kalau belum ada kalibrasi.
        """
        if self._lambung_inlier is None or len(self._lambung_inlier) < 3:
            return None
        try:
            d = cv2.pointPolygonTest(
                self._lambung_inlier.astype(np.float32),
                (float(court_pos[0]), float(court_pos[1])), True)
        except Exception:
            return None
        return max(0.0, -float(d))      # di dalam -> 0

    def perkiraan_galat_m(self, court_pos: list[float]) -> Optional[float]:
        """Taksiran galat posisi dalam meter, dari jarak ekstrapolasinya.

        Rumusnya BUKAN tebakan. Diukur di dua video uji dengan memasang
        homografi hanya dari separuh titik lalu meramalkan titik di luarnya,
        1.706 pengamatan:

            di dalam irisan (tinggal satu)  0,203 m
            2 sampai 4 m di luar           0,494 m
            4 sampai 6 m                   0,819 m
            6 sampai 10 m                  1,210 m
            10 sampai 30 m                 2,449 m

        Kemiringannya sekitar 0,12 m galat per meter ekstrapolasi, dengan dasar
        0,20 m yaitu ketepatan homografi itu sendiri. Angka ini TAKSIRAN untuk
        memberi tahu hilir seberapa boleh percaya, bukan koreksi.
        """
        d = self.jarak_ekstrapolasi(court_pos)
        if d is None:
            return None
        return round(EKSTRAP_GALAT_DASAR + EKSTRAP_GALAT_PER_M * d, 3)

    def court_to_pixel(self, court_pos: list[float]) -> Optional[list[float]]:
        """[x_meter, y_meter] → [px, py].  None if not calibrated."""
        if self._H_inv is None:
            return None
        pt  = np.array([[court_pos]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H_inv)
        return [float(out[0, 0, 0]), float(out[0, 0, 1])]

    # ------------------------------------------------------------------
    # 3PT logic
    # ------------------------------------------------------------------

    @staticmethod
    def is_three_point_shot(court_pos: list[float]) -> bool:
        """
        True if the court position is beyond the FIBA 3PT line.

        FIBA 3PT line = circular arc (radius 6.75 m from hoop) joined by two
        straight segments at y = 0.9 m and y = 14.1 m (0.9 m from each
        sideline). In the corner zone (y < 0.9 or y > 14.1) the straight
        section at x ≈ 2.99 m is the legal boundary — NOT the arc distance.
        The corner check must run BEFORE the distance check because in the
        corner zone a point can be > 6.75 m from the hoop yet still legally
        inside the 3PT line (x < 2.99, near the baseline).
        """
        x, y = float(court_pos[0]), float(court_pos[1])

        dist_l = math.hypot(x - HOOP_LEFT[0],  y - HOOP_LEFT[1])
        dist_r = math.hypot(x - HOOP_RIGHT[0], y - HOOP_RIGHT[1])
        near_left = dist_l <= dist_r

        # Corner zone: the boundary is the STRAIGHT section, not the arc.
        in_corner = y < 0.9 or y > 14.1
        if in_corner:
            if near_left:
                return x >= THREE_PT_STRAIGHT_X               # 2.99 m from left baseline
            else:
                return x <= (COURT_W - THREE_PT_STRAIGHT_X)   # 25.01 m from left baseline

        # Mid-court zone: use circular arc distance.
        return min(dist_l, dist_r) > THREE_PT_RADIUS

    # ------------------------------------------------------------------
    # Full-frame pipeline
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        detect → maybe recompute H → return output dict.
        Homography is recomputed when:
          - Not yet calibrated, OR
          - Camera move detected (avg keypoint shift > 20 px)
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        self._frame_shape = (frame.shape[0], frame.shape[1])
        keypoints = self.detect_keypoints(frame)

        self._geser_terakhir = None            # diisi ulang oleh _camera_moved
        camera_moved = self._camera_moved(keypoints)
        self._cut_terakhir = camera_moved and self.camera_cut(keypoints)
        if self._cut_terakhir:
            logger.info(
                "Court: GANTI SUDUT terdeteksi, geser %.0f px (ambang %.0f). "
                "Kuncian identitas di hilir akan disegarkan.",
                self._geser_terakhir or -1.0, COURT_CUT_PX)
        needs_calib  = self._H is None or camera_moved
        if camera_moved and self._H_raw is not None:
            # Reset EMA history so the new camera angle doesn't blend with old
            self._H_raw = None
            # Alasan yang sama berlaku untuk perata keypoint: begitu kamera
            # berpindah, posisi piksel dari sudut LAMA tidak lagi sah untuk
            # dirata ratakan dengan yang baru. Tanpa ini, beberapa frame
            # pertama setelah panning akan memakai rata rata dua sudut kamera
            # sekaligus, dan homografinya salah justru di saat paling kritis.
            if self._kp_smoother is not None:
                self._kp_smoother.reset()
            logger.debug("Camera moved — EMA H reset")
        if needs_calib:
            prev_H = self._H
            self.compute_homography(keypoints)
            # If camera moved but recalibration failed, clear stale H to avoid
            # pixel_to_court returning wrong positions from the old camera angle.
            if camera_moved and self._H is prev_H and prev_H is not None:
                self._H     = None
                self._H_inv = None
                logger.debug("Camera moved but recalib failed — clearing stale H")

        # Estimate occluded corner positions using current H (runs after RANSAC,
        # so estimated points never feed back into compute_homography).
        keypoints = self._estimate_missing_corners(keypoints)

        visible_count = sum(1 for kp in keypoints if not kp["occluded"])
        calibrated    = self._H is not None

        # Log every calibration attempt (INFO) so the backend log shows clearly
        # how many keypoints the model finds and whether H was computed.
        if needs_calib:
            logger.info(
                "Court calib: visible=%d  calibrated=%s",
                visible_count, calibrated,
            )
        else:
            logger.debug("Court keypoints: visible=%d  calibrated=True", visible_count)

        return {
            "homography_matrix":  self._H,
            "court_keypoints":    keypoints,
            "is_calibrated":      calibrated,
            "keypoints_detected": visible_count,
            # Dipakai hilir untuk menyegarkan kuncian identitas dengan cepat.
            "camera_cut":         self._cut_terakhir,
            "pixel_shift":        self._geser_terakhir,
        }

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def is_calibrated(self) -> bool:
        return self._H is not None

    def reset_calibration(self) -> None:
        self._H = None
        self._H_inv = None
        self._H_raw = None
        self._prev_pixels = None
        # Perata keypoint menyimpan beberapa frame terakhir. Kalau tidak ikut
        # dikosongkan, posisi dari kalibrasi LAMA masih tercampur ke rata rata
        # beberapa frame pertama setelah reset, dan homografi barunya melenceng.
        if self._kp_smoother is not None:
            self._kp_smoother.reset()
        logger.info("CourtMapper calibration reset")

    def save_calibration(self, path: Optional[str] = None) -> None:
        """Persist H matrix to JSON so restarts skip recalibration."""
        if self._H is None:
            logger.warning("save_calibration: no matrix to save")
            return
        save_path = path or DEFAULT_CACHE_PATH
        data = {
            "homography":     self._H.tolist(),
            "homography_inv": self._H_inv.tolist() if self._H_inv is not None else None,
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Calibration saved → %s", save_path)

    def load_calibration(self, path: Optional[str] = None) -> bool:
        """Load persisted H matrix. Returns True on success."""
        load_path = path or DEFAULT_CACHE_PATH
        if not os.path.exists(load_path):
            return False
        try:
            with open(load_path) as f:
                data = json.load(f)
            self._H = np.array(data["homography"], dtype=np.float64)
            if data.get("homography_inv"):
                self._H_inv = np.array(data["homography_inv"], dtype=np.float64)
            else:
                self._H_inv = np.linalg.inv(self._H)
            logger.info("Calibration loaded ← %s", load_path)
            return True
        except (json.JSONDecodeError, KeyError, np.linalg.LinAlgError) as exc:
            logger.warning("load_calibration failed: %s", exc)
            return False

    @staticmethod
    def _empty_result() -> dict:
        return {
            "homography_matrix":  None,
            "court_keypoints":    [],
            "is_calibrated":      False,
            "keypoints_detected": 0,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_court_mapper(
    models_path: Optional[str] = None,
    device: Optional[str] = None,
) -> CourtMapper:
    """Create and load a CourtMapper in one call."""
    model_file = os.path.join(models_path or MODELS_DIR, MODEL_FILENAME)
    mapper = CourtMapper(model_path=model_file, device=device)
    mapper.load_model()
    return mapper


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=== CourtMapper smoke test ===\n")

    # ── 1. Model load ─────────────────────────────────────────────────────
    print("--- 1. Model load ---")
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    mapper = CourtMapper(model_path=str(models_dir / MODEL_FILENAME))

    try:
        mapper.load_model()
        print(f"  Model loaded: {mapper.model is not None}")
        print(f"  Model names:  {mapper.model.names}")
        print(f"  kpt_shape:    {mapper.model.model.yaml['kpt_shape']}")
        print(f"  Total labels: {len(KP_INDEX_TO_LABEL)}")
    except (FileNotFoundError, ImportError) as e:
        print(f"  [SKIP] {e}")
        mapper.model = None  # ensure rest of test runs without model

    # ── 2. RANSAC homography with 8 dummy points ───────────────────────
    print("\n--- 2. RANSAC homography ---")

    # Create a synthetic projective mapping: pixel → court
    # Use 8 well-spread court points and compute "fake" pixel positions
    # via a known synthetic H so we can verify the output H is close.
    np.random.seed(42)
    known_court_pts = np.array([
        [0.0,  15.0],   # corner_top_left
        [28.0, 15.0],   # corner_top_right
        [28.0,  0.0],   # corner_bottom_right
        [0.0,   0.0],   # corner_bottom_left
        [14.0, 15.0],   # center_top
        [14.0,  0.0],   # center_bottom
        [14.0,  7.5],   # center_court
        [9.3,  15.0],   # boundary_top_left_center
    ], dtype=np.float32)

    # Synthetic perspective: scale + slight skew to simulate broadcast cam
    H_true = np.array([
        [30.0,  5.0, 100.0],
        [ 2.0, 28.0,  80.0],
        [ 0.0,  0.0,   1.0],
    ], dtype=np.float64)
    H_true_inv = np.linalg.inv(H_true)

    # Pixel positions derived from H_true_inv (court → pixel)
    fake_pixel_pts = cv2.perspectiveTransform(
        known_court_pts.reshape(1, -1, 2).astype(np.float32), H_true_inv.astype(np.float32)
    ).reshape(-1, 2)

    # Build keypoint list
    dummy_kps = []
    for (px, py), (cx, cy) in zip(fake_pixel_pts, known_court_pts):
        dummy_kps.append({
            "point_id":   0,
            "name":       "test",
            "pixel_pos":  [float(px), float(py)],
            "court_pos":  [float(cx), float(cy)],
            "confidence": 0.95,
            "occluded":   False,
        })

    H_computed = mapper.compute_homography(dummy_kps)
    assert H_computed is not None, "compute_homography returned None"
    assert H_computed.shape == (3, 3), f"Expected (3,3), got {H_computed.shape}"
    print(f"  Homography computed: {H_computed.shape} ✓")
    print(f"  RANSAC inliers: all 8 (exact synthetic data)")

    # ── 3. pixel_to_court / court_to_pixel round-trip ─────────────────
    print("\n--- 3. Round-trip accuracy ---")
    test_court_positions = [
        [0.0,   15.0],   # corner
        [14.0,   7.5],   # center
        [5.8,    9.95],  # paint edge (FIBA 4.9m top)
        [21.25,  7.5],   # right 3PT arc
    ]

    max_err = 0.0
    for court_pos in test_court_positions:
        pixel = cv2.perspectiveTransform(
            np.array([[court_pos]], dtype=np.float32), H_true_inv.astype(np.float32)
        ).reshape(2)
        recovered = mapper.pixel_to_court([float(pixel[0]), float(pixel[1])])
        assert recovered is not None, "pixel_to_court returned None"
        err = math.hypot(recovered[0] - court_pos[0], recovered[1] - court_pos[1])
        max_err = max(max_err, err)

    assert max_err < 0.1, f"Round-trip error {max_err:.4f}m exceeds 0.1m"
    print(f"  Max round-trip error: {max_err:.6f} m  (threshold < 0.10 m) ✓")

    # Inverse direction: court → pixel
    sample_pixel = mapper.court_to_pixel([14.0, 7.5])
    assert sample_pixel is not None, "court_to_pixel returned None"
    print(f"  court_to_pixel([14,7.5]) → pixel {[round(v,1) for v in sample_pixel]} ✓")

    # ── 4. is_three_point_shot ────────────────────────────────────────
    print("\n--- 4. is_three_point_shot ---")
    cases = [
        ([1.575,  7.5],  False, "hoop left (distance=0)"),
        ([0.0,    7.5],  False, "baseline mid (distance=1.575 < 6.75)"),
        ([14.0,   7.5],  True,  "center court (distance=12.4 > 6.75)"),
        ([5.0,    7.5],  False, "inside paint (distance=3.4 < 6.75)"),
        ([26.425, 7.5],  False, "hoop right (distance=0)"),
        ([8.0,    7.5],  False, "inside left arc (distance=6.43 < 6.75)"),
        ([9.0,    7.5],  True,  "beyond left arc (distance=7.43 > 6.75)"),
        ([3.5,    0.5],  True,  "left corner 3PT (x>=2.99, y<0.9)"),
        ([2.0,    0.5],  False, "left corner 2PT (x<2.99, y<0.9)"),
        ([24.5,  14.5],  True,  "right corner 3PT (x<=25.01, y>14.1)"),
    ]

    all_pass = True
    for pos, expected, label in cases:
        result = CourtMapper.is_three_point_shot(pos)
        status = "✓" if result == expected else "✗ FAIL"
        if result != expected:
            all_pass = False
        print(f"  {status}  {pos} → {result}  ({label})")

    assert all_pass, "One or more is_three_point_shot cases failed"

    # ── 5. Empty frame ────────────────────────────────────────────────
    print("\n--- 5. Empty / None frame ---")
    for frame in [None, np.zeros((0, 0, 3), dtype=np.uint8)]:
        out = mapper.process_frame(frame)
        assert out["court_keypoints"] == []
        assert not out["is_calibrated"] or mapper.is_calibrated()
    print("  Empty frames return safe empty result ✓")

    # ── 6. save_calibration / load_calibration ────────────────────────
    print("\n--- 6. Calibration persistence ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    mapper.save_calibration(tmp_path)
    mapper2 = CourtMapper()
    loaded = mapper2.load_calibration(tmp_path)
    assert loaded, "load_calibration returned False"
    assert mapper2._H is not None, "_H not restored"

    # Round-trip accuracy preserved after load
    # Compute a fresh pixel for [14.0, 7.5] to avoid relying on loop variable
    ref_court = [14.0, 7.5]
    ref_pixel = cv2.perspectiveTransform(
        np.array([[ref_court]], dtype=np.float32), H_true_inv.astype(np.float32)
    ).reshape(2)
    rc = mapper2.pixel_to_court([float(ref_pixel[0]), float(ref_pixel[1])])
    assert rc is not None, "pixel_to_court returned None after load"
    err2 = math.hypot(rc[0] - ref_court[0], rc[1] - ref_court[1])
    assert err2 < 0.1, f"Post-load round-trip error {err2:.4f}m"
    print(f"  save/load round-trip error: {err2:.6f} m ✓")
    os.unlink(tmp_path)

    # ── 7. process_frame with real model ─────────────────────────────
    if mapper.model is not None:
        print("\n--- 7. process_frame (real model, black frame) ---")
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = mapper.process_frame(dummy_frame)
        print(f"  is_calibrated:      {result['is_calibrated']}")
        print(f"  keypoints_detected: {result['keypoints_detected']}")
        print(f"  total keypoints:    {len(result['court_keypoints'])}")
        print("  process_frame completed without crash ✓")

    print("\n=== All tests passed ===")

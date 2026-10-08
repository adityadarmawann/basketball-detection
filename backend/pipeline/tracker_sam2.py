"""
Pelacak berbasis SAM2, alternatif ByteTrack. BUKAN bawaan.

Saklarnya TRACKER_METHOD=bytetrack|sam2. Bawaannya bytetrack.

Kenapa ada: ByteTrack hanya bisa menukar satu hal dengan hal lain. Setelan
longgar memberi track panjang tapi satu track_id menempel ke banyak pemain
(terukur 2,38 nomor punggung per track). Setelan ketat yang dipakai sekarang
memberi track murni tapi pendek (umur median 1,2 detik, 5.295 track_id dalam
17,5 menit). SAM2 melacak BENTUK MASKER, bukan tumpang tindih kotak, jadi bisa
memberi track yang panjang sekaligus murni. Terukur: 8 dari 8 objek bertahan
108 frame tanpa satu pun hilang.

Harganya, diukur di RTX 3060 Ti dengan autocast bfloat16:
    186 ms per frame, lawan 2 ms ByteTrack
    memori GPU puncak 1,11 GB
    resolusi masukan TERKUNCI 1024, modelnya menolak ukuran lain
Jadi ini untuk pemrosesan luring saat butuh ketelitian, bukan untuk siaran.

Tiga hal yang WAJIB diperhatikan dan sudah ditangani di sini:

1. SAM2 tidak membuat objek baru sendiri. Objek harus DIDAFTARKAN dari kotak
   detektor. Jadi pemain yang masuk layar setelah pendaftaran tidak terlacak
   sampai pendaftaran berikutnya.

2. Pendaftaran JANGAN dilakukan tepat di frame ganti sudut kamera. Di frame itu
   gambar paling kabur dan kamera sedang bergerak, sehingga detektor hanya
   menemukan sedikit pemain dan SAM2 akan memegang sedikit objek saja sampai
   ganti sudut berikutnya. Sudah terbukti merusak: percobaan yang mendaftar di
   frame ganti sudut hanya memegang 2,4 pemain per frame, padahal detektor
   menemukan 8,8. Karena itu pendaftaran DITUNDA SAM2_REGISTER_DELAY frame.

3. Di rekaman campus league sudut kamera berganti sekali per tiga detik, jadi
   umur track praktis dibatasi jarak antar perpindahan, bukan oleh SAM2. Yang
   menyatukan identitas antar sudut tetap nomor punggung.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL        = os.getenv("SAM2_MODEL", "sam2_t.pt")
MAX_OBJ      = int(os.getenv("SAM2_MAX_OBJ", "16"))
IMGSZ        = int(os.getenv("SAM2_IMGSZ", "1024"))   # terkunci, jangan diubah
DTYPE        = os.getenv("SAM2_DTYPE", "bfloat16")    # 4 kali lebih cepat dari fp32
REG_DELAY    = int(os.getenv("SAM2_REGISTER_DELAY", "3"))
# Daftar ulang kalau objek yang terpegang turun di bawah pecahan ini dari jumlah
# yang terdeteksi, selama beberapa frame berturut turut. Menangani pemain yang
# baru masuk layar tanpa menunggu ganti sudut berikutnya.
TOPUP_FRAC   = float(os.getenv("SAM2_TOPUP_FRAC", "0.6"))
TOPUP_N      = int(os.getenv("SAM2_TOPUP_FRAMES", "5"))
MIN_MASK_PX  = int(os.getenv("SAM2_MIN_MASK_PX", "50"))


class SAM2PlayerTracker:
    """Keluarannya sebentuk dengan PlayerTracker.update(), jadi pipeline di
    hilirnya tidak perlu tahu pelacak mana yang dipakai."""

    def __init__(self, frame_rate: int = 25) -> None:
        self.frame_rate = frame_rate
        self._pred = None
        self._torch = None
        self._siap = False
        self._peta: dict = {}          # indeks objek SAM2 -> track_id global
        self._id_berikut = 1
        self._tunda = 0
        self._kurang = 0
        self._n_pegang = 0
        # Dipakai video_processor dan jersey_ocr, harus ada agar antarmukanya sama.
        self.track_jersey_map: dict = {}
        self._newly_inherited: dict = {}

    # ── daur hidup ────────────────────────────────────────────────────────────

    def is_initialized(self) -> bool:
        return self._pred is not None

    def _buat(self):
        from ultralytics.models.sam import SAM2DynamicInteractivePredictor
        from ultralytics.utils.checks import check_imgsz
        import torch
        self._torch = torch
        p = SAM2DynamicInteractivePredictor(
            overrides=dict(conf=0.25, task="segment", mode="predict",
                           imgsz=IMGSZ, model=MODEL, save=False, verbose=False),
            max_obj_num=MAX_OBJ)
        # setup_source biasanya yang menetapkan imgsz; karena inference dipanggil
        # langsung, imgsz harus disetel sendiri.
        p.setup_model(model=None, verbose=False)
        p.imgsz = check_imgsz(p.args.imgsz, stride=32, min_dim=2)
        return p

    def _autocast(self):
        import torch
        if DTYPE in ("bfloat16", "float16") and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=getattr(torch, DTYPE))
        from contextlib import nullcontext
        return nullcontext()

    def _siapkan(self, frame):
        # Predictor ultralytics menunggu tensor hasil prapemrosesan, dan memakai
        # self.batch[1][0] untuk mengetahui ukuran gambar aslinya.
        self._pred.batch = [None, [frame]]
        return self._pred.preprocess([frame])

    # ── pendaftaran ulang ─────────────────────────────────────────────────────

    def on_camera_cut(self) -> None:
        """Sudut kamera berganti: jadwalkan pendaftaran ulang, TAPI tunda dulu
        sampai tampilan tenang. Lihat alasannya di docstring modul."""
        self._tunda = max(1, REG_DELAY)
        self._siap = False

    def _daftar(self, frame, kotak: list) -> None:
        if not kotak:
            return
        if self._pred is None:
            self._pred = self._buat()
        kotak = kotak[:MAX_OBJ]
        idx = list(range(len(kotak)))
        try:
            with self._autocast():
                self._pred.inference(self._siapkan(frame), bboxes=kotak,
                                     obj_ids=idx, update_memory=True)
        except Exception as e:
            logger.warning("SAM2 gagal mendaftarkan objek (%s)", e)
            self._siap = False
            return
        self._peta = {i: self._id_berikut + i for i in idx}
        self._id_berikut += len(idx)
        self._siap = True
        self._kurang = 0
        logger.debug("SAM2: %d objek didaftarkan", len(idx))

    # ── antarmuka utama ───────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, detections_dict: dict) -> dict:
        """Satu frame BGR plus keluaran detektor, kembalikan dict sebentuk
        PlayerTracker.update()."""
        kosong = {"tracked_players": [], "tracked_ball": None,
                  "tracked_referees": []}
        if frame is None or frame.size == 0:
            return kosong

        pemain = (detections_dict or {}).get("players") or []
        kotak = [p["bbox"] for p in pemain]

        # jadwal pendaftaran
        perlu = False
        if self._tunda > 0:
            self._tunda -= 1
            perlu = self._tunda == 0
        elif not self._siap:
            perlu = True
        elif kotak and self._n_pegang < len(kotak) * TOPUP_FRAC:
            self._kurang += 1
            if self._kurang >= TOPUP_N:
                perlu = True
        else:
            self._kurang = 0

        if perlu:
            self._daftar(frame, kotak)
            # Frame pendaftaran tetap mengeluarkan kotak detektor apa adanya,
            # supaya tidak ada frame yang kosong posisi pemainnya.
            return self._dari_deteksi(detections_dict)

        if not self._siap or self._pred is None:
            return self._dari_deteksi(detections_dict)

        try:
            with self._autocast():
                m, _ = self._pred.inference(self._siapkan(frame))
            mask = (m > 0.0).cpu().numpy()
        except Exception as e:
            logger.warning("SAM2 gagal melacak (%s); daftar ulang", e)
            self._siap = False
            return self._dari_deteksi(detections_dict)

        H, W = mask.shape[-2:]
        sx, sy = frame.shape[1] / W, frame.shape[0] / H
        hasil, n = [], 0
        for i in range(mask.shape[0]):
            ys, xs = np.nonzero(mask[i])
            if len(xs) < MIN_MASK_PX:
                continue
            n += 1
            x1, x2 = float(xs.min() * sx), float(xs.max() * sx)
            y1, y2 = float(ys.min() * sy), float(ys.max() * sy)
            tid = self._peta.get(i)
            if tid is None:
                continue
            e = {"track_id": tid, "bbox": [x1, y1, x2, y2], "confidence": 1.0,
                 "class": "player", "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0]}
            j = self.track_jersey_map.get(tid)
            if j is not None:
                e["jersey_number"] = j
            hasil.append(e)
        self._n_pegang = n
        self._newly_inherited = {}
        return {"tracked_players": hasil,
                "tracked_ball": self._bola(detections_dict),
                "tracked_referees": []}

    # ── bantuan ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bola(d: dict):
        b = (d or {}).get("ball")
        return dict(b) if b else None

    def _dari_deteksi(self, d: dict) -> dict:
        """Cadangan saat SAM2 belum siap: pakai kotak detektor apa adanya dengan
        id sementara, supaya frame tidak pernah kosong."""
        out = []
        for i, p in enumerate((d or {}).get("players") or []):
            x1, y1, x2, y2 = p["bbox"]
            out.append({"track_id": -(i + 1), "bbox": list(p["bbox"]),
                        "confidence": float(p.get("confidence", 0.0)),
                        "class": "player",
                        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0]})
        return {"tracked_players": out, "tracked_ball": self._bola(d),
                "tracked_referees": []}

    def initialize(self, frame_shape: tuple) -> None:
        """Dipanggil video_processor sekali sebelum update() pertama. SAM2 tidak
        perlu disiapkan per ukuran frame, tapi metodenya harus ada agar
        antarmukanya sama dengan PlayerTracker."""
        return None

    def get_jersey_number(self, track_id: int) -> Optional[int]:
        return self.track_jersey_map.get(track_id)

    def update_jersey(self, track_id: int, jersey_number: int) -> None:
        self.track_jersey_map[track_id] = jersey_number

    def reset(self) -> None:
        self._siap = False
        self._peta = {}
        self._tunda = 0
        self._kurang = 0
        self._n_pegang = 0
        self.track_jersey_map.clear()
        self._newly_inherited = {}


def create_sam2_tracker(frame_rate: int = 25) -> SAM2PlayerTracker:
    return SAM2PlayerTracker(frame_rate=frame_rate)

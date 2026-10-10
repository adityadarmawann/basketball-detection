"""
16_ukur_pelacakan.py

Mengukur MUTU PELACAKAN tanpa menjalankan pipeline penuh. Hanya detektor dan
pelacak, jadi sekitar 5 menit untuk video 4 menit, bukan 20 menit. Dipakai
untuk menyapu setelan dengan cepat sebelum memutuskan uji penuh.

Yang diukur:
  1. Jumlah track_id dan umur identitas    -> pecah atau tidak
  2. Lompatan POSISI antar frame berurutan -> track menempel ke dua orang?
  3. Lompatan WARNA jersey antar frame     -> ukuran kemurnian yang lebih tajam
  4. Cakupan BOLA, mentah lawan keluaran   -> berapa celah yang terisi pelacak

Catatan penting soal kemurnian: BANDINGKAN LAJU PER TRANSISI, jangan persentase
track yang memuat lompatan. Setelan yang menghasilkan track lebih panjang
otomatis punya lebih banyak transisi per track, jadi persentasenya naik tanpa
kemurniannya benar benar turun. Kesalahan itu pernah hampir membuat kesimpulan
terbalik.

Contoh:
    DETECTOR_METHOD=rfdetr BYTETRACK_MATCH_THRESH=0.7 \
    VIDEO=/path/ke/potongan.mp4 python eksperimen/16_ukur_pelacakan.py
"""

import os
import sys
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))
os.chdir(AKAR)

# WAJIB sebelum mengimpor pipeline: modul modul itu membaca os.getenv saat
# DIIMPOR, jadi .env harus sudah masuk lebih dulu.
from dotenv import load_dotenv            # noqa: E402
load_dotenv(AKAR / ".env")

from pipeline.detector import create_detector, DETECTOR_METHOD   # noqa: E402
from pipeline.tracker import create_tracker                       # noqa: E402

VIDEO   = os.environ["VIDEO"]
LANGKAH = int(os.getenv("LANGKAH", "2"))   # 60 fps sumber jadi 30 fps, sama
                                           # dengan yang dipakai pipeline


def main():
    det = create_detector()
    trk = create_tracker()
    trk.frame_rate = 30

    cap = cv2.VideoCapture(VIDEO)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f0 = cap.read()
    if not ok:
        print("video tidak bisa dibaca"); return
    if hasattr(trk, "initialize"):
        trk.initialize(f0.shape)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    POS   = defaultdict(list)    # tid -> [(frame, cx, cy)]
    WARNA = defaultdict(list)    # tid -> [(frame, LAB dada)]
    n_fr = bola_mentah = bola_keluar = bola_kira = 0
    celah = []
    jeda = 0
    t0 = time.perf_counter()

    for i in range(0, n, LANGKAH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        n_fr += 1
        H, W = fr.shape[:2]
        d = det.detect(fr)

        ada = d.get("ball") is not None
        bola_mentah += ada
        if ada:
            if jeda:
                celah.append(jeda)
            jeda = 0
        else:
            jeda += 1

        hasil = trk.update(fr, d)
        b = hasil.get("tracked_ball")
        if b is not None:
            bola_keluar += 1
            bola_kira += 1 if b.get("predicted") else 0

        for p in hasil.get("tracked_players", []):
            tid = p["track_id"]
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            POS[tid].append((n_fr, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
            # potongan DADA saja: 25-55 persen tinggi, 25-75 persen lebar.
            # Bagian ini paling stabil warnanya, tidak terganggu lantai
            # maupun kepala.
            h, w = y2 - y1, x2 - x1
            a = max(0, y1 + int(h * 0.25)); bb = min(H, y1 + int(h * 0.55))
            c = max(0, x1 + int(w * 0.25)); dd = min(W, x1 + int(w * 0.75))
            if bb > a and dd > c:
                pot = fr[a:bb, c:dd]
                if pot.size:
                    lab = cv2.cvtColor(pot, cv2.COLOR_BGR2LAB)
                    WARNA[tid].append((n_fr, np.median(lab.reshape(-1, 3), axis=0)))
    cap.release()

    umur = np.array([len(v) for v in POS.values()]) if POS else np.array([0])

    # Lompatan POSISI, hanya antar frame yang benar benar berurutan.
    lmp = []
    for v in POS.values():
        for a, b in zip(v, v[1:]):
            if b[0] - a[0] == 1:
                lmp.append(math.hypot(b[1] - a[1], b[2] - a[2]))
    lmp = np.array(lmp) if lmp else np.array([0.0])

    # Lompatan WARNA, ukuran kemurnian. Dilaporkan sebagai LAJU PER TRANSISI
    # supaya tidak bias terhadap track yang lebih panjang.
    lw = []
    for v in WARNA.values():
        for a, b in zip(v, v[1:]):
            if b[0] - a[0] == 1:
                lw.append(float(np.linalg.norm(b[1] - a[1])))
    lw = np.array(lw) if lw else np.array([0.0])
    laju = (lw > 40).sum() / max(len(lw), 1) * 100

    cl = np.array(celah) if celah else np.array([0])
    lama = time.perf_counter() - t0

    print(
        f"RINGKAS det={DETECTOR_METHOD} trk={os.getenv('TRACKER_METHOD','bytetrack')} "
        f"mt={os.getenv('BYTETRACK_MATCH_THRESH','bawaan')} gmc={os.getenv('GMC_ENABLED','0')} "
        f"ball={os.getenv('BALL_TRACK','bawaan')}\n"
        f"  frame {n_fr} | track {len(POS)} | umur median {np.median(umur):.0f}f "
        f"({np.median(umur)/30:.2f}s) | >=5s {(umur>=150).sum()} | >=10s {(umur>=300).sum()}\n"
        f"  lompat posisi p95 {np.percentile(lmp,95):.0f}px | "
        f"KEMURNIAN lompat warna>40 {laju:.1f}% dari {len(lw)} transisi\n"
        f"  bola mentah {bola_mentah} ({bola_mentah/max(n_fr,1)*100:.1f}%) -> "
        f"keluaran {bola_keluar} ({bola_keluar/max(n_fr,1)*100:.1f}%), perkiraan {bola_kira} | "
        f"celah median {np.median(cl):.0f}f p90 {np.percentile(cl,90):.0f}f maks {cl.max()}f\n"
        f"  waktu {lama/60:.1f} menit"
    )


if __name__ == "__main__":
    main()

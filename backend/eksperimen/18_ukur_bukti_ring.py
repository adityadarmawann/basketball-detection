"""
18_ukur_bukti_ring.py

Mengukur MUTU BUKTI di sekitar ring, bukan mutu pelacakan. Dipakai untuk
menjawab: apakah penghambat skor masih bisa diperbaiki TANPA melatih ulang
model?

Yang diukur:
  1. GETARAN titik tengah ring antar frame berurutan -> perlu dihaluskan?
  2. UKURAN kotak bola dalam piksel, di frame utuh    -> objek terlalu kecil?
  3. Keyakinan kelas ball-in-basket                    -> layak jadi penentu?
  4. Cakupan bola SAAT BOLA DEKAT RING                 -> ini yang menentukan
     skor, bukan cakupan rata rata sepanjang video

Contoh:
    DETECTOR_METHOD=rfdetr VIDEO=/path/klip.mp4 MULAI=70 DETIK=40 \
    python eksperimen/18_ukur_bukti_ring.py
"""

import os
import sys
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))
os.chdir(AKAR)

from dotenv import load_dotenv            # noqa: E402
load_dotenv(AKAR / ".env")

from pipeline.detector import create_detector, DETECTOR_METHOD   # noqa: E402

VIDEO   = os.environ["VIDEO"]
MULAI   = float(os.getenv("MULAI", "70"))
DETIK   = float(os.getenv("DETIK", "40"))
LANGKAH = int(os.getenv("LANGKAH", "2"))
DEKAT   = float(os.getenv("DEKAT", "120"))   # piksel, "dekat ring"


def main():
    det = create_detector()
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    awal = int(MULAI * fps)
    akhir = int((MULAI + DETIK) * fps)

    getar = []            # geser titik tengah ring antar frame berurutan
    ring_lebar = []
    bola_lebar = []
    konf_bib = []
    n_fr = ada_bola = 0
    n_dekat = ada_dekat = 0
    sebelum = None
    kelas_lain = defaultdict(list)

    for i in range(awal, akhir, LANGKAH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        n_fr += 1
        d = det.detect(fr)

        rings = d.get("hoops") or []
        pusat = None
        if rings:
            r = min(rings, key=lambda h: -(h.get("confidence") or 0))
            x1, y1, x2, y2 = r["bbox"]
            pusat = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            ring_lebar.append(x2 - x1)
            if sebelum is not None:
                getar.append(math.hypot(pusat[0] - sebelum[0], pusat[1] - sebelum[1]))
        sebelum = pusat if pusat else sebelum

        b = d.get("ball")
        if b is not None:
            ada_bola += 1
            x1, y1, x2, y2 = b["bbox"]
            bola_lebar.append(x2 - x1)

        # Cakupan SAAT DEKAT RING. Dipakai titik tengah ring sebagai jangkar,
        # jadi hitungan ini hanya untuk frame yang ringnya terlihat.
        if pusat is not None and b is not None:
            bx = (b["bbox"][0] + b["bbox"][2]) / 2.0
            by = (b["bbox"][1] + b["bbox"][3]) / 2.0
            if math.hypot(bx - pusat[0], by - pusat[1]) < DEKAT:
                n_dekat += 1
                ada_dekat += 1
        elif pusat is not None and b is None:
            n_dekat += 0   # tidak bisa tahu dekat atau tidak kalau bola hilang

        # Kelas ball-in-basket. Ambangnya ikut RFDETR_CONF, jadi jalankan
        # dengan RFDETR_CONF rendah kalau mau tahu sebaran sebenarnya.
        for x in (d.get("balls_in_basket") or []):
            konf_bib.append(x.get("confidence") or 0)
    cap.release()

    def r(a, nama, satuan="px"):
        a = np.array(a) if len(a) else np.array([0.0])
        return (f"  {nama:<34s} median {np.median(a):6.1f} {satuan}  "
                f"p90 {np.percentile(a,90):6.1f}  maks {a.max():6.1f}  n={len(a)}")

    print(f"\n=== bukti di sekitar ring | {DETECTOR_METHOD} | {n_fr} frame "
          f"({MULAI:.0f}s + {DETIK:.0f}s) ===")
    print(r(getar, "GETARAN titik tengah ring"))
    print(r(ring_lebar, "lebar kotak ring"))
    print(r(bola_lebar, "LEBAR KOTAK BOLA"))
    print(f"  bola terdeteksi {ada_bola}/{n_fr} = {ada_bola/max(n_fr,1)*100:.1f}%")
    print(f"  ball-in-basket: {len(konf_bib)} deteksi"
          + (f", konf maks {max(konf_bib):.3f}" if konf_bib else ", TIDAK ADA satu pun"))
    if bola_lebar:
        sk = 576.0 / 1280.0
        print(f"\n  Bola selebar {np.median(bola_lebar):.1f} px di frame 1280 lebar "
              f"menjadi {np.median(bola_lebar)*sk:.1f} px setelah diperkecil ke 576.")


if __name__ == "__main__":
    main()

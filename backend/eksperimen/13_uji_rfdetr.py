"""
13_uji_rfdetr.py

Menguji detektor RF-DETR hasil latihan (best-object-basket-detect-v1-DETR.pt)
lewat PABRIK detektor yang sebenarnya, bukan model telanjang, supaya saklar
DETECTOR_METHOD, pemuatan, bentuk keluaran, dan jalur cadangan ikut teruji.

Dijalankan dua kali untuk membandingkan:
    DETECTOR_METHOD=yolo   python eksperimen/13_uji_rfdetr.py
    DETECTOR_METHOD=rfdetr python eksperimen/13_uji_rfdetr.py

Yang diperiksa:
  1. Detektor yang termuat memang yang diminta (bukan diam diam jatuh ke YOLO).
  2. Bentuk keluaran detect() sama persis, kunci demi kunci, supaya pelacak,
     homografi, pose, dan statistik tidak perlu tahu bedanya.
  3. Berapa objek per kelas yang ditemukan, dan berapa lama per frame.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.detector import create_detector, DETECTOR_METHOD   # noqa: E402

VIDEO = os.getenv(
    "VIDEO",
    "/home/paul/computer-vision/campus-league/basketball/smart-vision-cl/"
    "test-video/test-Quarter-1_UNESA-UBAYA.mp4")
N_FRAME = int(os.getenv("N_FRAME", "12"))

# Kunci yang WAJIB ada di keluaran detect(), apa pun mesinnya. Diambil dari
# yang benar benar dibaca video_processor, event_engine, dan stats_calculator.
KUNCI_WAJIB = ["players", "ball", "referees", "hoops"]


def main():
    print("=" * 72)
    print(f"  DETECTOR_METHOD = {DETECTOR_METHOD}")
    print("=" * 72)

    det = create_detector()
    nama_kelas_mesin = type(det).__name__
    print(f"  detektor termuat : {nama_kelas_mesin}")
    if DETECTOR_METHOD == "rfdetr" and nama_kelas_mesin != "RFDetrDetector":
        print("  PERHATIAN: diminta rfdetr tapi yang termuat BUKAN RFDetrDetector.")
        print("  Artinya jalur cadangan ke YOLO terpakai. Lihat peringatan di atas.")
    if hasattr(det, "_class_names"):
        print(f"  kelas            : {det._class_names}")

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"  video tidak bisa dibuka: {VIDEO}")
        return
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  video            : {int(cap.get(3))}x{int(cap.get(4))}, {n} frame\n")

    # Pemanasan tidak dihitung: kompilasi dan alokasi hanya terjadi sekali.
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.1))
    ok, f0 = cap.read()
    if ok:
        det.detect(f0)

    jumlah = {"players": 0, "ball": 0, "referees": 0, "hoops": 0,
              "numbers": 0, "balls_in_basket": 0, "rims": 0}
    aksi = {}
    waktu = []
    kunci_hilang = set()

    for i in np.linspace(n * 0.1, n * 0.9, N_FRAME, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        t0 = time.perf_counter()
        out = det.detect(frame)
        waktu.append(time.perf_counter() - t0)

        for k in KUNCI_WAJIB:
            if k not in out:
                kunci_hilang.add(k)
        for k in jumlah:
            v = out.get(k)
            if v is None:
                continue
            jumlah[k] += 1 if isinstance(v, dict) else len(v)
        for p in out.get("players", []):
            a = p.get("rf_class") or p.get("class")
            aksi[a] = aksi.get(a, 0) + 1
    cap.release()

    print("  BENTUK KELUARAN")
    if kunci_hilang:
        print(f"    KUNCI HILANG: {sorted(kunci_hilang)}  <- alur akan putus")
    else:
        print(f"    semua kunci wajib ada: {KUNCI_WAJIB}")

    print(f"\n  OBJEK DITEMUKAN di {len(waktu)} frame")
    for k, v in jumlah.items():
        print(f"    {k:18s} {v:5d}   ({v/max(len(waktu),1):5.2f} per frame)")

    if aksi:
        print("\n  RINCIAN KELAS PEMAIN")
        for k, v in sorted(aksi.items(), key=lambda x: -x[1]):
            print(f"    {str(k):24s} {v:5d}")

    print(f"\n  WAKTU per frame: {np.mean(waktu)*1000:7.1f} ms   "
          f"(min {np.min(waktu)*1000:.0f}, maks {np.max(waktu)*1000:.0f})")
    # 4 kuarter x 10 menit di 25 fps = 40*60*25 = 60.000 frame.
    n_frame_pertandingan = 4 * 10 * 60 * 25
    print(f"  Perkiraan {n_frame_pertandingan:,} frame (4x10 menit di 25 fps): "
          f"{np.mean(waktu)*n_frame_pertandingan/60:.1f} menit, "
          f"deteksi SAJA (tahap lain belum dihitung)")


if __name__ == "__main__":
    main()

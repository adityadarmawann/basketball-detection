"""
Ukur sebaran pergeseran piksel keypoint lapangan, untuk menyetel COURT_CUT_PX.

Ambang yang terlalu tinggi membuat ganti sudut tidak pernah terdeteksi, jadi
kuncian nomor menempel ke orang yang salah. Ambang yang terlalu rendah membuat
kuncian dilepas terus menerus, jadi tidak ada nomor yang pernah terkonfirmasi.
Jadi angkanya harus diambil dari rekaman, bukan ditebak.

Pemakaian:
    python3 -u eksperimen/05_ukur_ganti_sudut.py <video> <n_frame>
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

COURT_EVERY_N = int(os.getenv("COURT_EVERY_N", "5"))


def main() -> int:
    video = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    from pipeline.court import CourtMapper, CAMERA_MOVE_PX, COURT_CUT_PX

    crt = CourtMapper()
    crt.load_model()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    geser, fid, n_kalib = [], 0, 0
    while fid < n:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % COURT_EVERY_N == 0:
            r = crt.process_frame(frame)
            if r.get("is_calibrated"):
                n_kalib += 1
            g = r.get("pixel_shift")
            if g is not None:
                geser.append(g)
        fid += 1
    cap.release()

    if not geser:
        print("tidak ada pengukuran geser")
        return 1
    g = np.array(geser)
    detik = n / fps
    print(f"video {detik:.0f} detik, {len(g)} pengukuran, {n_kalib} terkalibrasi")
    print(f"\nsebaran geser piksel rata rata per pengukuran:")
    for q in (50, 75, 90, 95, 99):
        print(f"  persentil {q:>2d} : {np.percentile(g, q):8.1f} px")
    print(f"  maksimum    : {g.max():8.1f} px")
    print(f"\nambang yang berlaku sekarang:")
    print(f"  CAMERA_MOVE_PX = {CAMERA_MOVE_PX:.0f}  -> "
          f"{(g > CAMERA_MOVE_PX).sum()}/{len(g)} "
          f"({(g > CAMERA_MOVE_PX).mean()*100:.0f}%) dianggap kamera bergerak")
    print(f"  COURT_CUT_PX   = {COURT_CUT_PX:.0f}  -> "
          f"{(g > COURT_CUT_PX).sum()}/{len(g)} "
          f"({(g > COURT_CUT_PX).mean()*100:.0f}%) dianggap GANTI SUDUT")
    print(f"\nkalau dianggap ganti sudut, sesering apa kuncian nomor dilepas:")
    for amb in (60, 80, 100, 120, 150, 200, 300):
        k = int((g > amb).sum())
        per_detik = k / detik if detik else 0
        print(f"  ambang {amb:>3d} px : {k:>4d} kali  "
              f"= 1 kali per {detik/k:.1f} detik" if k else
              f"  ambang {amb:>3d} px : {k:>4d} kali  (tidak pernah)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

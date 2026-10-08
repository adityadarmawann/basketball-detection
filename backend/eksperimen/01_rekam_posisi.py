"""
Tahap 1: satu lintasan CV untuk merekam aliran posisi lapangan ke disk.

Hanya detektor, pelacak, dan homografi yang dijalankan. Pose, OCR, aksi, dan
mesin kejadian dilewati karena tidak mempengaruhi perhitungan jarak, dan
melewatinya memotong waktu proses lebih dari separuh.

Hasilnya satu berkas .npz berisi (frame, track_id, x, y, ts_ms). Berkas itu
lalu diputar ulang lewat StatsCalculator dengan dua posisi saklar, jadi
pembandingannya memakai masukan yang sama persis, bukan dua lintasan CV yang
hasil deteksinya bisa sedikit berbeda.

Pemakaian:
    python3 -u eksperimen/01_rekam_posisi.py <video> <keluaran.npz> [detik_maks]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from pipeline.detector import BasketballDetector
from pipeline.tracker import PlayerTracker
from pipeline.court import CourtMapper

PROCESS_STRIDE = int(os.getenv("PROCESS_STRIDE", "2"))
COURT_EVERY_N  = int(os.getenv("COURT_EVERY_N", "5"))


def main() -> int:
    video = sys.argv[1]
    keluar = sys.argv[2]
    detik_maks = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"GAGAL membuka {video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    batas = int(detik_maks * fps) if detik_maks > 0 else total
    print(f"video     : {os.path.basename(video)}")
    print(f"ukuran    : {w}x{h}  fps {fps:.2f}  total {total} frame")
    print(f"diproses  : {batas} frame pertama, setiap {PROCESS_STRIDE} frame")

    det = BasketballDetector()
    det.load_model()
    trk = PlayerTracker(frame_rate=int(round(fps / PROCESS_STRIDE)))
    crt = CourtMapper()
    crt.load_model()
    det.warmup(h, w)
    crt.warmup(h, w)
    print(f"model     : siap")

    baris = []
    frame_id = 0
    court_terakhir = None
    n_kalibrasi = 0
    t_mulai = time.time()

    while frame_id < batas:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % PROCESS_STRIDE != 0:
            frame_id += 1
            continue
        ts_ms = frame_id * 1000.0 / fps

        d = det.detect(frame)
        tr = trk.update(frame, d)
        pemain = tr.get("tracked_players", [])

        if frame_id % COURT_EVERY_N == 0:
            court_terakhir = crt.process_frame(frame)
        court = court_terakhir or {}

        if court.get("is_calibrated"):
            n_kalibrasi += 1
            for p in pemain:
                c = p.get("center")
                if not c:
                    continue
                cp = crt.pixel_to_court(c)
                if cp is not None:
                    baris.append((frame_id, int(p["track_id"]),
                                  float(cp[0]), float(cp[1]), ts_ms))

        frame_id += 1
        if frame_id % 250 == 0:
            lewat = time.time() - t_mulai
            print(f"  frame {frame_id}/{batas}  sampel {len(baris)}  "
                  f"{frame_id/max(lewat,1e-6):.1f} frame/detik", flush=True)

    cap.release()
    arr = np.array(baris, dtype=np.float64) if baris else np.zeros((0, 5))
    np.savez_compressed(keluar, posisi=arr, fps=fps, stride=PROCESS_STRIDE)
    lewat = time.time() - t_mulai
    print(f"\nselesai   : {lewat/60:.1f} menit")
    print(f"sampel    : {len(baris)} posisi, {len(set(arr[:,1].astype(int))) if len(arr) else 0} track_id")
    print(f"kalibrasi : {n_kalibrasi} frame terkalibrasi")
    print(f"tersimpan : {keluar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

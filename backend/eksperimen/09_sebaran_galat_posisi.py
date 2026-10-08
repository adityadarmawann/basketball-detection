"""
Proses satu pertandingan penuh, catat sebaran taksiran galat posisi pemain.

Pertanyaan yang dijawab: dari semua posisi pemain yang tercatat, berapa persen
yang taksiran galatnya di atas 1 meter? Jawaban itu yang menentukan apakah
statistik jarak dan kecepatan perlu membuang posisi yang tidak andal.

Hanya detektor, pelacak, dan homografi yang dijalankan. Pose, OCR, aksi, dan
mesin kejadian dilewati karena tidak mempengaruhi posisi lapangan.

Pemakaian:
    python3 -u eksperimen/09_sebaran_galat_posisi.py <video> <keluaran.npz>
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

STRIDE = int(os.getenv("PROCESS_STRIDE", "2"))
COURT_EVERY = int(os.getenv("COURT_EVERY_N", "3"))


def main() -> int:
    video, keluar = sys.argv[1], sys.argv[2]

    from pipeline.detector import BasketballDetector
    from pipeline.tracker import PlayerTracker
    from pipeline.court import CourtMapper

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(4)), int(cap.get(3))
    print(f"{os.path.basename(video)}  {w}x{h}  fps {fps:.0f}  {total} frame "
          f"({total/fps/60:.1f} menit)")

    det = BasketballDetector(); det.load_model()
    trk = PlayerTracker(frame_rate=int(round(fps / STRIDE)))
    crt = CourtMapper(); crt.load_model()
    det.warmup(h, w); crt.warmup(h, w)

    baris = []
    fid, court, n_kal, t0 = 0, None, 0, time.time()
    while fid < total:
        ok, fr = cap.read()
        if not ok:
            break
        if fid % STRIDE != 0:
            fid += 1
            continue
        d = det.detect(fr)
        tr = trk.update(fr, d)
        if fid % COURT_EVERY == 0:
            court = crt.process_frame(fr)
        if court and court.get("is_calibrated"):
            n_kal += 1
            for p in tr.get("tracked_players", []):
                c = p.get("center")
                if not c:
                    continue
                cp = crt.pixel_to_court(c)
                if cp is None:
                    continue
                ex = crt.jarak_ekstrapolasi(cp)
                er = crt.perkiraan_galat_m(cp)
                baris.append((fid, int(p["track_id"]), cp[0], cp[1],
                              -1.0 if ex is None else ex,
                              -1.0 if er is None else er))
        fid += 1
        if fid % 2000 == 0:
            lewat = time.time() - t0
            print(f"  frame {fid}/{total}  posisi {len(baris)}  "
                  f"{fid/max(lewat,1e-6):.0f} frame/detik  sisa "
                  f"{(total-fid)/max(fid/max(lewat,1e-6),1e-6)/60:.1f} menit", flush=True)
    cap.release()
    arr = np.array(baris, dtype=np.float64) if baris else np.zeros((0, 6))
    np.savez_compressed(keluar, data=arr, fps=fps, stride=STRIDE,
                        n_kalibrasi=n_kal)
    print(f"\nselesai {(time.time()-t0)/60:.1f} menit")
    print(f"posisi tercatat : {len(arr)}")
    print(f"frame kalibrasi : {n_kal}")
    print(f"tersimpan       : {keluar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

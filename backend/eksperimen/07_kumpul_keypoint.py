"""
Kumpulkan prediksi keypoint MENTAH dari video ke .npz, sekali jalan.

Tujuannya agar penyetelan ambang bisa dilakukan berkali kali tanpa mengulang
inferensi model. Yang disimpan adalah keluaran model apa adanya, 33 x (x, y,
keyakinan) per frame, tanpa penyaringan apa pun.

Pemakaian:
    python3 -u eksperimen/07_kumpul_keypoint.py <video> <keluaran.npz> [langkah]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def main() -> int:
    video, keluar = sys.argv[1], sys.argv[2]
    langkah = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    dev = os.getenv("KP_DEVICE", "cpu")

    from ultralytics import YOLO
    m = YOLO("models/court_keypoints.pt")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{os.path.basename(video)}  {int(cap.get(3))}x{int(cap.get(4))}  "
          f"fps {fps:.0f}  {total} frame, sampel tiap {langkah} frame")

    frames, kps = [], []
    f, t0 = 0, time.time()
    while f < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        r = m.predict(fr, verbose=False, conf=0.01, device=dev)
        k = np.zeros((33, 3), np.float32)
        if r and r[0].keypoints is not None:
            d = r[0].keypoints.data.cpu().numpy()
            if len(d):
                k = d[np.argmax(d[:, :, 2].sum(1))]
        frames.append(f)
        kps.append(k)
        f += langkah
        if len(frames) % 100 == 0:
            print(f"  {len(frames)} sampel, {time.time()-t0:.0f}s", flush=True)
    cap.release()
    np.savez_compressed(keluar, frames=np.array(frames),
                        kps=np.array(kps, np.float32), fps=fps)
    print(f"tersimpan {keluar}: {len(frames)} sampel, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

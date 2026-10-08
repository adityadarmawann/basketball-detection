"""
Ukur perilaku sudut kamera: berapa titik yang terlihat, sisi mana, dan sesering
apa sisinya berpindah.

Latar belakangnya: kamera campus league mengikuti bola, jadi hampir tidak pernah
menyorot seluruh lapangan. Yang terlihat bergantian antara sisi kiri, tengah,
dan sisi kanan, dan perpindahannya cepat. Setiap kali sisinya berganti, ORANG
yang terlihat juga berbeda, sehingga kuncian identitas di hilir harus disegarkan.

Yang diukur per sampel:
    jumlah keypoint di atas ambang
    HIMPUNAN label yang terlihat, bukan cuma jumlahnya
    sisi lapangan yang tersorot, dari rata rata koordinat x label yang terlihat
    geser piksel terhadap sampel sebelumnya

Pemakaian:
    python3 -u eksperimen/06_ukur_sudut_kamera.py <video> [langkah_frame]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

AMBANG = float(os.getenv("UKUR_CONF", "0.35"))
DEV = os.getenv("UKUR_DEVICE", "cpu")


def main() -> int:
    video = sys.argv[1]
    langkah = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    from ultralytics import YOLO
    import pipeline.court as C

    m = YOLO("models/court_keypoints.pt")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video   : {os.path.basename(video)}")
    print(f"ukuran  : {int(cap.get(3))}x{int(cap.get(4))}  fps {fps:.0f}  "
          f"{total} frame  {total/fps/60:.1f} menit")
    print(f"sampel  : tiap {langkah} frame = {langkah/fps:.2f} detik, "
          f"total {total//langkah} sampel\n")

    sampel = []
    f = 0
    while f < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        r = m.predict(fr, verbose=False, conf=0.05, device=DEV)
        k = None
        if r and r[0].keypoints is not None:
            d = r[0].keypoints.data.cpu().numpy()
            if len(d):
                k = d[np.argmax(d[:, :, 2].sum(1))]
        if k is None:
            sampel.append({"frame": f, "n": 0, "label": [], "x": None, "piks": {}})
        else:
            terlihat, piks = [], {}
            for i, lab in enumerate(C.KP_INDEX_TO_LABEL):
                if k[i, 2] >= AMBANG:
                    terlihat.append(lab)
                    piks[lab] = [float(k[i, 0]), float(k[i, 1])]
            xs = [C.LABEL_TO_COURT[l][0] for l in terlihat]
            sampel.append({"frame": f, "n": len(terlihat), "label": terlihat,
                           "x": float(np.mean(xs)) if xs else None, "piks": piks})
        f += langkah
        if len(sampel) % 50 == 0:
            print(f"  {len(sampel)} sampel, frame {f}/{total}", flush=True)
    cap.release()

    out = os.getenv("UKUR_OUT", "/tmp/sudut.json")
    json.dump({"fps": fps, "langkah": langkah, "ambang": AMBANG,
               "sampel": sampel}, open(out, "w"))
    print(f"\ntersimpan: {out}  ({len(sampel)} sampel)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

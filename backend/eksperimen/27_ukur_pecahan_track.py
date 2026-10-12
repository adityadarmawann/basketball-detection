"""
27_ukur_pecahan_track.py

Mengukur POLA PECAHNYA track, bukan cuma jumlahnya. Menjawab satu pertanyaan
yang menentukan: kalau sebuah track berakhir, apakah ada track BARU yang muncul
sesaat kemudian di tempat yang berdekatan, yaitu orang yang sama dengan
identitas baru.

Kalau banyak, menyambung pecahan itu layak dikerjakan dan hasilnya bisa
diperkirakan dari sini. Kalau sedikit, berarti track berakhir karena orangnya
memang keluar layar atau kamera berpindah, dan menyambung tidak akan menolong.

Detektor dan pelacak saja, tanpa pipeline penuh, jadi sekitar 5 menit.

Contoh:
    DETECTOR_METHOD=rfdetr VIDEO=/home/paul/q1_4menit.mp4 \
      python eksperimen/27_ukur_pecahan_track.py
"""

import math
import os
import sys
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
from pipeline.tracker import create_tracker                       # noqa: E402

VIDEO   = os.environ["VIDEO"]
LANGKAH = int(os.getenv("LANGKAH", "2"))
FPS_EFF = 30.0

# Ambang "mungkin orang yang sama": jeda waktu dan jarak piksel.
JEDA_F   = int(os.getenv("JEDA_F", "30"))      # 1 detik di 30 fps
JARAK_PX = float(os.getenv("JARAK_PX", "150"))
# Beda warna jersey maksimum supaya dianggap orang yang sama.
WARNA_D  = float(os.getenv("WARNA_D", "40"))


def main():
    det = create_detector()
    trk = create_tracker()
    trk.frame_rate = int(FPS_EFF)

    cap = cv2.VideoCapture(VIDEO)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f0 = cap.read()
    if not ok:
        print("  video tidak terbaca"); return
    if hasattr(trk, "initialize"):
        trk.initialize(f0.shape)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Per track: frame pertama/terakhir, posisi pertama/terakhir, warna dada.
    info = {}
    n_fr = 0
    beda_frame = []          # beda gambar antar frame, untuk menandai ganti sudut
    frame_abu_sebelum = None

    for i in range(0, n_total, LANGKAH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        n_fr += 1
        H, W = fr.shape[:2]

        kecil = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
        if frame_abu_sebelum is not None:
            beda_frame.append(float(np.mean(cv2.absdiff(kecil, frame_abu_sebelum))))
        else:
            beda_frame.append(0.0)
        frame_abu_sebelum = kecil

        hasil = trk.update(fr, det.detect(fr))
        for p in hasil.get("tracked_players", []):
            tid = p["track_id"]
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            c = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            h, w = y2 - y1, x2 - x1
            a = max(0, y1 + int(h * 0.25)); b = min(H, y1 + int(h * 0.55))
            cc = max(0, x1 + int(w * 0.25)); dd = min(W, x1 + int(w * 0.75))
            lab = None
            if b > a and dd > cc:
                pot = fr[a:b, cc:dd]
                if pot.size:
                    lab = np.median(
                        cv2.cvtColor(pot, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)
            if tid not in info:
                info[tid] = {"awal": n_fr, "akhir": n_fr, "p_awal": c,
                             "p_akhir": c, "lab_awal": lab, "lab_akhir": lab,
                             "n": 1}
            else:
                d = info[tid]
                d["akhir"] = n_fr; d["p_akhir"] = c; d["n"] += 1
                if lab is not None:
                    d["lab_akhir"] = lab
    cap.release()

    # Simpan data mentahnya supaya jendela pencarian bisa DISAPU tanpa
    # menjalankan GPU lagi. Jendela 1 detik ternyata terlalu sempit: track_buffer
    # ByteTrack 90 frame (3 detik), jadi pengganti sebuah track bisa muncul jauh
    # setelah pengamatan terakhirnya.
    import json
    simpan = os.getenv("SIMPAN_KE", "/home/paul/rekam_pecahan_track.json")
    with open(simpan, "w") as fh:
        json.dump({
            "n_frame": n_fr,
            "beda_frame": [round(v, 3) for v in beda_frame],
            "track": {str(t): {
                "awal": d["awal"], "akhir": d["akhir"], "n": d["n"],
                "p_awal": [round(v, 1) for v in d["p_awal"]],
                "p_akhir": [round(v, 1) for v in d["p_akhir"]],
                "lab_awal": None if d["lab_awal"] is None else [round(float(v), 1) for v in d["lab_awal"]],
                "lab_akhir": None if d["lab_akhir"] is None else [round(float(v), 1) for v in d["lab_akhir"]],
            } for t, d in info.items()},
        }, fh)
    print(f"  data mentah disimpan: {simpan}")

    bd = np.array(beda_frame)
    potong = set(int(i) for i in np.where(bd > 25)[0])   # ganti sudut keras
    print(f"\n  === {n_fr} frame | {len(info)} track | "
          f"{len(potong)} ganti sudut keras ===")

    umur = np.array([d["n"] for d in info.values()])
    print(f"  umur track: median {np.median(umur):.0f}f "
          f"({np.median(umur)/FPS_EFF:.2f}s), >=5 detik "
          f"{(umur >= 5*FPS_EFF).sum()}")

    # Untuk tiap track yang BERAKHIR, cari track yang MULAI sesudahnya dalam
    # jendela waktu dan dekat secara posisi serta warna.
    mulai_pada = defaultdict(list)
    for tid, d in info.items():
        mulai_pada[d["awal"]].append(tid)

    bisa_sambung = 0
    sebab_potong = 0
    tanpa_calon  = 0
    jarak_sambung = []
    for tid, d in info.items():
        akhir = d["akhir"]
        if akhir >= n_fr - 2:
            continue                      # masih hidup di ujung video
        calon = None
        for f in range(akhir + 1, min(akhir + JEDA_F, n_fr) + 1):
            for t2 in mulai_pada.get(f, []):
                if t2 == tid:
                    continue
                d2 = info[t2]
                j = math.dist(d["p_akhir"], d2["p_awal"])
                if j > JARAK_PX:
                    continue
                if (d["lab_akhir"] is not None and d2["lab_awal"] is not None
                        and float(np.linalg.norm(d["lab_akhir"] - d2["lab_awal"])) > WARNA_D):
                    continue
                calon = (t2, j, f - akhir)
                break
            if calon:
                break
        if calon:
            bisa_sambung += 1
            jarak_sambung.append(calon[1])
        elif any(akhir <= c <= akhir + 3 for c in potong):
            sebab_potong += 1
        else:
            tanpa_calon += 1

    n_berakhir = bisa_sambung + sebab_potong + tanpa_calon
    if n_berakhir:
        print(f"\n  dari {n_berakhir} track yang BERAKHIR di tengah video:")
        print(f"    ADA calon sambungan      {bisa_sambung:4d} "
              f"({bisa_sambung/n_berakhir*100:.0f}%)  <- bisa disambung")
        print(f"    berakhir saat ganti sudut{sebab_potong:4d} "
              f"({sebab_potong/n_berakhir*100:.0f}%)  <- memang batas wajar")
        print(f"    tidak ada calon          {tanpa_calon:4d} "
              f"({tanpa_calon/n_berakhir*100:.0f}%)  <- keluar layar / gagal deteksi")
    if jarak_sambung:
        js = np.array(jarak_sambung)
        print(f"    jarak ke calon: median {np.median(js):.0f} px, "
              f"p90 {np.percentile(js,90):.0f} px")
        print(f"\n  PERKIRAAN kalau disambung: {len(info)} track jadi "
              f"sekitar {len(info) - bisa_sambung}")


if __name__ == "__main__":
    main()

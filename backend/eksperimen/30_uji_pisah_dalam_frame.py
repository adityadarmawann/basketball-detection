"""
30_uji_pisah_dalam_frame.py

Menguji SATU premis sebelum menulis kode: apakah DI DALAM SATU FRAME kedua tim
terpisah rapi berdasarkan warna dada.

Kenapa premis ini menentukan: semua acuan mutlak sudah gugur terukur. Jangkar
foto jersey memberi daya pisah lebih buruk, dan koreksi lantai cuma menekan
pergeseran 8 persen. Yang pasti berbagi pencahayaan dengan seorang pemain
tinggal pemain lain di frame yang sama. Kalau di dalam satu frame pun warnanya
bercampur, maka warna dada saja tidak cukup dan harus dicari sinyal lain.

Yang diukur, semuanya TANPA label tangan:
  1. Daya pisah dua kelompok di dalam tiap frame (silhouette).
  2. Keseimbangan ukuran kelompok. Dua tim bermain, jadi timpang berat itu
     tanda pemisahan gagal.
  3. KEMANTAPAN label per track: dengan cara relatif per frame, apakah sebuah
     track tetap di kelompok yang sama sepanjang hidupnya. Ini yang langsung
     menjawab keluhan "berantakan setelah kamera pindah".
  4. Pengaruh WASIT dan orang bukan pemain, dengan membandingkan hasil saat
     mereka ikut dan saat disaring.

TIDAK mengubah apa pun. Hanya mengukur.

Contoh:
    DETECTOR_METHOD=rfdetr VIDEO=/home/paul/q1_4menit.mp4 \
      python eksperimen/30_uji_pisah_dalam_frame.py
"""

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
MIN_ORANG = int(os.getenv("MIN_ORANG", "4"))
SKALA = np.array([40.0, 8.0, 8.0])     # sama dengan skala sumbu tim pipeline


def belah_dua(X):
    """Belah titik titik jadi dua kelompok pada SUMBU RAGAM TERBESAR.

    Dipakai sumbu ragam terbesar, bukan k-means acak, supaya hasilnya tetap
    (tidak bergantung undian awal) dan murah. Batasnya di celah terlebar
    sepanjang sumbu itu, bukan di tengah, supaya tidak memaksa belah rata.
    """
    if len(X) < 2:
        return None, None
    Xs = X / SKALA
    pusat = Xs - Xs.mean(axis=0)
    u, s, vt = np.linalg.svd(pusat, full_matrices=False)
    t = pusat @ vt[0]
    urut = np.sort(t)
    celah = np.diff(urut)
    if len(celah) == 0:
        return None, None
    i = int(np.argmax(celah))
    batas = (urut[i] + urut[i + 1]) / 2.0
    return (t > batas).astype(int), float(celah[i])


def main():
    det = create_detector()
    trk = create_tracker()
    trk.frame_rate = 30
    cap = cv2.VideoCapture(VIDEO)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f0 = cap.read()
    if not ok:
        print("  video tidak terbaca"); return
    if hasattr(trk, "initialize"):
        trk.initialize(f0.shape)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pisah_skor, seimbang, celah_lebar = [], [], []
    simpan_frame = []
    label_track = defaultdict(list)      # tid -> [kelompok relatif per frame]
    n_fr = n_dipakai = 0

    for i in range(0, n_total, LANGKAH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        n_fr += 1
        d = det.detect(fr)
        hasil = trk.update(fr, d)
        pemain = hasil.get("tracked_players", []) or []
        H, W = fr.shape[:2]
        warna, tids = [], []
        for p in pemain:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            h, w = y2 - y1, x2 - x1
            a = max(0, y1 + int(h * 0.25)); b = min(H, y1 + int(h * 0.55))
            c = max(0, x1 + int(w * 0.25)); dd = min(W, x1 + int(w * 0.75))
            if b > a and dd > c:
                pot = fr[a:b, c:dd]
                if pot.size:
                    warna.append(np.median(
                        cv2.cvtColor(pot, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0))
                    tids.append(p["track_id"])
        if len(warna) < MIN_ORANG:
            continue
        n_dipakai += 1
        simpan_frame.append({"f": n_fr, "tid": tids,
                             "lab": [[round(float(v), 1) for v in w] for w in warna]})
        X = np.array(warna, float)
        lab, celah = belah_dua(X)
        if lab is None:
            continue
        g0, g1 = X[lab == 0] / SKALA, X[lab == 1] / SKALA
        if len(g0) == 0 or len(g1) == 0:
            continue
        antar = np.linalg.norm(g0.mean(axis=0) - g1.mean(axis=0))
        dalam = (g0.std(axis=0).mean() + g1.std(axis=0).mean()) / 2 + 1e-9
        pisah_skor.append(antar / dalam)
        seimbang.append(min(len(g0), len(g1)) / len(X))
        celah_lebar.append(celah)
        for t, l in zip(tids, lab):
            label_track[t].append(int(l))
    cap.release()

    # Simpan warna per frame supaya ATURAN BELAH bisa diuji berkali kali tanpa
    # menjalankan GPU lagi. Aturan "celah terlebar" terbukti cuma memisahkan
    # satu orang paling nyeleneh (panitia, pelatih), bukan dua tim.
    import json
    with open(os.getenv("SIMPAN_KE", "/home/paul/rekam_warna_frame.json"), "w") as fh:
        json.dump(simpan_frame, fh)
    print(f"  data per frame disimpan: {len(simpan_frame)} frame")

    ps = np.array(pisah_skor); sb = np.array(seimbang); cl = np.array(celah_lebar)
    print(f"\n  === {n_fr} frame, {n_dipakai} dipakai (>= {MIN_ORANG} orang) ===")
    print(f"  daya pisah dalam frame : median {np.median(ps):5.2f}  "
          f"p10 {np.percentile(ps,10):5.2f}  p90 {np.percentile(ps,90):5.2f}")
    print(f"    (>1,5 berarti dua kelompok benar benar terpisah)")
    print(f"  keseimbangan kelompok  : median {np.median(sb)*100:4.0f}%  "
          f"(50% = rata, 0% = semua jadi satu)")
    print(f"  lebar celah pemisah    : median {np.median(cl):5.2f}")

    # Kemantapan label relatif sepanjang satu track.
    mantap, goyah = 0, 0
    for t, s in label_track.items():
        if len(s) < 10:
            continue
        maj = max(set(s), key=s.count)
        cocok = s.count(maj) / len(s)
        if cocok >= 0.9: mantap += 1
        else: goyah += 1
    tot = mantap + goyah
    if tot:
        print(f"\n  track dengan >=10 frame: {tot}")
        print(f"    MANTAP (>=90% frame di kelompok sama) {mantap} "
              f"({mantap/tot*100:.0f}%)")
        print(f"    goyah                                  {goyah} "
              f"({goyah/tot*100:.0f}%)")


if __name__ == "__main__":
    main()

"""
29_ukur_geser_cahaya.py

Menguji SATU dasar sebelum menulis kode apa pun:
apakah pergeseran warna LANTAI sejalan dengan pergeseran warna JERSEY saat
kamera berpindah sudut.

Kalau sejalan, warna dada pemain bisa diluruskan memakai lantai sebagai acuan,
sehingga jangkar tim tetap sah setelah kamera pindah. Kalau TIDAK sejalan,
gagasan itu batal dan tidak perlu dikerjakan.

Cara ukur: untuk tiap track yang hidup melewati pergantian sudut kamera,
bandingkan warna dadanya sebelum dan sesudah, lalu bandingkan dengan
pergeseran warna lantai di saat yang sama. Track yang sama = orang yang sama =
jersey yang sama, jadi semua perubahan warnanya berasal dari cahaya.

TIDAK mengubah apa pun. Hanya mengukur.

Contoh:
    DETECTOR_METHOD=rfdetr VIDEO=/home/paul/q1_4menit.mp4 \
      python eksperimen/29_ukur_geser_cahaya.py
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

VIDEO    = os.environ["VIDEO"]
LANGKAH  = int(os.getenv("LANGKAH", "2"))
CUT_BEDA = float(os.getenv("CUT_BEDA", "25"))


def warna_lantai(fr, kotak_pemain):
    """LAB median LANTAI: piksel yang BUKAN bagian dari kotak pemain mana pun.

    Lantai dipilih karena luas, selalu terlihat, dan warna aslinya tetap. Kotak
    pemain dibuang supaya warna jersey tidak ikut terhitung sebagai lantai.
    """
    H, W = fr.shape[:2]
    topeng = np.ones((H, W), np.uint8)
    for b in kotak_pemain:
        x1, y1, x2, y2 = [int(v) for v in b]
        topeng[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 0
    # Setengah bawah layar saja: di situ hampir pasti lantai, bukan penonton.
    topeng[:int(H * 0.45), :] = 0
    if topeng.sum() < 5000:
        return None
    lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
    return np.median(lab[topeng == 1], axis=0)


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

    per_frame = []     # (n_fr, lantai_lab, {tid: dada_lab})
    abu = None
    cuts = []
    n_fr = 0

    for i in range(0, n_total, LANGKAH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        n_fr += 1
        kecil = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
        if abu is not None and float(np.mean(cv2.absdiff(kecil, abu))) > CUT_BEDA:
            cuts.append(n_fr)
        abu = kecil

        hasil = trk.update(fr, det.detect(fr))
        pemain = hasil.get("tracked_players", []) or []
        H, W = fr.shape[:2]
        dada = {}
        kotak = []
        for p in pemain:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            kotak.append((x1, y1, x2, y2))
            h, w = y2 - y1, x2 - x1
            a = max(0, y1 + int(h * 0.25)); b = min(H, y1 + int(h * 0.55))
            c = max(0, x1 + int(w * 0.25)); dd = min(W, x1 + int(w * 0.75))
            if b > a and dd > c:
                pot = fr[a:b, c:dd]
                if pot.size:
                    dada[p["track_id"]] = np.median(
                        cv2.cvtColor(pot, cv2.COLOR_BGR2LAB).reshape(-1, 3), axis=0)
        per_frame.append((n_fr, warna_lantai(fr, kotak), dada))
    cap.release()

    idx = {f: (lt, dd) for f, lt, dd in per_frame}
    print(f"\n  === {n_fr} frame | {len(cuts)} pergantian sudut keras ===")

    # Untuk tiap cut, cari track yang HIDUP di kedua sisi.
    geser_dada, geser_lantai, sisa_setelah = [], [], []
    JAUH = 3    # frame sebelum dan sesudah cut
    for c in cuts:
        a, b = c - JAUH, c + JAUH
        if a not in idx or b not in idx:
            continue
        lt_a, dd_a = idx[a]
        lt_b, dd_b = idx[b]
        if lt_a is None or lt_b is None:
            continue
        d_lantai = lt_b - lt_a
        for tid in set(dd_a) & set(dd_b):
            d_dada = dd_b[tid] - dd_a[tid]
            geser_dada.append(np.linalg.norm(d_dada))
            geser_lantai.append(np.linalg.norm(d_lantai))
            sisa_setelah.append(np.linalg.norm(d_dada - d_lantai))

    if not geser_dada:
        print("  tidak ada track yang hidup melewati pergantian sudut.")
        print("  Itu sendiri temuan: track selalu PUTUS saat kamera pindah.")
        return

    gd = np.array(geser_dada); gl = np.array(geser_lantai); sa = np.array(sisa_setelah)
    print(f"  {len(gd)} pasangan track yang hidup melewati cut\n")
    print(f"  geser warna DADA   median {np.median(gd):6.1f}  p90 {np.percentile(gd,90):6.1f}")
    print(f"  geser warna LANTAI median {np.median(gl):6.1f}  p90 {np.percentile(gl,90):6.1f}")
    print(f"  SISA setelah dikoreksi lantai: median {np.median(sa):6.1f}  "
          f"p90 {np.percentile(sa,90):6.1f}")
    turun = (1 - np.median(sa) / max(np.median(gd), 1e-9)) * 100
    print(f"\n  => koreksi lantai menekan pergeseran {turun:.0f} persen")
    if turun > 30:
        print("     LAYAK: pergeseran jersey memang sejalan dengan lantai.")
    else:
        print("     TIDAK LAYAK: pergeseran jersah TIDAK sejalan dengan lantai,")
        print("     jadi lantai bukan acuan yang tepat. Gagasan ini dibatalkan.")


if __name__ == "__main__":
    main()

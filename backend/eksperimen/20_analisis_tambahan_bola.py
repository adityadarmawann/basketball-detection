"""
20_analisis_tambahan_bola.py

Menjawab satu pertanyaan: KENAPA menurunkan ambang bola menaikkan cakupan tapi
MENURUNKAN skor.

Dugaannya: detektor memilih SATU bola per frame, yang keyakinannya tertinggi.
Di ambang 0,25, frame yang cuma punya kandidat lemah menghasilkan TIDAK ADA
bola, dan itu celah yang jujur. Di ambang 0,10, frame yang sama menghasilkan
bola PALSU di posisi yang salah. Jadi menurunkan ambang menukar celah jujur
dengan posisi salah, dan posisi salah lebih merusak gerbang bola masuk daripada
celah.

Caranya: dari rekaman skrip 19 (posisi dan keyakinan per frame), deteksi dipisah
jadi "kuat" (c >= AMBANG_KUAT) dan "tambahan" (AMBANG_REKAM <= c < AMBANG_KUAT).
Lalu diukur seberapa jauh tiap deteksi dari lintasan tetangganya yang KUAT.
Kalau yang tambahan jauh lebih sering melompat, itu posisi salah, bukan
tangkapan tambahan.

Tidak memakai GPU sama sekali.

Contoh:
    REKAM_DIR=/home/paul/rekam_bola python eksperimen/20_analisis_tambahan_bola.py
"""

import json
import math
import os
from pathlib import Path

import numpy as np

REKAM_DIR    = Path(os.getenv("REKAM_DIR", "/home/paul/rekam_bola"))
AMBANG_KUAT  = float(os.getenv("AMBANG_KUAT", "0.25"))
JENDELA_F    = int(os.getenv("JENDELA_F", "3"))     # cari tetangga kuat +-3 langkah


def muat(nama):
    d = json.load(open(REKAM_DIR / f"{nama}.json"))
    # Hanya baris yang benar benar ada deteksinya.
    return d, [b for b in d["baris"] if b["c"] > 0]


def jarak_ke_lintasan(baris, kuat_per_f, f, x, y):
    """Jarak ke tetangga KUAT terdekat dalam jendela, atau None kalau tidak ada.

    Dipakai sebagai ukuran "apakah posisi ini masuk akal menurut lintasan".
    Tetangga yang dipakai hanya yang KUAT, supaya pembandingnya tidak ikut
    tercemar deteksi lemah yang sedang dinilai.
    """
    terdekat = None
    for d in range(1, JENDELA_F + 1):
        for ff in (f - d, f + d):
            k = kuat_per_f.get(ff)
            if k is not None:
                j = math.hypot(x - k[0], y - k[1])
                if terdekat is None or j < terdekat:
                    terdekat = j
        if terdekat is not None:
            return terdekat
    return None


def olah(nama):
    meta, baris = muat(nama)
    kuat_per_f = {b["f"]: (b["x"], b["y"]) for b in baris if b["c"] >= AMBANG_KUAT}

    j_kuat, j_tambahan = [], []
    for b in baris:
        # Untuk yang KUAT, tetangganya dihitung tanpa dirinya sendiri.
        tanpa_diri = dict(kuat_per_f)
        tanpa_diri.pop(b["f"], None)
        j = jarak_ke_lintasan(baris, tanpa_diri, b["f"], b["x"], b["y"])
        if j is None:
            continue
        (j_kuat if b["c"] >= AMBANG_KUAT else j_tambahan).append(j)

    n_kuat = sum(1 for b in baris if b["c"] >= AMBANG_KUAT)
    n_tbh  = len(baris) - n_kuat
    total  = len(meta["baris"])

    print(f"\n=== {nama} | {total} frame diperiksa | ambang kuat {AMBANG_KUAT} ===")
    print(f"  deteksi KUAT     (c >= {AMBANG_KUAT}) : {n_kuat:5d} = {n_kuat/total*100:5.1f}%")
    print(f"  deteksi TAMBAHAN (c <  {AMBANG_KUAT}) : {n_tbh:5d} = {n_tbh/total*100:5.1f}%")

    for label, arr in (("KUAT", j_kuat), ("TAMBAHAN", j_tambahan)):
        if not arr:
            print(f"  {label:<9s}: tidak ada yang punya tetangga kuat")
            continue
        a = np.array(arr)
        # 60 px kira kira tiga lebar bola. Lebih dari itu, posisinya tidak
        # mungkin bola yang sama dalam 3 langkah frame.
        print(f"  jarak ke lintasan, {label:<9s}: median {np.median(a):6.1f} px  "
              f"p90 {np.percentile(a,90):7.1f}  "
              f">60px {(a>60).sum()/len(a)*100:5.1f}%  n={len(a)}")

    if j_kuat and j_tambahan:
        pk = (np.array(j_kuat) > 60).sum() / len(j_kuat) * 100
        pt = (np.array(j_tambahan) > 60).sum() / len(j_tambahan) * 100
        print(f"\n  KESIMPULAN {nama}: deteksi tambahan melompat {pt:.1f}% lawan "
              f"{pk:.1f}% pada yang kuat", end="")
        print(f" ({pt/max(pk,0.01):.1f}x lebih sering)" if pk > 0 else "")


if __name__ == "__main__":
    for nama in ("rfdetr", "yolo"):
        if (REKAM_DIR / f"{nama}.json").exists():
            olah(nama)

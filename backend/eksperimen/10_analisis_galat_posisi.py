"""
Baca hasil skrip 09, jawab: berapa persen posisi yang taksiran galatnya besar,
dan berapa persen JARAK TEMPUH yang dibangun dari posisi tidak andal.

Yang kedua lebih menentukan daripada yang pertama. Posisi yang galatnya besar
bisa saja banyak tapi jarang dipakai menghitung jarak, atau sebaliknya sedikit
tapi justru menyumbang banyak langkah.

Pemakaian:
    python3 -u eksperimen/10_analisis_galat_posisi.py <galat_posisi.npz>
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

AMBANG = [0.5, 1.0, 1.5, 2.0, 3.0]


def main() -> int:
    d = np.load(sys.argv[1])
    a = d["data"]
    fps, stride = float(d["fps"]), int(d["stride"])
    if len(a) == 0:
        print("tidak ada posisi tercatat")
        return 1
    fid, tid, x, y, ex, er = (a[:, 0], a[:, 1].astype(int), a[:, 2], a[:, 3],
                              a[:, 4], a[:, 5])
    sah = er >= 0
    print(f"=== {len(a)} posisi pemain tercatat, {len(set(tid))} track_id ===")
    print(f"frame terkalibrasi: {int(d['n_kalibrasi'])}\n")

    print("SEBARAN JARAK EKSTRAPOLASI (meter di luar bukti keypoint)")
    e = ex[sah]
    for q in (50, 75, 90, 95, 99):
        print(f"  persentil {q:>2d}: {np.percentile(e, q):6.2f} m")
    print(f"  maksimum   : {e.max():6.2f} m")
    print(f"  tepat nol (ditopang bukti langsung): {(e <= 0.01).mean()*100:5.1f}%\n")

    print("SEBARAN TAKSIRAN GALAT POSISI (meter)")
    g = er[sah]
    for q in (50, 75, 90, 95, 99):
        print(f"  persentil {q:>2d}: {np.percentile(g, q):6.3f} m")
    print(f"  maksimum   : {g.max():6.3f} m\n")

    print("BERAPA PERSEN POSISI YANG GALATNYA DI ATAS AMBANG")
    for t in AMBANG:
        print(f"  di atas {t:.1f} m : {(g > t).mean()*100:5.1f}%  "
              f"({int((g > t).sum())} dari {len(g)})")
    print()

    # ── berapa persen JARAK TEMPUH dibangun dari posisi tidak andal ──────────
    print("BERAPA PERSEN JARAK TEMPUH DIBANGUN DARI POSISI TIDAK ANDAL")
    print("  (satu langkah dihitung tidak andal kalau SALAH SATU ujungnya di atas ambang)")
    urut = np.lexsort((fid, tid))
    t_, f_, x_, y_, g_ = tid[urut], fid[urut], x[urut], y[urut], er[urut]
    sama = t_[1:] == t_[:-1]
    dekat = (f_[1:] - f_[:-1]) <= stride * 2        # langkah berurutan saja
    pakai = sama & dekat & (g_[1:] >= 0) & (g_[:-1] >= 0)
    lang = np.hypot(x_[1:] - x_[:-1], y_[1:] - y_[:-1])[pakai]
    gmax = np.maximum(g_[1:], g_[:-1])[pakai]
    total = lang.sum()
    print(f"  total langkah terhitung: {len(lang)}, jarak kotor {total:.0f} m")
    for t in AMBANG:
        m = gmax > t
        print(f"  dari langkah dengan galat di atas {t:.1f} m : "
              f"{m.mean()*100:5.1f}% langkah, {lang[m].sum()/total*100:5.1f}% jarak")
    return 0


if __name__ == "__main__":
    sys.exit(main())

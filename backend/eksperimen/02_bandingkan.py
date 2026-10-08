"""
Tahap 2: putar ulang aliran posisi lewat StatsCalculator, bandingkan dua metode.

Masukannya sama persis untuk dua duanya, karena berasal dari satu berkas .npz
hasil satu lintasan CV. Jadi selisih angkanya murni dari metode perhitungan,
bukan dari deteksi yang berbeda.

Pemakaian:
    python3 -u eksperimen/02_bandingkan.py <posisi.npz>
"""
import os, sys, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def jalankan(npz, metode: str) -> dict:
    """Satu putaran penuh dengan satu posisi saklar. Modul dimuat ulang supaya
    konstanta tingkat modul terbaca sesuai env."""
    os.environ["PATH_CLEAN_METHOD"] = metode
    import pipeline.stats_calculator as S
    importlib.reload(S)
    assert S.PATH_CLEAN_METHOD == metode, S.PATH_CLEAN_METHOD

    fps = float(npz["fps"])
    stride = int(npz["stride"])
    sc = S.StatsCalculator(fps=fps / stride)

    pos = npz["posisi"]
    # kelompokkan per frame, karena _update_tracking_stats menerima satu frame
    urut = pos[np.argsort(pos[:, 0], kind="stable")]
    batas = np.searchsorted(urut[:, 0], np.unique(urut[:, 0]), side="left")
    batas = list(batas) + [len(urut)]
    for i in range(len(batas) - 1):
        blok = urut[batas[i]:batas[i + 1]]
        snap = {int(r[1]): (r[2], r[3], r[4]) for r in blok}
        sc._update_tracking_stats(snap)

    hasil = {}
    for tid in sorted(sc._all_tids):
        n = sc._playing_frames.get(tid, 0)
        if n < 30:                      # lewati pecahan sangat pendek
            continue
        r = sc._calc_mpi_raw(tid)
        hasil[tid] = {
            "frame": n,
            "jarak": r["distance_m"],
            "v_rata": r["avg_speed_kmh"],
            "v_maks": r["max_speed_kmh"],
            "percepatan": r["acceleration_ms2"],
        }
    return hasil


def main() -> int:
    npz = np.load(sys.argv[1])
    print(f"sampel    : {len(npz['posisi'])} posisi, fps {float(npz['fps']):.0f}, "
          f"stride {int(npz['stride'])}")

    lama = jalankan(npz, "none")
    baru = jalankan(npz, "roboflow")
    sama = sorted(set(lama) & set(baru))
    print(f"track dibandingkan: {len(sama)} (hanya yang >= 30 frame)\n")

    print(f"{'track':>6} {'frame':>6} | {'jarak lama':>11} {'jarak baru':>11} {'turun':>7} "
          f"| {'v.maks lama':>12} {'v.maks baru':>12} | {'acc lama':>9} {'acc baru':>9}")
    print("-" * 110)
    tot_l = tot_b = 0.0
    accs_l, accs_b, vmax_l, vmax_b = [], [], [], []
    for tid in sama:
        a, b = lama[tid], baru[tid]
        tot_l += a["jarak"]; tot_b += b["jarak"]
        accs_l.append(a["percepatan"]); accs_b.append(b["percepatan"])
        vmax_l.append(a["v_maks"]);     vmax_b.append(b["v_maks"])
        turun = (a["jarak"] - b["jarak"]) / a["jarak"] * 100 if a["jarak"] else 0.0
        if len(sama) <= 25 or tid in sama[:12] or tid in sama[-6:]:
            print(f"{tid:>6} {a['frame']:>6} | {a['jarak']:>11.2f} {b['jarak']:>11.2f} "
                  f"{turun:>6.1f}% | {a['v_maks']:>12.2f} {b['v_maks']:>12.2f} "
                  f"| {a['percepatan']:>9.1f} {b['percepatan']:>9.1f}")
    if len(sama) > 25:
        print(f"{'...':>6}  ({len(sama) - 18} track lain tidak dicetak)")

    print("-" * 110)
    print(f"\nRINGKASAN atas {len(sama)} track")
    print(f"  total jarak          lama {tot_l:9.1f} m   baru {tot_b:9.1f} m   "
          f"turun {(tot_l-tot_b)/tot_l*100:.1f}%")
    print(f"  percepatan median    lama {np.median(accs_l):9.1f}     baru {np.median(accs_b):9.1f}     "
          f"m/s2   (manusia wajar < 10)")
    print(f"  percepatan maks      lama {max(accs_l):9.1f}     baru {max(accs_b):9.1f}     m/s2")
    print(f"  kecepatan maks medi. lama {np.median(vmax_l):9.2f}     baru {np.median(vmax_b):9.2f}     "
          f"km/jam (sprint manusia < 30)")
    print(f"  kecepatan maks tert. lama {max(vmax_l):9.2f}     baru {max(vmax_b):9.2f}     km/jam")

    n_tak_wajar_l = sum(1 for a in accs_l if a > 10)
    n_tak_wajar_b = sum(1 for a in accs_b if a > 10)
    print(f"\n  track dengan percepatan di atas 10 m/s2 (mustahil untuk manusia):")
    print(f"    lama {n_tak_wajar_l}/{len(sama)} ({n_tak_wajar_l/len(sama)*100:.0f}%)   "
          f"baru {n_tak_wajar_b}/{len(sama)} ({n_tak_wajar_b/len(sama)*100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

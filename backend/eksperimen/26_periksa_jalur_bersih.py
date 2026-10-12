"""
26_periksa_jalur_bersih.py

Menjawab: saat PATH_CLEAN_METHOD=roboflow menyala, BERAPA BANYAK track yang
benar benar memakai jalur bersih, dan berapa yang diam diam jatuh ke jalur
mentah.

Kenapa perlu: `_calc_mpi_raw` hanya memakai hasil jalur bersih kalau
`_clean_readout` mengembalikan kecepatan yang TIDAK kosong. Kalau kosong, ia
jatuh ke jalur mentah TANPA peringatan apa pun. Terukur: memperbaiki jalur
mentah mengubah percepatan median 53,7 jadi 12,4, padahal kalau jalur bersih
bekerja angka itu mestinya tidak berubah sama sekali. Jadi ada yang jatuh.

Menjalankan pipeline penuh sebentar lalu memeriksa keadaan dalam StatsCalculator.

Contoh:
    DETECTOR_METHOD=rfdetr VIDEO=/home/paul/q1_4menit.mp4 N=3000 \
      RENDER_OUTPUT_VIDEO=0 python eksperimen/26_periksa_jalur_bersih.py
"""

import os
import sys
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))
os.chdir(AKAR)

from dotenv import load_dotenv            # noqa: E402
load_dotenv(AKAR / ".env")

import cv2                                # noqa: E402
from pipeline.video_processor import create_video_processor   # noqa: E402

VIDEO = os.environ["VIDEO"]
N     = int(os.getenv("N", "3000"))


def main():
    vp = create_video_processor(mongo_db=None, redis_client=None)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    vp._source_fps = fps
    vp._target_fps = max(1, round(fps / 2))
    if vp._stats:
        vp._stats.fps = vp._target_fps

    # WAJIB melangkah 2 frame seperti pipeline. Kalau setiap frame dibaca
    # padahal statistik diberi tahu fps efektifnya 30, jeda sebenarnya jadi
    # 16,7 ms bukan 33 ms, dan hasilnya TIDAK sebanding dengan run produksi.
    # Batas langkah jalur bersih juga ikut bergeser, sehingga jumlah track yang
    # jatuh ke jalur mentah akan terbaca lebih kecil dari yang sebenarnya.
    LANGKAH = 2
    n = iv = 0
    while n < N:
        ok, fr = cap.read()
        if not ok:
            break
        if iv % LANGKAH == 0:
            try:
                vp._process_frame(fr, n, int(iv / fps * 1000))
            except Exception as e:
                print(f"  galat frame {n}: {type(e).__name__} {e}")
                break
            n += 1
        iv += 1
    cap.release()
    print(f"  {n} frame diproses dari {iv} frame video (langkah {LANGKAH})")

    st = vp._stats
    if st is None:
        print("  StatsCalculator tidak ada"); return

    print(f"\n  === {n} frame | jalur bersih menyala: {st._clean_on} ===")
    tids = [t for t in st._all_tids if st._playing_frames.get(t, 0) >= 20]
    print(f"  track dengan >=20 frame: {len(tids)}")

    pakai_bersih, jatuh_mentah = [], []
    acc_bersih, acc_mentah = [], []
    for t in tids:
        _, c_ms = st._clean_readout(t)
        if c_ms:
            pakai_bersih.append(t)
            acc_bersih.append(st._accel_from_ms(c_ms))
        else:
            jatuh_mentah.append(t)
            acc_mentah.append(st._calc_acceleration(t))

    print(f"  MEMAKAI jalur bersih : {len(pakai_bersih)}")
    print(f"  JATUH ke jalur mentah: {len(jatuh_mentah)}")
    for nama, arr in (("jalur bersih", acc_bersih), ("jalur mentah", acc_mentah)):
        if arr:
            a = np.array(arr)
            print(f"    percepatan {nama:<13s} median {np.median(a):6.1f} m/s2  "
                  f"mustahil >10 {(a>10).sum()}/{len(a)}")

    # Kenapa yang jatuh itu jatuh: periksa isi penampungnya.
    if jatuh_mentah:
        print("\n  SEBAB yang jatuh, 10 contoh:")
        for t in jatuh_mentah[:10]:
            buf = st._clean_buf.get(t) or []
            terkunci = len(st._clean_speeds.get(t, []))
            seg = st._clean_segment(buf, len(buf)) if len(buf) >= 3 else None
            print(f"    track {t:5d} | titik di penampung {len(buf):4d} | "
                  f"terkunci {terkunci:4d} | "
                  f"potongan {'None (semua langkah melebihi batas)' if seg is None else 'ada'}")
        batas = 36.0 / 3.6 / max(float(st.fps), 1e-6)
        print(f"  batas langkah = {batas:.3f} m per frame pada fps {st.fps}")


if __name__ == "__main__":
    main()

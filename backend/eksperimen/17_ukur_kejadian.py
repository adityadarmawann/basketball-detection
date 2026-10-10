"""
17_ukur_kejadian.py

Menjalankan PIPELINE PENUH lalu mencocokkan tiap kejadian MADE_FG dengan
KEBENARAN dari papan skor, satu per satu menurut WAKTUNYA.

KENAPA PER KEJADIAN, BUKAN TOTAL SKOR
Skor total MENIPU. Pernah terjadi: dua konfigurasi sama sama memberi 12 poin
lawan kebenaran 11, dan sempat disimpulkan "5 dari 5 bola masuk benar".
Setelah dicocokkan per waktu, ternyata deteksi palsu dan yang terlewat SALING
MENIADAKAN. Jangan pernah menilai dari total lagi.

KEBENARAN
Dibaca manual dari papan skor video, ditulis di TRUTH di bawah sebagai
(detik, poin). Papan skor TELAT 3 sampai 5 detik dari bola masuk dan tidak
tetap, jadi jendela pencocokan dibuat longgar 8 detik. JAM TEMBAKAN lebih bisa
dipercaya untuk memastikan: reset ke 24 berarti ganti penguasaan, reset ke 14
berarti rebound serangan setelah bola kena ring.

Contoh:
    DETECTOR_METHOD=rfdetr VIDEO=/path/ke/potongan.mp4 RENDER_OUTPUT_VIDEO=0 \
    python eksperimen/17_ukur_kejadian.py
"""

import os
import sys
import math
import time
import asyncio
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))
os.chdir(AKAR)

from dotenv import load_dotenv            # noqa: E402
load_dotenv(AKAR / ".env")

import pipeline.event_engine as EE        # noqa: E402

# Kebenaran untuk potongan detik 60-300 dari Q1-UNESA-UBAYA-CUT-2-17.mp4.
# Waktu di sini sudah DIMUNDURKAN kira kira 4 detik dari saat papan skor
# berubah, untuk memperkirakan saat bola benar benar masuk.
TRUTH = [(19, 2), (59, 2), (90, 2), (119, 2), (146, 3)]
JENDELA = float(os.getenv("JENDELA_COCOK", "8"))

HOOP_KIRI  = (1.575, 7.5)
HOOP_KANAN = (26.425, 7.5)

REKAM = []
_asli = EE.EventEngine.process


def process(self, frame_data):
    hasil = _asli(self, frame_data)
    for e in (hasil or []):
        if str(e.get("type", "")).upper() == "MADE_FG":
            REKAM.append({k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in e.items()})
    return hasil


EE.EventEngine.process = process

from pipeline.video_processor import create_video_processor   # noqa: E402
from pipeline.detector import DETECTOR_METHOD                 # noqa: E402


def jam(ms):
    s = (ms or 0) / 1000.0
    return f"{int(s)//60}:{int(s)%60:04.1f}"


async def main():
    vp = create_video_processor(mongo_db=None, redis_client=None)
    t0 = time.perf_counter()
    await vp.process_video(video_path=os.environ["VIDEO"],
                           match_id=f"ukur_{DETECTOR_METHOD}", lock_quarter=True)
    lama = time.perf_counter() - t0

    benar = palsu = nilai_benar = 0
    ketemu = set()
    total = 0
    print(f"\n  === {DETECTOR_METHOD} | {len(REKAM)} deteksi | {lama/60:.1f} menit ===")
    for e in REKAM:
        t = (e.get("timestamp_ms") or 0) / 1000.0
        tiga = bool(e.get("is_three"))
        total += 3 if tiga else 2
        cp = e.get("court_pos")
        d = (min(math.dist(cp, HOOP_KIRI), math.dist(cp, HOOP_KANAN))
             if isinstance(cp, (list, tuple)) and len(cp) == 2 else 0.0)
        dekat = sorted((abs(t - tt), tt, pp) for tt, pp in TRUTH)
        d0, tt, pp = dekat[0]
        if d0 <= JENDELA and tt not in ketemu:
            benar += 1
            ketemu.add(tt)
            ok = (3 if tiga else 2) == pp
            nilai_benar += ok
            print(f"     {jam(e.get('timestamp_ms')):>7s} "
                  f"{'3 angka' if tiga else '2 angka'} {d:5.1f}m  "
                  f"cocok {tt//60}:{tt%60:02d} (beda {d0:.0f}s)  "
                  f"nilai {'BENAR' if ok else f'salah, harusnya {pp}'}")
        else:
            palsu += 1
            print(f"     {jam(e.get('timestamp_ms')):>7s} "
                  f"{'3 angka' if tiga else '2 angka'} {d:5.1f}m  PALSU")

    lewat = [f"{tt//60}:{tt%60:02d}" for tt, _ in TRUTH if tt not in ketemu]
    benar_total = sum(p for _, p in TRUTH)
    print(f"\n     asli ditemukan {benar}/{len(TRUTH)}   nilai 2/3 benar {nilai_benar}   "
          f"palsu {palsu}   terlewat {len(lewat)} ({', '.join(lewat) or 'tidak ada'})")
    print(f"     poin sistem {total}  lawan KEBENARAN {benar_total}")
    print(f"     skor mesin  {vp._events_engine.score}")


if __name__ == "__main__":
    asyncio.run(main())

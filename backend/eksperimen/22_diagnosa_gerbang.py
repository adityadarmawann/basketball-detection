"""
22_diagnosa_gerbang.py

Menjawab DUA pertanyaan sekaligus dari satu kali jalan:
  1. Kenapa bola masuk di 1:30 dan 2:26 TIDAK PERNAH terdeteksi di setelan
     apa pun.
  2. Kenapa palsu di 3:52 LOLOS di setelan apa pun, termasuk gerbang tepi rim.

Caranya: membungkus `EventEngine._detect_field_goal` dan merekam keadaan dalam
tiap kali dipanggil, lalu mencetak garis waktu di sekitar saat saat itu. Yang
dicari adalah DI SYARAT MANA gerbangnya berhenti.

Cara membaca hasilnya:
  - `dwell` tidak pernah mencapai ambangnya  -> bola tidak pernah MASUK zona
    ring. Masalahnya deteksi atau posisi bola, bukan gerbangnya.
  - `dwell` mencapai ambang tapi tidak ada kejadian -> gerbangnya menolak di
    syarat "datang dari ATAS" atau "konfirmasi di BAWAH rim".
  - `bola=X` berarti frame itu tidak ada bola sama sekali.
  - `kira` berarti posisi bola hasil PERKIRAAN pelacak, bukan deteksi nyata.
    Konfirmasi di bawah rim menolak yang perkiraan, itu memang disengaja.

Contoh:
    DETECTOR_METHOD=rfdetr VIDEO=/home/paul/q1_4menit.mp4 RENDER_OUTPUT_VIDEO=0 \
      python eksperimen/22_diagnosa_gerbang.py
"""

import math
import os
import sys
import asyncio
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))
os.chdir(AKAR)

from dotenv import load_dotenv            # noqa: E402
load_dotenv(AKAR / ".env")

import pipeline.event_engine as EE        # noqa: E402

# Saat yang diperiksa, dalam DETIK pada potongan 240 detik.
# 90 dan 146 = bola masuk yang selalu terlewat. 232 = palsu yang selalu lolos.
SAAT = [int(x) for x in os.getenv("SAAT", "90,146,232").split(",")]
LEBAR = float(os.getenv("LEBAR", "6"))     # detik sebelum dan sesudah

REKAM = []
_asli = EE.EventEngine._detect_field_goal


def _bungkus(self, ball, hoops, is_court, players, quarter, ts_ms):
    sebelum_dwell = self._ball_in_hoop_frames
    hasil = _asli(self, ball, hoops, is_court, players, quarter, ts_ms)

    bola_px = ball.get("center") if ball else None
    jarak = None
    if bola_px is not None:
        ring = self._nearest_hoop_px_center(bola_px)
        if ring is not None:
            jarak = math.dist(bola_px, ring)

    REKAM.append({
        "t": (ts_ms or 0) / 1000.0,
        "ada_bola": bola_px is not None,
        "kira": bool(ball.get("predicted")) if ball else False,
        "n_ring": len(hoops or []),
        "umur_cache": self._hoop_cache_age,
        "jarak": jarak,
        "dwell_sebelum": sebelum_dwell,
        "dwell": self._ball_in_hoop_frames,
        "bawah_ok": self._fg_below_ok,
        "n_hist": len(self._ball_px_hist),
        "kejadian": len(hasil or []),
    })
    return hasil


EE.EventEngine._detect_field_goal = _bungkus

from pipeline.video_processor import create_video_processor   # noqa: E402
from pipeline.detector import DETECTOR_METHOD                 # noqa: E402


def jam(t):
    return f"{int(t)//60}:{t%60:04.1f}"


async def main():
    vp = create_video_processor(mongo_db=None, redis_client=None)
    await vp.process_video(video_path=os.environ["VIDEO"],
                           match_id=f"diag_{DETECTOR_METHOD}", lock_quarter=True)

    ambang = vp._events_engine._fg_zone_dwell_f
    print(f"\n  ambang dwell = {ambang} frame | radius = {EE.FG_HOOP_RADIUS_PX} px"
          f" | gerbang rim = {EE.FG_RIM_GATE}")
    print(f"  {len(REKAM)} frame terekam\n")

    for s in SAAT:
        jendela = [r for r in REKAM if abs(r["t"] - s) <= LEBAR]
        print(f"  ═══ sekitar {jam(float(s))} ({len(jendela)} frame) ═══")
        if not jendela:
            print("     tidak ada frame terekam di jendela ini\n")
            continue

        dwell_maks = max(r["dwell"] for r in jendela)
        n_bola = sum(r["ada_bola"] for r in jendela)
        n_kira = sum(r["kira"] for r in jendela)
        dekat = [r["jarak"] for r in jendela if r["jarak"] is not None]
        print(f"     bola ada {n_bola}/{len(jendela)} ({n_kira} perkiraan) | "
              f"dwell maks {dwell_maks} dari {ambang} | "
              f"jarak terdekat {min(dekat):.0f} px" if dekat else
              f"     bola ada {n_bola}/{len(jendela)} | dwell maks {dwell_maks}")

        # Cetak hanya frame yang MENARIK: ada kejadian, dwell bergerak, atau
        # bola cukup dekat ke ring. Mencetak semuanya membuat tidak terbaca.
        for r in jendela:
            menarik = (r["kejadian"] or r["dwell"] or r["dwell_sebelum"]
                       or (r["jarak"] is not None and r["jarak"] < EE.FG_HOOP_RADIUS_PX * 3))
            if not menarik:
                continue
            j = f"{r['jarak']:6.0f}" if r["jarak"] is not None else "     X"
            print(f"     {jam(r['t']):>7s} bola={'Y' if r['ada_bola'] else 'X'}"
                  f"{'(kira)' if r['kira'] else '      '} "
                  f"ring={r['n_ring']} umur={r['umur_cache']:2d} jarak={j} "
                  f"dwell={r['dwell_sebelum']}->{r['dwell']} "
                  f"bawah={'Y' if r['bawah_ok'] else 'X'} hist={r['n_hist']:2d}"
                  f"{'  *** KEJADIAN ***' if r['kejadian'] else ''}")

        if dwell_maks < ambang:
            print(f"     => dwell tidak pernah capai {ambang}: bola tidak pernah "
                  f"MASUK zona ring. Masalah DETEKSI atau POSISI bola.")
        elif not any(r["kejadian"] for r in jendela):
            print(f"     => dwell cukup tapi TIDAK ada kejadian: gerbang menolak "
                  f"di syarat 'datang dari ATAS' atau 'konfirmasi di BAWAH rim'.")
        print()


if __name__ == "__main__":
    asyncio.run(main())

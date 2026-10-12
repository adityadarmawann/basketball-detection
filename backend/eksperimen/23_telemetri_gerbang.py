"""
23_telemetri_gerbang.py

Merekam KEADAAN GERBANG bola masuk per frame ke satu berkas JSON supaya
ambang untuk saklar baru bisa dipilih dari DATA, bukan dikira kira.

Bedanya dengan 22_diagnosa_gerbang.py: skrip 22 hanya mencetak garis waktu di
sekitar beberapa saat yang sudah diketahui, dan hanya memakai jarak ke TITIK
TENGAH ring. Skrip ini merekam SELURUH klip beserta KOTAK RIM mentahnya, jadi
sesudahnya bisa dihitung:
  - jarak bola ke pusat KOTAK RIM, bukan ke titik tengah ring
  - posisi tegak bola relatif tepi bawah kotak rim
  - panjang satu rentetan dwell, dari mulai sampai putus
  - apakah frame penentu itu deteksi NYATA atau perkiraan pelacak

Skrip ini TIDAK mengubah perilaku gerbang. Dia hanya membungkus dan mencatat.
Yang boleh memutuskan terima atau tolak tetap 17_ukur_kejadian.py.

Contoh:
    VIDEO=/home/paul/q1_4menit.mp4 \
      /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/23_telemetri_gerbang.py
"""

import json
import os
import sys
import time
import asyncio
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))
os.chdir(AKAR)

from dotenv import load_dotenv            # noqa: E402
load_dotenv(AKAR / ".env")

import pipeline.event_engine as EE        # noqa: E402

KELUARAN = os.getenv("TELEMETRI_OUT", "/home/paul/telemetri_gerbang.json")

REKAM: list = []
KEJADIAN: list = []

_asli_fg = EE.EventEngine._detect_field_goal
_asli_proc = EE.EventEngine.process


def _bungkus_fg(self, ball, hoops, is_court, players, quarter, ts_ms):
    dwell_sebelum = self._ball_in_hoop_frames
    hasil = _asli_fg(self, ball, hoops, is_court, players, quarter, ts_ms)

    bola_px = ball.get("center") if ball else None
    # Acuan ring dan kotak rim dibaca SESUDAH panggilan aslinya. Aman, karena
    # keduanya diperbarui di process() sebelum gerbang dipanggil dan gerbang
    # sendiri tidak menyentuhnya.
    rim = self._nearest_rim_bbox_px(bola_px) if bola_px else None
    ring = self._nearest_hoop_px_center(bola_px) if bola_px else None

    REKAM.append({
        "f": self.frame_count,
        "t": round((ts_ms or 0) / 1000.0, 3),
        "b": [round(float(bola_px[0]), 1), round(float(bola_px[1]), 1)] if bola_px else None,
        "nyata": (not bool(ball.get("predicted"))) if ball else False,
        "rim": [round(float(v), 1) for v in rim] if rim else None,
        "ring": [round(float(ring[0]), 1), round(float(ring[1]), 1)] if ring else None,
        "umur_rim": self._rim_cache_age,
        "umur_ring": self._hoop_cache_age,
        "d0": dwell_sebelum,
        "d1": self._ball_in_hoop_frames,
        "kej": len(hasil or []),
    })
    return hasil


def _bungkus_proc(self, frame_data):
    hasil = _asli_proc(self, frame_data)
    for e in (hasil or []):
        if str(e.get("type", "")).upper() in ("MADE_FG", "MADE_FT"):
            KEJADIAN.append({
                "t": round((e.get("timestamp_ms") or 0) / 1000.0, 2),
                "jenis": e.get("type"),
                "tiga": bool(e.get("is_three")),
            })
    return hasil


EE.EventEngine._detect_field_goal = _bungkus_fg
EE.EventEngine.process = _bungkus_proc

from pipeline.video_processor import create_video_processor   # noqa: E402
from pipeline.detector import DETECTOR_METHOD, create_detector  # noqa: E402


async def main():
    print(f"  detektor kelas = {type(create_detector()).__name__}", flush=True)
    vp = create_video_processor(mongo_db=None, redis_client=None)
    t0 = time.perf_counter()
    await vp.process_video(video_path=os.environ["VIDEO"],
                           match_id=f"telem_{DETECTOR_METHOD}", lock_quarter=True)
    lama = time.perf_counter() - t0

    ee = vp._events_engine
    meta = {
        "detektor": DETECTOR_METHOD,
        "fps": ee.fps,
        "dwell_ambang": ee._fg_zone_dwell_f,
        "radius_px": EE.FG_HOOP_RADIUS_PX,
        "rim_gate": EE.FG_RIM_GATE,
        "confirm_below": EE.FG_CONFIRM_BELOW,
        "below_margin": EE.FG_BELOW_MARGIN_PX,
        "min_dy": ee._fg_min_dy,
        "menit": round(lama / 60.0, 2),
    }
    with open(KELUARAN, "w") as f:
        json.dump({"meta": meta, "frame": REKAM, "kejadian": KEJADIAN}, f)
    print(f"\n  {meta}")
    print(f"  {len(REKAM)} frame, {len(KEJADIAN)} kejadian skor -> {KELUARAN}")
    for k in KEJADIAN:
        print(f"     {int(k['t'])//60}:{k['t']%60:04.1f}  {k['jenis']} "
              f"{'3 angka' if k['tiga'] else '2 angka'}")


if __name__ == "__main__":
    asyncio.run(main())

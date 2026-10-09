"""
15_jalankan_pipeline.py
Menjalankan PIPELINE PENUH (video_processor) pada satu video, memakai apa pun
yang disetel di .env. Tanpa backend, tanpa MongoDB.

    VIDEO=... MATCH_ID=... python eksperimen/15_jalankan_pipeline.py
"""
import os, sys, asyncio, time, json
from pathlib import Path
AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))

# WAJIB sebelum mengimpor pipeline: modul modul itu membaca os.getenv saat
# DIIMPOR, jadi .env harus sudah masuk lebih dulu. main.py melakukan ini juga.
from dotenv import load_dotenv
load_dotenv(AKAR / ".env")

from pipeline.video_processor import create_video_processor, TARGET_FPS   # noqa: E402
from pipeline.detector import DETECTOR_METHOD                             # noqa: E402

VIDEO    = os.environ["VIDEO"]
MATCH_ID = os.getenv("MATCH_ID", "uji")

async def main():
    print("=" * 70)
    print(f"  DETECTOR_METHOD = {DETECTOR_METHOD}")
    print(f"  video           = {VIDEO}")
    print(f"  match_id        = {MATCH_ID}")
    print("=" * 70)
    vp = create_video_processor(mongo_db=None, redis_client=None)
    t0 = time.perf_counter()
    await vp.process_video(video_path=VIDEO, match_id=MATCH_ID, lock_quarter=True)
    lama = time.perf_counter() - t0
    print(f"\n  SELESAI dalam {lama/60:.1f} menit, status={vp._status}")

    # Ringkas apa yang dihasilkan
    stats = getattr(vp, "_stats", None)
    ev    = getattr(vp, "_events_engine", None)
    trk   = getattr(vp, "_tracker", None)
    peta  = dict(getattr(trk, "track_jersey_map", {}) or {}) if trk else {}
    nomor = sorted({v for v in peta.values() if v is not None})
    print(f"  nomor punggung terkunci : {len(peta)} track -> {len(nomor)} nomor unik")
    print(f"  nomor                   : {nomor}")
    if ev is not None:
        print(f"  skor                    : {getattr(ev, 'score', None)}")
        evs = getattr(ev, "events", None) or getattr(ev, "_events", None)
        if evs is not None:
            print(f"  kejadian tercatat       : {len(evs)}")
            from collections import Counter
            jenis = Counter(e.get("type") for e in evs if isinstance(e, dict))
            for k, v in jenis.most_common(10):
                print(f"      {str(k):22s} {v}")
    out = Path(os.getenv("OUTPUT_PATH", "./output_dev")) / MATCH_ID
    if out.exists():
        print(f"\n  berkas keluaran di {out}:")
        for p in sorted(out.iterdir()):
            print(f"    {p.name}  ({p.stat().st_size/1024:.0f} KB)")

asyncio.run(main())

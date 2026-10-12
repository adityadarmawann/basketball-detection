"""
28_ukur_tim_dan_bbox.py

Menjawab tiga keluhan nyata, dengan angka, bukan kesan:
  1. bbox MELAYANG: kotak yang muncul padahal tidak ada deteksi di frame itu
     (posisi hasil perkiraan pelacak).
  2. pemain TANPA bbox: deteksi yang ada tapi tidak dapat track_id.
  3. label TIM berantakan setelah kamera berpindah: berapa kali label A/B
     sebuah track BERUBAH, dan apakah perubahannya menumpuk di sekitar
     pergantian sudut kamera.

Detektor, pelacak, dan penentu tim saja. Tanpa pose, aksi, kejadian, statistik.

Contoh:
    DETECTOR_METHOD=rfdetr VIDEO=/home/paul/q1_4menit.mp4 \
      python eksperimen/28_ukur_tim_dan_bbox.py
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
from pipeline.jersey_ocr import create_jersey_ocr                 # noqa: E402

VIDEO   = os.environ["VIDEO"]
LANGKAH = int(os.getenv("LANGKAH", "2"))
CUT_BEDA = float(os.getenv("CUT_BEDA", "25"))   # beda gambar = ganti sudut keras


def main():
    det = create_detector()
    trk = create_tracker()
    trk.frame_rate = 30
    ocr = create_jersey_ocr()

    cap = cv2.VideoCapture(VIDEO)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f0 = cap.read()
    if not ok:
        print("  video tidak terbaca"); return
    if hasattr(trk, "initialize"):
        trk.initialize(f0.shape)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tim_per_track = defaultdict(list)     # tid -> [(frame, tim)]
    n_fr = 0
    n_deteksi = n_bertrack = n_melayang = 0
    cut_frames = []
    abu_sebelum = None

    for i in range(0, n_total, LANGKAH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        n_fr += 1

        kecil = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY)
        if abu_sebelum is not None:
            if float(np.mean(cv2.absdiff(kecil, abu_sebelum))) > CUT_BEDA:
                cut_frames.append(n_fr)
        abu_sebelum = kecil

        d = det.detect(fr)
        n_deteksi += len(d.get("players") or [])
        hasil = trk.update(fr, d)
        pemain = hasil.get("tracked_players", []) or []
        n_bertrack += len(pemain)
        # bbox MELAYANG: pelacak menandai posisi hasil perkiraan.
        n_melayang += sum(1 for p in pemain if p.get("predicted"))

        try:
            ocr.process_frame(fr, pemain)
        except Exception:
            pass
        for p in pemain:
            t = p.get("team")
            if t in ("A", "B"):
                tim_per_track[p["track_id"]].append((n_fr, t))
    cap.release()

    print(f"\n  === {n_fr} frame | {DETECTOR_METHOD} | "
          f"{len(cut_frames)} ganti sudut keras ===")
    print(f"  deteksi pemain          {n_deteksi}")
    print(f"  dapat track_id          {n_bertrack}"
          f"  ({n_bertrack/max(n_deteksi,1)*100:.1f}% dari deteksi)")
    print(f"  TANPA track_id          {max(0, n_deteksi-n_bertrack)}"
          f"  <- pemain tanpa bbox")
    print(f"  bbox MELAYANG (perkiraan) {n_melayang}"
          f"  ({n_melayang/max(n_bertrack,1)*100:.1f}% dari bbox)")

    # Stabilitas label tim.
    n_track = len(tim_per_track)
    tukar_total = 0
    tukar_dekat_cut = 0
    track_bertukar = 0
    for tid, seri in tim_per_track.items():
        tukar = [f for (f, t), (f2, t2) in zip(seri, seri[1:]) if t != t2]
        if tukar:
            track_bertukar += 1
            tukar_total += len(tukar)
            for f in tukar:
                if any(abs(f - c) <= 3 for c in cut_frames):
                    tukar_dekat_cut += 1
    n_trans = sum(max(0, len(s) - 1) for s in tim_per_track.values())
    print(f"\n  track dengan label tim  {n_track}")
    print(f"  track yang BERTUKAR tim {track_bertukar}"
          f"  ({track_bertukar/max(n_track,1)*100:.0f}%)")
    print(f"  pertukaran label        {tukar_total} dari {n_trans} transisi"
          f"  ({tukar_total/max(n_trans,1)*100:.2f}%)")
    print(f"    di antaranya DEKAT ganti sudut: {tukar_dekat_cut}"
          f"  ({tukar_dekat_cut/max(tukar_total,1)*100:.0f}%)")


if __name__ == "__main__":
    main()

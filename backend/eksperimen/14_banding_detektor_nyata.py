"""
14_banding_detektor_nyata.py

Membandingkan DETECTOR_METHOD=yolo lawan rfdetr dengan menjalankan RANTAI NYATA
sistem, bukan model telanjang dan bukan hitungan kotak:

    frame -> detector.detect() -> tracker.update() -> attach_detection_classes()
          -> jersey_ocr.process()

Yang diukur adalah hal hal yang benar benar dipedulikan sistem, bukan jumlah
kotak per frame:
  1. Berapa persen frame bolanya ketemu          (penguasaan dan tembakan)
  2. Berapa NOMOR PUNGGUNG yang akhirnya terkunci (identitas pemain)
  3. Berapa banyak track_id untuk 10 pemain       (kestabilan pelacakan)
  4. Berapa kali kelas ball-in-basket muncul      (deteksi tembakan masuk)
  5. Waktu per frame

Video 60 fps diturunkan ke 25 fps persis seperti pipeline (TARGET_FPS=25),
supaya fisika pelacak dan jumlah frame sebanding.

Dijalankan dua kali:
    DETECTOR_METHOD=yolo   python eksperimen/14_banding_detektor_nyata.py
    DETECTOR_METHOD=rfdetr python eksperimen/14_banding_detektor_nyata.py
"""

import os
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.detector import create_detector, DETECTOR_METHOD      # noqa: E402
from pipeline.tracker import create_tracker                          # noqa: E402
from pipeline.jersey_ocr import create_jersey_ocr                    # noqa: E402

try:
    from pipeline.detector_rfdetr import attach_detection_classes
except Exception:
    attach_detection_classes = None

VIDEO  = os.getenv("VIDEO", "/home/paul/computer-vision/campus-league/basketball/"
                            "smart-vision-cl/test-video/test-Quarter-1_UNESA-UBAYA.mp4")
DETIK  = float(os.getenv("DETIK", "90"))     # berapa detik video yang diproses
MULAI  = float(os.getenv("MULAI", "60"))     # mulai dari detik ke berapa
TARGET_FPS = 25                               # sama dengan pipeline


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"video tidak bisa dibuka: {VIDEO}")
        return
    fps_sumber = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Turunkan ke 25 fps persis seperti pipeline: ambil tiap langkah frame.
    langkah = max(1, round(fps_sumber / TARGET_FPS))
    f_mulai = int(MULAI * fps_sumber)
    n_proses = int(DETIK * fps_sumber / langkah)

    print("=" * 74)
    print(f"  DETECTOR_METHOD = {DETECTOR_METHOD}")
    print("=" * 74)
    print(f"  video   : {W}x{H}, {fps_sumber:.0f} fps, {n_total} frame")
    print(f"  potongan: detik {MULAI:.0f} sampai {MULAI+DETIK:.0f}, langkah {langkah} "
          f"(jadi {TARGET_FPS} fps), {n_proses} frame diproses")

    det = create_detector()
    print(f"  detektor: {type(det).__name__}")
    if DETECTOR_METHOD == "rfdetr" and type(det).__name__ != "RFDetrDetector":
        print("  PERHATIAN: jatuh ke YOLO, hasil di bawah BUKAN RF-DETR.")

    trk = create_tracker()
    trk.frame_rate = TARGET_FPS
    if hasattr(trk, "initialize"):
        try:
            trk.initialize((H, W, 3))
        except Exception:
            pass
    ocr = create_jersey_ocr()
    print(f"  pelacak : {type(trk).__name__}   OCR: {type(ocr).__name__}\n")

    frame_bola = 0
    n_pemain = n_wasit = n_nomor_kotak = n_bib = 0
    track_ids = set()
    waktu = []
    n_frame = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_mulai)
    idx = f_mulai
    while n_frame < n_proses:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        idx += langkah

        t0 = time.perf_counter()
        detections = det.detect(frame)
        tracking = trk.update(frame, detections)
        tracked = tracking.get("tracked_players", [])
        if attach_detection_classes is not None:
            attach_detection_classes(detections, tracked)
        if tracked:
            ocr.process(frame, tracked, tracker=trk)
        waktu.append(time.perf_counter() - t0)

        frame_bola += 1 if detections.get("ball") else 0
        n_pemain   += len(detections.get("players", []))
        n_wasit    += len(detections.get("referees", []))
        n_nomor_kotak += len(detections.get("numbers", []) or [])
        n_bib      += len(detections.get("balls_in_basket", []) or [])
        for t in tracked:
            track_ids.add(t.get("track_id"))

        n_frame += 1
        if n_frame % 250 == 0:
            print(f"    ... {n_frame}/{n_proses} frame, "
                  f"{len(getattr(trk,'track_jersey_map',{}) or {})} nomor terkunci")
    cap.release()

    peta = dict(getattr(trk, "track_jersey_map", {}) or {})
    nomor_unik = sorted({v for v in peta.values() if v is not None})

    print(f"\n  HASIL dari {n_frame} frame\n")
    print(f"  1. BOLA ketemu di            : {frame_bola}/{n_frame} frame "
          f"({frame_bola/max(n_frame,1)*100:.1f} %)")
    print(f"  2. NOMOR PUNGGUNG terkunci   : {len(peta)} track mendapat nomor")
    print(f"     nomor unik yang terbaca   : {nomor_unik}")
    print(f"     jumlah nomor unik         : {len(nomor_unik)}")
    print(f"  3. track_id dipakai          : {len(track_ids)}  "
          f"(makin sedikit makin stabil; sebenarnya cuma 10 pemain)")
    print(f"  4. ball-in-basket muncul     : {n_bib} kali")
    print(f"  5. kotak nomor dari detektor : {n_nomor_kotak} "
          f"({n_nomor_kotak/max(n_frame,1):.2f} per frame)")
    print()
    print(f"  pemain per frame             : {n_pemain/max(n_frame,1):.2f}")
    print(f"  wasit per frame              : {n_wasit/max(n_frame,1):.2f}")
    print(f"  waktu per frame              : {np.mean(waktu)*1000:.1f} ms "
          f"(rantai deteksi+pelacak+OCR saja)")

    hasil = {
        "metode": DETECTOR_METHOD, "frame": n_frame,
        "bola_persen": frame_bola / max(n_frame, 1) * 100,
        "nomor_terkunci": len(peta), "nomor_unik": nomor_unik,
        "track_id": len(track_ids), "ball_in_basket": n_bib,
        "kotak_nomor": n_nomor_kotak,
        "pemain_per_frame": n_pemain / max(n_frame, 1),
        "wasit_per_frame": n_wasit / max(n_frame, 1),
        "ms_per_frame": float(np.mean(waktu) * 1000),
    }
    out = Path(os.getenv("KELUARAN", f"/tmp/banding_{DETECTOR_METHOD}.json"))
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(f"\n  ditulis: {out}")


if __name__ == "__main__":
    main()

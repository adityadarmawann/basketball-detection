"""
Debug script: extract frame at 5:51, run YOLO to get player bboxes,
then run all 4 OCR variants on each player and print full detail.
"""

import cv2
import numpy as np
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH  = "/home/iot/Videos/test-Quarter-1_UNESA-UBAYA.mp4"
TARGET_SEC  = 5 * 60 + 51   # 5:51

sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

# ── 1. Extract frame ──────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit(f"[ERROR] Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
target_frame_idx = int(TARGET_SEC * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
ret, frame = cap.read()
cap.release()

if not ret:
    sys.exit(f"[ERROR] Cannot read frame at {TARGET_SEC}s (frame #{target_frame_idx})")

h_frame, w_frame = frame.shape[:2]
print(f"Frame: {w_frame}×{h_frame} @ {fps:.1f}fps  →  frame #{target_frame_idx} ({TARGET_SEC}s)")
cv2.imwrite("/tmp/ocr_debug_frame.jpg", frame)
print("Saved: /tmp/ocr_debug_frame.jpg")

# ── 2. Detect players with YOLO ───────────────────────────────────────────────
print("\n[STEP 1] Running YOLO player detection...")
try:
    import torch
    from ultralytics import YOLO

    model_path = os.path.join(BACKEND_DIR, "models/best-object-basketball.pt")
    detector   = YOLO(model_path)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    det_results = detector(frame, device=device, conf=0.3, verbose=False)
    player_bboxes = []
    for r in det_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])
            player_bboxes.append((x1, y1, x2, y2, det_conf))

    print(f"  Detected: {len(player_bboxes)} players")

except Exception as e:
    print(f"  [WARN] YOLO failed: {e}")
    print("  Falling back to manual crops from screenshot coordinates...")
    # Rough estimates from the screenshot (proportional to 1920×1080)
    player_bboxes = [
        (int(w_frame*0.22), int(h_frame*0.08), int(w_frame*0.42), int(h_frame*0.98), 0.9),
        (int(w_frame*0.35), int(h_frame*0.08), int(w_frame*0.55), int(h_frame*0.98), 0.9),
    ]
    print(f"  Using {len(player_bboxes)} manual fallback crops")

# Draw all bbox on frame for reference
frame_annot = frame.copy()
for i, (x1, y1, x2, y2, dc) in enumerate(player_bboxes):
    cv2.rectangle(frame_annot, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_annot, f"P{i+1}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite("/tmp/ocr_debug_detections.jpg", frame_annot)
print("  Saved: /tmp/ocr_debug_detections.jpg")

# ── 3. Load PaddleOCR ────────────────────────────────────────────────────────
print("\n[STEP 2] Loading PaddleOCR...")
from pipeline.jersey_ocr import (
    JerseyOCR, _preprocess_for_ocr, _otsu_variant, _adaptive_variant,
    OCR_MIN_SCORE, OCR_HEIGHT_PX
)

ocr_inst = JerseyOCR()
ocr_inst.load_ocr()
print("  PaddleOCR loaded.")

# ── 4. OCR each player ────────────────────────────────────────────────────────
print("\n[STEP 3] OCR per player")
print("=" * 72)

for i, (x1, y1, x2, y2, det_conf) in enumerate(player_bboxes):
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w_frame, x2); y2c = min(h_frame, y2)
    if x2c <= x1c or y2c <= y1c:
        continue

    player_crop = frame[y1c:y2c, x1c:x2c]
    ph, pw = player_crop.shape[:2]

    # Same heuristic as _back_crop_heuristic
    ny1 = int(ph * 0.10); ny2 = int(ph * 0.75)
    nx1 = int(pw * 0.05); nx2 = int(pw * 0.95)
    number_crop = player_crop[ny1:ny2, nx1:nx2]
    if number_crop.size == 0:
        print(f"\nPlayer {i+1}: crop empty, skip")
        continue

    # Blur score
    gray  = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY)
    blur  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_bright = float(gray.mean())

    print(f"\nPlayer {i+1}  bbox=({x1c},{y1c})→({x2c},{y2c})  size={pw}×{ph}px  det_conf={det_conf:.2f}")
    print(f"  Number region: ({nx1},{ny1})→({nx2},{ny2})  size={number_crop.shape[1]}×{number_crop.shape[0]}px")
    print(f"  Blur(Laplacian): {blur:.1f}   Brightness mean: {mean_bright:.1f}")

    # Save crops
    cv2.imwrite(f"/tmp/ocr_p{i+1}_full.jpg",   player_crop)
    cv2.imwrite(f"/tmp/ocr_p{i+1}_number.jpg",  number_crop)

    # All 4 variants
    processed = _preprocess_for_ocr(number_crop)
    variants  = [
        ("CLAHE",      processed),
        ("CLAHE_inv",  cv2.bitwise_not(processed)),
        ("Otsu",       _otsu_variant(number_crop)),
        ("Adaptive",   _adaptive_variant(number_crop)),
    ]

    # Save variant images for visual inspection
    for vname, vimg in variants:
        cv2.imwrite(f"/tmp/ocr_p{i+1}_{vname}.jpg", vimg)

    print(f"  OCR variants (threshold={OCR_MIN_SCORE}):")
    found_any = False
    for vname, vimg in variants:
        try:
            ocr_results = ocr_inst._ocr.predict(vimg)
        except Exception as exc:
            print(f"    [{vname:10s}] ERROR: {exc}")
            continue

        if not ocr_results:
            print(f"    [{vname:10s}] (no output)")
            continue

        for res in ocr_results:
            texts  = res.get("rec_texts",  []) or []
            scores = res.get("rec_scores", []) or []
            if not texts:
                print(f"    [{vname:10s}] (empty rec_texts)")
                continue
            for text, score in zip(texts, scores):
                found_any = True
                ok      = score >= OCR_MIN_SCORE
                parsed  = JerseyOCR._parse_jersey_number(text) if ok else None
                status  = f"✓ → jersey={parsed}" if ok else f"✗ below threshold"
                print(f"    [{vname:10s}] '{text}'  score={score:.3f}  {status}")

    if not found_any:
        print(f"    ALL VARIANTS: no text detected at all")

print("\n" + "=" * 72)
print("Saved crops:")
print("  /tmp/ocr_debug_frame.jpg        — original frame")
print("  /tmp/ocr_debug_detections.jpg   — frame with player boxes")
for i in range(len(player_bboxes)):
    print(f"  /tmp/ocr_p{i+1}_full.jpg / _number.jpg / _CLAHE.jpg / _Otsu.jpg / _Adaptive.jpg")
print("=" * 72)

"""
Test jersey_no.pt: apakah model bisa localize number bbox lebih akurat dari heuristic?
"""
import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from ultralytics import YOLO
from pipeline.jersey_ocr import JerseyOCR, _preprocess_for_ocr, _otsu_variant, _adaptive_variant, OCR_MIN_SCORE

MODEL_PATH = "models/jersey_no.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
jersey_model = YOLO(MODEL_PATH)
print(f"jersey_no.pt loaded. classes={jersey_model.names}, device={device}")

ocr_inst = JerseyOCR()
ocr_inst.load_ocr()

def run_ocr_on_crop(crop, label=""):
    if crop is None or crop.size == 0:
        return []
    variants = [
        ("CLAHE",     _preprocess_for_ocr(crop)),
        ("CLAHE_inv", cv2.bitwise_not(_preprocess_for_ocr(crop))),
        ("Otsu",      _otsu_variant(crop)),
        ("Adaptive",  _adaptive_variant(crop)),
    ]
    found = []
    for vname, vimg in variants:
        try:
            res = ocr_inst._ocr.predict(vimg)
            if res:
                for r in res:
                    for text, score in zip(r.get("rec_texts",[]) or [], r.get("rec_scores",[]) or []):
                        parsed = JerseyOCR._parse_jersey_number(text)
                        if score >= OCR_MIN_SCORE:
                            found.append((vname, text, score, parsed))
        except: pass
    return found

players = [
    ("white-14", "/tmp/ocr_p4_full.jpg"),
    ("red-11",   "/tmp/ocr_p2_full.jpg"),
]

for name, path in players:
    full = cv2.imread(path)
    h, w = full.shape[:2]
    print(f"\n{'='*60}")
    print(f"Player: {name}  full={w}x{h}")

    # Run jersey_no.pt on the full player crop
    results = jersey_model(full, conf=0.15, verbose=False, device=device)

    model_bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            model_bboxes.append((x1, y1, x2, y2, conf))
            print(f"  jersey_no.pt detected: ({x1},{y1})→({x2},{y2})  conf={conf:.3f}")

    if not model_bboxes:
        print("  jersey_no.pt: nothing detected (even at conf=0.15)")
        print("  → will need heuristic fallback")

    # Heuristic fallback (current approach)
    ny1 = int(h*.10); ny2 = int(h*.75)
    nx1 = int(w*.05); nx2 = int(w*.95)
    heuristic_crop = full[ny1:ny2, nx1:nx2]

    # Chest-only alternative
    cy1 = int(h*.15); cy2 = int(h*.55)
    cx1 = int(w*.10); cx2 = int(w*.90)
    chest_crop = full[cy1:cy2, cx1:cx2]

    print(f"\n  --- OCR comparison ---")

    # 1. Model bbox crop (if found)
    for i, (x1, y1, x2, y2, conf) in enumerate(model_bboxes):
        # Expand slightly for context
        pad = 4
        x1e = max(0, x1-pad); y1e = max(0, y1-pad)
        x2e = min(w, x2+pad); y2e = min(h, y2+pad)
        model_crop = full[y1e:y2e, x1e:x2e]
        cv2.imwrite(f"/tmp/jersey_model_{name}_bbox{i}.jpg", model_crop)
        ocr_res = run_ocr_on_crop(model_crop)
        print(f"  [jersey_no.pt bbox {i}]  size={model_crop.shape[1]}x{model_crop.shape[0]}  conf={conf:.3f}")
        if ocr_res:
            for vname, text, score, parsed in ocr_res:
                print(f"    OK [{vname}] '{text}'  score={score:.3f}  jersey={parsed}")
        else:
            print("    → nothing detected")

    # 2. Current heuristic
    ocr_res = run_ocr_on_crop(heuristic_crop)
    print(f"  [heuristic 10-75%]  size={heuristic_crop.shape[1]}x{heuristic_crop.shape[0]}")
    if ocr_res:
        for vname, text, score, parsed in ocr_res:
            print(f"    OK [{vname}] '{text}'  score={score:.3f}  jersey={parsed}")
    else:
        print("    -> nothing")

    # 3. Chest strip
    ocr_res = run_ocr_on_crop(chest_crop)
    print(f"  [chest 15-55%]  size={chest_crop.shape[1]}x{chest_crop.shape[0]}")
    if ocr_res:
        for vname, text, score, parsed in ocr_res:
            print(f"    OK [{vname}] '{text}'  score={score:.3f}  jersey={parsed}")
    else:
        print("    -> nothing")

    # Draw model results on image
    annotated = full.copy()
    for x1, y1, x2, y2, conf in model_bboxes:
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(annotated, f"{conf:.2f}", (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    # Draw heuristic
    cv2.rectangle(annotated, (nx1,ny1), (nx2,ny2), (255,0,0), 1)  # blue = heuristic
    cv2.rectangle(annotated, (cx1,cy1), (cx2,cy2), (0,0,255), 1)  # red = chest
    cv2.imwrite(f"/tmp/jersey_model_{name}_annotated.jpg", annotated)
    print(f"  Annotated (green=model, blue=heuristic, red=chest): /tmp/jersey_model_{name}_annotated.jpg")

"""
Test apakah jersey_no.pt detect number region setelah upscale 2x.
Bandingkan: model bbox vs heuristic chest strip vs wide strip.
"""
import cv2, numpy as np, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from pipeline.jersey_ocr import JerseyOCR, _preprocess_for_ocr, _otsu_variant, _adaptive_variant, OCR_MIN_SCORE

device = "cuda" if torch.cuda.is_available() else "cpu"
jersey_model = YOLO("models/jersey_no.pt")

ocr_inst = JerseyOCR()
ocr_inst.load_model()   # loads jersey_no.pt into self.model
ocr_inst.load_ocr()

def run_ocr(crop):
    if crop is None or crop.size == 0:
        return None, 0.0
    variants = [
        _preprocess_for_ocr(crop),
        cv2.bitwise_not(_preprocess_for_ocr(crop)),
        _otsu_variant(crop),
        _adaptive_variant(crop),
    ]
    best_num, best_sc = None, 0.0
    for vimg in variants:
        try:
            res = ocr_inst._ocr.predict(vimg)
            if res:
                for r in res:
                    for text, score in zip(r.get("rec_texts",[]) or [], r.get("rec_scores",[]) or []):
                        parsed = JerseyOCR._parse_jersey_number(text)
                        if score >= OCR_MIN_SCORE and parsed is not None and score > best_sc:
                            best_num, best_sc = parsed, score
        except: pass
    return best_num, best_sc

players = [
    ("white-14", "/tmp/ocr_p4_full.jpg"),
    ("red-11",   "/tmp/ocr_p2_full.jpg"),
]

for name, path in players:
    full = cv2.imread(path)
    h, w = full.shape[:2]
    print(f"\n{'='*65}")
    print(f"Player: {name}  crop={w}x{h}px")

    # ── Test jersey_no.pt at various upscales ──────────────────────────
    print(f"\n  jersey_no.pt detections:")
    for scale in [1.0, 1.5, 2.0, 3.0]:
        up_h = max(64, int(h * scale))
        up_w = max(32, int(w * scale))
        upscaled = cv2.resize(full, (up_w, up_h), interpolation=cv2.INTER_LINEAR)

        det = jersey_model(upscaled, conf=0.10, verbose=False, device=device)
        boxes = []
        for r in det:
            for box in r.boxes:
                x1u,y1u,x2u,y2u = [float(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                # Map back to original coords
                y1o = max(0, int(y1u / scale)); y2o = min(h, int(y2u / scale))
                x1o = max(0, int(x1u / scale)); x2o = min(w, int(x2u / scale))
                boxes.append((x1o, y1o, x2o, y2o, conf))

        if boxes:
            for (x1o,y1o,x2o,y2o,conf) in boxes:
                model_crop = full[y1o:y2o, x1o:x2o]
                num, sc = run_ocr(model_crop)
                print(f"    scale={scale:.1f}x  bbox=({x1o},{y1o})→({x2o},{y2o})  conf={conf:.3f}  "
                      f"size={x2o-x1o}x{y2o-y1o}  OCR→ jersey={num} (score={sc:.3f})")
                cv2.imwrite(f"/tmp/l1_{name}_s{scale:.0f}x_bbox.jpg", model_crop)
        else:
            print(f"    scale={scale:.1f}x  → no detection")

    # ── Baseline: heuristic crops ──────────────────────────────────────
    print(f"\n  Baseline crops:")
    chest = full[int(h*.15):int(h*.55), int(w*.10):int(w*.90)]
    wide  = full[int(h*.10):int(h*.75), int(w*.05):int(w*.95)]
    num_c, sc_c = run_ocr(chest)
    num_w, sc_w = run_ocr(wide)
    print(f"    chest(15-55%): size={chest.shape[1]}x{chest.shape[0]}  OCR→ jersey={num_c} (score={sc_c:.3f})")
    print(f"    wide (10-75%): size={wide.shape[1]}x{wide.shape[0]}   OCR→ jersey={num_w} (score={sc_w:.3f})")

    # ── Use the new _detect_number_bbox via process pipeline ───────────
    print(f"\n  _detect_number_bbox (new code):")
    bbox = ocr_inst._detect_number_bbox(full)
    if bbox:
        y1, y2, x1, x2 = bbox
        num_bbox_crop = full[y1:y2, x1:x2]
        num, sc = run_ocr(num_bbox_crop)
        source = "model" if ocr_inst.model is not None else "heuristic"
        print(f"    → ({x1},{y1})→({x2},{y2})  size={x2-x1}x{y2-y1}  OCR→ jersey={num} (score={sc:.3f})")
        cv2.imwrite(f"/tmp/l1_{name}_final_bbox.jpg", num_bbox_crop)
    else:
        print(f"    → None returned")

"""
Test crop region 15-55% vs 10-75% across multiple frames and poses.
Samples 20 timestamps spread across the video.
"""
import cv2, numpy as np, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from pipeline.jersey_ocr import JerseyOCR, _preprocess_for_ocr, _otsu_variant, _adaptive_variant, OCR_MIN_SCORE

VIDEO_PATH   = "/home/iot/Videos/test-Quarter-1_UNESA-UBAYA.mp4"
DETECTOR_PT  = "models/best-object-basketball.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

detector  = YOLO(DETECTOR_PT)
ocr_inst  = JerseyOCR()
ocr_inst.load_ocr()

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_sec    = total_frames / fps
print(f"Video: {total_frames} frames @ {fps:.0f}fps = {total_sec:.0f}s")

# Sample evenly across video
sample_times = [int(total_sec * i / 20) for i in range(1, 21)]  # 20 timestamps

def has_digit(ocr_result_list):
    return any(parsed is not None for _, _, _, parsed in ocr_result_list)

def run_ocr_crop(crop):
    if crop is None or crop.size == 0:
        return []
    variants = [
        _preprocess_for_ocr(crop),
        cv2.bitwise_not(_preprocess_for_ocr(crop)),
        _otsu_variant(crop),
        _adaptive_variant(crop),
    ]
    found = []
    for vimg in variants:
        try:
            res = ocr_inst._ocr.predict(vimg)
            if res:
                for r in res:
                    for text, score in zip(r.get("rec_texts",[]) or [], r.get("rec_scores",[]) or []):
                        parsed = JerseyOCR._parse_jersey_number(text)
                        if score >= OCR_MIN_SCORE:
                            found.append(("v", text, score, parsed))
        except: pass
    return found

# Stats
stats = {
    "total_players":  0,
    "chest_wins":     0,  # chest got digit, old didn't
    "old_wins":       0,  # old got digit, chest didn't
    "both_got":       0,  # both got a digit
    "both_missed":    0,  # both missed
    "chest_only_digits": [],
    "old_only_digits":   [],
}

print(f"\nTesting {len(sample_times)} timestamps...\n")
print(f"{'Time':>6}  {'Players':>7}  {'Chest(15-55%)':>15}  {'Old(10-75%)':>13}  {'Result'}")
print("-" * 70)

for t in sample_times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if not ret:
        continue

    h_f, w_f = frame.shape[:2]
    det = detector(frame, device=device, conf=0.35, verbose=False)
    players = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                int(b.xyxy[0][2]), int(b.xyxy[0][3])) for r in det for b in r.boxes]

    chest_hits = 0
    old_hits   = 0
    n = 0

    for (x1, y1, x2, y2) in players:
        x1c = max(0,x1); y1c = max(0,y1)
        x2c = min(w_f,x2); y2c = min(h_f,y2)
        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue
        ph, pw = crop.shape[:2]
        if ph < 30 or pw < 15:
            continue
        n += 1
        stats["total_players"] += 1

        # chest strip (15%-55%)
        chest = crop[int(ph*.15):int(ph*.55), int(pw*.10):int(pw*.90)]
        # old heuristic (10%-75%)
        old   = crop[int(ph*.10):int(ph*.75), int(pw*.05):int(pw*.95)]

        chest_res = run_ocr_crop(chest)
        old_res   = run_ocr_crop(old)

        c_digit = has_digit(chest_res)
        o_digit = has_digit(old_res)

        if c_digit: chest_hits += 1
        if o_digit: old_hits   += 1

        if c_digit and not o_digit:
            stats["chest_wins"] += 1
            best = max((r for r in chest_res if r[3] is not None), key=lambda x: x[2], default=None)
            if best: stats["chest_only_digits"].append(best[3])
        elif o_digit and not c_digit:
            stats["old_wins"] += 1
            best = max((r for r in old_res if r[3] is not None), key=lambda x: x[2], default=None)
            if best: stats["old_only_digits"].append(best[3])
        elif c_digit and o_digit:
            stats["both_got"] += 1
        else:
            stats["both_missed"] += 1

    mm, ss = divmod(t, 60)
    result = ""
    if chest_hits > old_hits:   result = "CHEST BETTER"
    elif old_hits > chest_hits: result = "OLD BETTER"
    elif chest_hits == old_hits and chest_hits > 0: result = "tied"
    else: result = "both miss"

    print(f"{mm:02d}:{ss:02d}   {n:>7}  {chest_hits:>10}/{n}     {old_hits:>7}/{n}     {result}")

cap.release()

print(f"\n{'='*70}")
print(f"SUMMARY across {stats['total_players']} total player-frames:")
print(f"  Chest(15-55%) better (got digit, old didn't): {stats['chest_wins']}")
print(f"  Old(10-75%) better  (got digit, chest didn't): {stats['old_wins']}")
print(f"  Both got a digit:   {stats['both_got']}")
print(f"  Both missed:        {stats['both_missed']}")
if stats['old_only_digits']:
    from collections import Counter
    print(f"  Numbers ONLY old caught: {Counter(stats['old_only_digits']).most_common(10)}")
if stats['chest_only_digits']:
    from collections import Counter
    print(f"  Numbers ONLY chest caught: {Counter(stats['chest_only_digits']).most_common(10)}")

import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pipeline.jersey_ocr import JerseyOCR, _preprocess_for_ocr, _otsu_variant, _adaptive_variant, OCR_MIN_SCORE

ocr_inst = JerseyOCR()
ocr_inst.load_ocr()

def test_ocr(label, crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    processed = _preprocess_for_ocr(crop)
    variants = [
        ("CLAHE",     processed),
        ("CLAHE_inv", cv2.bitwise_not(processed)),
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
                        found.append((vname, text, score, parsed))
        except: pass
    print(f"\n  [{label}]  size={crop.shape[1]}x{crop.shape[0]}  bright={brightness:.0f}")
    if found:
        for vname, text, score, parsed in found:
            ok = "OK" if score >= OCR_MIN_SCORE else "--"
            print(f"    {ok} [{vname}] '{text}'  score={score:.3f}  jersey={parsed}")
    else:
        print("    -> nothing detected")

for idx, (name, path) in enumerate([("white-14", "/tmp/ocr_p4_full.jpg"), ("red-11", "/tmp/ocr_p2_full.jpg")]):
    full = cv2.imread(path)
    h, w = full.shape[:2]
    print(f"\n{'='*55}")
    print(f"Player {name}  full={w}x{h}")
    regions = {
        "current  10-75%, 5-95%":  full[int(h*.10):int(h*.75), int(w*.05):int(w*.95)],
        "tighter  20-65%,15-85%":  full[int(h*.20):int(h*.65), int(w*.15):int(w*.85)],
        "center   25-60%,20-80%":  full[int(h*.25):int(h*.60), int(w*.20):int(w*.80)],
        "chest    15-50%,10-90%":  full[int(h*.15):int(h*.50), int(w*.10):int(w*.90)],
        "mid-body 30-70%,10-90%":  full[int(h*.30):int(h*.70), int(w*.10):int(w*.90)],
    }
    for rname, crop in regions.items():
        if crop.size > 0:
            test_ocr(rname, crop)
            key = rname.split()[0]
            cv2.imwrite(f"/tmp/t_p{idx+1}_{key}.jpg", crop)

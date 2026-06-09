#!/usr/bin/env python3
"""
Standalone court keypoint detection + homography visualizer.

Side-by-side view:
  Left  — video frame with keypoint overlay
  Right — bird-eye FIBA court (400 × 214 px)

Usage:
    python backend/scripts/test_court_standalone.py <video_path>

Controls:
    Q / ESC  — quit
    SPACE    — pause / resume
    S        — save screenshot to court_test_NNNN.png
"""

import cv2
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.court import CourtMapper, COURT_W, COURT_H

# ── Validate args ─────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python backend/scripts/test_court_standalone.py <video_path>")
    sys.exit(1)

VIDEO_PATH  = sys.argv[1]
MODELS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

# ── Bird-eye canvas config ────────────────────────────────────────────────────
BIRD_W, BIRD_H = 400, 214           # px — proportional to 28 m × 15 m
SCALE_X        = BIRD_W / COURT_W   # ≈ 14.29 px/m
SCALE_Y        = BIRD_H / COURT_H   # ≈ 14.27 px/m

RECALIB_EVERY  = 30   # force homography recompute every N frames

# ── Colors (BGR) ─────────────────────────────────────────────────────────────
C_LINE  = (200, 200, 200)   # court lines — light gray
C_GOOD  = (  0, 220,   0)   # keypoint conf ≥ 0.5 — green
C_WEAK  = (  0, 140, 255)   # keypoint conf  < 0.5 — orange
C_CAL   = (  0, 200,   0)   # status text when calibrated
C_NOCAL = (  0, 100, 255)   # status text when not calibrated

# ── Coordinate helpers ────────────────────────────────────────────────────────

def c2b(x_m: float, y_m: float) -> tuple[int, int]:
    """Court metres → bird-eye canvas pixel (top-left origin, y flipped)."""
    return (
        int(round(x_m * SCALE_X)),
        int(round((COURT_H - y_m) * SCALE_Y)),
    )


# ── FIBA court drawing ────────────────────────────────────────────────────────

def draw_fiba_court(img: np.ndarray) -> None:
    """Render FIBA court markings on a black 400×214 canvas."""
    T = 1  # line thickness

    def seg(x1, y1, x2, y2):
        cv2.line(img, c2b(x1, y1), c2b(x2, y2), C_LINE, T)

    def rect(x1, y1, x2, y2):
        cv2.rectangle(img, c2b(x1, y1), c2b(x2, y2), C_LINE, T)

    def circ(xm, ym, rm):
        cv2.circle(img, c2b(xm, ym), int(round(rm * SCALE_X)), C_LINE, T)

    def arc_poly(hx, hy, r, y_lo, y_hi, face_right):
        """3PT arc as polyline, clipped to y ∈ [y_lo, y_hi]."""
        pts = []
        for deg in range(0, 361):
            rad = math.radians(deg)
            x = hx + r * math.cos(rad)
            y = hy + r * math.sin(rad)
            in_front = (x > hx) if face_right else (x < hx)
            if y_lo <= y <= y_hi and in_front:
                pts.append(c2b(x, y))
        if pts:
            cv2.polylines(img, [np.array(pts, np.int32)], False, C_LINE, T)

    # Outer boundary
    rect(0, 0, COURT_W, COURT_H)

    # Center line + center circle (r = 1.8 m)
    seg(14.0, 0, 14.0, COURT_H)
    circ(14.0, 7.5, 1.8)

    # Paint areas (keys)
    rect(0,    3.65,  5.8,  11.35)
    rect(22.2, 3.65,  COURT_W, 11.35)

    # Free-throw circles (r = 1.8 m)
    circ( 5.8, 7.5, 1.8)
    circ(22.2, 7.5, 1.8)

    # Hoops (filled dot)
    cv2.circle(img, c2b( 1.28,  7.47), 4, C_LINE, -1)
    cv2.circle(img, c2b(26.425, 7.5 ), 4, C_LINE, -1)

    # 3PT arcs (polyline, y clamped to 0.9 – 14.1 m)
    arc_poly( 1.28,  7.47, 6.75, 0.9, 14.1, face_right=True)
    arc_poly(26.425, 7.5,  6.75, 0.9, 14.1, face_right=False)

    # 3PT straight sections (horizontal, from baseline to arc junction)
    sx = 2.99                              # x where arc meets endline section
    seg(0,              0.9,  sx,              0.9)   # left — bottom
    seg(0,              14.1, sx,              14.1)  # left — top
    seg(COURT_W - sx,   0.9,  COURT_W,         0.9)   # right — bottom
    seg(COURT_W - sx,   14.1, COURT_W,         14.1)  # right — top


# ── Initialise model ──────────────────────────────────────────────────────────
model_file = os.path.join(MODELS_PATH, "court_keypoints.pt")
print(f"Model  : {model_file}")
print(f"Video  : {VIDEO_PATH}")

mapper = CourtMapper(model_path=model_file)
try:
    mapper.load_model()
    print("Model loaded OK\n")
except FileNotFoundError as exc:
    print(f"ERROR  : {exc}")
    sys.exit(1)

# ── Open video ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR  : Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"Frames : {total}   FPS: {src_fps:.1f}")
print("Controls: Q/ESC=quit  SPACE=pause/resume  S=save screenshot\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
frame_idx    = 0
paused       = False
shot_n       = 0
last_kps     = []
last_frame   = None
status_text  = "NOT CALIBRATED"
status_color = C_NOCAL

while True:
    # ── Advance video (skip when paused) ─────────────────────────────────
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
        last_frame = frame

        # Detect keypoints
        kps = mapper.detect_keypoints(frame)

        # Recompute homography periodically or when not yet calibrated
        needs_recalib = (
            not mapper.is_calibrated()
            or frame_idx % RECALIB_EVERY == 0
        )
        if needs_recalib:
            mapper.compute_homography(kps)

        last_kps = kps
        frame_idx += 1

        # Update status
        n_vis = sum(1 for kp in kps if not kp["occluded"])
        if mapper.is_calibrated():
            status_text  = f"CALIBRATED  kp={n_vis}"
            status_color = C_CAL
        else:
            status_text  = f"NOT CALIBRATED  kp={n_vis}"
            status_color = C_NOCAL

    if last_frame is None:
        continue

    # ── Video overlay ─────────────────────────────────────────────────────
    vis = last_frame.copy()

    for kp in last_kps:
        px = int(kp["pixel_pos"][0])
        py = int(kp["pixel_pos"][1])
        color = C_GOOD if not kp["occluded"] else C_WEAK
        label = str(kp["point_id"])

        # Dot with thin black outline for contrast
        cv2.circle(vis, (px, py), 5, (0, 0, 0), -1)
        cv2.circle(vis, (px, py), 4, color,      -1)

        # Label with black stroke then colored fill
        cv2.putText(vis, label, (px + 7, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (px + 7, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color,      1, cv2.LINE_AA)

    # Status in top-left
    cv2.putText(vis, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0),    4, cv2.LINE_AA)
    cv2.putText(vis, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color,  1, cv2.LINE_AA)

    cv2.putText(vis, f"Frame {frame_idx}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Resize to bird-eye height for side-by-side
    scale_h   = BIRD_H / vis.shape[0]
    vis_small = cv2.resize(vis, (int(vis.shape[1] * scale_h), BIRD_H))

    # ── Bird-eye view ─────────────────────────────────────────────────────
    bird = np.zeros((BIRD_H, BIRD_W, 3), dtype=np.uint8)
    draw_fiba_court(bird)

    for kp in last_kps:
        # Prefer transformed position; fall back to ground-truth court pos
        if mapper.is_calibrated():
            cp = mapper.pixel_to_court(kp["pixel_pos"])
        else:
            cp = kp.get("court_pos")

        if cp is None:
            continue
        cx, cy = cp
        if not (0 <= cx <= COURT_W and 0 <= cy <= COURT_H):
            continue

        bx, by = c2b(cx, cy)
        color  = C_GOOD if not kp["occluded"] else C_WEAK
        label  = str(kp["point_id"])

        cv2.circle(bird, (bx, by), 4, (0, 0, 0), -1)
        cv2.circle(bird, (bx, by), 3, color,      -1)

        cv2.putText(bird, label, (bx + 4, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)

    # ── Combine and show ──────────────────────────────────────────────────
    combined = np.hstack([vis_small, bird])
    cv2.imshow("Court Test  [ Q=quit  SPACE=pause  S=screenshot ]", combined)

    key = cv2.waitKey(1 if not paused else 40) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        print("Quit.")
        break
    elif key == ord(' '):
        paused = not paused
        print("PAUSED" if paused else "RESUMED")
    elif key in (ord('s'), ord('S')):
        fname = f"court_test_{shot_n:04d}.png"
        cv2.imwrite(fname, combined)
        print(f"Screenshot saved: {fname}")
        shot_n += 1

cap.release()
cv2.destroyAllWindows()
print(f"Done. Processed {frame_idx} frames.")

"""
Court homography visual test.

Draws on each frame:
  - Detected keypoints (green=visible, red=occluded) with label ID
  - Projected FIBA court grid in pixel space (when calibrated)
  - Calibration status overlay

Usage:
  python tools/test_court_homography.py <input_video> <output_video> [--every N]
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np

# Allow running from project root or from tools/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.court import (
    CourtMapper, LABEL_TO_COURT, COURT_W, COURT_H,
    HOOP_LEFT, HOOP_RIGHT, THREE_PT_RADIUS, THREE_PT_STRAIGHT_X,
)

# ── Court wireframe lines in metric space ─────────────────────────────────────
# Each entry: list of [x, y] points forming a polyline (in meters).
# Origin: bottom-left corner of court (x=baseline→baseline, y=sideline→sideline).

def _arc_pts(cx, cy, r, a_start, a_end, n=40):
    """Sample n points on a circle arc from a_start to a_end degrees."""
    angles = np.linspace(np.radians(a_start), np.radians(a_end), n)
    return [[cx + r * np.cos(a), cy + r * np.sin(a)] for a in angles]

def _build_court_lines():
    lines = []

    # Outer boundary
    lines.append([[0, 0], [28, 0], [28, 15], [0, 15], [0, 0]])

    # Center line
    lines.append([[14, 0], [14, 15]])

    # Center circle
    lines.append(_arc_pts(14, 7.5, 1.8, 0, 360))

    for side in ("left", "right"):
        bx  = 0.0  if side == "left" else 28.0
        hx  = HOOP_LEFT[0]  if side == "left" else HOOP_RIGHT[0]
        ftx = 5.8  if side == "left" else 22.2
        sign = 1   if side == "left" else -1

        # FIBA inner paint box (4.9m wide, y=5.05..9.95)
        lines.append([[bx, 5.05], [ftx, 5.05], [ftx, 9.95], [bx, 9.95]])
        # Outer paint box (CL court, y=3.65..11.35)
        lines.append([[bx, 3.65], [ftx, 3.65], [ftx, 11.35], [bx, 11.35]])

        # Free throw circle (half facing court)
        a0, a1 = (270, 90) if side == "left" else (90, 270)
        lines.append(_arc_pts(ftx, 7.5, 1.8, a0, a1))

        # Restricted area (no-charge arc, r=1.25m)
        a0, a1 = (90, 270) if side == "left" else (270, 90)
        lines.append(_arc_pts(hx, 7.5, 1.25, a0, a1))

        # 3PT line: corner straights
        corner_y_top = 14.1
        corner_y_bot = 0.9
        lines.append([[bx, corner_y_top],
                      [bx + sign * THREE_PT_STRAIGHT_X, corner_y_top]])
        lines.append([[bx, corner_y_bot],
                      [bx + sign * THREE_PT_STRAIGHT_X, corner_y_bot]])

        # 3PT arc
        # Arc centre = hoop; arc from corner junction to corner junction
        # At corner: x-distance from hoop = THREE_PT_STRAIGHT_X - (hx if left else 28-hx)
        # theta where arc meets corner straight: cos(t) = dx/r
        dx = abs((bx + sign * THREE_PT_STRAIGHT_X) - hx)
        theta = np.degrees(np.arccos(min(dx / THREE_PT_RADIUS, 1.0)))
        if side == "left":
            a0, a1 = 90 - theta, 270 + theta   # left: arc faces right
        else:
            a0, a1 = 270 - theta, 90 + theta
        lines.append(_arc_pts(hx, 7.5, THREE_PT_RADIUS, a0, a1))

        # Hoop ring (r=0.23m)
        lines.append(_arc_pts(hx, 7.5, 0.23, 0, 360, 20))

    return lines

COURT_LINES = _build_court_lines()


def project_lines(mapper: CourtMapper, frame_h: int, frame_w: int):
    """Return list of pixel polylines for all court wireframe lines."""
    px_lines = []
    for line in COURT_LINES:
        pts = []
        for x_m, y_m in line:
            px = mapper.court_to_pixel([x_m, y_m])
            if px is None:
                continue
            px_i, py_i = int(round(px[0])), int(round(px[1]))
            if -200 <= px_i <= frame_w + 200 and -200 <= py_i <= frame_h + 200:
                pts.append((px_i, py_i))
        if len(pts) >= 2:
            px_lines.append(pts)
    return px_lines


def draw_frame(frame, keypoints, mapper, frame_idx, fps, calibrated, calib_count):
    h, w = frame.shape[:2]
    out = frame.copy()

    # ── Court wireframe ───────────────────────────────────────────────────────
    if calibrated:
        for pts in project_lines(mapper, h, w):
            for i in range(len(pts) - 1):
                cv2.line(out, pts[i], pts[i + 1], (0, 255, 255), 1, cv2.LINE_AA)

    # ── Keypoints ─────────────────────────────────────────────────────────────
    for kp in keypoints:
        x, y = kp["pixel_pos"]
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < w and 0 <= py < h):
            continue
        lid  = kp["point_id"]
        conf = kp["confidence"]
        occ  = kp["occluded"]

        ransac_out = kp.get("ransac_outlier", False)
        if occ:
            color = (0, 80, 255)    # red  — below confidence threshold (occluded)
        elif ransac_out:
            color = (0, 165, 255)   # orange — detected but RANSAC rejected (wrong pos)
        else:
            color = (0, 220, 0)     # green — RANSAC inlier (used in H matrix)

        radius = 4 if (occ or ransac_out) else 5
        cv2.circle(out, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), radius, (255, 255, 255), 1, cv2.LINE_AA)

        label_txt = f"{lid}"
        cv2.putText(out, label_txt, (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, label_txt, (px + 6, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Show confidence for visible points
        if not occ:
            cv2.putText(out, f"{conf:.2f}", (px + 6, py + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Status overlay ────────────────────────────────────────────────────────
    inliers  = sum(1 for kp in keypoints if not kp["occluded"] and not kp.get("ransac_outlier", False))
    outliers = sum(1 for kp in keypoints if not kp["occluded"] and kp.get("ransac_outlier", False))
    occluded = sum(1 for kp in keypoints if kp["occluded"])
    total    = len(keypoints)
    ts_sec   = frame_idx / max(fps, 1)
    ts_str   = f"{int(ts_sec//60):02d}:{ts_sec%60:05.2f}"

    status_color = (0, 220, 0) if calibrated else (0, 80, 255)
    status_txt   = "CALIBRATED" if calibrated else "NOT CALIBRATED"

    cv2.rectangle(out, (0, 0), (430, 96), (0, 0, 0), -1)
    cv2.putText(out, f"Frame {frame_idx}  [{ts_str}]",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    cv2.putText(out, f"H: {status_txt}  (#{calib_count})",
                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1)
    # Legend dots
    for cx, col, lbl in [(8, (0,220,0), f"inlier:{inliers}"),
                          (100, (0,165,255), f"outlier:{outliers}"),
                          (210, (0,80,255), f"occluded:{occluded}")]:
        cv2.circle(out, (cx + 4, 52), 5, col, -1)
        cv2.putText(out, lbl, (cx + 14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
    cv2.putText(out, f"total detected: {total-occluded}/{total}",
                (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(out, "green=H inlier  orange=RANSAC reject  red=not detected",
                (8, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input",  help="Input video path")
    ap.add_argument("output", help="Output video path")
    ap.add_argument("--every", type=int, default=1,
                    help="Process every Nth frame (default=1, all frames)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (0=all)")
    args = ap.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    model_file = ROOT / "backend" / "models" / "court_keypoints.pt"
    print(f"Loading {model_file} ...")
    mapper = CourtMapper(str(model_file))
    mapper.load_model()
    print("Model loaded.")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.input}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h}  {fps:.1f}fps  {total} frames")

    # ── Writer ────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # ── Process ───────────────────────────────────────────────────────────────
    frame_idx   = 0
    calib_count = 0
    written     = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.max_frames and frame_idx >= args.max_frames:
            break

        # Always run court detection on every processed frame so the
        # output video shows the real detection cadence, but skip HEAVY
        # homography recompute on non-strided frames.
        result     = mapper.process_frame(frame)
        keypoints  = result["court_keypoints"]
        calibrated = result["is_calibrated"]

        if calibrated and (frame_idx == 0 or not hasattr(main, "_prev_calib")):
            calib_count += 1
        main._prev_calib = calibrated  # type: ignore[attr-defined]

        annotated = draw_frame(frame, keypoints, mapper,
                               frame_idx, fps, calibrated, calib_count)
        writer.write(annotated)
        written += 1

        if frame_idx % 100 == 0:
            ts  = frame_idx / fps
            pct = frame_idx / max(total, 1) * 100
            print(f"  [{int(ts//60):02d}:{ts%60:04.1f}]  frame {frame_idx}/{total} "
                  f"({pct:.0f}%)  visible={result['keypoints_detected']}  "
                  f"calibrated={calibrated}")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nDone. {written} frames written → {out_path}")
    print(f"Total calibration events: {calib_count}")


if __name__ == "__main__":
    main()

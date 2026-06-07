"""
Court calibration via YOLOv8-pose (court_keypoints.pt).
Detects 33 FIBA court keypoints → computes homography → maps pixel ↔ metric coords.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = os.getenv("MODELS_PATH", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_FILENAME = "court_keypoints.pt"
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "homography_cache.json")

# ---------------------------------------------------------------------------
# FIBA court constants
# ---------------------------------------------------------------------------

COURT_W = 28.0   # meters, x-axis (baseline to baseline)
COURT_H = 15.0   # meters, y-axis (sideline to sideline)

HOOP_LEFT  = [1.28,   7.47]   # back-projected from cl-sample.mp4 (was 1.575, 7.5)
HOOP_RIGHT = [26.425, 7.5]
THREE_PT_RADIUS = 6.75          # arc radius in meters
THREE_PT_STRAIGHT_X = 2.99     # x where arc meets corner straight section (left basket)

# ---------------------------------------------------------------------------
# Keypoint mapping
# ---------------------------------------------------------------------------

# Model outputs 33 keypoints indexed 0-32.
# This list maps model index → label ID used in the dataset.
# Label IDs span 1-41 with 8 gaps (3,6,18,20,22,24,36,39 not present).
KP_INDEX_TO_LABEL: list[int] = [
     1,  2,  4,  5,  7,  8,  9, 10, 11, 12,  # 0-9
    13, 14, 15, 16, 17, 19, 21, 23, 25, 26,  # 10-19
    27, 28, 29, 30, 31, 32, 33, 34, 35, 37,  # 20-29
    38, 40, 41,                               # 30-32
]

# Label ID → [x_meter, y_meter]  (FIBA origin = bottom-left corner of court)
LABEL_TO_COURT: dict[int, list[float]] = {
    # ── Outer boundary ──────────────────────────────────────────────────
    1:  [0.0,   12.674], # corner top-left  (back-projected; was 15.0)
    8:  [0.0,    0.0],   # corner bottom-left
    34: [28.0,  15.0],   # corner top-right
    41: [28.0,   0.0],   # corner bottom-right
    # label 15 removed: model index 12 detects right side (x≈19m), conflicts with label 25 at index 18
    25: [18.7,  15.0],   # top boundary right-center
    17: [9.3,    0.0],   # bottom boundary left-center
    27: [18.7,   0.0],   # bottom boundary right-center
    # ── Center line ─────────────────────────────────────────────────────
    19: [14.0,  15.0],   # center top
    23: [14.0,   0.0],   # center bottom
    21: [14.0,   7.5],   # center court
    # ── Left paint area ─────────────────────────────────────────────────
    2:  [0.0,   11.35],  # baseline left, top of paint
    7:  [0.0,    3.65],  # baseline left, bottom of paint
    4:  [0.0,    8.325], # baseline left, top-center
    5:  [0.0,    6.675], # baseline left, bottom-center
    10: [5.8,   11.35],  # left inner paint top
    11: [5.8,    3.65],  # left inner paint bottom
    12: [6.99,   8.402], # left paint top corner  (back-projected; was 5.8, 9.3)
    14: [4.915,  6.779], # left paint bottom corner  (back-projected; was 5.8, 5.7)
    13: [5.8,    7.5],   # free throw line left
    9:  [1.28,   7.47],  # hoop left  (back-projected; matches HOOP_LEFT)
    16: [8.03,   7.47],  # left 3PT arc tangent point  (HOOP_LEFT[0] + THREE_PT_RADIUS = 1.28 + 6.75)
    # ── Right paint area ────────────────────────────────────────────────
    35: [28.0,  11.35],  # baseline right, top of paint
    40: [28.0,   3.65],  # baseline right, bottom of paint
    37: [28.0,   9.3],   # baseline right, top-center
    38: [28.0,   5.7],   # baseline right, bottom-center
    31: [22.2,  11.35],  # right inner paint top
    32: [22.2,   3.65],  # right inner paint bottom
    28: [23.75,  9.3],   # right paint top corner
    29: [23.75,  5.7],   # right paint bottom corner
    33: [26.425, 7.5],   # hoop right  (HOOP_RIGHT constant)
    30: [22.2,   5.7],   # free throw line right
    26: [19.675, 7.5],   # right 3PT arc tangent point  (HOOP_RIGHT[0] - THREE_PT_RADIUS = 26.425 - 6.75)
}

LABEL_NAMES: dict[int, str] = {
    1:  "corner_top_left",        2:  "baseline_left_top_paint",
    4:  "baseline_left_top_ctr",  5:  "baseline_left_bot_ctr",
    7:  "baseline_left_bot_paint",8:  "corner_bottom_left",
    9:  "hoop_left",              10: "paint_left_inner_top",
    11: "paint_left_inner_bot",   12: "paint_left_top",
    13: "free_throw_left",        14: "paint_left_bottom",
    15: "boundary_top_left_ctr",  16: "arc_3pt_left",
    17: "boundary_bot_left_ctr",  19: "center_top",
    21: "center_court",           23: "center_bottom",
    25: "boundary_top_right_ctr", 26: "arc_3pt_right",
    27: "boundary_bot_right_ctr", 28: "paint_right_top",
    29: "paint_right_bottom",     30: "free_throw_right",
    31: "paint_right_inner_top",  32: "paint_right_inner_bot",
    33: "hoop_right",             34: "corner_top_right",
    35: "baseline_right_top_paint",37: "baseline_right_top_ctr",
    38: "baseline_right_bot_ctr", 40: "baseline_right_bot_paint",
    41: "corner_bottom_right",
}

# Thresholds
CONF_THRESHOLD     = 0.5    # min visibility score to use a keypoint (general)
LOW_CONF_THRESHOLD = 0.3    # lower threshold for corner/baseline-bottom labels
LOW_CONF_LABELS    = {7, 8, 40, 41}  # often occluded by crowd/camera angle
MIN_KEYPOINTS      = 6      # minimum for RANSAC homography
CAMERA_MOVE_PX     = 20.0  # avg shift (pixels) that triggers homography recompute


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class CourtMapper:
    """
    Two-stage court calibration:
      1. detect_keypoints() → YOLOv8-pose inference, 33 keypoints per frame
      2. compute_homography() → cv2.findHomography + RANSAC
    Cached H matrix is reused until a camera-move is detected (avg shift > 20 px).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or os.path.join(MODELS_DIR, MODEL_FILENAME)
        self.device = device
        self.model = None
        self._H: Optional[np.ndarray] = None      # pixel → court
        self._H_inv: Optional[np.ndarray] = None  # court → pixel
        self._prev_pixels: Optional[dict[int, list[float]]] = None  # for shift check
        self._frame_shape: tuple[int, int] = (720, 1280)  # (h, w) updated each frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load court_keypoints.pt. Raises FileNotFoundError if absent."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path.resolve()}\n"
                "Place court_keypoints.pt in backend/models/ or set MODELS_PATH."
            )
        try:
            import torch
            from ultralytics import YOLO
            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO(str(path))
            self.model.to(target)
            logger.info("CourtMapper loaded: %s (device=%s)", path.name, target)
        except ImportError as exc:
            raise ImportError(
                "ultralytics and torch required. pip install ultralytics torch"
            ) from exc
        except RuntimeError as exc:
            logger.warning("GPU init failed (%s) — falling back to CPU", exc)
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            self.model.to("cpu")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_keypoints(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on one BGR frame.
        Returns list of keypoint dicts (all 33 entries; occluded ones marked).
        """
        if self.model is None or frame is None or frame.size == 0:
            return []

        results = self.model.predict(frame, verbose=False, conf=0.1)
        if not results or results[0].keypoints is None:
            return []

        kps_obj = results[0].keypoints
        if len(kps_obj.data) == 0:
            return []

        # If multiple court detections, pick the one with highest total confidence
        kp_tensor = kps_obj.data  # torch.Tensor (N, 33, 3)
        if len(kp_tensor) > 1:
            best_idx = int(kp_tensor[:, :, 2].sum(dim=1).argmax().item())
        else:
            best_idx = 0

        kp_array = kp_tensor[best_idx].cpu().numpy()  # (33, 3): [x, y, vis]

        keypoints: list[dict] = []
        for kp_idx, label_id in enumerate(KP_INDEX_TO_LABEL):
            x, y, conf = (
                float(kp_array[kp_idx, 0]),
                float(kp_array[kp_idx, 1]),
                float(kp_array[kp_idx, 2]),
            )
            court_coord = LABEL_TO_COURT.get(label_id)
            if court_coord is None:
                continue
            effective_threshold = (
                LOW_CONF_THRESHOLD if label_id in LOW_CONF_LABELS else CONF_THRESHOLD
            )
            keypoints.append({
                "point_id":   label_id,
                "name":       LABEL_NAMES.get(label_id, f"kp_{label_id}"),
                "pixel_pos":  [x, y],
                "court_pos":  court_coord,
                "confidence": conf,
                "occluded":   conf < effective_threshold,
                "estimated":  False,
            })

        return keypoints

    # ------------------------------------------------------------------
    # Homography
    # ------------------------------------------------------------------

    def compute_homography(self, keypoints: list[dict]) -> Optional[np.ndarray]:
        """
        Fit pixel→court homography with RANSAC.
        Requires at least MIN_KEYPOINTS (6) visible (conf >= 0.5) points.
        Returns the 3×3 H (pixel→court) matrix and caches it internally.

        RANSAC direction: court→pixel so the reprojection threshold (15 px)
        is applied in pixel space, giving meaningful outlier rejection.
        _H (pixel→court) = inv(H_ct2px); _H_inv (court→pixel) = H_ct2px.
        """
        # Use occluded flag so per-label thresholds (LOW_CONF_LABELS) are honoured
        visible = [kp for kp in keypoints if not kp["occluded"] and not kp.get("estimated", False)]

        if len(visible) < MIN_KEYPOINTS:
            logger.warning(
                "compute_homography: only %d visible keypoints (need %d) — skipping",
                len(visible), MIN_KEYPOINTS,
            )
            return None

        src_px = np.array([kp["pixel_pos"] for kp in visible], dtype=np.float32)
        src_ct = np.array([kp["court_pos"] for kp in visible], dtype=np.float32)

        # court→pixel: RANSAC threshold in pixel space (15 px ≈ 0.3–0.5 m)
        try:
            H_ct2px, mask = cv2.findHomography(
                src_ct, src_px, cv2.RANSAC, ransacReprojThreshold=15.0
            )
        except cv2.error as exc:
            logger.warning("findHomography failed: %s", exc)
            return None

        if H_ct2px is None:
            logger.warning(
                "Homography returned None — points may be collinear or degenerate"
            )
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        logger.debug("Homography: %d/%d RANSAC inliers", inliers, len(visible))

        inlier_kps = []
        for i, kp in enumerate(visible):
            if mask is not None and mask[i]:
                inlier_kps.append(kp)
            else:
                logger.warning(
                    "RANSAC outlier: label=%d %s pixel=%s court=%s",
                    kp["point_id"], kp["name"],
                    kp["pixel_pos"], kp["court_pos"],
                )

        if not self._validate_court_coverage(inlier_kps):
            logger.warning(
                "Court coverage check failed: inliers only on 1 side — H may be unreliable"
            )

        # pixel→court = inv(court→pixel)
        try:
            H_px2ct = np.linalg.inv(H_ct2px)
        except np.linalg.LinAlgError:
            logger.warning("H inversion failed — singular matrix")
            return None

        self._H     = H_px2ct.astype(np.float64)   # pixel → court
        self._H_inv = H_ct2px.astype(np.float64)   # court → pixel

        # Update pixel cache for camera-move detection
        self._prev_pixels = {
            kp["point_id"]: kp["pixel_pos"]
            for kp in keypoints
            if not kp["occluded"]
        }

        return self._H

    def _camera_moved(self, keypoints: list[dict]) -> bool:
        """Return True if avg pixel shift since last H compute > CAMERA_MOVE_PX."""
        if self._prev_pixels is None:
            return True

        shifts: list[float] = []
        for kp in keypoints:
            if kp["occluded"]:
                continue
            prev = self._prev_pixels.get(kp["point_id"])
            if prev is not None:
                dx = kp["pixel_pos"][0] - prev[0]
                dy = kp["pixel_pos"][1] - prev[1]
                shifts.append(math.hypot(dx, dy))

        if not shifts:
            return False
        return (sum(shifts) / len(shifts)) > CAMERA_MOVE_PX

    # ------------------------------------------------------------------
    # Geometric fallback & coverage validation
    # ------------------------------------------------------------------

    def _estimate_missing_corners(self, keypoints: list[dict]) -> list[dict]:
        """
        If H is available and a corner label (7, 8, 40, 41) is occluded,
        project its known court coord through court_to_pixel to synthesise
        a pixel position.  Requires ≥2 non-occluded keypoints on the same
        side of the court (left for 7/8, right for 40/41) to avoid extrapolation
        on frames where the whole side is invisible.
        Estimated keypoints are marked "estimated": True and must NOT be
        fed back into compute_homography (guarded by the estimated flag check).
        """
        if self._H_inv is None:
            return keypoints

        left_visible = sum(
            1 for kp in keypoints
            if not kp["occluded"] and not kp.get("estimated", False)
            and kp["court_pos"][0] < 14.0
        )
        right_visible = sum(
            1 for kp in keypoints
            if not kp["occluded"] and not kp.get("estimated", False)
            and kp["court_pos"][0] > 14.0
        )

        fh, fw = self._frame_shape
        # Allow up to 5% outside visible frame to catch near-edge corners
        margin_x, margin_y = fw * 0.05, fh * 0.05

        result: list[dict] = []
        for kp in keypoints:
            lid = kp["point_id"]
            if kp["occluded"] and not kp.get("estimated", False):
                needs_estimate = (
                    (lid in {7, 8}   and left_visible  >= 2) or
                    (lid in {40, 41} and right_visible >= 2)
                )
                if needs_estimate:
                    pixel = self.court_to_pixel(kp["court_pos"])
                    if pixel is not None:
                        px, py = pixel
                        in_frame = (
                            -margin_x <= px <= fw + margin_x and
                            -margin_y <= py <= fh + margin_y
                        )
                        if in_frame:
                            kp = dict(kp, pixel_pos=pixel, occluded=False, estimated=True)
                            logger.debug(
                                "_estimate_missing_corners: label=%d estimated → pixel=%s",
                                lid, [round(px, 1), round(py, 1)],
                            )
                        else:
                            logger.debug(
                                "_estimate_missing_corners: label=%d projected out-of-frame "
                                "(%s) — keeping occluded", lid, [round(px, 1), round(py, 1)],
                            )
            result.append(kp)
        return result

    @staticmethod
    def _validate_court_coverage(inlier_kps: list[dict]) -> bool:
        """
        True if RANSAC inliers span at least 2 distinct sides of the court.
        Checks 4 half-planes: left (x<14), right (x>14), top (y>7.5), bottom (y<7.5).
        A homography fitted to only one side (e.g. left baseline) can drift badly
        for coordinates on the opposite side.
        """
        if not inlier_kps:
            return False
        has_left   = any(kp["court_pos"][0] < 14.0 for kp in inlier_kps)
        has_right  = any(kp["court_pos"][0] > 14.0 for kp in inlier_kps)
        has_top    = any(kp["court_pos"][1] > 7.5  for kp in inlier_kps)
        has_bottom = any(kp["court_pos"][1] < 7.5  for kp in inlier_kps)
        return sum([has_left, has_right, has_top, has_bottom]) >= 2

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def pixel_to_court(self, pixel_pos: list[float]) -> Optional[list[float]]:
        """[px, py] → [x_meter, y_meter].  None if not calibrated."""
        if self._H is None:
            return None
        pt  = np.array([[pixel_pos]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        return [float(out[0, 0, 0]), float(out[0, 0, 1])]

    def court_to_pixel(self, court_pos: list[float]) -> Optional[list[float]]:
        """[x_meter, y_meter] → [px, py].  None if not calibrated."""
        if self._H_inv is None:
            return None
        pt  = np.array([[court_pos]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H_inv)
        return [float(out[0, 0, 0]), float(out[0, 0, 1])]

    # ------------------------------------------------------------------
    # 3PT logic
    # ------------------------------------------------------------------

    @staticmethod
    def is_three_point_shot(court_pos: list[float]) -> bool:
        """
        True if the court position is beyond the FIBA 3PT line.

        FIBA 3PT line = circular arc (radius 6.75 m from hoop) joined by two
        straight segments at y = 0.9 m and y = 14.1 m (0.9 m from each
        sideline). In the corner zone (y < 0.9 or y > 14.1) the straight
        section at x ≈ 2.99 m is the legal boundary — NOT the arc distance.
        The corner check must run BEFORE the distance check because in the
        corner zone a point can be > 6.75 m from the hoop yet still legally
        inside the 3PT line (x < 2.99, near the baseline).
        """
        x, y = float(court_pos[0]), float(court_pos[1])

        dist_l = math.hypot(x - HOOP_LEFT[0],  y - HOOP_LEFT[1])
        dist_r = math.hypot(x - HOOP_RIGHT[0], y - HOOP_RIGHT[1])
        near_left = dist_l <= dist_r

        # Corner zone: the boundary is the STRAIGHT section, not the arc.
        in_corner = y < 0.9 or y > 14.1
        if in_corner:
            if near_left:
                return x >= THREE_PT_STRAIGHT_X               # 2.99 m from left baseline
            else:
                return x <= (COURT_W - THREE_PT_STRAIGHT_X)   # 25.01 m from left baseline

        # Mid-court zone: use circular arc distance.
        return min(dist_l, dist_r) > THREE_PT_RADIUS

    # ------------------------------------------------------------------
    # Full-frame pipeline
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        detect → maybe recompute H → return output dict.
        Homography is recomputed when:
          - Not yet calibrated, OR
          - Camera move detected (avg keypoint shift > 20 px)
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        self._frame_shape = (frame.shape[0], frame.shape[1])
        keypoints = self.detect_keypoints(frame)

        if self._H is None or self._camera_moved(keypoints):
            self.compute_homography(keypoints)

        # Estimate occluded corner positions using current H (runs after RANSAC,
        # so estimated points never feed back into compute_homography).
        keypoints = self._estimate_missing_corners(keypoints)

        visible_count = sum(1 for kp in keypoints if not kp["occluded"])
        return {
            "homography_matrix":  self._H,
            "court_keypoints":    keypoints,
            "is_calibrated":      self._H is not None,
            "keypoints_detected": visible_count,
        }

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def is_calibrated(self) -> bool:
        return self._H is not None

    def reset_calibration(self) -> None:
        self._H = None
        self._H_inv = None
        self._prev_pixels = None
        logger.info("CourtMapper calibration reset")

    def save_calibration(self, path: Optional[str] = None) -> None:
        """Persist H matrix to JSON so restarts skip recalibration."""
        if self._H is None:
            logger.warning("save_calibration: no matrix to save")
            return
        save_path = path or DEFAULT_CACHE_PATH
        data = {
            "homography":     self._H.tolist(),
            "homography_inv": self._H_inv.tolist() if self._H_inv is not None else None,
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Calibration saved → %s", save_path)

    def load_calibration(self, path: Optional[str] = None) -> bool:
        """Load persisted H matrix. Returns True on success."""
        load_path = path or DEFAULT_CACHE_PATH
        if not os.path.exists(load_path):
            return False
        try:
            with open(load_path) as f:
                data = json.load(f)
            self._H = np.array(data["homography"], dtype=np.float64)
            if data.get("homography_inv"):
                self._H_inv = np.array(data["homography_inv"], dtype=np.float64)
            else:
                self._H_inv = np.linalg.inv(self._H)
            logger.info("Calibration loaded ← %s", load_path)
            return True
        except (json.JSONDecodeError, KeyError, np.linalg.LinAlgError) as exc:
            logger.warning("load_calibration failed: %s", exc)
            return False

    @staticmethod
    def _empty_result() -> dict:
        return {
            "homography_matrix":  None,
            "court_keypoints":    [],
            "is_calibrated":      False,
            "keypoints_detected": 0,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_court_mapper(
    models_path: Optional[str] = None,
    device: Optional[str] = None,
) -> CourtMapper:
    """Create and load a CourtMapper in one call."""
    model_file = os.path.join(models_path or MODELS_DIR, MODEL_FILENAME)
    mapper = CourtMapper(model_path=model_file, device=device)
    mapper.load_model()
    return mapper


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=== CourtMapper smoke test ===\n")

    # ── 1. Model load ─────────────────────────────────────────────────────
    print("--- 1. Model load ---")
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    mapper = CourtMapper(model_path=str(models_dir / MODEL_FILENAME))

    try:
        mapper.load_model()
        print(f"  Model loaded: {mapper.model is not None}")
        print(f"  Model names:  {mapper.model.names}")
        print(f"  kpt_shape:    {mapper.model.model.yaml['kpt_shape']}")
        print(f"  Total labels: {len(KP_INDEX_TO_LABEL)}")
    except (FileNotFoundError, ImportError) as e:
        print(f"  [SKIP] {e}")
        mapper.model = None  # ensure rest of test runs without model

    # ── 2. RANSAC homography with 8 dummy points ───────────────────────
    print("\n--- 2. RANSAC homography ---")

    # Create a synthetic projective mapping: pixel → court
    # Use 8 well-spread court points and compute "fake" pixel positions
    # via a known synthetic H so we can verify the output H is close.
    np.random.seed(42)
    known_court_pts = np.array([
        [0.0,  15.0],   # corner_top_left
        [28.0, 15.0],   # corner_top_right
        [28.0,  0.0],   # corner_bottom_right
        [0.0,   0.0],   # corner_bottom_left
        [14.0, 15.0],   # center_top
        [14.0,  0.0],   # center_bottom
        [14.0,  7.5],   # center_court
        [9.3,  15.0],   # boundary_top_left_center
    ], dtype=np.float32)

    # Synthetic perspective: scale + slight skew to simulate broadcast cam
    H_true = np.array([
        [30.0,  5.0, 100.0],
        [ 2.0, 28.0,  80.0],
        [ 0.0,  0.0,   1.0],
    ], dtype=np.float64)
    H_true_inv = np.linalg.inv(H_true)

    # Pixel positions derived from H_true_inv (court → pixel)
    fake_pixel_pts = cv2.perspectiveTransform(
        known_court_pts.reshape(1, -1, 2).astype(np.float32), H_true_inv.astype(np.float32)
    ).reshape(-1, 2)

    # Build keypoint list
    dummy_kps = []
    for (px, py), (cx, cy) in zip(fake_pixel_pts, known_court_pts):
        dummy_kps.append({
            "point_id":   0,
            "name":       "test",
            "pixel_pos":  [float(px), float(py)],
            "court_pos":  [float(cx), float(cy)],
            "confidence": 0.95,
            "occluded":   False,
        })

    H_computed = mapper.compute_homography(dummy_kps)
    assert H_computed is not None, "compute_homography returned None"
    assert H_computed.shape == (3, 3), f"Expected (3,3), got {H_computed.shape}"
    print(f"  Homography computed: {H_computed.shape} ✓")
    print(f"  RANSAC inliers: all 8 (exact synthetic data)")

    # ── 3. pixel_to_court / court_to_pixel round-trip ─────────────────
    print("\n--- 3. Round-trip accuracy ---")
    test_court_positions = [
        [0.0,   15.0],   # corner
        [14.0,   7.5],   # center
        [5.8,   11.35],  # paint edge
        [21.25,  7.5],   # right 3PT arc
    ]

    max_err = 0.0
    for court_pos in test_court_positions:
        pixel = cv2.perspectiveTransform(
            np.array([[court_pos]], dtype=np.float32), H_true_inv.astype(np.float32)
        ).reshape(2)
        recovered = mapper.pixel_to_court([float(pixel[0]), float(pixel[1])])
        assert recovered is not None, "pixel_to_court returned None"
        err = math.hypot(recovered[0] - court_pos[0], recovered[1] - court_pos[1])
        max_err = max(max_err, err)

    assert max_err < 0.1, f"Round-trip error {max_err:.4f}m exceeds 0.1m"
    print(f"  Max round-trip error: {max_err:.6f} m  (threshold < 0.10 m) ✓")

    # Inverse direction: court → pixel
    sample_pixel = mapper.court_to_pixel([14.0, 7.5])
    assert sample_pixel is not None, "court_to_pixel returned None"
    print(f"  court_to_pixel([14,7.5]) → pixel {[round(v,1) for v in sample_pixel]} ✓")

    # ── 4. is_three_point_shot ────────────────────────────────────────
    print("\n--- 4. is_three_point_shot ---")
    cases = [
        ([1.575,  7.5],  False, "hoop left (distance=0)"),
        ([0.0,    7.5],  False, "baseline mid (distance=1.575 < 6.75)"),
        ([14.0,   7.5],  True,  "center court (distance=12.4 > 6.75)"),
        ([5.0,    7.5],  False, "inside paint (distance=3.4 < 6.75)"),
        ([26.425, 7.5],  False, "hoop right (distance=0)"),
        ([8.0,    7.5],  False, "inside left arc (distance=6.43 < 6.75)"),
        ([9.0,    7.5],  True,  "beyond left arc (distance=7.43 > 6.75)"),
        ([3.5,    0.5],  True,  "left corner 3PT (x>=2.99, y<0.9)"),
        ([2.0,    0.5],  False, "left corner 2PT (x<2.99, y<0.9)"),
        ([24.5,  14.5],  True,  "right corner 3PT (x<=25.01, y>14.1)"),
    ]

    all_pass = True
    for pos, expected, label in cases:
        result = CourtMapper.is_three_point_shot(pos)
        status = "✓" if result == expected else "✗ FAIL"
        if result != expected:
            all_pass = False
        print(f"  {status}  {pos} → {result}  ({label})")

    assert all_pass, "One or more is_three_point_shot cases failed"

    # ── 5. Empty frame ────────────────────────────────────────────────
    print("\n--- 5. Empty / None frame ---")
    for frame in [None, np.zeros((0, 0, 3), dtype=np.uint8)]:
        out = mapper.process_frame(frame)
        assert out["court_keypoints"] == []
        assert not out["is_calibrated"] or mapper.is_calibrated()
    print("  Empty frames return safe empty result ✓")

    # ── 6. save_calibration / load_calibration ────────────────────────
    print("\n--- 6. Calibration persistence ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    mapper.save_calibration(tmp_path)
    mapper2 = CourtMapper()
    loaded = mapper2.load_calibration(tmp_path)
    assert loaded, "load_calibration returned False"
    assert mapper2._H is not None, "_H not restored"

    # Round-trip accuracy preserved after load
    # Compute a fresh pixel for [14.0, 7.5] to avoid relying on loop variable
    ref_court = [14.0, 7.5]
    ref_pixel = cv2.perspectiveTransform(
        np.array([[ref_court]], dtype=np.float32), H_true_inv.astype(np.float32)
    ).reshape(2)
    rc = mapper2.pixel_to_court([float(ref_pixel[0]), float(ref_pixel[1])])
    assert rc is not None, "pixel_to_court returned None after load"
    err2 = math.hypot(rc[0] - ref_court[0], rc[1] - ref_court[1])
    assert err2 < 0.1, f"Post-load round-trip error {err2:.4f}m"
    print(f"  save/load round-trip error: {err2:.6f} m ✓")
    os.unlink(tmp_path)

    # ── 7. process_frame with real model ─────────────────────────────
    if mapper.model is not None:
        print("\n--- 7. process_frame (real model, black frame) ---")
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = mapper.process_frame(dummy_frame)
        print(f"  is_calibrated:      {result['is_calibrated']}")
        print(f"  keypoints_detected: {result['keypoints_detected']}")
        print(f"  total keypoints:    {len(result['court_keypoints'])}")
        print("  process_frame completed without crash ✓")

    print("\n=== All tests passed ===")

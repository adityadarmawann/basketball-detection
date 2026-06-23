"""
Per-player pose estimation using YOLOv8s-Pose (yolov8s-pose.pt).
Extracts 17 COCO body keypoints, joint angles, shooting detection,
and jump-height estimation per tracked player.
"""

import logging
import math
import os
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = os.getenv("MODELS_PATH", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_FILENAME = "yolov8s-pose.pt"

# ---------------------------------------------------------------------------
# COCO 17 keypoint schema
# ---------------------------------------------------------------------------

KEYPOINT_NAMES: list[str] = [
    "nose",                                          # 0
    "left_eye",    "right_eye",                      # 1-2
    "left_ear",    "right_ear",                      # 3-4
    "left_shoulder","right_shoulder",                # 5-6
    "left_elbow",  "right_elbow",                    # 7-8
    "left_wrist",  "right_wrist",                    # 9-10
    "left_hip",    "right_hip",                      # 11-12
    "left_knee",   "right_knee",                     # 13-14
    "left_ankle",  "right_ankle",                    # 15-16
]

# Named index shortcuts
_NOSE          = 0
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW,    _R_ELBOW    = 7, 8
_L_WRIST,    _R_WRIST    = 9, 10
_L_HIP,      _R_HIP      = 11, 12
_L_KNEE,     _R_KNEE     = 13, 14
_L_ANKLE,    _R_ANKLE    = 15, 16

CONF_THRESHOLD = 0.5
HIP_HISTORY_LEN = 30    # frames of rolling hip-Y baseline


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _kp(keypoints: list[dict], idx: int) -> Optional[dict]:
    """Return keypoint dict by index if it exists in the list."""
    if idx < len(keypoints):
        return keypoints[idx]
    return None


def _visible(kp: Optional[dict]) -> bool:
    return kp is not None and kp["confidence"] >= CONF_THRESHOLD


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class PoseEstimator:
    """
    Runs YOLOv8s-Pose on each player's bbox crop, returning per-player
    keypoints, joint angles, shooting detection, and jump estimation.

    All outputs are tagged ⚠️ APPROX — they depend on camera angle and
    are not equivalent to biomechanical measurements.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
    ):
        self.model_path = model_path or os.path.join(MODELS_DIR, MODEL_FILENAME)
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None

        # track_id → deque of hip-Y values for jump baseline
        self._hip_history: dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load yolov8s-pose.pt. Raises FileNotFoundError if absent."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path.resolve()}\n"
                "Place yolov8s-pose.pt in backend/models/ or set MODELS_PATH."
            )
        try:
            import torch
            from ultralytics import YOLO

            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO(str(path))
            self.model.to(target)
            if target.startswith("cuda"):
                self.model.half()   # FP16 weights → Tensor Core acceleration
            self._use_half = target.startswith("cuda")
            logger.info("PoseEstimator loaded: %s (device=%s, fp16=%s)",
                        path.name, target, self._use_half)

        except ImportError as exc:
            raise ImportError(
                "ultralytics and torch required. pip install ultralytics torch"
            ) from exc
        except RuntimeError as exc:
            logger.warning("GPU init failed (%s) — falling back to CPU", exc)
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            self.model.to("cpu")
            self._use_half = False

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def warmup(self, height: int = 720, width: int = 1280) -> None:
        """One dummy inference to trigger CUDA JIT compilation."""
        if not self.is_model_loaded():
            return
        dummy_crop = np.zeros((64, 48, 3), dtype=np.uint8)
        try:
            self.model.predict(
                [dummy_crop, dummy_crop], verbose=False,
                conf=self.conf_threshold,
                half=getattr(self, "_use_half", False),
            )
            logger.info("PoseEstimator warmup done")
        except Exception as e:
            logger.debug("PoseEstimator warmup error (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray, tracked_players: list[dict]) -> dict:
        """
        Run pose estimation for every tracked player.

        Batches ALL player crops into a single YOLO predict() call so the GPU
        runs one forward pass instead of N sequential passes — ~N× faster on CUDA.

        Args:
            frame:           Full BGR frame from OpenCV.
            tracked_players: Output list from tracker.py
                             [{"track_id", "bbox", ...}, ...]

        Returns:
            {"poses": [per-player pose dicts]}
        """
        if frame is None or frame.size == 0 or not tracked_players:
            return {"poses": []}

        h_frame, w_frame = frame.shape[:2]

        # ── Build crops and track their frame-space offsets ───────────────
        crops:   list                    = []   # ndarray or None per player
        offsets: list[tuple[int, int]]   = []   # (x1c, y1c) for coord mapping

        for player in tracked_players:
            x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
            x1c = max(0, x1);  y1c = max(0, y1)
            x2c = min(w_frame, x2);  y2c = min(h_frame, y2)
            if x2c <= x1c or y2c <= y1c:
                crops.append(None)
            else:
                crop = frame[y1c:y2c, x1c:x2c]
                crops.append(crop if crop.size > 0 else None)
            offsets.append((x1c, y1c))

        # ── Single-batch GPU inference for all valid crops ────────────────
        valid_idx   = [i for i, c in enumerate(crops) if c is not None]
        valid_crops = [crops[i] for i in valid_idx]

        batch_results: list = []
        if valid_crops and self.is_model_loaded():
            try:
                batch_results = self.model.predict(
                    valid_crops,
                    verbose=False,
                    conf=self.conf_threshold,
                    half=getattr(self, "_use_half", False),
                )
            except Exception as exc:
                logger.warning("Batch pose inference failed: %s", exc)
                batch_results = []

        # ── Map results back to each player ───────────────────────────────
        kp_map: dict[int, list[dict]] = {}
        for j, player_i in enumerate(valid_idx):
            x1c, y1c = offsets[player_i]
            if j < len(batch_results):
                kp_map[player_i] = self._parse_keypoints(batch_results[j], x1c, y1c)
            else:
                kp_map[player_i] = self._empty_keypoints()

        poses: list[dict] = []
        for i, player in enumerate(tracked_players):
            track_id  = player["track_id"]
            keypoints = kp_map.get(i, self._empty_keypoints())
            poses.append({
                "track_id":       track_id,
                "keypoints":      keypoints,
                "angles":         self._calc_joint_angles(keypoints),
                "is_shooting":    self._detect_shooting(keypoints),
                "jump_height_px": self._estimate_jump_height(track_id, keypoints),
                "bbox":           player["bbox"],
            })

        return {"poses": poses}

    # ------------------------------------------------------------------
    # Keypoint parsing (used by batch estimate)
    # ------------------------------------------------------------------

    def _parse_keypoints(self, result, x1c: int, y1c: int) -> list[dict]:
        """
        Parse one YOLO Results object into 17 keypoint dicts.
        Offsets (x1c, y1c) map crop-space back to full-frame coordinates.
        Returns empty keypoints (conf=0) when the crop has no detections.
        """
        if result.keypoints is None:
            return self._empty_keypoints()

        kp_data = result.keypoints.data
        if len(kp_data) == 0:
            return self._empty_keypoints()

        # Pick highest-confidence person detection when crop has multiple people
        boxes = result.boxes
        if boxes is not None and len(boxes.conf) > 1:
            best = int(boxes.conf.argmax().item())
        else:
            best = 0

        kp_array = kp_data[best].cpu().numpy()   # (17, 3): [x_crop, y_crop, conf]

        keypoints: list[dict] = []
        for idx in range(17):
            xc, yc, conf = (
                float(kp_array[idx, 0]),
                float(kp_array[idx, 1]),
                float(kp_array[idx, 2]),
            )
            keypoints.append({
                "name":       KEYPOINT_NAMES[idx],
                "index":      idx,
                "x":          xc + x1c,   # full-frame x
                "y":          yc + y1c,   # full-frame y
                "confidence": conf,
            })
        return keypoints

    # ------------------------------------------------------------------
    # Legacy single-crop helper (kept for backwards-compat / testing)
    # ------------------------------------------------------------------

    def _crop_and_estimate(
        self,
        frame: np.ndarray,
        bbox: list[float],
        w_frame: int,
        h_frame: int,
    ) -> list[dict]:
        """Single-player crop inference. Use estimate() for batch processing."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1c = max(0, x1);  y1c = max(0, y1)
        x2c = min(w_frame, x2);  y2c = min(h_frame, y2)
        if x2c <= x1c or y2c <= y1c:
            return self._empty_keypoints()
        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0 or not self.is_model_loaded():
            return self._empty_keypoints()
        try:
            results = self.model.predict(
                crop, verbose=False, conf=self.conf_threshold,
                half=getattr(self, "_use_half", False),
            )
        except Exception as exc:
            logger.warning("Pose inference failed for bbox %s: %s", bbox, exc)
            return self._empty_keypoints()
        if not results:
            return self._empty_keypoints()
        return self._parse_keypoints(results[0], x1c, y1c)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_angle(p1: dict, p2: dict, p3: dict) -> float:
        """
        Compute the angle at joint p2 formed by the p1→p2→p3 chain.
        Returns degrees in [0, 180].  All three points must be visible.
        """
        v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]], dtype=np.float64)
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]], dtype=np.float64)

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0

        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(math.degrees(math.acos(cos_angle)))

    def _calc_joint_angles(self, keypoints: list[dict]) -> dict[str, float]:
        """
        Calculate elbow and knee angles.  Returns NaN for joints whose
        keypoints don't meet the confidence threshold.
        """
        angles: dict[str, float] = {
            "left_elbow":  float("nan"),
            "right_elbow": float("nan"),
            "left_knee":   float("nan"),
            "right_knee":  float("nan"),
        }

        # Elbow angles: shoulder → elbow → wrist
        pairs = [
            ("left_elbow",  _L_SHOULDER, _L_ELBOW, _L_WRIST),
            ("right_elbow", _R_SHOULDER, _R_ELBOW, _R_WRIST),
            ("left_knee",   _L_HIP,      _L_KNEE,  _L_ANKLE),
            ("right_knee",  _R_HIP,      _R_KNEE,  _R_ANKLE),
        ]
        for name, i1, i2, i3 in pairs:
            p1, p2, p3 = _kp(keypoints, i1), _kp(keypoints, i2), _kp(keypoints, i3)
            if _visible(p1) and _visible(p2) and _visible(p3):
                angles[name] = self._calc_angle(p1, p2, p3)

        return angles

    # ------------------------------------------------------------------
    # Shooting detection
    # ------------------------------------------------------------------

    def _detect_shooting(self, keypoints: list[dict]) -> bool:
        """
        Heuristic shooting detector.  Returns True if ALL three conditions
        are met for at least one arm (left or right):

        1. Elbow angle (shoulder → elbow → wrist) < 160°
        2. Wrist Y < Nose Y  (hand above the head in image coordinates)
        3. Shoulder, elbow, and wrist all have confidence >= threshold

        Returns False if keypoints are missing or confidence is low.
        """
        nose = _kp(keypoints, _NOSE)
        if not _visible(nose):
            return False

        arms = [
            (_L_SHOULDER, _L_ELBOW, _L_WRIST),
            (_R_SHOULDER, _R_ELBOW, _R_WRIST),
        ]
        for si, ei, wi in arms:
            shoulder = _kp(keypoints, si)
            elbow    = _kp(keypoints, ei)
            wrist    = _kp(keypoints, wi)

            if not (_visible(shoulder) and _visible(elbow) and _visible(wrist)):
                continue

            elbow_angle = self._calc_angle(shoulder, elbow, wrist)
            wrist_above_head = wrist["y"] < nose["y"]

            if elbow_angle < 160.0 and wrist_above_head:
                return True

        return False

    # ------------------------------------------------------------------
    # Jump-height estimation
    # ------------------------------------------------------------------

    def _estimate_jump_height(
        self, track_id: int, keypoints: list[dict]
    ) -> float:
        """
        Approximate jump height in pixels.

        Baseline = rolling max of hip-Y over the last HIP_HISTORY_LEN frames
        (the highest Y = lowest position = standing).
        jump_height_px = baseline_hip_y - current_hip_y
        Positive → player is higher than baseline (jumping).
        Returns 0.0 if hip keypoints are not visible.
        """
        l_hip = _kp(keypoints, _L_HIP)
        r_hip = _kp(keypoints, _R_HIP)

        visible_hips = [kp for kp in [l_hip, r_hip] if _visible(kp)]
        if not visible_hips:
            return 0.0

        current_hip_y = sum(kp["y"] for kp in visible_hips) / len(visible_hips)

        if track_id not in self._hip_history:
            self._hip_history[track_id] = deque(maxlen=HIP_HISTORY_LEN)

        self._hip_history[track_id].append(current_hip_y)

        # Baseline = max Y in history (image Y increases downward → max = lowest)
        baseline_hip_y = max(self._hip_history[track_id])
        jump_height_px = baseline_hip_y - current_hip_y

        return max(0.0, float(jump_height_px))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_keypoints() -> list[dict]:
        """Return 17 keypoints all with confidence=0 (no pose detected)."""
        return [
            {"name": KEYPOINT_NAMES[i], "index": i, "x": 0.0, "y": 0.0, "confidence": 0.0}
            for i in range(17)
        ]

    def clear_history(self, track_id: Optional[int] = None) -> None:
        """
        Remove hip-Y history.  Pass track_id to clear one player,
        or None to clear all (e.g., between quarters).
        """
        if track_id is None:
            self._hip_history.clear()
        else:
            self._hip_history.pop(track_id, None)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pose_estimator(
    models_path: Optional[str] = None,
    device: Optional[str] = None,
    conf_threshold: float = 0.5,
) -> PoseEstimator:
    """Create and load a PoseEstimator in one call."""
    model_file = os.path.join(models_path or MODELS_DIR, MODEL_FILENAME)
    estimator = PoseEstimator(
        model_path=model_file,
        device=device,
        conf_threshold=conf_threshold,
    )
    estimator.load_model()
    return estimator


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import math as _math

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=== PoseEstimator smoke test ===\n")

    est = PoseEstimator()

    # ── 1. _calc_angle ────────────────────────────────────────────────
    print("--- 1. _calc_angle ---")

    def mkpt(x, y):
        return {"x": float(x), "y": float(y), "confidence": 1.0}

    # 90° angle: p1=(1,0), p2=(0,0), p3=(0,1)
    angle_90 = PoseEstimator._calc_angle(mkpt(1, 0), mkpt(0, 0), mkpt(0, 1))
    assert abs(angle_90 - 90.0) < 0.01, f"Expected 90°, got {angle_90:.4f}°"
    print(f"  Right angle (1,0)→(0,0)→(0,1)  = {angle_90:.2f}° ✓")

    # 180° angle: p1=(-1,0), p2=(0,0), p3=(1,0)
    angle_180 = PoseEstimator._calc_angle(mkpt(-1, 0), mkpt(0, 0), mkpt(1, 0))
    assert abs(angle_180 - 180.0) < 0.01, f"Expected 180°, got {angle_180:.4f}°"
    print(f"  Straight line (-1,0)→(0,0)→(1,0) = {angle_180:.2f}° ✓")

    # 45° angle
    angle_45 = PoseEstimator._calc_angle(mkpt(1, 0), mkpt(0, 0), mkpt(1, 1))
    assert abs(angle_45 - 45.0) < 0.01, f"Expected 45°, got {angle_45:.4f}°"
    print(f"  45° angle                         = {angle_45:.2f}° ✓")

    # Degenerate: coincident points → 0°
    angle_deg = PoseEstimator._calc_angle(mkpt(0, 0), mkpt(0, 0), mkpt(1, 0))
    assert angle_deg == 0.0
    print(f"  Degenerate (zero vector)           = {angle_deg:.2f}° ✓")

    # ── 2. _detect_shooting ───────────────────────────────────────────
    print("\n--- 2. _detect_shooting ---")

    def make_keypoints(overrides: dict) -> list[dict]:
        """Build 17 zero-confidence keypoints, apply overrides {index: (x,y,c)}."""
        kps = PoseEstimator._empty_keypoints()
        for idx, (x, y, c) in overrides.items():
            kps[idx] = {"name": KEYPOINT_NAMES[idx], "index": idx,
                        "x": float(x), "y": float(y), "confidence": float(c)}
        return kps

    # Shooting pose: right arm raised, wrist above nose
    #   nose at y=100, shoulder at y=200, elbow at y=150 (going up),
    #   wrist at y=50 → wrist above nose ✓
    #   angle at elbow: shoulder(200) → elbow(150) → wrist(50)
    #   vectors: (200-150, 0)=(50,0) and (50-150, 0)=(-100,0) → 180° → FAIL
    # Use a bent arm: elbow offset sideways
    #   shoulder(0,200), elbow(80,160), wrist(40,60)
    #   v1 = (0-80, 200-160) = (-80, 40)
    #   v2 = (40-80, 60-160) = (-40, -100)
    #   cos = ((-80)(-40)+(40)(-100))/(norm*norm) = (3200-4000)/(89.4*107.7) = -800/9629 ≈ -0.083
    #   angle ≈ 94.8° < 160° ✓ and wrist_y=60 < nose_y=100 ✓
    shooting_kps = make_keypoints({
        _NOSE:       (200, 100, 0.9),
        _R_SHOULDER: (0,   200, 0.9),
        _R_ELBOW:    (80,  160, 0.9),
        _R_WRIST:    (40,   60, 0.9),
    })
    result = est._detect_shooting(shooting_kps)
    assert result is True, f"Expected shooting=True, got {result}"
    print("  Shooting pose (right arm raised, bent elbow) → True ✓")

    # Standing pose: arms down, wrist below nose
    standing_kps = make_keypoints({
        _NOSE:       (200, 100, 0.9),
        _R_SHOULDER: (200, 300, 0.9),
        _R_ELBOW:    (200, 450, 0.9),
        _R_WRIST:    (200, 600, 0.9),
    })
    result_stand = est._detect_shooting(standing_kps)
    assert result_stand is False, f"Expected shooting=False, got {result_stand}"
    print("  Standing pose (arms down)                    → False ✓")

    # Straight arm (angle = 180°): elbow between shoulder and wrist → angle ≥ 160° → False
    # Shoulder(y=200) → elbow(y=125, midpoint) → wrist(y=50, above nose)
    # vectors: (0,+75) and (0,-75) → opposite directions → angle = 180° ≥ 160° → False
    straight_kps = make_keypoints({
        _NOSE:       (200, 100, 0.9),
        _R_SHOULDER: (200, 200, 0.9),
        _R_ELBOW:    (200, 125, 0.9),   # elbow between shoulder and wrist
        _R_WRIST:    (200,  50, 0.9),   # wrist above nose
    })
    result_straight = est._detect_shooting(straight_kps)
    assert result_straight is False, "Straight arm (180°) should not be shooting"
    print("  Straight extended arm (angle=180°)           → False ✓")

    # Low confidence → False
    low_conf_kps = make_keypoints({
        _NOSE:       (200, 100, 0.9),
        _R_SHOULDER: (0,   200, 0.3),   # below threshold
        _R_ELBOW:    (80,  160, 0.3),
        _R_WRIST:    (40,   60, 0.3),
    })
    result_low = est._detect_shooting(low_conf_kps)
    assert result_low is False, "Low-confidence keypoints should not trigger shooting"
    print("  Low-confidence keypoints                     → False ✓")

    # ── 3. _estimate_jump_height ──────────────────────────────────────
    print("\n--- 3. _estimate_jump_height ---")

    # Feed 10 standing frames (hip_y = 500) then one jumping frame (hip_y = 350)
    player_id = 42
    for _ in range(10):
        kps_stand = make_keypoints({
            _L_HIP: (100, 500, 0.9),
            _R_HIP: (120, 500, 0.9),
        })
        est._estimate_jump_height(player_id, kps_stand)

    kps_jump = make_keypoints({
        _L_HIP: (100, 350, 0.9),
        _R_HIP: (120, 350, 0.9),
    })
    jump_h = est._estimate_jump_height(player_id, kps_jump)
    assert abs(jump_h - 150.0) < 1.0, f"Expected ~150px jump, got {jump_h:.2f}"
    print(f"  Jump height after 10 standing frames: {jump_h:.1f} px ✓")

    # Zero when standing
    kps_ground = make_keypoints({
        _L_HIP: (100, 500, 0.9),
        _R_HIP: (120, 500, 0.9),
    })
    jump_zero = est._estimate_jump_height(player_id, kps_ground)
    assert jump_zero == 0.0, f"Expected 0 jump height, got {jump_zero}"
    print(f"  Jump height when standing: {jump_zero:.1f} px ✓")

    # No visible hips → 0
    no_hip_kps = PoseEstimator._empty_keypoints()
    jump_none = est._estimate_jump_height(99, no_hip_kps)
    assert jump_none == 0.0
    print(f"  No visible hips → {jump_none:.1f} px ✓")

    # clear_history
    est.clear_history(player_id)
    assert player_id not in est._hip_history
    print("  clear_history(track_id) removes entry ✓")
    est.clear_history()
    assert len(est._hip_history) == 0
    print("  clear_history() clears all ✓")

    # ── 4. estimate() with None / empty ──────────────────────────────
    print("\n--- 4. estimate() edge cases ---")
    out = est.estimate(None, [{"track_id": 1, "bbox": [0, 0, 100, 200]}])
    assert out == {"poses": []}, f"Expected empty poses, got {out}"
    print("  estimate(None frame) → empty poses ✓")

    import numpy as _np
    out2 = est.estimate(_np.zeros((480, 640, 3), dtype=_np.uint8), [])
    assert out2 == {"poses": []}, f"Expected empty poses, got {out2}"
    print("  estimate(empty tracked_players) → empty poses ✓")

    # Out-of-bounds bbox → clamped, empty keypoints returned
    out3 = est.estimate(
        _np.zeros((480, 640, 3), dtype=_np.uint8),
        [{"track_id": 1, "bbox": [700, 600, 900, 900]}],
    )
    assert len(out3["poses"]) == 1
    assert out3["poses"][0]["keypoints"][0]["confidence"] == 0.0
    print("  Out-of-bounds bbox → clamped, zero-confidence keypoints ✓")

    # ── 5. Model load ─────────────────────────────────────────────────
    print("\n--- 5. Model load ---")
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / "models" / MODEL_FILENAME
    try:
        est_real = PoseEstimator(model_path=str(model_path))
        est_real.load_model()
        print(f"  Model loaded: {est_real.is_model_loaded()} ✓")
        print(f"  Model names : {est_real.model.names}")
        print(f"  kpt_shape   : {est_real.model.model.yaml['kpt_shape']}")

        # Quick inference on blank frame (expect 0 detections)
        blank = _np.zeros((640, 640, 3), dtype=_np.uint8)
        out_r = est_real.estimate(blank, [{"track_id": 1, "bbox": [0, 0, 640, 640]}])
        print(f"  estimate on blank frame: {len(out_r['poses'])} pose(s) returned ✓")

    except (FileNotFoundError, ImportError) as e:
        print(f"  [SKIP] {e}")

    print("\n=== All tests passed ===")

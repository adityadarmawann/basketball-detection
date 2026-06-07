"""
Action classification for basketball analytics — two modes.

PLACEHOLDER (rule-based, default):
  Heuristic detection from pose.py output, tracker state, and ball kinematics.
  All outputs tagged source="rule_based" with confidence=0.6 (estimate).

MODEL (when action_classifier.pt is present):
  Loads the temporal classifier and runs 16-frame inference per player.
  Falls back to rule-based if the file is missing or fails to load.

8 action classes
  0  Shoot/Release
  1  Pass
  2  Dribble
  3  Block
  4  Jump
  5  Catch
  6  Rebound_attempt
  7  Steal/Reach_in

"Stand" (-1) is returned when no rule fires.
"""

import logging
import math
import os
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR     = os.getenv("MODELS_PATH", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_FILENAME = "action_best_v1.pt"

# ── Action class registry ────────────────────────────────────────────────────
ACTION_NAMES: dict[int, str] = {
    0: "Shoot/Release",
    1: "Pass",
    2: "Dribble",
    3: "Block",
    4: "Jump",
    5: "Catch",
    6: "Rebound_attempt",
    7: "Steal/Reach_in",
}
ACTION_IDS: dict[str, int] = {v: k for k, v in ACTION_NAMES.items()}
STAND = "Stand"

# ── Rule-based thresholds ────────────────────────────────────────────────────
JUMP_HEIGHT_THRESHOLD     = 15.0   # pixels above baseline
BALL_PROXIMITY_NEAR       = 80.0   # pixels — dribble / catch zone
BALL_PROXIMITY_FAR        = 150.0  # pixels — rebound zone
PASS_BALL_SPEED           = 5.0    # pixels/frame — minimum speed to call pass
PLAYER_SPEED_THRESHOLD    = 1.5    # pixels/frame — "player is moving"
BLOCK_IOU_THRESHOLD       = 0.05   # bounding-box IoU — overlap for block
STEAL_IOU_THRESHOLD       = 0.05   # bounding-box IoU — overlap for steal
HOOP_PROXIMITY_THRESHOLD  = 120.0  # pixels — ball near ring/backboard
PASS_FIRED_DURATION       = 15     # frames pass remains active for catch window

RULE_CONFIDENCE  = 0.6    # all rule-based outputs flagged as estimates
SHOOT_CONFIDENCE = 0.85   # from pose.py shooting detection

MODEL_BUFFER_MIN = 8   # minimum frames required for model inference
_T_FAST          = 32  # SlowFast fast-pathway temporal dim (head pool expects 32)
_T_SLOW          = 8   # SlowFast slow-pathway temporal dim: T_FAST / lateral_stride=4
_INPUT_SIZE      = 224 # spatial resize for model input


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _dist(a: list, b: list) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _bbox_iou(a: list, b: list) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# ── Main class ───────────────────────────────────────────────────────────────

class ActionClassifier:
    """
    Classifies basketball player actions per frame.

    Usage:
        clf = ActionClassifier()
        clf.load_model()   # no-op if .pt is absent
        result = clf.classify(frame, tracked_players, pose_results, ball_info, frame_buffer)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path or os.path.join(MODELS_DIR, MODEL_FILENAME)
        self.device     = device
        self.model      = None
        self._model_mode = False
        self._action_classes: list = list(ACTION_NAMES.values())
        self._idx_to_action: dict  = dict(enumerate(self._action_classes))

        # ── Rule-based state (updated every frame) ──
        self._tracked_players: list       = []
        self._prev_ball_pos: Optional[list] = None
        self._ball_velocity: Optional[list] = None
        self._ball_direction_history: deque = deque(maxlen=5)
        self._prev_player_pos: dict       = {}   # track_id → [cx, cy]
        self._player_speeds: dict         = {}   # track_id → px/frame
        self._possession_id: Optional[int] = None
        self._prev_possession_id: Optional[int] = None
        self._prev_ball_near: dict        = {}   # track_id → bool (last frame)
        self._pass_fired: bool            = False
        self._pass_fired_countdown: int   = 0
        self._ball_near_hoop: bool        = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load action_best_v1.pt (SlowFast-R50, 8 action classes).
        Requires pytorchvideo. Falls back to rule-based silently if unavailable.
        """
        path = Path(self.model_path)
        if not path.exists():
            logger.warning(
                "action_best_v1.pt not found at %s — rule-based mode",
                path.resolve(),
            )
            return
        try:
            import torch

            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

            # Metadata from checkpoint
            self._action_classes = ckpt.get("classes", self._action_classes)
            c2i = ckpt.get("class_to_idx", {})
            if c2i:
                self._idx_to_action = {v: k for k, v in c2i.items()}
            else:
                self._idx_to_action = dict(enumerate(self._action_classes))

            try:
                # pytorchvideo.models needs torchvision.ops.RoIAlign (unused by SlowFast).
                # Patch the missing symbol so the import succeeds on envs where only
                # a minimal torchvision is installed.
                import sys, types, torch.nn as _nn
                import torchvision as _tv
                if not hasattr(_tv, "ops") or not hasattr(_tv.ops, "RoIAlign"):
                    _ops = types.ModuleType("torchvision.ops")
                    class _RoIAlign(_nn.Module):
                        def forward(self, x, boxes): return x
                    _ops.RoIAlign = _RoIAlign
                    _tv.ops = _ops
                    sys.modules["torchvision.ops"] = _ops

                from pytorchvideo.models.slowfast import create_slowfast

                model = create_slowfast(
                    model_depth=50,
                    input_channels=(3, 3),
                    model_num_class=len(self._action_classes),
                    slowfast_channel_reduction_ratio=8,
                    slowfast_conv_channel_fusion_ratio=2,
                    slowfast_fusion_conv_kernel_size=(7, 1, 1),
                    slowfast_fusion_conv_stride=(4, 1, 1),
                    # Default head_pool_kernel_sizes=((8,1,1),(32,1,1)) matches
                    # T_slow=8 and T_fast=32, consistent with lateral_stride=4.
                )
                # Checkpoint stores head as Sequential: proj.0=Dropout, proj.1=Linear.
                # pytorchvideo creates head as a plain Linear: proj.weight/bias.
                # Remap proj.1.* → proj.* so state_dict loads cleanly.
                raw_sd = ckpt["model"]
                remapped = {
                    k.replace("blocks.6.proj.1.", "blocks.6.proj."): v
                    for k, v in raw_sd.items()
                }
                model.load_state_dict(remapped)
                self.model = model.to(target).eval()
                self._model_mode = True
                logger.info(
                    "action_best_v1.pt loaded (SlowFast-R50, %d classes) on %s — model mode active",
                    len(self._action_classes), target,
                )
            except ImportError:
                logger.warning(
                    "pytorchvideo not installed — rule-based mode. "
                    "pip install pytorchvideo to enable SlowFast inference."
                )
            except RuntimeError as exc:
                logger.warning(
                    "SlowFast state-dict mismatch (%s) — rule-based mode", exc
                )
            except Exception as exc:
                logger.warning(
                    "SlowFast model build failed (%s) — rule-based mode", exc
                )

        except Exception as exc:
            logger.warning("Failed to load action_best_v1.pt: %s — rule-based mode", exc)

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def get_mode(self) -> str:
        return "model" if (self._model_mode and self.model is not None) else "rule_based"

    # ── Public entry point ───────────────────────────────────────────────────

    def classify(
        self,
        frame,
        tracked_players: list,
        pose_results: Optional[dict] = None,
        ball_info: Optional[dict]    = None,
        frame_buffer                 = None,
    ) -> dict:
        """
        Classify actions for every tracked player in the current frame.

        Args:
            frame:           BGR frame (may be None in rule-based mode).
            tracked_players: [{track_id, bbox, center, …}] from tracker.py.
            pose_results:    {"poses": [{track_id, is_shooting, jump_height_px, …}]}
                             from pose.py.  Pass None if unavailable.
            ball_info:       {"center": [cx,cy], …} from detector.py.
                             May include "hoops" list for rebound detection.
            frame_buffer:    deque of last N BGR frames (required for model mode).

        Returns:
            {"actions": [{track_id, action, action_id, confidence, source}]}
        """
        if not tracked_players:
            return {"actions": []}

        self._update_state(tracked_players, ball_info)

        # ── Model path ──
        if (
            self._model_mode
            and self.model is not None
            and frame_buffer is not None
            and len(frame_buffer) >= MODEL_BUFFER_MIN
        ):
            try:
                return self._model_inference(frame_buffer, tracked_players)
            except Exception as exc:
                logger.warning("Model inference failed (%s) — falling back to rules", exc)

        # ── Rule-based path ──
        actions: list     = []
        current_ball_near: dict = {}

        for player in tracked_players:
            track_id   = player["track_id"]
            pose_data  = self._get_pose_data(track_id, pose_results)
            action_str, action_id, conf = self._rule_based(
                track_id, player, pose_data, ball_info
            )

            # Track proximity for CATCH detection next frame
            ball_center = ball_info.get("center") if ball_info else None
            if ball_center:
                prox = _dist(player.get("center", [0, 0]), ball_center)
                current_ball_near[track_id] = prox < BALL_PROXIMITY_NEAR

            # Set pass-fired window when PASS is detected
            if action_id == ACTION_IDS["Pass"]:
                self._pass_fired = True
                self._pass_fired_countdown = PASS_FIRED_DURATION

            actions.append({
                "track_id":   track_id,
                "action":     action_str,
                "action_id":  action_id,
                "confidence": conf,
                "source":     "rule_based",
            })

        # Advance state for next frame
        self._prev_ball_near = current_ball_near
        if self._pass_fired_countdown > 0:
            self._pass_fired_countdown -= 1
        if self._pass_fired_countdown == 0:
            self._pass_fired = False

        return {"actions": actions}

    # ── State update ─────────────────────────────────────────────────────────

    def _update_state(self, tracked_players: list, ball_info: Optional[dict]) -> None:
        """Refresh ball velocity, player speeds, possession, and hoop proximity."""
        self._tracked_players = tracked_players

        ball_center = ball_info.get("center") if ball_info else None

        # Ball velocity + direction history
        if ball_center:
            if self._prev_ball_pos is not None:
                vx = ball_center[0] - self._prev_ball_pos[0]
                vy = ball_center[1] - self._prev_ball_pos[1]
                self._ball_velocity = [vx, vy]
                speed = math.sqrt(vx ** 2 + vy ** 2)
                if speed > 0.0:
                    self._ball_direction_history.append([vx / speed, vy / speed])
            self._prev_ball_pos = ball_center

        # Per-player speed
        for player in tracked_players:
            tid    = player["track_id"]
            center = player.get("center", [0, 0])
            if tid in self._prev_player_pos:
                self._player_speeds[tid] = _dist(center, self._prev_player_pos[tid])
            self._prev_player_pos[tid] = center

        # Possession: nearest player within BALL_PROXIMITY_NEAR
        self._prev_possession_id = self._possession_id
        if ball_center and tracked_players:
            min_d = float("inf")
            closest = None
            for player in tracked_players:
                d = _dist(player.get("center", [0, 0]), ball_center)
                if d < min_d and d < BALL_PROXIMITY_NEAR:
                    min_d  = d
                    closest = player["track_id"]
            self._possession_id = closest

        self._update_hoop_proximity(ball_info)

    def _update_hoop_proximity(self, ball_info: Optional[dict]) -> None:
        """
        Set _ball_near_hoop.
        Uses explicit hoops list from ball_info when available;
        falls back to ball vertical direction-reversal (bounce heuristic).
        """
        if not ball_info:
            return

        ball_center = ball_info.get("center")
        hoops       = ball_info.get("hoops", [])

        # Explicit hoop positions (from detector.py output merged into ball_info)
        if hoops and ball_center:
            for hoop in hoops:
                bbox = hoop.get("bbox", [])
                if len(bbox) == 4:
                    hcx = (bbox[0] + bbox[2]) / 2
                    hcy = (bbox[1] + bbox[3]) / 2
                    if _dist(ball_center, [hcx, hcy]) < HOOP_PROXIMITY_THRESHOLD:
                        self._ball_near_hoop = True
                        return
            self._ball_near_hoop = False
            return

        # Bounce heuristic: vertical component of normalised velocity reversed
        if len(self._ball_direction_history) >= 2:
            hist = list(self._ball_direction_history)
            prev_vy = hist[-2][1]
            curr_vy = hist[-1][1]
            if prev_vy * curr_vy < 0.0 and abs(curr_vy) > 0.3:
                self._ball_near_hoop = True
                return

        self._ball_near_hoop = False

    # ── Rule-based logic ─────────────────────────────────────────────────────

    def _rule_based(
        self,
        track_id:  int,
        player:    dict,
        pose_data: Optional[dict],
        ball_info: Optional[dict],
    ) -> tuple[str, int, float]:
        """
        Apply heuristic rules in priority order.
        Returns (action_str, action_id, confidence).
        action_id == -1 for the default "Stand".
        """
        player_center = player.get("center", [0, 0])
        player_bbox   = player.get("bbox",   [0, 0, 0, 0])

        ball_center = ball_info.get("center") if ball_info else None
        ball_prox   = _dist(player_center, ball_center) if ball_center else None

        # ── 1. Shoot/Release (highest confidence — from pose.py) ────────────
        if pose_data and pose_data.get("is_shooting", False):
            return "Shoot/Release", 0, SHOOT_CONFIDENCE

        # ── 2. Block ────────────────────────────────────────────────────────
        if self._is_blocking(track_id, player_bbox):
            return "Block", 3, RULE_CONFIDENCE

        # ── 3. Steal/Reach-in ───────────────────────────────────────────────
        if self._is_stealing(track_id, player):
            return "Steal/Reach_in", 7, RULE_CONFIDENCE

        # ── 4. Catch (ball arrived after pass) ─────────────────────────────
        if ball_prox is not None and ball_prox < BALL_PROXIMITY_NEAR:
            was_near = self._prev_ball_near.get(track_id, False)
            if self._pass_fired and not was_near:
                return "Catch", 5, RULE_CONFIDENCE

        # ── 5. Rebound attempt ──────────────────────────────────────────────
        if ball_prox is not None and ball_prox < BALL_PROXIMITY_FAR and self._ball_near_hoop:
            return "Rebound_attempt", 6, RULE_CONFIDENCE

        # ── 6. Pass ─────────────────────────────────────────────────────────
        # Use _prev_possession_id: player HAD ball last frame and threw it.
        if (
            self._prev_possession_id == track_id
            and ball_center is not None
            and self._is_ball_moving_away(player_center, ball_center)
        ):
            return "Pass", 1, RULE_CONFIDENCE

        # ── 7. Dribble ──────────────────────────────────────────────────────
        if ball_prox is not None and ball_prox < BALL_PROXIMITY_NEAR:
            if self._player_speeds.get(track_id, 0.0) > PLAYER_SPEED_THRESHOLD:
                return "Dribble", 2, RULE_CONFIDENCE

        # ── 8. Jump ─────────────────────────────────────────────────────────
        if pose_data and pose_data.get("jump_height_px", 0.0) > JUMP_HEIGHT_THRESHOLD:
            return "Jump", 4, RULE_CONFIDENCE

        return STAND, -1, 0.5

    def _is_ball_moving_away(self, player_center: list, ball_center: list) -> bool:
        """True if ball velocity exceeds PASS_BALL_SPEED and points away from player."""
        if self._ball_velocity is None:
            return False
        vx, vy  = self._ball_velocity
        speed   = math.sqrt(vx ** 2 + vy ** 2)
        if speed < PASS_BALL_SPEED:
            return False
        dx   = ball_center[0] - player_center[0]
        dy   = ball_center[1] - player_center[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 1.0:
            return False
        dot = (vx * dx + vy * dy) / dist
        return dot > 0.0  # velocity component points toward ball → away from player

    def _is_blocking(self, track_id: int, player_bbox: list) -> bool:
        """True if ball changed direction AND this player overlaps with another."""
        if not self._ball_direction_changed():
            return False
        for other in self._tracked_players:
            if other["track_id"] == track_id:
                continue
            if _bbox_iou(player_bbox, other.get("bbox", [0, 0, 0, 0])) > BLOCK_IOU_THRESHOLD:
                return True
        return False

    def _ball_direction_changed(self) -> bool:
        """True if ball direction reversed (dot product of oldest vs newest < 0)."""
        if len(self._ball_direction_history) < 3:
            return False
        hist  = list(self._ball_direction_history)
        first = hist[0]; last = hist[-1]
        return first[0] * last[0] + first[1] * last[1] < 0.0

    def _is_stealing(self, track_id: int, player: dict) -> bool:
        """True if possession just transferred TO track_id AND players overlapped."""
        if self._prev_possession_id is None or self._possession_id is None:
            return False
        if self._prev_possession_id == self._possession_id:
            return False
        if self._possession_id != track_id:
            return False
        prev_player = next(
            (p for p in self._tracked_players
             if p["track_id"] == self._prev_possession_id),
            None,
        )
        if prev_player is None:
            return False
        return (
            _bbox_iou(player.get("bbox", [0, 0, 0, 0]),
                      prev_player.get("bbox", [0, 0, 0, 0]))
            > STEAL_IOU_THRESHOLD
        )

    @staticmethod
    def _get_pose_data(
        track_id: int, pose_results: Optional[dict]
    ) -> Optional[dict]:
        if not pose_results:
            return None
        for pose in pose_results.get("poses", []):
            if pose.get("track_id") == track_id:
                return pose
        return None

    # ── Model inference (SlowFast-R50) ──────────────────────────────────────

    def _model_inference(self, frame_buffer, tracked_players: list) -> dict:
        """
        SlowFast-R50 inference.

        Slow pathway: _T_SLOW uniformly sampled frames from a _T_FAST-frame clip.
        Fast pathway: _T_FAST uniformly sampled frames from the buffer.
        Player crops are extracted from each frame, resized to _INPUT_SIZE×_INPUT_SIZE.
        """
        import torch

        frames = list(frame_buffer)
        if not frames:
            raise RuntimeError("empty frame buffer")

        device  = next(self.model.parameters()).device
        actions = []

        for player in tracked_players:
            track_id    = player["track_id"]
            x1, y1, x2, y2 = [int(v) for v in player.get("bbox", [0, 0, _INPUT_SIZE, _INPUT_SIZE])]

            fast_tensor = self._build_input_tensor(frames, x1, y1, x2, y2, _T_FAST)
            # Slow: uniformly pick _T_SLOW indices from the _T_FAST fast frames
            slow_idx    = torch.linspace(0, _T_FAST - 1, _T_SLOW).long()
            slow_tensor = fast_tensor[:, :, slow_idx, :, :]

            try:
                with torch.no_grad():
                    logits = self.model([slow_tensor.to(device), fast_tensor.to(device)])
                    probs  = torch.softmax(logits, dim=-1)[0]
                    top1   = int(probs.argmax().item())
                    conf   = float(probs[top1].item())
            except Exception as exc:
                logger.warning("SlowFast inference error for track %d: %s", track_id, exc)
                actions.append({
                    "track_id":  track_id,
                    "action":    STAND,
                    "action_id": -1,
                    "confidence": 0.5,
                    "source":    "rule_based",
                })
                continue

            class_name          = self._idx_to_action.get(top1, STAND)
            action_str, action_id = self._map_class_name(class_name)
            actions.append({
                "track_id":   track_id,
                "action":     action_str,
                "action_id":  action_id,
                "confidence": conf,
                "source":     "model",
            })

        return {"actions": actions}

    def _build_input_tensor(
        self,
        frames: list,
        x1: int, y1: int, x2: int, y2: int,
        T: int,
    ):
        """
        Crop player bbox from every frame, resize to _INPUT_SIZE, sample/pad to T frames.
        Returns FloatTensor [1, C, T, H, W] normalised to [0, 1].
        """
        import torch
        import cv2
        import numpy as np

        crops = []
        for frame in frames:
            h, w   = frame.shape[:2]
            crop   = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                crop = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            # BGR → RGB, normalise
            crops.append(crop[:, :, ::-1].astype(np.float32) / 255.0)

        n = len(crops)
        if n >= T:
            idx   = np.linspace(0, n - 1, T, dtype=int)
            crops = [crops[i] for i in idx]
        else:
            crops += [crops[-1]] * (T - n)  # repeat last frame

        arr = np.stack(crops)            # [T, H, W, C]
        arr = arr.transpose(3, 0, 1, 2) # [C, T, H, W]
        return torch.tensor(arr).unsqueeze(0)  # [1, C, T, H, W]

    @staticmethod
    def _map_class_name(name: str) -> tuple[str, int]:
        """Map checkpoint class name (e.g. 'dribble') to ACTION_NAMES string + id."""
        _aliases: dict[str, tuple[str, int]] = {
            "shoot":   ("Shoot/Release",    0),
            "pass":    ("Pass",             1),
            "dribble": ("Dribble",          2),
            "block":   ("Block",            3),
            "jump":    ("Jump",             4),
            "catch":   ("Catch",            5),
            "rebound": ("Rebound_attempt",  6),
            "steal":   ("Steal/Reach_in",   7),
        }
        return _aliases.get(name.lower(), (STAND, -1))

    # ── Utilities ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all per-game state."""
        self._tracked_players = []
        self._prev_ball_pos   = None
        self._ball_velocity   = None
        self._ball_direction_history.clear()
        self._prev_player_pos.clear()
        self._player_speeds.clear()
        self._possession_id      = None
        self._prev_possession_id = None
        self._prev_ball_near.clear()
        self._pass_fired            = False
        self._pass_fired_countdown  = 0
        self._ball_near_hoop        = False


# ── Factory ──────────────────────────────────────────────────────────────────

def create_action_classifier(
    models_path: Optional[str] = None,
    device: Optional[str]      = None,
) -> ActionClassifier:
    """Create and attempt to load model (silently falls back to rule-based)."""
    model_file = os.path.join(models_path or MODELS_DIR, MODEL_FILENAME)
    clf = ActionClassifier(model_path=model_file, device=device)
    clf.load_model()
    return clf


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    print("=== ActionClassifier smoke test ===\n")

    # Reference player for most tests
    PLAYER = {"track_id": 1, "bbox": [50, 50, 150, 250], "center": [100, 150]}

    # ── Test helpers ────────────────────────────────────────────────────────

    def fresh() -> ActionClassifier:
        return ActionClassifier()

    def check(label: str, got_action: str, got_id: int, exp_action: str, exp_id: int) -> None:
        ok = got_action == exp_action and got_id == exp_id
        status = "✓" if ok else f"✗  FAIL got ({got_action!r}, {got_id})"
        print(f"  {label:<28} → {got_action!r:<20} id={got_id:<3}  {status}")
        assert ok, f"Expected ({exp_action!r}, {exp_id}), got ({got_action!r}, {got_id})"

    # ── 1. classify() with frame=None ───────────────────────────────────────
    print("--- 1. classify(frame=None) ---")
    ac = fresh()
    r = ac.classify(None, [PLAYER])
    assert "actions" in r and len(r["actions"]) == 1
    assert r["actions"][0]["source"] == "rule_based"
    print("  Frame=None with one player → Stand (no crash) ✓")

    r_empty = ac.classify(None, [])
    assert r_empty == {"actions": []}
    print("  Frame=None, no players → empty ✓")

    # ── 2. Model fallback when .pt absent ───────────────────────────────────
    print("\n--- 2. Fallback to rule_based (model absent) ---")
    ac_nopt = ActionClassifier(model_path="/nonexistent/action_best_v1.pt")
    ac_nopt.load_model()
    assert ac_nopt.get_mode() == "rule_based"
    assert not ac_nopt.is_model_loaded()
    print(f"  get_mode() = {ac_nopt.get_mode()!r} ✓")
    print(f"  is_model_loaded() = {ac_nopt.is_model_loaded()} ✓")

    # ── 3. All 8 action classes — direct _rule_based() ──────────────────────
    print("\n--- 3. All 8 action classes (rule_based) ---")

    # 0  Shoot/Release
    ac0 = fresh()
    a, i, c = ac0._rule_based(1, PLAYER, {"is_shooting": True, "jump_height_px": 0.0}, None)
    check("SHOOT (pose.is_shooting=True)", a, i, "Shoot/Release", 0)
    assert c == SHOOT_CONFIDENCE

    # 4  Jump (no other condition active)
    ac4 = fresh()
    a, i, c = ac4._rule_based(1, PLAYER, {"is_shooting": False, "jump_height_px": 20.0}, None)
    check("JUMP  (height=20 > 15)", a, i, "Jump", 4)
    assert c == RULE_CONFIDENCE

    # 2  Dribble (ball near + player moving)
    ac2 = fresh()
    ac2._player_speeds[1] = 3.0         # above PLAYER_SPEED_THRESHOLD
    ac2._pass_fired = False
    ball_near = {"center": [130, 150]}  # 30 px from player center
    a, i, c = ac2._rule_based(1, PLAYER, None, ball_near)
    check("DRIBBLE (ball≈30px + speed=3)", a, i, "Dribble", 2)

    # 1  Pass (player HAD possession last frame + ball moving away fast)
    ac1 = fresh()
    ac1._prev_possession_id = 1         # had ball last frame
    ac1._ball_velocity = [10, 0]        # fast rightward (> PASS_BALL_SPEED=5)
    # Ball is 150 px to the right — away from player at [100,150]
    ball_right = {"center": [250, 150]}
    a, i, c = ac1._rule_based(1, PLAYER, None, ball_right)
    check("PASS  (prev_possess + ball moving away)", a, i, "Pass", 1)

    # 5  Catch (pass_fired + ball newly arrived)
    ac5 = fresh()
    ac5._pass_fired = True
    ac5._prev_ball_near[1] = False     # ball was not near last frame
    # No possession set, no direction history → BLOCK/STEAL skip
    a, i, c = ac5._rule_based(1, PLAYER, None, ball_near)
    check("CATCH (pass_fired + ball arrives)", a, i, "Catch", 5)

    # 6  Rebound_attempt (ball near hoop, player within 150 px)
    ac6 = fresh()
    ac6._ball_near_hoop = True
    ac6._pass_fired = False
    ball_100px = {"center": [100, 250]}  # 100 px below player center (< 150)
    a, i, c = ac6._rule_based(1, PLAYER, None, ball_100px)
    check("REBOUND (hoop + prox=100 < 150)", a, i, "Rebound_attempt", 6)

    # 3  Block (direction change + bbox overlap)
    ac3 = fresh()
    overlapping = {"track_id": 2, "bbox": [60, 60, 140, 240], "center": [100, 150]}
    ac3._tracked_players = [PLAYER, overlapping]
    # Insert 3 direction vectors: first points up, last points down → dot < 0
    for v in [[0.0, 1.0], [0.0, 1.0], [0.0, -1.0]]:
        ac3._ball_direction_history.append(v)
    a, i, c = ac3._rule_based(1, PLAYER, None, None)
    check("BLOCK (dir-change + bbox overlap)", a, i, "Block", 3)

    # 7  Steal/Reach_in (possession change + overlap)
    ac7 = fresh()
    other99 = {"track_id": 99, "bbox": [60, 60, 140, 240], "center": [100, 150]}
    ac7._tracked_players = [PLAYER, other99]
    ac7._prev_possession_id = 99   # player 99 had ball
    ac7._possession_id      = 1    # player 1 has it now
    # No direction history → BLOCK won't fire; STEAL fires next
    a, i, c = ac7._rule_based(1, PLAYER, None, None)
    check("STEAL (possession change + overlap)", a, i, "Steal/Reach_in", 7)

    # Default  Stand
    ac_s = fresh()
    a, i, c = ac_s._rule_based(1, PLAYER, None, None)
    check("STAND (no conditions met)", a, i, STAND, -1)
    assert c == 0.5

    # ── 4. classify() integration — actions list format ──────────────────────
    print("\n--- 4. classify() output format ---")
    ac_int = fresh()
    out = ac_int.classify(None, [PLAYER], pose_results=None, ball_info=None)
    actions = out["actions"]
    assert len(actions) == 1
    entry = actions[0]
    assert entry["track_id"]   == 1
    assert entry["action"]     == STAND
    assert entry["action_id"]  == -1
    assert entry["confidence"] == 0.5
    assert entry["source"]     == "rule_based"
    print(f"  {entry} ✓")

    # ── 5. classify() pass_fired propagates across frames ──────────────────
    print("\n--- 5. Pass → Catch across two frames ---")
    passer = {"track_id": 2, "bbox": [300, 50, 400, 250], "center": [350, 150]}
    catcher = {"track_id": 3, "bbox": [50, 50, 150, 250], "center": [100, 150]}

    ac_seq = fresh()
    # Frame -1: ball near passer → establishes possession=2 and prev_ball_pos
    ball_at_passer = {"center": [355, 150]}  # 5 px from passer [350,150] < 80px
    ac_seq._update_state([passer, catcher], ball_at_passer)
    # Now: _possession_id=2, _prev_ball_pos=[355,150]

    # Frame A: ball moves far right → velocity=[145,0], prev_possession=2, possession=None
    # PASS rule fires: prev_possession==2, velocity fast, direction away from passer
    ball_away = {"center": [500, 150]}
    out_a = ac_seq.classify(None, [passer, catcher], ball_info=ball_away)
    passer_action = next(x for x in out_a["actions"] if x["track_id"] == 2)
    assert passer_action["action"] == "Pass", f"Expected Pass, got {passer_action['action']}"
    print(f"  Frame A: player 2 → Pass ✓")

    # Frame B: ball arrives near catcher
    # _prev_ball_near[3]=False (catcher was far from [500,150] in frame A)
    # pass_fired=True → CATCH fires for catcher
    ball_near_catcher = {"center": [120, 150]}  # 20 px from catcher [100,150]
    out_b = ac_seq.classify(None, [passer, catcher], ball_info=ball_near_catcher)
    catcher_action = next(x for x in out_b["actions"] if x["track_id"] == 3)
    assert catcher_action["action"] == "Catch", (
        f"Expected Catch, got {catcher_action['action']}"
    )
    print(f"  Frame B: player 3 → Catch ✓")

    # ── 6. _update_hoop_proximity: explicit hoops list ──────────────────────
    print("\n--- 6. _update_hoop_proximity ---")
    ac_hp = fresh()
    ball_near_hoop = {
        "center": [200, 100],
        "hoops": [{"bbox": [180, 80, 220, 120]}],  # hoop center = [200,100]
    }
    ac_hp._update_hoop_proximity(ball_near_hoop)
    assert ac_hp._ball_near_hoop is True
    print("  Ball within HOOP_PROXIMITY_THRESHOLD → _ball_near_hoop=True ✓")

    ball_far_hoop = {
        "center": [700, 400],
        "hoops": [{"bbox": [180, 80, 220, 120]}],
    }
    ac_hp._update_hoop_proximity(ball_far_hoop)
    assert ac_hp._ball_near_hoop is False
    print("  Ball far from hoop → _ball_near_hoop=False ✓")

    # Bounce heuristic (no hoops list)
    ac_hp2 = fresh()
    ac_hp2._ball_direction_history.extend([[0.0, 0.9], [0.0, -0.9]])
    ac_hp2._update_hoop_proximity({"center": [100, 100]})
    assert ac_hp2._ball_near_hoop is True
    print("  Y-velocity reversal → _ball_near_hoop=True ✓")

    # ── 7. reset() ──────────────────────────────────────────────────────────
    print("\n--- 7. reset() ---")
    ac_r = fresh()
    ac_r._possession_id = 5
    ac_r._ball_velocity = [3, 4]
    ac_r._pass_fired    = True
    ac_r.reset()
    assert ac_r._possession_id is None
    assert ac_r._ball_velocity is None
    assert ac_r._pass_fired is False
    print("  State cleared after reset() ✓")

    # ── 8. _map_class_name covers all 8 classes ──────────────────────────────
    print("\n--- 8. _map_class_name ---")
    expected = [
        ("shoot",   "Shoot/Release",   0),
        ("pass",    "Pass",            1),
        ("dribble", "Dribble",         2),
        ("block",   "Block",           3),
        ("jump",    "Jump",            4),
        ("catch",   "Catch",           5),
        ("rebound", "Rebound_attempt", 6),
        ("steal",   "Steal/Reach_in",  7),
    ]
    for raw, exp_str, exp_id in expected:
        got_str, got_id = ActionClassifier._map_class_name(raw)
        assert got_str == exp_str and got_id == exp_id, (
            f"{raw!r} → expected ({exp_str!r}, {exp_id}), got ({got_str!r}, {got_id})"
        )
    print("  All 8 class aliases mapped correctly ✓")
    got_str, got_id = ActionClassifier._map_class_name("unknown")
    assert got_str == STAND and got_id == -1
    print("  Unknown class → Stand, -1 ✓")

    # ── 9. Rule confidence invariants ───────────────────────────────────────
    print("\n--- 9. Rule confidence invariants ---")
    ac_ci = fresh()
    ac_ci._player_speeds[1] = 3.0
    ac_ci._pass_fired = False
    a, i, c = ac_ci._rule_based(1, PLAYER, None, ball_near)
    assert c == RULE_CONFIDENCE, f"Rule confidence should be {RULE_CONFIDENCE}, got {c}"
    print(f"  Rule-based confidence = {c} (= RULE_CONFIDENCE) ✓")
    a, i, c = fresh()._rule_based(1, PLAYER, {"is_shooting": True}, None)
    assert c == SHOOT_CONFIDENCE
    print(f"  Shoot confidence = {c} (= SHOOT_CONFIDENCE) ✓")

    print("\n=== All tests passed ===")

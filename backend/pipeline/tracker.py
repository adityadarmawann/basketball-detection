"""
Player tracking via BoxMOT (ByteTrack default, BoT-SORT optional).
Assigns stable track IDs to players and referees across frames.
Ball is a pass-through from detector — not tracked via BoxMOT.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# BoxMOT output column indices: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
_COL_X1   = 0
_COL_Y1   = 1
_COL_X2   = 2
_COL_Y2   = 3
_COL_ID   = 4
_COL_CONF = 5

# Class indices used when building the [x1,y1,x2,y2,conf,cls] input array.
# Values match the label ordering in best-object-basketball.pt:
#   0=player, 1=ball, 2=referee, 3=hoop, 4=backboard
_CLS_PLAYER  = 0
_CLS_REFEREE = 2


def _center(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


class PlayerTracker:
    """
    Wraps ByteTrack (or BoT-SORT) from BoxMOT.

    Two independent tracker instances run in parallel:
      - player_tracker  → all "player" detections (10 players on court)
      - referee_tracker → all "referee" detections (less critical)

    Ball detection is forwarded as-is from detector.py — BoxMOT is not
    needed because there is only one ball and no ID-swap risk.

    Track ID history (track_id → jersey_number) is maintained here so
    jersey_ocr.py can register mappings and any downstream consumer can
    look them up via get_jersey_number().
    """

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        reid_model=None,
        device: Optional[str] = None,
        frame_rate: int = 25,
    ):
        """
        Args:
            tracker_type: "bytetrack" (default, no ReID — fast) or "botsort"
                          (with ReID — more robust under heavy occlusion).
            reid_model:   Pre-loaded ReID model object; only used by BoT-SORT.
                          Pass None for ByteTrack or ReID-less BoT-SORT.
            device:       "cuda" | "cpu" | None (auto). Passed to ReID model
                          only; ByteTrack is pure numpy and ignores it.
            frame_rate:   Expected FPS — used to tune internal track buffer.
        """
        if tracker_type not in ("bytetrack", "botsort"):
            raise ValueError(
                f"Unknown tracker_type {tracker_type!r}. "
                "Choose 'bytetrack' or 'botsort'."
            )

        self.tracker_type = tracker_type
        self.reid_model = reid_model
        self.device = device
        self.frame_rate = frame_rate

        self._player_tracker = None
        self._referee_tracker = None
        self._frame_shape: Optional[tuple] = None

        # Populated by update_jersey() after jersey_ocr.py resolves numbers.
        self.track_jersey_map: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_tracker(self):
        """Instantiate a fresh tracker. Raises ImportError if BoxMOT missing."""
        try:
            from boxmot.trackers import ByteTrack, BotSort

            if self.tracker_type == "bytetrack":
                return ByteTrack(track_buffer=25, frame_rate=self.frame_rate)
            else:  # botsort
                return BotSort(track_buffer=30, frame_rate=self.frame_rate)

        except ImportError as exc:
            raise ImportError(
                "BoxMOT is required. Install via: pip install boxmot"
            ) from exc

    def initialize(self, frame_shape: tuple) -> None:
        """
        Create tracker instances sized to frame_shape.
        Must be called once before the first update().

        Args:
            frame_shape: (H, W, C) from frame.shape.
        """
        self._frame_shape = frame_shape
        try:
            self._player_tracker  = self._build_tracker()
            self._referee_tracker = self._build_tracker()
            self._unavailable = False
            logger.info(
                "PlayerTracker ready: type=%s  shape=%s  fps=%d",
                self.tracker_type, frame_shape, self.frame_rate,
            )
        except ImportError as e:
            logger.warning("BoxMOT not available — tracking disabled: %s", e)
            self._player_tracker  = None
            self._referee_tracker = None
            self._unavailable = True

    def is_initialized(self) -> bool:
        return self._player_tracker is not None

    def reset(self) -> None:
        """
        Re-create tracker state from scratch.
        Call between quarters or when a new video clip starts so old
        track IDs do not bleed across sessions.
        """
        if self._frame_shape:
            self.initialize(self._frame_shape)
        self.track_jersey_map.clear()
        logger.info("PlayerTracker reset (quarter boundary or new clip)")

    # ------------------------------------------------------------------
    # Jersey mapping (populated externally by jersey_ocr.py)
    # ------------------------------------------------------------------

    def update_jersey(self, track_id: int, jersey_number: int) -> None:
        """
        Store a confirmed track_id → jersey_number association.
        Called by jersey_ocr.py once OCR resolves a number for a track.
        """
        self.track_jersey_map[track_id] = jersey_number

    def get_jersey_number(self, track_id: int) -> Optional[int]:
        return self.track_jersey_map.get(track_id)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, detections_dict: dict) -> dict:
        """
        Consume one BGR frame + detector output, return tracked dict.

        Args:
            frame:           BGR numpy array (H, W, 3).
            detections_dict: Output from BasketballDetector.detect().

        Returns:
            {
              "tracked_players":  [{"track_id", "bbox", "confidence",
                                    "class", "center", ??"jersey_number"}, ...],
              "tracked_ball":     {"bbox", "center", "confidence"} | None,
              "tracked_referees": [{"track_id", "bbox", "confidence",
                                    "class", "center"}, ...],
            }
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        if not self.is_initialized():
            if getattr(self, '_unavailable', False):
                return self._empty_result()
            self.initialize(frame.shape)
            if not self.is_initialized():
                return self._empty_result()

        # Players
        player_arr = self._detections_to_array(
            detections_dict.get("players", []), _CLS_PLAYER
        )
        tracked_players = self._run_tracker(
            self._player_tracker, player_arr, frame, "player"
        )

        # Referees
        ref_arr = self._detections_to_array(
            detections_dict.get("referees", []), _CLS_REFEREE
        )
        tracked_refs = self._run_tracker(
            self._referee_tracker, ref_arr, frame, "referee"
        )

        # Ball — pass-through, no tracker needed
        tracked_ball = self._passthrough_ball(detections_dict.get("ball"))

        return {
            "tracked_players":  tracked_players,
            "tracked_ball":     tracked_ball,
            "tracked_referees": tracked_refs,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_tracker(
        self,
        tracker,
        dets: np.ndarray,
        frame: np.ndarray,
        class_label: str,
    ) -> list[dict]:
        """Feed a detection array into a tracker, return parsed list."""
        if dets.shape[0] == 0:
            # BoxMOT still expects a call each frame to advance its Kalman
            # filters for lost tracks. Use the canonical empty array shape.
            dets = np.empty((0, 6), dtype=np.float32)

        try:
            tracks = tracker.update(dets, frame)  # → np.ndarray (N, 8)
        except Exception as exc:
            logger.warning(
                "Tracker.update() failed (class=%s): %s — returning empty for this frame",
                class_label, exc,
            )
            return []

        return self._parse_tracks(tracks, class_label)

    def _detections_to_array(
        self, detections: list[dict], cls_idx: int
    ) -> np.ndarray:
        """
        Convert list of detector dicts → BoxMOT input shape [N, 6].
        Columns: [x1, y1, x2, y2, confidence, class_index]
        """
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        rows = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            rows.append([x1, y1, x2, y2, det["confidence"], cls_idx])
        return np.array(rows, dtype=np.float32)

    def _parse_tracks(self, tracks: np.ndarray, class_label: str) -> list[dict]:
        """
        Parse BoxMOT output [N, 8] → list of pipeline dicts.
        Columns in: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        """
        results: list[dict] = []
        if tracks is None or len(tracks) == 0:
            return results

        seen_ids: set[int] = set()
        for row in tracks:
            x1  = float(row[_COL_X1])
            y1  = float(row[_COL_Y1])
            x2  = float(row[_COL_X2])
            y2  = float(row[_COL_Y2])
            tid = int(row[_COL_ID])
            conf = float(row[_COL_CONF])

            if tid in seen_ids:
                # Duplicate in the same frame → ID swap event. Log but keep going.
                logger.warning(
                    "Duplicate track_id=%d in class=%s — possible ID swap, skipping duplicate",
                    tid, class_label,
                )
                continue
            seen_ids.add(tid)

            entry: dict = {
                "track_id":   tid,
                "bbox":       [x1, y1, x2, y2],
                "confidence": conf,
                "class":      class_label,
                "center":     _center(x1, y1, x2, y2),
            }

            jersey = self.track_jersey_map.get(tid)
            if jersey is not None:
                entry["jersey_number"] = jersey

            results.append(entry)

        return results

    @staticmethod
    def _passthrough_ball(ball_raw: Optional[dict]) -> Optional[dict]:
        if ball_raw is None:
            return None
        x1, y1, x2, y2 = ball_raw["bbox"]
        return {
            "bbox":       ball_raw["bbox"],
            "center":     _center(x1, y1, x2, y2),
            "confidence": ball_raw["confidence"],
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "tracked_players":  [],
            "tracked_ball":     None,
            "tracked_referees": [],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tracker(
    tracker_type: str = "bytetrack",
    device: Optional[str] = None,
    frame_rate: int = 25,
) -> PlayerTracker:
    """
    Create a PlayerTracker instance (not yet initialized).
    Call tracker.initialize(frame.shape) before the first update().
    """
    return PlayerTracker(
        tracker_type=tracker_type,
        device=device,
        frame_rate=frame_rate,
    )


# ---------------------------------------------------------------------------
# Smoke test (no model required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== PlayerTracker smoke test ===\n")

    # ── 1. _detections_to_array ──────────────────────────────────────────
    tracker = PlayerTracker()

    fake_dets = [
        {"bbox": [100.0, 200.0, 200.0, 400.0], "confidence": 0.91, "class": "player"},
        {"bbox": [300.0, 150.0, 380.0, 360.0], "confidence": 0.78, "class": "player"},
    ]
    arr = tracker._detections_to_array(fake_dets, _CLS_PLAYER)
    assert arr.shape == (2, 6), f"Expected (2,6), got {arr.shape}"
    assert arr[0, 4] == 0.91
    assert arr[0, 5] == _CLS_PLAYER
    print("[PASS] _detections_to_array: shape and values correct")

    empty_arr = tracker._detections_to_array([], _CLS_PLAYER)
    assert empty_arr.shape == (0, 6)
    print("[PASS] _detections_to_array: empty list → (0,6) array")

    # ── 2. _parse_tracks ─────────────────────────────────────────────────
    # Simulate ByteTrack output: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
    fake_tracks = np.array(
        [
            [100.0, 200.0, 200.0, 400.0, 1.0, 0.91, 0.0, 0.0],
            [300.0, 150.0, 380.0, 360.0, 2.0, 0.78, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    parsed = tracker._parse_tracks(fake_tracks, "player")
    assert len(parsed) == 2
    assert parsed[0]["track_id"] == 1
    assert parsed[1]["track_id"] == 2
    assert parsed[0]["center"] == [150.0, 300.0]
    assert parsed[0]["class"] == "player"
    print("[PASS] _parse_tracks: 2 tracks parsed with correct ids and centers")

    # Duplicate track_id in same frame (ID-swap simulation)
    dup_tracks = np.array(
        [[100.0, 200.0, 200.0, 400.0, 5.0, 0.9, 0.0, 0.0],
         [110.0, 210.0, 210.0, 410.0, 5.0, 0.7, 0.0, 1.0]],
        dtype=np.float32,
    )
    dup_parsed = tracker._parse_tracks(dup_tracks, "player")
    assert len(dup_parsed) == 1, "Duplicate ID should be skipped"
    print("[PASS] _parse_tracks: duplicate track_id skipped with warning")

    # ── 3. Ball pass-through ─────────────────────────────────────────────
    ball_raw = {"bbox": [400.0, 300.0, 430.0, 330.0], "confidence": 0.95}
    ball_out = PlayerTracker._passthrough_ball(ball_raw)
    assert ball_out["center"] == [415.0, 315.0]
    assert ball_out["confidence"] == 0.95
    print("[PASS] _passthrough_ball: center computed correctly")

    assert PlayerTracker._passthrough_ball(None) is None
    print("[PASS] _passthrough_ball: None → None")

    # ── 4. update() with None/empty frame ────────────────────────────────
    result = tracker.update(None, {"players": fake_dets})
    assert result["tracked_players"] == []
    assert result["tracked_ball"] is None
    print("[PASS] update(None frame): returns empty result without crash")

    # ── 5. Jersey mapping ────────────────────────────────────────────────
    tracker.update_jersey(1, 23)
    tracker.update_jersey(2, 11)
    assert tracker.get_jersey_number(1) == 23
    assert tracker.get_jersey_number(99) is None
    # Jersey appears in _parse_tracks output after mapping
    parsed_with_jersey = tracker._parse_tracks(fake_tracks, "player")
    assert parsed_with_jersey[0].get("jersey_number") == 23
    assert parsed_with_jersey[1].get("jersey_number") == 11
    print("[PASS] update_jersey / get_jersey_number: mapping stored and applied")

    # ── 6. reset() clears jersey map ─────────────────────────────────────
    tracker._frame_shape = (720, 1280, 3)
    try:
        tracker.reset()
        assert tracker.track_jersey_map == {}
        print("[PASS] reset(): jersey map cleared")
    except ImportError as exc:
        print(f"[SKIP] reset() skipped — BoxMOT not in this Python env: {exc}")

    # ── 7. create_tracker factory ─────────────────────────────────────────
    t2 = create_tracker(tracker_type="bytetrack", frame_rate=30)
    assert isinstance(t2, PlayerTracker)
    assert t2.tracker_type == "bytetrack"
    assert t2.frame_rate == 30
    print("[PASS] create_tracker: factory returns correct PlayerTracker")

    # ── 8. BoxMOT live round-trip (if available) ─────────────────────────
    print("\n--- BoxMOT live round-trip ---")
    try:
        live_tracker = create_tracker()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        live_tracker.initialize(dummy_frame.shape)
        print(f"  Tracker initialized (type={live_tracker.tracker_type})")

        detections_dict = {
            "players": fake_dets,
            "ball": ball_raw,
            "referees": [],
        }
        out = live_tracker.update(dummy_frame, detections_dict)
        print(f"  tracked_players : {len(out['tracked_players'])} (expected ≥0 on black frame)")
        print(f"  tracked_ball    : {out['tracked_ball']}")
        print(f"  tracked_referees: {len(out['tracked_referees'])}")
        print("[PASS] BoxMOT live round-trip completed without crash")

    except ImportError as exc:
        print(f"[SKIP] BoxMOT not available in this environment: {exc}")

    print("\n=== All tests passed ===")

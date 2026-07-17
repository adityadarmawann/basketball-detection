"""
Player tracking via BoxMOT (ByteTrack default, BoT-SORT optional).
Assigns stable track IDs to players and referees across frames.
Ball is a pass-through from detector — not tracked via BoxMOT.
"""

import logging
import os
from typing import Optional

import cv2
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


# processed frames to keep the lost-track pooling cache. CRITICAL (workflow #2):
# BoxMOT revives lost tracks with the SAME id for the whole buffer_size window
# (= int(frame_rate/30 * track_buffer)), so tracker.py's partial-vote pooling —
# which only fires for BRAND-NEW ids — is inert until a new id spawns. This cache
# MUST outlive buffer_size or pooling never re-stitches the extra fragments that
# the stricter match_thresh produces. Default 120 > buffer_size 90 (buffer 90 @
# 30fps). If BYTETRACK_BUFFER is raised, raise this with it.
_LOST_TRACK_MAX_AGE = int(os.getenv("LOST_TRACK_MAX_AGE", "120"))  # was 60
# IoU required to associate a brand-new track with a recently-lost UNCONFIRMED
# track for partial-vote pooling (Fix #2). Higher than the confirmed-jersey
# inheritance threshold (0.25) because pooling wrong votes corrupts a number.
_PARTIAL_INHERIT_IOU = 0.35

# How long a briefly-lost player keeps a PREDICTED bbox on screen (processed frames).
# A jumping player defeats the association gate twice over: vertical acceleration breaks
# the Kalman IoU match, and the mid-air detection is motion-blurred into the low-score
# band, which ByteTrack forbids from STARTING tracks — so the shooter's box vanishes
# exactly during the shot, and score attribution falls back to whoever else is nearest
# the ball. The lost track still lives inside ByteTrack (track_buffer=90) and its Kalman
# mean keeps advancing every frame (GMC-warped, so it follows camera pans too); coasting
# just keeps DRAWING it. When the player lands and is re-detected, the same track_id
# re-associates and the box snaps back to reality.
# ~0.5 s of airtime: 15 frames at the 30 fps processed rate. 0 disables.
_COAST_FRAMES = int(os.getenv("TRACK_COAST_FRAMES", "15"))


def _center(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def _box_iou(a: list, b: list) -> float:
    """Axis-aligned IoU of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ── Global Motion Compensation (workflow fix B) ──────────────────────────────
# The camera pans continuously, so ByteTrack's camera-uncompensated Kalman
# prediction drifts a track's box onto a different nearby player (the root cause).
# GMC estimates the frame-to-frame global affine from STATIC background (moving
# players masked out) and warps every track's Kalman state so predictions follow
# the pan. Env-gated DEFAULT-OFF (needs validation) and every call is guarded so
# it can NEVER break the default path.
_GMC_ENABLED = os.getenv("GMC_ENABLED", "0") == "1"
_GMC_SCALE   = float(os.getenv("GMC_SCALE", "0.2"))     # downscale for cheap optical flow


def _apply_gmc(stracks, H) -> None:
    """Warp each track's 8-dim xyah Kalman state (+velocities) by affine H (2x3).
    Ported from BoT-SORT multi_gmc; safe no-op on empty/None."""
    if H is None or not stracks:
        return
    R = H[:2, :2]
    R8x8 = np.kron(np.eye(4, dtype=float), R)
    t = H[:2, 2]
    for st in stracks:
        if getattr(st, "mean", None) is None:
            continue
        mean = R8x8.dot(st.mean)
        mean[:2] += t
        st.mean = mean
        st.covariance = R8x8.dot(st.covariance).dot(R8x8.T)


class _GMC:
    """Masked sparse-optical-flow global motion compensation.

    Self-contained (does NOT use boxmot's SparseOptFlow, which has a
    keypoint-refresh bug that produces plausible-but-wrong affines over long
    pans — workflow trap #1). Fix: re-detect features EVERY frame. Returns a 2x3
    affine (full-res coords) mapping prev→cur; identity on any failure.
    """

    def __init__(self, scale: float = 0.2, max_corners: int = 200, min_inliers: int = 20):
        self.scale = float(scale)
        self.max_corners = int(max_corners)
        self.min_inliers = int(min_inliers)
        self.prev_gray = None
        self.prev_pts = None

    def _detect(self, gray, mask):
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=0.01,
            minDistance=max(2, int(8 * self.scale)), mask=mask, blockSize=3,
        )

    def _mask(self, shape, det_boxes):
        m = np.full(shape, 255, dtype=np.uint8)
        if det_boxes is not None and len(det_boxes):
            for b in det_boxes:
                x1, y1, x2, y2 = (np.asarray(b[:4], dtype=float) * self.scale).astype(int)
                m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
        return m

    def apply(self, frame, det_boxes):
        H = np.eye(2, 3, dtype=np.float64)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gray = cv2.resize(gray, (max(1, int(w * self.scale)), max(1, int(h * self.scale))))
        mask = self._mask(gray.shape, det_boxes)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 8:
            self.prev_gray = gray
            self.prev_pts = self._detect(gray, mask)
            return H

        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None)
        if cur_pts is not None and status is not None:
            good_prev = self.prev_pts[status.ravel() == 1]
            good_cur = cur_pts[status.ravel() == 1]
            if len(good_prev) >= self.min_inliers:
                A, inl = cv2.estimateAffinePartial2D(good_prev, good_cur, method=cv2.RANSAC)
                if A is not None and inl is not None and int(inl.sum()) >= self.min_inliers:
                    A = A.astype(np.float64)
                    A[0, 2] /= self.scale          # rescale translation to full-res
                    A[1, 2] /= self.scale
                    # Sanity gate (workflow trap #1): a real camera pan is a SMALL,
                    # near-rigid per-frame motion. A low-texture court yields noisy
                    # estimates that would jump tracks and SHATTER ids, so reject any
                    # affine that is not ~unit-scale with a plausible translation.
                    sc   = float(np.sqrt(abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])))
                    tmag = float(np.hypot(A[0, 2], A[1, 2]))
                    if 0.9 <= sc <= 1.1 and tmag <= 0.3 * w:
                        H = A

        # ALWAYS refresh features (the bug fix) so keypoints don't deplete over a long pan.
        self.prev_gray = gray
        self.prev_pts = self._detect(gray, mask)
        return H

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None


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

        # Jersey inheritance: when ByteTrack drops a track and re-assigns a new
        # track_id to the same physical player, the new ID inherits the jersey from
        # the old one without waiting for another OCR confirmation.
        self._player_last_seen: dict[int, list] = {}    # track_id → bbox (last frame)
        self._lost_jersey_cache: dict[int, tuple] = {}  # track_id → (bbox, jersey, age)
        self._newly_inherited: dict[int, int] = {}      # new_tid → old_tid (current frame only)
        # Fix #2: ALL recently-lost tracks (confirmed or not) for partial-vote
        # pooling — lets jersey_ocr carry unconfirmed number votes across a
        # camera-cut re-ID instead of restarting from zero.
        self._lost_track_cache: dict[int, tuple] = {}   # track_id → (bbox, age)
        # GMC (fix B, env-gated default-off): per-frame camera-motion affine.
        self._gmc = None
        self._gmc_H = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_tracker(self):
        """Instantiate a fresh tracker. Raises ImportError if BoxMOT missing."""
        try:
            # boxmot 10.x: classes live in submodule paths, not the top-level trackers package
            from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
            from boxmot.trackers.botsort.bot_sort import BoTSORT

            if self.tracker_type == "bytetrack":
                # ── Workflow fix A: purer tracks under camera motion ──────────────
                # match_thresh is an IoU DISTANCE threshold (1 - IoU): 0.8 → match at
                # IoU≥0.2 (very lenient); 0.5 → IoU≥0.5 (genuine overlap required).
                # The old 0.8 + track_buffer=450 (15s revival tail) let one track_id
                # weld onto many players under continuous pan (measured 2.38 distinct
                # jerseys/track). Stricter match + shorter buffer keep each track PURE
                # (≈1 player); the extra fragments are re-stitched by the vote-pooling
                # (needs _LOST_TRACK_MAX_AGE > buffer_size, see above).
                # Do NOT go below 0.5: the low-score 2nd-assoc (0.5) and unconfirmed
                # gate (0.7) are hardcoded in byte_tracker; below 0.5 inverts strictness.
                # Env-gated so 30/0.45 "strict" preset is A/B-testable without redeploy.
                _mt = float(os.getenv("BYTETRACK_MATCH_THRESH", "0.5"))
                _tb = int(os.getenv("BYTETRACK_BUFFER", "90"))
                logger.info("ByteTrack params: match_thresh=%.2f  track_buffer=%d", _mt, _tb)
                return BYTETracker(track_buffer=_tb, frame_rate=self.frame_rate, match_thresh=_mt)
            else:  # botsort
                return BoTSORT(
                    model_weights=None, device="cpu", fp16=False,
                    track_buffer=30, frame_rate=self.frame_rate,
                )

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
        if _GMC_ENABLED and self._gmc is None:
            self._gmc = _GMC(scale=_GMC_SCALE)
            logger.info("GMC (camera-motion compensation) ENABLED (scale=%.2f)", _GMC_SCALE)
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
        self._player_last_seen.clear()
        self._lost_jersey_cache.clear()
        self._lost_track_cache.clear()
        if self._gmc is not None:
            self._gmc.reset()          # new clip → drop previous-frame features
            self._gmc_H = None
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
        # GMC: estimate this frame's camera motion ONCE (players masked as moving),
        # then _run_tracker warps each tracker's Kalman states before association.
        self._gmc_H = None
        if self._gmc is not None:
            try:
                self._gmc_H = self._gmc.apply(frame, player_arr)
            except Exception as _e:
                logger.debug("GMC apply failed: %s", _e)
                self._gmc_H = None
        tracked_players = self._run_tracker(
            self._player_tracker, player_arr, frame, "player"
        )

        # ── Jersey inheritance: recover jersey when ByteTrack re-assigns track_id ──
        # ByteTrack drops a track after track_buffer frames of absence and creates
        # a new track_id when the player reappears. We use the last known bbox of
        # lost tracks to spatially match new IDs and transfer the confirmed jersey
        # immediately — without waiting for another OCR confirmation cycle.
        current_player_ids = {p["track_id"] for p in tracked_players}

        # Age existing lost-track cache and prune stale entries
        self._lost_jersey_cache = {
            tid: (bbox, jersey, age + 1)
            for tid, (bbox, jersey, age) in self._lost_jersey_cache.items()
            if age < _LOST_TRACK_MAX_AGE
        }

        # Register newly disappeared player tracks that had a confirmed jersey
        for tid, prev_bbox in self._player_last_seen.items():
            if tid not in current_player_ids and tid in self.track_jersey_map:
                self._lost_jersey_cache.setdefault(
                    tid, (prev_bbox, self.track_jersey_map[tid], 0)
                )

        # Inherit jersey for brand-new track IDs via spatial proximity (IoU ≥ 0.25)
        self._newly_inherited.clear()
        new_track_ids = current_player_ids - set(self._player_last_seen.keys())
        if new_track_ids and self._lost_jersey_cache:
            for player in tracked_players:
                tid = player["track_id"]
                if tid not in new_track_ids or tid in self.track_jersey_map:
                    continue
                best_iou, best_jersey, best_ltid = 0.0, None, None
                for _ltid, (lost_bbox, lost_jersey, _age) in self._lost_jersey_cache.items():
                    iou = _box_iou(player["bbox"], lost_bbox)
                    if iou > best_iou:
                        best_iou, best_jersey, best_ltid = iou, lost_jersey, _ltid
                if best_iou >= 0.25 and best_jersey is not None:
                    self.track_jersey_map[tid] = best_jersey
                    player["jersey_number"] = best_jersey
                    self._newly_inherited[tid] = best_ltid
                    logger.info(
                        "Jersey inherited: new tid=%d ← old tid=%d #%d (IoU=%.2f)",
                        tid, best_ltid, best_jersey, best_iou,
                    )

        # ── Partial-vote inheritance (Fix #2): recover UNCONFIRMED fragments ──
        # The block above only re-links tracks whose jersey was already confirmed.
        # A player still mid-voting when the camera cuts gets a fresh track_id and
        # loses the partial votes. Register every disappeared track and map new IDs
        # to them spatially so jersey_ocr can POOL the partial votes. We only add to
        # _newly_inherited (team + votes carry via inherit_team); track_jersey_map is
        # left untouched — no number is asserted, just voting momentum preserved.
        self._lost_track_cache = {
            tid: (bbox, age + 1)
            for tid, (bbox, age) in self._lost_track_cache.items()
            if age < _LOST_TRACK_MAX_AGE
        }
        for tid, prev_bbox in self._player_last_seen.items():
            if tid not in current_player_ids:
                self._lost_track_cache.setdefault(tid, (prev_bbox, 0))

        for player in tracked_players:
            tid = player["track_id"]
            # Only brand-new tracks not already handled by confirmed inheritance
            # and not carrying a confirmed jersey.
            if (tid not in new_track_ids
                    or tid in self._newly_inherited
                    or tid in self.track_jersey_map):
                continue
            best_iou, best_old = 0.0, None
            for _ltid, (lost_bbox, _age) in self._lost_track_cache.items():
                if _ltid in current_player_ids:
                    continue
                iou = _box_iou(player["bbox"], lost_bbox)
                if iou > best_iou:
                    best_iou, best_old = iou, _ltid
            if best_iou >= _PARTIAL_INHERIT_IOU and best_old is not None:
                self._newly_inherited[tid] = best_old
                self._lost_track_cache.pop(best_old, None)
                logger.debug(
                    "Partial-vote assoc: new tid=%d ← old tid=%d (IoU=%.2f)",
                    tid, best_old, best_iou,
                )

        # Update last-seen positions for next frame's inheritance check
        self._player_last_seen = {p["track_id"]: p["bbox"] for p in tracked_players}

        # ── Coasting: keep a briefly-lost player visible on prediction ───────────
        # Placed AFTER the inheritance bookkeeping on purpose: the lost caches and
        # _player_last_seen must only ever see REAL observations, never predicted
        # boxes. Coasted entries are flagged so downstream skips OCR/colour voting
        # on them (a predicted crop may sit off the player and would poison votes);
        # the replay keeps drawing the box and the event engine keeps seeing the
        # airborne shooter.
        if _COAST_FRAMES > 0 and self._player_tracker is not None:
            try:
                fh, fw = frame.shape[:2]
                cur_fid = int(getattr(self._player_tracker, "frame_id", 0) or 0)
                for st in list(getattr(self._player_tracker, "lost_stracks", [])):
                    tid = int(st.track_id)
                    if tid in current_player_ids or not getattr(st, "is_activated", False):
                        continue
                    gap = cur_fid - int(getattr(st, "end_frame", cur_fid) or cur_fid)
                    if not (0 < gap <= _COAST_FRAMES):
                        continue
                    try:
                        x1, y1, x2, y2 = [float(v) for v in st.xyxy]
                    except Exception:
                        continue
                    bw, bh = x2 - x1, y2 - y1
                    if bw < 8 or bh < 16:                      # degenerate prediction
                        continue
                    cx1, cy1 = max(0.0, x1), max(0.0, y1)
                    cx2, cy2 = min(float(fw), x2), min(float(fh), y2)
                    if (cx2 - cx1) < bw * 0.4 or (cy2 - cy1) < bh * 0.4:
                        continue                               # drifted mostly off-frame
                    # Someone already covers this spot (same player re-id'd, or another
                    # player moved in) → don't draw a duplicate box on top of them.
                    if any(_box_iou([cx1, cy1, cx2, cy2], p["bbox"]) > 0.3
                           for p in tracked_players):
                        continue
                    tracked_players.append({
                        "track_id":   tid,
                        "bbox":       [cx1, cy1, cx2, cy2],
                        "confidence": float(getattr(st, "score", 0.0) or 0.0),
                        "class":      "player",
                        "center":     [(cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0],
                        "coasted":    True,
                    })
            except Exception as _e:
                logger.debug("Coasting skipped: %s", _e)

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

        # GMC (fix B): warp this tracker's Kalman states by the camera-motion affine
        # BEFORE its internal multi_predict/association, so predictions follow the pan.
        if getattr(self, "_gmc_H", None) is not None:
            try:
                _apply_gmc(
                    list(getattr(tracker, "tracked_stracks", []))
                    + list(getattr(tracker, "lost_stracks", [])),
                    self._gmc_H,
                )
            except Exception as _e:
                logger.debug("GMC warp failed (class=%s): %s", class_label, _e)

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

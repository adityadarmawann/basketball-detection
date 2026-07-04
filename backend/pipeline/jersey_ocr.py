"""
Jersey number detection (2-layer) + team color classification.

Layer 1 — jersey_no.pt:
  Fine-tuned YOLOv8 with 1 class ("number").
  Locates the jersey number bbox within the player crop.
  Falls back to heuristic torso-center crop if model confidence is too low.

Layer 2 — PaddleOCR 3.5.0:
  Reads the digit(s) from the localized crop, filters to 0-99.
  Voting (>= VOTE_THRESHOLD frames) stabilises the assignment before it is
  registered with the tracker.

Team classification: K-Means (K=2) on HSV pixel samples from each player's
torso. Run once at calibration; user can override via set_team_reference().
"""

import logging
import os
import re
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR            = os.getenv("MODELS_PATH", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_FILENAME         = "jersey_no.pt"
DIGIT_MODEL_FILENAME   = "best-detect-num-v2.pt"
DIGIT_CONF_THRESHOLD   = 0.15   # minimum per-digit detection confidence
                                 # voting + roster whitelist handle noise; low threshold
                                 # catches distant/angled digits that would otherwise be missed

VOTE_THRESHOLD        = 2    # OCR reads required to confirm a jersey number
                             # at ocr_interval=2 (0.13s), 2 votes ≈ 0.27s — faster confirmation;
                             # roster whitelist + MAX_VOTE_HISTORY absorb stray misreads
CONF_UPGRADE_MIN      = 0.60 # absolute digit-model confidence floor to trigger a flush
CONF_UPGRADE_DELTA    = 0.25 # new read must beat _best_read_conf by this margin to flush
MAX_VOTE_HISTORY      = 20   # rolling window — 20 reads per player (OCR_SAMPLE_EVERY=1)
                             # smaller window lets wrong reads be forgotten faster
TRANSFER_MIN_LONG_VOTES = 3  # 2-digit candidate needs this many reads before 1-digit votes
                             # are reinterpreted as partial reads; raised to 3 so "10→1"
                             # partial reads accumulate enough evidence before reassigning
OCR_SAMPLE_EVERY      = 1    # run OCR on every process() call per player
                             # roster whitelist removes false positives; voting absorbs noise
HIGH_CONF_EARLY_EXIT  = 0.85 # stop trying OCR variants once any one exceeds this score
CONF_THRESHOLD    = 0.30  # jersey_no.pt detection confidence (lowered from 0.40 — distant/small numbers)
OCR_HEIGHT_PX     = 160   # larger resize → OCR reads small/distant jersey numbers better
OCR_MIN_SCORE     = 0.35  # slightly more permissive — voting absorbs noise
BLUR_THRESHOLD    = 0.0   # disabled — blur filter skips too many fast-player frames;
                          # voting window (VOTE_THRESHOLD=2, MAX=40) handles noise instead


# ---------------------------------------------------------------------------
# Heuristic back-region crop (fallback when no fine-tuned model)
# ---------------------------------------------------------------------------

def _back_crop_heuristic(player_h: int, player_w: int) -> tuple[int, int, int, int]:
    """
    Return (y1, y2, x1, x2) of the jersey number area within the player crop.

    Chest-focused strip (15%-55% height, 10%-90% width):
    - Targets the jersey chest/back number zone directly
    - Avoids the top-of-frame area that bleeds into background sponsor banners
    - Avoids the leg area below the jersey hem
    - Slightly inset horizontally to reduce arm-edge and adjacent-player noise
    - Tested against indoor basketball footage: correctly reads jersey numbers
      that the wider (10%-75%) heuristic missed due to background contamination
    """
    y1 = int(player_h * 0.15)
    y2 = int(player_h * 0.55)
    x1 = int(player_w * 0.10)
    x2 = int(player_w * 0.90)
    return y1, y2, x1, x2


# ---------------------------------------------------------------------------
# Blur detection (25fps motion blur guard)
# ---------------------------------------------------------------------------

def _blur_score(crop: np.ndarray) -> float:
    """
    Laplacian variance sharpness score. Higher = sharper.
    At 25fps, sprinting player (~6m/s) moves 15-25px per frame
    making jersey numbers unreadable. Skip blurry crops entirely.

    Typical indoor court 25fps values:
      > 200  sharp  (standing / walking)
      80-200 ok     (light jog)
      < 80   blurry (sprint / rapid cut) → skip OCR
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_too_blurry(crop: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    """Return True if crop is too blurry to attempt OCR."""
    if crop is None or crop.size == 0:
        return True
    return _blur_score(crop) < threshold


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _otsu_variant(crop: np.ndarray) -> np.ndarray:
    """
    Binary (Otsu) version of the crop — effective for solid-colour jerseys where
    CLAHE-enhanced grayscale retains too much background gradient.
    Returns a 3-channel BGR image padded to match _preprocess_for_ocr output size.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((OCR_HEIGHT_PX, OCR_HEIGHT_PX, 3), dtype=np.uint8)

    scale  = OCR_HEIGHT_PX / h
    new_w  = max(48, int(w * scale))
    binary = cv2.resize(binary, (new_w, OCR_HEIGHT_PX), interpolation=cv2.INTER_NEAREST)

    pad    = max(8, new_w // 4)
    binary = cv2.copyMakeBorder(binary, 6, 6, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _adaptive_variant(crop: np.ndarray) -> np.ndarray:
    """
    Adaptive threshold with THRESH_BINARY_INV — handles both dark and light jerseys.

    For white jerseys (dark numbers on light background):
      Local mean is high → dark number pixels fall below (mean - C) → output 255 (white)
      Result: WHITE numbers on BLACK background → OCR reads well.

    For dark jerseys (light numbers on dark background):
      Local mean is low → light number pixels exceed (mean - C) → output 0 after INV
      Result: also produces readable contrast.

    Unlike global Otsu, adaptive handles uneven lighting across the jersey crop.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Block size 15 captures neighbourhood larger than a jersey digit stroke (~3-8px)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=10,
    )

    # Slight dilation reconnects broken digit strokes (common for bold jersey fonts)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((OCR_HEIGHT_PX, OCR_HEIGHT_PX, 3), dtype=np.uint8)

    scale  = OCR_HEIGHT_PX / h
    new_w  = max(48, int(w * scale))
    binary = cv2.resize(binary, (new_w, OCR_HEIGHT_PX), interpolation=cv2.INTER_NEAREST)

    pad    = max(8, new_w // 4)
    binary = cv2.copyMakeBorder(binary, 6, 6, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Grayscale → denoise → CLAHE contrast enhancement → sharpen →
    resize to OCR_HEIGHT_PX height with padding.
    Returns a 3-channel BGR image (PaddleOCR expects BGR).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Light blur — faster than fastNlMeansDenoising (~0.1ms vs ~10ms)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # CLAHE — clamp tileGridSize so tiles never exceed image size
    h_raw, w_raw = gray.shape[:2]
    tile_h = max(1, min(4, h_raw // 2))
    tile_w = max(1, min(4, w_raw // 2))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile_w, tile_h))
    enhanced = clahe.apply(gray)

    # Sharpen — unsharp mask
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    enhanced = cv2.addWeighted(enhanced, 1.8, blurred, -0.8, 0)

    # Resize to fixed height, keep aspect ratio
    h, w = enhanced.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((OCR_HEIGHT_PX, OCR_HEIGHT_PX, 3), dtype=np.uint8)
    scale = OCR_HEIGHT_PX / h
    new_w = max(1, int(w * scale))
    resized = cv2.resize(enhanced, (new_w, OCR_HEIGHT_PX), interpolation=cv2.INTER_CUBIC)

    # Guarantee minimum width (single digit "1" from far away can be very narrow)
    MIN_WIDTH = 48
    if resized.shape[1] < MIN_WIDTH:
        resized = cv2.resize(resized, (MIN_WIDTH, OCR_HEIGHT_PX), interpolation=cv2.INTER_CUBIC)

    # Add horizontal padding so OCR doesn't clip edge digits
    pad = max(8, resized.shape[1] // 4)
    padded = cv2.copyMakeBorder(resized, 6, 6, pad, pad, cv2.BORDER_CONSTANT, value=0)

    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class JerseyOCR:
    """
    Two-layer jersey number reader with HSV team classification.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: float = CONF_THRESHOLD,
    ):
        self.model_path     = model_path or os.path.join(MODELS_DIR, MODEL_FILENAME)
        self.device         = device
        self.conf_threshold = conf_threshold

        self.model        = None    # jersey_no.pt (YOLO localizer)
        self._digit_model = None    # best-detect-num-v2.pt (YOLO digit 0-9 detector)
        self._ocr         = None    # PaddleOCR (kept; currently inactive — see _run_ocr)
        self._number_class_idx: Optional[int] = None   # None → use heuristic

        # Voting state: track_id → deque of int candidates
        self._votes: dict[int, deque] = {}

        # Team classification
        # _track_team: current best team per track — updated every frame, NOT a permanent lock.
        # _track_team_votes: rolling deque["A"|"B"] per track — old reads fall off naturally.
        self._team_colors:       dict[str, list[float]] = {}   # "A"/"B" → [H, S, V]
        self._track_team:        dict[int, str]         = {}   # track_id → current best team
        self._track_team_votes:  dict[int, deque]       = {}   # track_id → deque["A"|"B"]

        # Per-track OCR scheduling: frame number of last OCR run per track_id.
        # Default -OCR_SAMPLE_EVERY so the first appearance always triggers OCR.
        self._frame_counter: int = 0
        self._last_ocr_frame: dict[int, int] = {}
        # Best digit-model confidence seen per track — used to detect zoom-in upgrades.
        self._best_read_conf: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load jersey_no.pt. Raises FileNotFoundError if absent."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path.resolve()}\n"
                "Place jersey_no.pt in backend/models/ or set MODELS_PATH."
            )
        try:
            import torch
            from ultralytics import YOLO

            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

            # Prefer TensorRT engine when on CUDA (2-4× faster than FP16 PyTorch)
            engine_path = Path(self.model_path).with_suffix(".engine")
            if target.startswith("cuda") and engine_path.exists():
                load_path = engine_path
                logger.info("TensorRT engine found — loading %s", engine_path.name)
            else:
                load_path = Path(self.model_path)

            if load_path.suffix == ".engine":
                # task="detect" required: manually-built TRT engines have no embedded metadata.
                # Without it, Ultralytics cannot cache the predictor → engine reload per call.
                self.model = YOLO(str(load_path), task="detect")
                self._use_half = False
            else:
                self.model = YOLO(str(load_path))
                self.model.to(target)
                if target.startswith("cuda"):
                    self.model.half()
                self._use_half = target.startswith("cuda")

            # Find "number" class index in the loaded model
            for idx, name in self.model.names.items():
                if "number" in name.lower() or name.lower() in ("no", "jersey", "num"):
                    self._number_class_idx = idx
                    break

            if self._number_class_idx is None:
                logger.warning(
                    "jersey_no.pt has no recognisable 'number' class (classes: %s). "
                    "Using heuristic torso-crop fallback for number localisation.",
                    list(self.model.names.values()),
                )
            else:
                logger.info(
                    "jersey_no.pt loaded: '%s' class at index %d",
                    self.model.names[self._number_class_idx],
                    self._number_class_idx,
                )

        except ImportError as exc:
            raise ImportError(
                "ultralytics and torch required. pip install ultralytics torch"
            ) from exc
        except RuntimeError as exc:
            logger.warning("GPU init failed (%s) — falling back to CPU", exc)
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            self.model.to("cpu")

    def load_ocr(self) -> None:
        """Initialise PaddleOCR 3.5.0. Raises ImportError if not installed."""
        try:
            from paddleocr import PaddleOCR

            # PaddleOCR 3.5.0 requires explicit disable of heavyweight pipelines
            import paddle
            _use_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            self._ocr = PaddleOCR(
                lang="en",
                device="gpu" if _use_gpu else "cpu",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=32,   # allow small crops (jersey from distance)
                text_det_limit_type="min",
                text_det_box_thresh=0.3,      # relaxed: catch faint digits
                text_det_unclip_ratio=2.0,    # wider bbox — prevents digit clipping
                text_rec_score_thresh=0.4,    # pre-filter low-conf results
            )
            logger.info("PaddleOCR device: %s", "GPU" if _use_gpu else "CPU")
            logger.info("PaddleOCR 3.5.0 initialised (lang=en, tuned for jersey numbers)")
        except ImportError as exc:
            raise ImportError(
                "paddleocr required. pip install paddleocr paddlepaddle"
            ) from exc

    def load_digit_model(self) -> None:
        """Load best-detect-num-v2.pt — YOLO digit detector (classes 0-9)."""
        path = Path(os.path.join(MODELS_DIR, DIGIT_MODEL_FILENAME))
        if not path.exists():
            logger.warning(
                "Digit model not found: %s — YOLO digit OCR disabled.", path.resolve()
            )
            return
        try:
            import torch
            from ultralytics import YOLO

            target = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

            # Prefer TensorRT engine when on CUDA
            engine_path = path.with_suffix(".engine")
            if target.startswith("cuda") and engine_path.exists():
                load_path = engine_path
                logger.info("TensorRT engine found — loading %s", engine_path.name)
            else:
                load_path = path

            if load_path.suffix == ".engine":
                self._digit_model = YOLO(str(load_path), task="detect")
                self._digit_use_half = False
            else:
                self._digit_model = YOLO(str(load_path))
                self._digit_model.to(target)
                if target.startswith("cuda"):
                    self._digit_model.half()
                self._digit_use_half = target.startswith("cuda")
            logger.info(
                "best-detect-num-v2.pt loaded on %s (classes: %s)",
                target,
                list(self._digit_model.names.values()),
            )
        except Exception as exc:
            logger.warning("Digit model load failed: %s — YOLO digit OCR disabled", exc)
            self._digit_model = None

    def is_model_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # Blur threshold tuning (runtime-adjustable for different venues)
    # ------------------------------------------------------------------

    def set_blur_threshold(self, threshold: float) -> None:
        """
        Adjust blur rejection threshold at runtime.

        Guidelines for 25fps indoor basketball:
          Bright venue, sharp camera  → 100-120
          Normal Campus League setup  →  80 (default)
          Dim venue / shaky camera    →  50-60
          Disable blur filter         →   0
        """
        global BLUR_THRESHOLD
        BLUR_THRESHOLD = threshold
        logger.info("Blur threshold updated to %.1f", threshold)

    def get_blur_stats(self) -> dict:
        """Return current blur threshold for monitoring/debug dashboard."""
        return {
            "blur_threshold": BLUR_THRESHOLD,
            "description": "Laplacian variance cutoff. Crops below this are skipped.",
        }

    def warmup(self, height: int = 720, width: int = 1280) -> None:
        """One dummy inference on jersey_no.pt + digit model to trigger CUDA JIT before real frames."""
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        dummy_crop  = np.zeros((OCR_HEIGHT_PX, OCR_HEIGHT_PX, 3), dtype=np.uint8)
        if self.model is not None:
            try:
                # First call with model.predict() triggers ultralytics model fusion
                # (BN+conv layer merge). Without this, _detect_numbers_fullframe()
                # hits a FP16/float32 dtype mismatch on the unfused model.
                self.model.predict(
                    dummy_frame, verbose=False,
                )
                self._detect_numbers_fullframe(dummy_frame)
                logger.info("jersey_no.pt warmup done (%dx%d dummy frame)", width, height)
            except Exception as e:
                logger.debug("jersey_no.pt warmup error (non-fatal): %s", e)
        if self._digit_model is not None:
            try:
                self._digit_model(dummy_crop, conf=DIGIT_CONF_THRESHOLD, verbose=False)
                logger.info("best-detect-num-v2.pt warmup done")
            except Exception as e:
                logger.debug("digit model warmup error (non-fatal): %s", e)
        # PaddleOCR warmup (inactive — re-enable if switching back to PaddleOCR)
        # if self._ocr is not None:
        #     try:
        #         self._ocr.predict(dummy_crop)
        #         logger.info("PaddleOCR warmup done")
        #     except Exception as e:
        #         logger.debug("PaddleOCR warmup error (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
        tracked_players: list[dict],
        tracker=None,           # PlayerTracker instance from tracker.py
        roster: dict | None = None,  # {jersey_str: {...}} — whitelist filter
    ) -> dict:
        """
        For each tracked player: localise number → OCR → vote → classify team.

        Digit-model inferences run sequentially (one crop per call) because the
        TRT engine is exported with static batch_size=1. Batching N crops raises
        a silent AssertionError that returns all-None, skipping every jersey vote.

        Players whose jersey number is already confirmed by the tracker skip digit
        OCR entirely — team classification still runs for new track_ids.

        Args:
            frame:           Full BGR frame.
            tracked_players: [{track_id, bbox, ...}] from tracker.py.
            tracker:         PlayerTracker; when provided, update_jersey() is
                             called automatically after a vote is confirmed.
            roster:          Match roster dict keyed by jersey number string.
                             When provided, OCR candidates not present in the
                             roster are silently discarded before entering the
                             vote deque — eliminates impossible readings.
        """
        if frame is None or frame.size == 0 or not tracked_players:
            return {"jersey_results": [], "team_colors": self._team_colors}

        self._frame_counter += 1
        h_frame, w_frame = frame.shape[:2]
        n = len(tracked_players)

        # ── Layer 1: full-frame jersey_no.pt (1 GPU call shared by all players) ──
        _frame_num_boxes = self._detect_numbers_fullframe(frame)
        logger.info(
            "JerseyOCR call #%d: %d players  jersey_no.pt=%d boxes",
            self._frame_counter, n, len(_frame_num_boxes),
        )

        # ── Pass 1: bucket each player — fast-path done vs. needs batch OCR ──
        # results[i] filled directly for fast-path players; None means pending.
        # pending maps slot index → (player, player_crop, number_crop).
        results: list[dict | None] = [None] * n
        pending: dict[int, tuple]  = {}

        for i, player in enumerate(tracked_players):
            track_id = player["track_id"]

            # Per-track OCR timer — same logic as original
            frames_since_ocr = self._frame_counter - self._last_ocr_frame.get(
                track_id, self._frame_counter - OCR_SAMPLE_EVERY
            )
            run_ocr = frames_since_ocr >= OCR_SAMPLE_EVERY

            # Fast path A: timer hasn't fired AND team already known
            if not run_ocr and track_id in self._track_team:
                confirmed_num, conf = self._vote_status(track_id)
                results[i] = {
                    "track_id":       track_id,
                    "jersey_number":  confirmed_num,
                    "confidence":     conf,
                    "team":           self._track_team[track_id],
                    "team_color_hsv": self._team_colors.get(self._track_team[track_id], []),
                    "blur_skipped":   False,
                }
                continue

            # Fast path B removed: OCR always continues even after jersey confirmation.
            # The rolling vote deque (maxlen=MAX_VOTE_HISTORY) naturally updates the
            # confirmed number when the camera zooms in and a higher-confidence read
            # arrives. Confidence-upgrade logic in Pass 2 flushes stale votes when a
            # significantly better read is detected (CONF_UPGRADE_MIN + CONF_UPGRADE_DELTA).

            # Extract player crop
            x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
            x1c = max(0, x1);    y1c = max(0, y1)
            x2c = min(w_frame, x2); y2c = min(h_frame, y2)
            if x2c <= x1c or y2c <= y1c:
                results[i] = self._empty_result(track_id)
                continue
            player_crop = frame[y1c:y2c, x1c:x2c]

            # Layer 1 result: find number_crop for this player
            # Priority 1: jersey_no.pt full-frame bbox overlapping this player
            # Priority 2: heuristic chest strip on player crop (fallback)
            number_crop = None
            if _frame_num_boxes:
                best_match, best_ratio = None, 0.0
                for (nx1f, ny1f, nx2f, ny2f, _nconf) in _frame_num_boxes:
                    ix1 = max(x1c, nx1f); iy1 = max(y1c, ny1f)
                    ix2 = min(x2c, nx2f); iy2 = min(y2c, ny2f)
                    if ix2 > ix1 and iy2 > iy1:
                        overlap   = (ix2 - ix1) * (iy2 - iy1)
                        num_area  = max(1, (nx2f - nx1f) * (ny2f - ny1f))
                        ratio     = overlap / num_area
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_match = (nx1f, ny1f, nx2f, ny2f)
                if best_match and best_ratio >= 0.25:
                    nx1f, ny1f, nx2f, ny2f = best_match
                    pad = 10
                    nc  = frame[
                        max(0, ny1f - pad):min(h_frame, ny2f + pad),
                        max(0, nx1f - pad):min(w_frame, nx2f + pad),
                    ]
                    if nc.size > 0:
                        number_crop = nc
                        logger.debug(
                            "track %d: jersey_no.pt full-frame match (overlap=%.2f)",
                            track_id, best_ratio,
                        )

            if number_crop is None:
                num_bbox = self._detect_number_bbox(player_crop)
                if num_bbox is None:
                    results[i] = self._empty_result(track_id)
                    continue
                ny1h, ny2h, nx1h, nx2h = num_bbox
                number_crop = player_crop[ny1h:ny2h, nx1h:nx2h]
                if number_crop.size == 0:
                    results[i] = self._empty_result(track_id)
                    continue

            # Blur guard — same logic as original
            _crop_gray = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY) \
                if number_crop.ndim == 3 else number_crop
            _effective_blur_thr = (
                BLUR_THRESHOLD * 0.40
                if float(_crop_gray.mean()) > 160
                else BLUR_THRESHOLD
            )
            _blur = _blur_score(_crop_gray)
            if _blur < _effective_blur_thr:
                logger.debug(
                    "track %d: blur score=%.1f < %.1f — OCR skipped",
                    track_id, _blur, _effective_blur_thr,
                )
                self._last_ocr_frame[track_id] = self._frame_counter
                team     = self._classify_team(player_crop, track_id)
                team_hsv = self._team_colors.get(team, []) if team else []
                num, conf = self._vote_status(track_id)
                results[i] = {
                    "track_id":       track_id,
                    "jersey_number":  num,
                    "confidence":     conf,
                    "team":           team,
                    "team_color_hsv": team_hsv,
                    "blur_skipped":   True,
                }
                continue

            # Schedule for batch digit OCR in Pass 2
            pending[i] = (player, player_crop, number_crop)

        # ── Batch digit OCR: all pending tight crops in ONE GPU call ─────────
        if pending:
            slot_indices = sorted(pending.keys())
            tight_crops  = [pending[i][2] for i in slot_indices]
            tight_cands  = self._batch_digit_ocr(tight_crops)

            # Small second batch for fallback wide crop where tight gave None
            fb_map: dict[int, tuple] = {}   # j (tight-list index) → (slot_idx, fb_crop)
            for j, slot_i in enumerate(slot_indices):
                if tight_cands[j] is not None:
                    continue
                _, player_crop, _ = pending[slot_i]
                ph, pw = player_crop.shape[:2]
                fb = player_crop[int(ph * 0.10):int(ph * 0.75),
                                 int(pw * 0.05):int(pw * 0.95)]
                if fb.size > 0:
                    fb_map[j] = (slot_i, fb)

            if fb_map:
                fb_keys  = sorted(fb_map.keys())
                fb_crops = [fb_map[j][1] for j in fb_keys]
                fb_cands = self._batch_digit_ocr(fb_crops)
                for k, j in enumerate(fb_keys):
                    if k < len(fb_cands) and fb_cands[k] is not None:
                        tight_cands[j] = fb_cands[k]
                        logger.debug(
                            "track %d: digit from fallback wide crop",
                            pending[fb_map[j][0]][0]["track_id"],
                        )

            logger.info(
                "JerseyOCR batch: %d/%d players sent to digit model",
                len(slot_indices), n,
            )

            # ── Pass 2: vote, confidence upgrade, team classify, build results ──
            for j, slot_i in enumerate(slot_indices):
                player, player_crop, _ = pending[slot_i]
                track_id  = player["track_id"]

                # Unpack (number, mean_conf) from digit model; default to no read.
                raw_result = tight_cands[j]
                candidate  = None
                read_conf  = 0.0
                if raw_result is not None:
                    candidate, read_conf = raw_result

                # Roster whitelist: drop candidates not worn by any player
                if candidate is not None and roster:
                    js = str(candidate)
                    in_roster = (js in roster or
                                 f"{js}_A" in roster or
                                 f"{js}_B" in roster)
                    if not in_roster:
                        logger.debug(
                            "track %d: OCR candidate %d rejected — not in roster",
                            track_id, candidate,
                        )
                        candidate  = None
                        read_conf  = 0.0

                # ── Confidence-upgrade flush ──────────────────────────────────
                # When the camera zooms in and produces a significantly more
                # confident read than anything seen before, flush the stale vote
                # deque so the new reading takes over quickly (VOTE_THRESHOLD
                # re-confirmation in ~0.27s) rather than being diluted by old votes.
                if candidate is not None and read_conf >= CONF_UPGRADE_MIN:
                    prev_best = self._best_read_conf.get(track_id, 0.0)
                    if read_conf > prev_best + CONF_UPGRADE_DELTA:
                        current_confirmed, _ = self._vote_status(track_id)
                        if current_confirmed is not None and candidate != current_confirmed:
                            self._votes.pop(track_id, None)
                            logger.info(
                                "track %d: confidence upgrade %.2f→%.2f %s→%d — flushing votes",
                                track_id, prev_best, read_conf, current_confirmed, candidate,
                            )
                    if read_conf > self._best_read_conf.get(track_id, 0.0):
                        self._best_read_conf[track_id] = read_conf

                self._last_ocr_frame[track_id] = self._frame_counter
                self._vote_number(track_id, candidate)
                confirmed, conf = self._vote_status(track_id)

                logger.info(
                    "track %d: candidate=%s  votes=%s  confirmed=%s",
                    track_id, candidate,
                    dict(Counter(self._votes.get(track_id, []))),
                    confirmed,
                )

                if confirmed is not None and tracker is not None:
                    tracker.update_jersey(track_id, confirmed)

                team     = self._classify_team(player_crop, track_id)
                team_hsv = self._team_colors.get(team, []) if team else []
                results[slot_i] = {
                    "track_id":       track_id,
                    "jersey_number":  confirmed,
                    "confidence":     conf,
                    "team":           team,
                    "team_color_hsv": team_hsv,
                    "blur_skipped":   False,
                }

        return {
            "jersey_results": [r for r in results if r is not None],
            "team_colors":    self._team_colors,
        }

    # ------------------------------------------------------------------
    # Layer 1 — number localisation
    # ------------------------------------------------------------------

    def _detect_numbers_fullframe(
        self, frame: np.ndarray
    ) -> list[tuple[int, int, int, int, float]]:
        """
        Run jersey_no.pt on the FULL FRAME and return all detected number bboxes.

        Returns list of (x1, y1, x2, y2, conf) in frame coordinates.

        Full-frame inference is far more reliable than per-player crop inference:
        the model was trained on full-frame context and gives confident detections
        (0.4-0.9) on complete frames vs near-zero on 60-160px player crops.
        Called once per process() call and shared across all players.
        """
        if self.model is None or frame is None or frame.size == 0:
            return []
        try:
            det = self.model(frame, conf=0.10, verbose=False)
            boxes = []
            for r in det:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if (
                        self._number_class_idx is not None
                        and cls != self._number_class_idx
                    ):
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    boxes.append((x1, y1, x2, y2, conf))
            # INFO level so we can see in logs whether the localiser fires at all
            logger.info("jersey_no.pt full-frame: %d number boxes found", len(boxes))
            return boxes
        except Exception as exc:
            logger.warning("jersey_no.pt full-frame inference error: %s", exc)
            return []

    def _detect_number_bbox(
        self, player_crop: np.ndarray
    ) -> Optional[tuple[int, int, int, int]]:
        """
        Heuristic fallback: returns (y1, y2, x1, x2) of the chest strip region
        within player_crop (15%-55% height, 10%-90% width).

        Used when jersey_no.pt full-frame detection finds no number box that
        overlaps with this player's bbox. See process() for the full cascade.
        """
        h, w = player_crop.shape[:2]
        if h < 20 or w < 10:
            return None
        return _back_crop_heuristic(h, w)

    # ------------------------------------------------------------------
    # Layer 2 — digit detection
    # ------------------------------------------------------------------

    def _run_yolo_digit_ocr(self, number_crop: np.ndarray) -> Optional[int]:
        """
        Run best-detect-num-v2.pt on the jersey number crop.

        Detects individual digits 0-9 as YOLO object classes, sorts detected
        boxes left-to-right by x-coordinate, concatenates their labels into a
        jersey number string (e.g. digit "1" at x=10, digit "9" at x=30 → "19" → 19).

        Returns None when the digit model is not loaded or no digits are detected.
        """
        if self._digit_model is None or number_crop is None or number_crop.size == 0:
            return None

        h, w = number_crop.shape[:2]
        if h < 8 or w < 4:
            return None

        # Upscale small crops so fine digit strokes are visible to the model
        if h < OCR_HEIGHT_PX:
            scale = OCR_HEIGHT_PX / h
            number_crop = cv2.resize(
                number_crop,
                (max(16, int(w * scale)), OCR_HEIGHT_PX),
                interpolation=cv2.INTER_CUBIC,
            )

        try:
            det = self._digit_model(number_crop, conf=DIGIT_CONF_THRESHOLD, verbose=False)
        except Exception as exc:
            logger.debug("YOLO digit model inference error: %s", exc)
            return None

        digit_boxes: list[tuple[float, str, float]] = []  # (x1, digit_char, conf)
        for r in det:
            for box in r.boxes:
                cls = int(box.cls[0])
                # Prefer model class name when it is a bare digit char; fallback to index
                raw_name   = self._digit_model.names.get(cls, str(cls))
                digit_char = raw_name if (len(raw_name) == 1 and raw_name.isdigit()) else str(cls)
                x1   = float(box.xyxy[0][0])
                conf = float(box.conf[0])
                digit_boxes.append((x1, digit_char, conf))
                logger.debug("digit: '%s' x1=%.0f conf=%.2f", digit_char, x1, conf)

        if not digit_boxes:
            logger.debug("digit model: no digits found in crop h=%d w=%d", *number_crop.shape[:2])
            return None

        # When >2 digits detected (background artifacts), keep the 2 most confident
        # before sorting left-to-right — avoids wrongly taking low-conf edge detections.
        if len(digit_boxes) > 2:
            digit_boxes.sort(key=lambda b: -b[2])
            digit_boxes = digit_boxes[:2]
        digit_boxes.sort(key=lambda b: b[0])
        number_str = "".join(b[1] for b in digit_boxes)

        try:
            number = int(number_str)
            if 0 <= number <= 99:
                logger.debug("YOLO digit OCR → jersey %d", number)
                return number
        except ValueError:
            pass

        return None

    def _batch_digit_ocr(self, crops: list) -> list:
        """
        Run best-detect-num-v2.pt on a batch of crops in a single GPU call.

        Replaces N sequential _run_yolo_digit_ocr() calls with one batched
        model.predict() — roughly N× faster on CUDA.
        Example: 10 players → 10 × 15 ms = 150 ms  →  1 × 25 ms = 25 ms.

        Returns list[Optional[tuple[int, float]]] index-aligned with crops.
        Each element is (jersey_number, mean_digit_confidence) or None.
        """
        if not crops or self._digit_model is None:
            return [None] * len(crops)

        # Upscale small crops (mirrors _run_yolo_digit_ocr's resize step)
        processed: list = []
        for crop in crops:
            if crop is None or crop.size == 0:
                processed.append(None)
                continue
            h, w = crop.shape[:2]
            if h < 8 or w < 4:
                processed.append(None)
                continue
            if h < OCR_HEIGHT_PX:
                scale = OCR_HEIGHT_PX / h
                crop = cv2.resize(
                    crop,
                    (max(16, int(w * scale)), OCR_HEIGHT_PX),
                    interpolation=cv2.INTER_CUBIC,
                )
            processed.append(crop)

        valid_idx   = [i for i, c in enumerate(processed) if c is not None]
        valid_crops = [processed[i] for i in valid_idx]

        if not valid_crops:
            return [None] * len(crops)

        # TRT engine compiled with static batch_size=1 — process sequentially.
        # Batching N crops at once raises AssertionError from ultralytics when
        # the engine max-batch != N; the exception is silent at DEBUG level and
        # returns all-None, causing every jersey number vote to be skipped.
        out = [None] * len(crops)
        for j, orig_idx in enumerate(valid_idx):
            try:
                det = self._digit_model(
                    valid_crops[j],
                    conf=DIGIT_CONF_THRESHOLD,
                    verbose=False,
                )
                out[orig_idx] = self._parse_digit_result(det[0])
            except Exception as exc:
                logger.debug("Digit model inference error (crop %d): %s", j, exc)
        return out

    def _parse_digit_result(self, det_result) -> Optional[tuple[int, float]]:
        """
        Parse one YOLO digit-model Result object into (jersey_number, mean_confidence)
        or None. mean_confidence is the average per-digit detection confidence — used
        by the confidence-upgrade logic in Pass 2 to detect zoom-in upgrades.
        """
        digit_boxes: list[tuple[float, str, float]] = []
        for box in det_result.boxes:
            cls        = int(box.cls[0])
            raw_name   = self._digit_model.names.get(cls, str(cls))
            digit_char = (raw_name
                          if (len(raw_name) == 1 and raw_name.isdigit())
                          else str(cls))
            x1   = float(box.xyxy[0][0])
            conf = float(box.conf[0])
            digit_boxes.append((x1, digit_char, conf))
            logger.debug("digit: '%s' x1=%.0f conf=%.2f", digit_char, x1, conf)

        if not digit_boxes:
            return None

        if len(digit_boxes) > 2:
            digit_boxes.sort(key=lambda b: -b[2])
            digit_boxes = digit_boxes[:2]
        digit_boxes.sort(key=lambda b: b[0])
        number_str  = "".join(b[1] for b in digit_boxes)
        mean_conf   = sum(b[2] for b in digit_boxes) / len(digit_boxes)

        try:
            number = int(number_str)
            if 0 <= number <= 99:
                logger.debug("YOLO digit OCR → jersey %d (conf=%.2f)", number, mean_conf)
                return number, mean_conf
        except ValueError:
            pass
        return None

    def _run_ocr(self, number_crop: np.ndarray) -> Optional[int]:
        """
        PaddleOCR pipeline — currently inactive; YOLO digit model is used instead.
        To re-enable: change _run_yolo_digit_ocr() calls in process() back to _run_ocr()
        and restore load_ocr() + warmup() calls in create_jersey_ocr().
        """
        # if self._ocr is None:
        #     return None
        # if number_crop.size == 0:
        #     return None
        #
        # processed = _preprocess_for_ocr(number_crop)
        # otsu      = _otsu_variant(number_crop)
        # adaptive  = _adaptive_variant(number_crop)
        #
        # # Four variants: CLAHE, inverted CLAHE, Otsu binary, adaptive binary.
        # candidates: list[tuple[int, float]] = []
        # for img in [processed, cv2.bitwise_not(processed), otsu, adaptive]:
        #     try:
        #         results = self._ocr.predict(img)
        #     except Exception as exc:
        #         logger.debug("PaddleOCR predict error: %s", exc)
        #         continue
        #     if not results:
        #         continue
        #     for res in results:
        #         try:
        #             texts  = res["rec_texts"]
        #             scores = res["rec_scores"]
        #         except (TypeError, KeyError):
        #             continue
        #         for text, score in zip(texts or [], scores or []):
        #             if score < OCR_MIN_SCORE:
        #                 continue
        #             number = self._parse_jersey_number(text)
        #             if number is not None:
        #                 candidates.append((number, score))
        #                 logger.debug("OCR: '%s' (score=%.2f) → jersey %d", text, score, number)
        #     if candidates and max(c[1] for c in candidates) >= HIGH_CONF_EARLY_EXIT:
        #         break
        #
        # if not candidates:
        #     return None
        # best_number, best_score = max(candidates, key=lambda x: x[1])
        # logger.debug("OCR best: jersey %d (score=%.2f)", best_number, best_score)
        # return best_number
        return None  # inactive — use _run_yolo_digit_ocr()

    # Common OCR character confusions for jersey number context.
    # Applied before digit extraction so "O3" → "03", "S7" → "57", etc.
    _OCR_CHAR_MAP = str.maketrans("OIlSZ", "01152")

    @staticmethod
    def _parse_jersey_number(text: str) -> Optional[int]:
        """Extract first integer 0-99 from OCR text. Returns None if invalid."""
        if not text:
            return None
        # Normalise: uppercase, fix common OCR confusions, then strip non-digits
        clean  = text.strip().upper().translate(JerseyOCR._OCR_CHAR_MAP)
        digits = re.sub(r"[^0-9]", "", clean)
        if not digits:
            return None
        try:
            number = int(digits[:2])   # at most 2 digits
            return number if 0 <= number <= 99 else None
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def _vote_number(self, track_id: int, candidate: Optional[int]) -> Optional[int]:
        """
        Accumulate candidate votes per track_id.
        - None candidates are ignored (not added to history), preserving existing votes.
        - Returns confirmed jersey number (via prefix-aware resolution) once
          >= VOTE_THRESHOLD votes exist, else None.
        """
        if candidate is not None:
            if track_id not in self._votes:
                self._votes[track_id] = deque(maxlen=MAX_VOTE_HISTORY)
            self._votes[track_id].append(candidate)

        return self._current_confirmed(track_id)

    def _resolve_votes(self, raw_counter: Counter, total: int) -> tuple[Optional[int], float]:
        """
        Prefix/suffix-aware vote resolution for jersey number ambiguity.

        OCR often produces partial digit reads from fast camera motion:
          "10" → "1"  (right digit clipped when camera pans left)
          "23" → "3"  (left digit clipped when camera pans right)
        Simple majority voting would lock onto the wrong single digit.

        Rule: if a 1-digit candidate AND a 2-digit candidate share a digit
        in the same position (prefix OR suffix), AND the 2-digit candidate
        has >= TRANSFER_MIN_LONG_VOTES independent reads (enough evidence it
        is real), then all 1-digit votes are reinterpreted as partial reads
        and transferred to the 2-digit candidate.

        True single-digit players (#1–#9) are protected: if no matching
        2-digit candidate has sufficient votes, no transfer occurs and the
        1-digit reading is kept as-is.

        Multiple 2-digit matches (rare): votes distributed proportionally
        to each match's existing count.
        """
        if not raw_counter:
            return None, 0.0

        adjusted = dict(raw_counter)

        for short_num, short_count in list(raw_counter.items()):
            if short_num is None or not (0 <= short_num <= 9):
                continue  # only process 1-digit candidates

            short_str = str(short_num)

            # Collect 2-digit candidates that contain this digit as first or last digit
            # and have enough reads to be considered genuine (not background contamination).
            matching_long = [
                (long_num, long_count)
                for long_num, long_count in raw_counter.items()
                if long_num is not None
                and 10 <= long_num <= 99
                and (str(long_num)[0] == short_str or str(long_num)[-1] == short_str)
                and long_count >= TRANSFER_MIN_LONG_VOTES
            ]

            if not matching_long:
                continue  # no 2-digit match with enough evidence → keep 1-digit votes

            # Distribute 1-digit votes proportionally across matching 2-digit candidates
            total_long = sum(c for _, c in matching_long)
            for long_num, long_count in matching_long:
                transfer = round(short_count * long_count / total_long)
                adjusted[long_num] = adjusted.get(long_num, 0) + transfer
            adjusted[short_num] = 0  # 1-digit votes fully absorbed

        valid = {k: v for k, v in adjusted.items() if k is not None and v > 0}
        if not valid:
            return None, 0.0

        best = max(valid, key=valid.get)
        best_count = valid[best]
        confirmed = best if best_count >= VOTE_THRESHOLD else None
        confidence = best_count / total if total > 0 else 0.0
        return confirmed, confidence

    def _current_confirmed(self, track_id: int) -> Optional[int]:
        if track_id not in self._votes or not self._votes[track_id]:
            return None
        votes = self._votes[track_id]
        confirmed, _ = self._resolve_votes(Counter(votes), len(votes))
        return confirmed

    def _vote_confidence(self, track_id: int) -> float:
        """Fraction of votes for the leading candidate [0.0, 1.0]."""
        if track_id not in self._votes or not self._votes[track_id]:
            return 0.0
        votes = self._votes[track_id]
        _, confidence = self._resolve_votes(Counter(votes), len(votes))
        return confidence

    def _vote_status(self, track_id: int) -> tuple[Optional[int], float]:
        """Return (confirmed_number, confidence) using prefix-aware vote resolution."""
        if track_id not in self._votes or not self._votes[track_id]:
            return None, 0.0
        votes = self._votes[track_id]
        return self._resolve_votes(Counter(votes), len(votes))

    def reset(self) -> None:
        """
        Full reset at quarter boundaries.

        Clears vote history AND per-track team cache so that newly assigned
        track IDs (ByteTrack re-creates its ID counter on reset) do not
        inherit stale data from the previous quarter.
        Team colour references (_team_colors) are kept — they are calibrated
        once per game and remain valid across quarters.
        """
        self._votes.clear()
        self._track_team.clear()
        self._track_team_votes.clear()
        self._frame_counter = 0
        self._last_ocr_frame.clear()
        self._best_read_conf.clear()
        logger.info("JerseyOCR reset (quarter boundary)")

    def reset_votes(self, track_id: Optional[int] = None) -> None:
        """Clear voting history for one track or all tracks."""
        if track_id is None:
            self._votes.clear()
        else:
            self._votes.pop(track_id, None)

    # ------------------------------------------------------------------
    # Team classification (K-Means HSV)
    # ------------------------------------------------------------------

    _TEAM_VOTE_HISTORY = 20   # rolling window — votes older than this are forgotten

    def _classify_team(
        self, player_crop: np.ndarray, track_id: int
    ) -> Optional[str]:
        """
        Assign track to "A" or "B" by comparing jersey chest color to calibrated
        team color library. Re-runs every frame — no permanent lock.

        Pipeline:
          1. Compute median HSV of player chest crop (20-50% height, 15-85% width).
          2. Measure HSV distance to each team center (_team_colors["A"/"B"]).
          3. Append winner to a rolling deque (max _TEAM_VOTE_HISTORY reads).
          4. Return current majority of the rolling window.

        Rolling window (not permanent lock) means:
          - A zoom-in that gives clearer color overrides stale wrong votes.
          - Lighting/angle changes are absorbed within ~20 frames (~1.3s at 15fps).
          - If a crop fails (partial occlusion), _track_team holds the last known
            result so the player's team doesn't flicker to None.
        """
        if "A" not in self._team_colors or "B" not in self._team_colors:
            return self._track_team.get(track_id)   # no calibration yet — last known

        dominant_hsv = self._dominant_hsv(player_crop)
        if dominant_hsv is None:
            return self._track_team.get(track_id)   # crop failed — last known, no new vote

        dist_a = self._hsv_distance(dominant_hsv, self._team_colors["A"])
        dist_b = self._hsv_distance(dominant_hsv, self._team_colors["B"])
        vote   = "A" if dist_a <= dist_b else "B"

        # Rolling deque: append vote; old reads fall off after _TEAM_VOTE_HISTORY entries
        if track_id not in self._track_team_votes:
            self._track_team_votes[track_id] = deque(maxlen=self._TEAM_VOTE_HISTORY)
        self._track_team_votes[track_id].append(vote)

        votes     = self._track_team_votes[track_id]
        count_a   = votes.count("A")
        count_b   = votes.count("B")
        result    = "A" if count_a >= count_b else "B"

        # Update current-best cache (used by Fast Path A and inherit_team)
        self._track_team[track_id] = result
        return result

    def calibrate_teams(
        self, frame: np.ndarray, tracked_players: list[dict]
    ) -> dict[str, list[float]]:
        """
        Cluster one representative HSV point per player (not raw pixels) so
        jersey colors dominate over background, court, and skin noise.

        Previous approach pooled all raw torso pixels → thousands of samples
        where court / shorts / skin contaminated the cluster centers.

        New approach:
          1. Tight chest crop (20-50% height) per player — skips shoulders & shorts
          2. Median HSV per player — robust to white numbers on coloured jerseys
          3. K-Means on N per-player medians (N ≈ players on frame, ~5-10 points)
          4. K-Means++ init + 20 restarts — stable on small datasets
        """
        if frame is None or not tracked_players:
            return self._team_colors

        h_frame, w_frame = frame.shape[:2]
        per_player: list[np.ndarray] = []   # one [H, S, V] median per player

        for player in tracked_players:
            x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
            crop = frame[max(0, y1):min(h_frame, y2),
                         max(0, x1):min(w_frame, x2)]
            if crop.size == 0:
                continue
            th, tw = crop.shape[:2]
            # Chest strip: 20%-50% height, 15%-85% width
            # Avoids head, shoulders (skin), shorts (black), arms
            chest = crop[int(th * 0.20):int(th * 0.50),
                         int(tw * 0.15):int(tw * 0.85)]
            if chest.size == 0:
                continue
            hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
            per_player.append(np.median(hsv, axis=0))  # median robust to number pixels

        if len(per_player) < 2:
            logger.warning("calibrate_teams: need ≥ 2 players, got %d", len(per_player))
            return self._team_colors

        samples = np.array(per_player, dtype=np.float32)

        # K-Means++ gives stable init on small N; 20 restarts + tighter eps
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centers = cv2.kmeans(
            samples, 2, None, criteria, 20, cv2.KMEANS_PP_CENTERS
        )

        # Guard: if one cluster has < 2 members it's likely a referee or
        # spectator, not a second team. K-Means would split "all jerseys" vs
        # "dark official" → every player lands in cluster A.
        # Skip calibration and retry on the next eligible frame.
        count_a = int(np.sum(labels == 0))
        count_b = int(np.sum(labels == 1))
        if min(count_a, count_b) < 2:
            logger.info(
                "calibrate_teams: skipped — unbalanced clusters (%d vs %d). "
                "Likely a referee/official in frame. Retrying next frame.",
                count_a, count_b,
            )
            return self._team_colors

        self._team_colors["A"] = centers[0].tolist()
        self._team_colors["B"] = centers[1].tolist()
        self._track_team.clear()
        self._track_team_votes.clear()

        logger.info(
            "Team colors calibrated (%d players, clusters %d+%d): A=%s  B=%s",
            len(per_player), count_a, count_b,
            [round(v, 1) for v in self._team_colors["A"]],
            [round(v, 1) for v in self._team_colors["B"]],
        )
        return self._team_colors

    def set_team_reference(self, team: str, hsv_color: list[float]) -> None:
        """Manual override from user UI. team='A' or 'B', hsv_color=[H,S,V]."""
        if team not in ("A", "B"):
            raise ValueError("team must be 'A' or 'B'")
        self._team_colors[team] = list(hsv_color)
        self._track_team.clear()
        self._track_team_votes.clear()

    def inherit_team(self, new_tid: int, old_tid: int) -> None:
        """Copy locked team or accumulated votes from old_tid to new_tid.

        Called when ByteTrack drops a track and re-assigns a new ID to the same
        physical player, so team colour doesn't flicker for 3 tentative frames.
        """
        if old_tid in self._track_team:
            self._track_team[new_tid] = self._track_team[old_tid]
        if old_tid in self._track_team_votes:
            self._track_team_votes[new_tid] = deque(
                self._track_team_votes[old_tid],
                maxlen=self._TEAM_VOTE_HISTORY,
            )

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_hsv(crop: np.ndarray) -> Optional[list[float]]:
        """
        Median HSV of the jersey chest region.

        Uses the same 20-50%/15-85% chest crop as calibrate_teams() so that
        classify_team() compares apples-to-apples against the calibration centers.
        Median is robust to white jersey numbers and dark shorts bleeding into crop.
        """
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return None
        chest = crop[int(h * 0.20):int(h * 0.50), int(w * 0.15):int(w * 0.85)]
        if chest.size == 0:
            return None
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
        return np.median(hsv.reshape(-1, 3).astype(np.float32), axis=0).tolist()

    @staticmethod
    def _hsv_distance(a: list[float], b: list[float]) -> float:
        """
        Hue-aware HSV distance, robust to arena lighting variation.

        Hue weight scales with the minimum saturation of the two colors:
        achromatic colors (white/gray/black, S≈0) have undefined/unreliable
        hue, so the hue term is suppressed when either color is achromatic.
        V weight is low (0.25) since brightness shifts under different arena
        lighting conditions should not flip a player's team assignment.
        """
        dh = abs(a[0] - b[0])
        dh = min(dh, 180.0 - dh)          # circular hue distance
        ds = abs(a[1] - b[1])
        dv = abs(a[2] - b[2])
        # Scale hue weight by saturation — 0 when achromatic, up to 2.0 when fully saturated
        hue_weight = 2.0 * min(a[1], b[1]) / 255.0
        return float(hue_weight * dh + ds + dv * 0.25)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(track_id: int) -> dict:
        return {
            "track_id":       track_id,
            "jersey_number":  None,
            "confidence":     0.0,
            "team":           None,
            "team_color_hsv": [],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_jersey_ocr(
    models_path: Optional[str] = None,
    device: Optional[str] = None,
) -> "JerseyOCR":
    """Create and load models in one call. Uses YOLO digit detector as primary OCR."""
    model_file = os.path.join(models_path or MODELS_DIR, MODEL_FILENAME)
    ocr = JerseyOCR(model_path=model_file, device=device)
    ocr.load_model()
    ocr.load_digit_model()
    # ocr.load_ocr()   # PaddleOCR — inactive; uncomment to switch back
    return ocr


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=== JerseyOCR smoke test ===\n")

    jerseyocr = JerseyOCR()

    # ── 1. _vote_number logic ─────────────────────────────────────────
    print("--- 1. _vote_number ---")
    tid = 10

    # 1 vote for 7 → not confirmed yet (need >= VOTE_THRESHOLD=2)
    jerseyocr._vote_number(tid, 7)
    assert jerseyocr._current_confirmed(tid) is None, "1 vote should not confirm"
    print("  1 vote for 7 → not confirmed ✓")

    # 2nd vote for 7 → confirmed (VOTE_THRESHOLD=2)
    result = jerseyocr._vote_number(tid, 7)
    assert result == 7, f"Expected 7 confirmed after 2 votes, got {result}"
    print(f"  2 votes for 7 → confirmed: {result} ✓ (VOTE_THRESHOLD={VOTE_THRESHOLD})")

    # Confidence
    conf = jerseyocr._vote_confidence(tid)
    assert conf > 0.5, f"Expected confidence > 0.5, got {conf:.2f}"
    print(f"  Vote confidence: {conf:.2f} ✓")

    # reset_votes for one track
    jerseyocr.reset_votes(tid)
    assert tid not in jerseyocr._votes
    print("  reset_votes(track_id) cleared ✓")

    # None candidate → no change, still returns None (no votes)
    r_none = jerseyocr._vote_number(99, None)
    assert r_none is None
    print("  None candidate → None ✓")

    # Voting window: only last MAX_VOTE_HISTORY candidates matter
    tid2 = 20
    for _ in range(MAX_VOTE_HISTORY):
        jerseyocr._vote_number(tid2, 23)
    # Now override with different number → old ones still dominate
    for _ in range(2):
        jerseyocr._vote_number(tid2, 99)
    confirmed2 = jerseyocr._current_confirmed(tid2)
    assert confirmed2 == 23, f"Expected 23, got {confirmed2}"
    print(f"  Rolling window: dominant=23 persists over 2 votes for 99 ✓")

    jerseyocr.reset_votes()
    assert len(jerseyocr._votes) == 0
    print("  reset_votes() clears all ✓")

    # ── 1b. prefix/suffix-aware _resolve_votes ────────────────────────
    print("\n--- 1b. _resolve_votes (prefix/suffix-aware) ---")
    from collections import deque as _deque

    # Case 1: "10" OCR'd as "1" — "1"×4 + "10"×3 → should confirm "10"
    jerseyocr._votes[81] = _deque([1, 1, 10, 1, 10, 10, 1], maxlen=MAX_VOTE_HISTORY)
    num, conf = jerseyocr._vote_status(81)
    assert num == 10, f"Expected 10 (prefix-aware: '1' transfers to '10'), got {num}"
    print(f"  '1'×4 + '10'×3 → prefix transfer → confirms: {num} ✓")

    # Case 2: True #1 — no 2-digit match with enough votes → stays as 1
    jerseyocr._votes[82] = _deque([1, 1, 1, 1, 1], maxlen=MAX_VOTE_HISTORY)
    num, conf = jerseyocr._vote_status(82)
    assert num == 1, f"Expected 1 (no 2-digit match), got {num}"
    print(f"  '1'×5 only → no transfer → confirms: {num} (true #1 protected) ✓")

    # Case 3: Suffix — "23" OCR'd as "3" — "3"×4 + "23"×3 → should confirm "23"
    jerseyocr._votes[83] = _deque([3, 3, 23, 3, 23, 23, 3], maxlen=MAX_VOTE_HISTORY)
    num, conf = jerseyocr._vote_status(83)
    assert num == 23, f"Expected 23 (suffix-aware: '3' transfers to '23'), got {num}"
    print(f"  '3'×4 + '23'×3 → suffix transfer → confirms: {num} ✓")

    # Case 4: Stray misread — "1"×5 + "10"×2 → not enough 2-digit evidence, no transfer
    jerseyocr._votes[84] = _deque([1, 1, 10, 1, 10, 1, 1], maxlen=MAX_VOTE_HISTORY)
    num, conf = jerseyocr._vote_status(84)
    # "10" only has 2 votes < TRANSFER_MIN_LONG_VOTES=3 → no transfer → "1" wins
    assert num == 1, f"Expected 1 (insufficient 2-digit evidence, TRANSFER_MIN=3), got {num}"
    print(f"  '1'×5 + '10'×2 → insufficient evidence → no transfer → confirms: {num} ✓")

    jerseyocr.reset_votes()

    # ── 2. _parse_jersey_number ───────────────────────────────────────
    print("\n--- 2. _parse_jersey_number ---")
    cases = [
        ("23",    23),
        (" 7 ",    7),
        ("00",     0),
        ("99",    99),
        ("100",   10),   # takes first 2 digits: "10"
        ("abc",  None),
        ("",     None),
        ("3b",     3),
        ("B5",     5),
        ("123",   12),   # takes first 2 digits
    ]
    for text, expected in cases:
        got = JerseyOCR._parse_jersey_number(text)
        status = "✓" if got == expected else f"✗ FAIL (got {got})"
        print(f"  '{text}' → {got}  {status}")
        assert got == expected, f"parse_jersey_number('{text}') = {got}, expected {expected}"

    # ── 3. _preprocess_for_ocr ───────────────────────────────────────
    print("\n--- 3. _preprocess_for_ocr ---")
    dummy_crop = np.random.randint(0, 255, (80, 60, 3), dtype=np.uint8)
    processed  = _preprocess_for_ocr(dummy_crop)
    assert processed.shape[0] >= OCR_HEIGHT_PX, f"Expected height >= {OCR_HEIGHT_PX}"
    assert processed.shape[2] == 3, "Expected 3 channels"
    print(f"  Input (80×60) → output {processed.shape[:2]} (base height={OCR_HEIGHT_PX}px + padding) ✓")

    tiny = np.zeros((2, 1, 3), dtype=np.uint8)
    result_tiny = _preprocess_for_ocr(tiny)
    assert result_tiny.shape[0] >= OCR_HEIGHT_PX
    print("  Tiny 2×1 crop → safe output ✓")

    # ── 4. _back_crop_heuristic ──────────────────────────────────────
    print("\n--- 4. _back_crop_heuristic ---")
    y1, y2, x1, x2 = _back_crop_heuristic(200, 100)
    assert y1 < y2 and x1 < x2
    assert y1 == 30 and y2 == 110  # 15%-55% of 200 (chest strip)
    assert x1 == 10 and x2 == 90   # 10%-90% of 100
    print(f"  200×100 player → chest strip: y=[{y1},{y2}] x=[{x1},{x2}] ✓")

    # ── 5. HSV team distance ──────────────────────────────────────────
    print("\n--- 5. _hsv_distance (circular hue) ---")
    # Same color → distance 0
    d0 = JerseyOCR._hsv_distance([10, 200, 180], [10, 200, 180])
    assert d0 == 0.0
    print(f"  Identical HSV → {d0:.2f} ✓")

    # Hue wrap-around: 5° vs 175° → distance = min(170, 10) * 2 = 20
    d_wrap = JerseyOCR._hsv_distance([5, 0, 0], [175, 0, 0])
    assert abs(d_wrap - 20.0) < 0.01, f"Expected 20.0, got {d_wrap:.4f}"
    print(f"  Hue 5° vs 175° wrap → {d_wrap:.2f} (circular) ✓")

    # ── 6. set_team_reference + classify_team ─────────────────────────
    print("\n--- 6. set_team_reference / _classify_team ---")
    jerseyocr.set_team_reference("A", [30.0, 200.0, 180.0])   # warm/yellow
    jerseyocr.set_team_reference("B", [100.0, 200.0, 180.0])  # cool/green

    # Create synthetic crops
    yellow_player = np.full((100, 60, 3), [30, 200, 150], dtype=np.uint8)
    green_player  = np.full((100, 60, 3), [100, 200, 150], dtype=np.uint8)
    # Convert from HSV to BGR for the crop
    yellow_bgr = cv2.cvtColor(yellow_player, cv2.COLOR_HSV2BGR)
    green_bgr  = cv2.cvtColor(green_player,  cv2.COLOR_HSV2BGR)

    team_y = jerseyocr._classify_team(yellow_bgr, track_id=1)
    team_g = jerseyocr._classify_team(green_bgr,  track_id=2)
    assert team_y == "A", f"Yellow player should be Team A, got {team_y}"
    assert team_g == "B", f"Green player should be Team B, got {team_g}"
    print(f"  Yellow crop → Team {team_y} ✓")
    print(f"  Green  crop → Team {team_g} ✓")

    # ── 7. process() with None / empty ───────────────────────────────
    print("\n--- 7. process() edge cases ---")
    out = jerseyocr.process(None, [{"track_id": 1, "bbox": [0, 0, 100, 200]}])
    assert out["jersey_results"] == []
    print("  process(None frame) → empty ✓")

    out2 = jerseyocr.process(np.zeros((480, 640, 3), dtype=np.uint8), [])
    assert out2["jersey_results"] == []
    print("  process(empty players) → empty ✓")

    # Out-of-bounds bbox → empty result for that player
    out3 = jerseyocr.process(
        np.zeros((480, 640, 3), dtype=np.uint8),
        [{"track_id": 5, "bbox": [700, 700, 900, 900]}],
    )
    assert len(out3["jersey_results"]) == 1
    assert out3["jersey_results"][0]["jersey_number"] is None
    print("  Out-of-bounds bbox → empty result ✓")

    # ── 8. calibrate_teams ───────────────────────────────────────────
    print("\n--- 8. calibrate_teams ---")
    frame_cal = np.zeros((480, 640, 3), dtype=np.uint8)
    # Two distinct player patches: yellow (team A) and blue (team B)
    frame_cal[50:250, 50:150]  = cv2.cvtColor(
        np.full((200, 100, 3), [30, 220, 200], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )
    frame_cal[50:250, 200:300] = cv2.cvtColor(
        np.full((200, 100, 3), [110, 220, 200], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )
    players_cal = [
        {"track_id": 11, "bbox": [50, 50, 150, 250]},
        {"track_id": 12, "bbox": [200, 50, 300, 250]},
    ]
    jerseyocr2 = JerseyOCR()
    colors = jerseyocr2.calibrate_teams(frame_cal, players_cal)
    assert "A" in colors and "B" in colors
    assert len(colors["A"]) == 3 and len(colors["B"]) == 3
    print(f"  Team A color (HSV): {[round(v,1) for v in colors['A']]} ✓")
    print(f"  Team B color (HSV): {[round(v,1) for v in colors['B']]} ✓")

    # ── 9. Model load ─────────────────────────────────────────────────
    print("\n--- 9. Model load ---")
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / "models" / MODEL_FILENAME
    try:
        ocr_real = JerseyOCR(model_path=str(model_path))
        ocr_real.load_model()
        print(f"  jersey_no.pt loaded ✓")
        print(f"  model.names (first 5): {dict(list(ocr_real.model.names.items())[:5])}")
        print(f"  number_class_idx: {ocr_real._number_class_idx} "
              f"({'fine-tuned' if ocr_real._number_class_idx is not None else 'fallback heuristic'})")
    except (FileNotFoundError, ImportError) as e:
        print(f"  [SKIP] {e}")

    print("\n--- 10. PaddleOCR load ---")
    try:
        jerseyocr3 = JerseyOCR()
        jerseyocr3.load_ocr()
        print("  PaddleOCR loaded ✓")
    except ImportError as e:
        print(f"  [SKIP] {e}")

    print("\n=== All tests passed ===")
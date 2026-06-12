"""
Jersey number detection (2-layer) + team color classification.

Layer 1 — jersey_no.pt:
  Fine-tuned YOLOv8 with 1 class ("number").
  Locates the jersey number bbox within the player crop.
  Falls back to heuristic torso-center crop if model confidence is too low.

Layer 2 — PaddleOCR 3.5.0:
  Reads the digit(s) from the localized crop, filters to 0-99.
  Voting (>= 3 frames) stabilises the assignment before it is registered
  with the tracker.

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

MODELS_DIR   = os.getenv("MODELS_PATH", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_FILENAME = "jersey_no.pt"

VOTE_THRESHOLD        = 2    # OCR reads required to confirm a jersey number (was 3)
MAX_VOTE_HISTORY      = 20   # rolling window of OCR reads per track_id
                             # at OCR_SAMPLE_EVERY=5 this covers ~100 video frames (~4s)
OCR_SAMPLE_EVERY      = 5    # run OCR once every N frames per player (staggered across players)
HIGH_CONF_EARLY_EXIT  = 0.85 # stop trying OCR variants once any one exceeds this score
CONF_THRESHOLD    = 0.40  # jersey_no.pt detection confidence
OCR_HEIGHT_PX     = 128   # resize crop to this height for OCR (was 112; better for 2-digit far)
OCR_MIN_SCORE     = 0.40  # minimum PaddleOCR confidence to accept
BLUR_THRESHOLD    = 20.0  # Laplacian variance below this = too blurry for OCR (was 55; accept motion)


# ---------------------------------------------------------------------------
# Heuristic back-region crop (fallback when no fine-tuned model)
# ---------------------------------------------------------------------------

def _back_crop_heuristic(player_h: int, player_w: int) -> tuple[int, int, int, int]:
    """
    Return (y1, y2, x1, x2) of the jersey number area within the player crop.
    Targets the chest/number region: skip head (top 20%), end at waist (~55%),
    wider width (10%-90%) to catch numbers near jersey edges.
    """
    y1 = int(player_h * 0.20)
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

        self.model = None           # jersey_no.pt (YOLO)
        self._ocr  = None           # PaddleOCR instance
        self._number_class_idx: Optional[int] = None   # None → use heuristic

        # Voting state: track_id → deque of int candidates
        self._votes: dict[int, deque] = {}

        # Team classification
        self._team_colors: dict[str, list[float]] = {}   # "A"/"B" → [H, S, V]
        self._track_team:  dict[int, str]         = {}   # track_id → "A"/"B"

        # Per-track OCR scheduling: frame number of last OCR run per track_id.
        # Default -OCR_SAMPLE_EVERY so the first appearance always triggers OCR.
        self._frame_counter: int = 0
        self._last_ocr_frame: dict[int, int] = {}

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
            self.model = YOLO(str(path))
            self.model.to(target)

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
            self._ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=32,   # allow small crops (jersey from distance)
                text_det_limit_type="min",
                text_det_box_thresh=0.3,      # relaxed: catch faint digits
                text_det_unclip_ratio=2.0,    # wider bbox — prevents digit clipping
                text_rec_score_thresh=0.4,    # pre-filter low-conf results
            )
            logger.info("PaddleOCR 3.5.0 initialised (lang=en, tuned for jersey numbers)")
        except ImportError as exc:
            raise ImportError(
                "paddleocr required. pip install paddleocr paddlepaddle"
            ) from exc

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

        Args:
            frame:           Full BGR frame.
            tracked_players: [{track_id, bbox, ...}] from tracker.py.
            tracker:         PlayerTracker; when provided, update_jersey() is
                             called automatically after a vote is confirmed.
            roster:          Match roster dict keyed by jersey number string.
                             When provided, OCR candidates not present in the
                             roster are silently discarded before entering the
                             vote deque — eliminates impossible readings like
                             numbers that no player actually wears.
        """
        if frame is None or frame.size == 0 or not tracked_players:
            return {"jersey_results": [], "team_colors": self._team_colors}

        self._frame_counter += 1
        h_frame, w_frame = frame.shape[:2]
        results: list[dict] = []

        for player in tracked_players:
            track_id = player["track_id"]

            # ── Per-track OCR timer ───────────────────────────────────────
            # Each track has its own independent timer based on when it last
            # ran OCR. New tracks default to -OCR_SAMPLE_EVERY so first
            # appearance always triggers OCR. Tracks that appear at different
            # times naturally stagger without needing a modulo offset.
            frames_since_ocr = self._frame_counter - self._last_ocr_frame.get(
                track_id, self._frame_counter - OCR_SAMPLE_EVERY
            )
            run_ocr = frames_since_ocr >= OCR_SAMPLE_EVERY

            # Fast path: timer hasn't fired yet AND team already known.
            # Use _vote_status() for a single Counter pass instead of calling
            # _current_confirmed() + _vote_confidence() separately.
            if not run_ocr and track_id in self._track_team:
                confirmed_num, conf = self._vote_status(track_id)
                results.append({
                    "track_id":       track_id,
                    "jersey_number":  confirmed_num,
                    "confidence":     conf,
                    "team":           self._track_team[track_id],
                    "team_color_hsv": self._team_colors.get(self._track_team[track_id], []),
                    "blur_skipped":   False,
                })
                continue

            # ── Crop extraction (needed for team classify or OCR) ─────────
            x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
            x1c = max(0, x1);    y1c = max(0, y1)
            x2c = min(w_frame, x2); y2c = min(h_frame, y2)
            if x2c <= x1c or y2c <= y1c:
                results.append(self._empty_result(track_id))
                continue

            player_crop = frame[y1c:y2c, x1c:x2c]

            # Layer 1 — locate number bbox
            num_bbox = self._detect_number_bbox(player_crop)
            if num_bbox is None:
                results.append(self._empty_result(track_id))
                continue

            ny1, ny2, nx1, nx2 = num_bbox
            number_crop = player_crop[ny1:ny2, nx1:nx2]
            if number_crop.size == 0:
                results.append(self._empty_result(track_id))
                continue

            # Blur guard — skip OCR on motion-blurred crops.
            # White jerseys have naturally lower Laplacian variance (large uniform
            # bright area → few edges), so we relax the threshold proportionally.
            # Mean > 160 = light jersey; use 40% of the normal threshold.
            # Compute gray once and reuse for mean, Laplacian, and debug log —
            # avoids 3× redundant BGR→GRAY + 2× Laplacian for the same crop.
            _crop_gray = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY) \
                if number_crop.ndim == 3 else number_crop
            _effective_blur_thr = (
                BLUR_THRESHOLD * 0.40
                if float(_crop_gray.mean()) > 160
                else BLUR_THRESHOLD
            )
            _blur = _blur_score(_crop_gray)   # gray already computed — no re-conversion
            if _blur < _effective_blur_thr:
                logger.debug(
                    "track %d: blur score=%.1f < %.1f — OCR skipped",
                    track_id, _blur, _effective_blur_thr,
                )
                self._last_ocr_frame[track_id] = self._frame_counter  # reset timer even on blur
                team = self._classify_team(player_crop, track_id)
                team_hsv = self._team_colors.get(team, []) if team else []
                num, conf = self._vote_status(track_id)
                results.append({
                    "track_id":       track_id,
                    "jersey_number":  num,
                    "confidence":     conf,
                    "team":           team,
                    "team_color_hsv": team_hsv,
                    "blur_skipped":   True,
                })
                continue

            # Layer 2 — OCR
            self._last_ocr_frame[track_id] = self._frame_counter
            candidate = self._run_ocr(number_crop)

            # Roster whitelist: discard any candidate not in the roster so it
            # never pollutes the vote deque.  Numbers like "0" or "99" that
            # no player wears are dropped here before any vote is cast.
            if candidate is not None and roster and str(candidate) not in roster:
                logger.debug(
                    "track %d: OCR candidate %d rejected — not in roster",
                    track_id, candidate,
                )
                candidate = None

            # _vote_number appends candidate; _vote_status reads confirmed+confidence
            # in a single Counter pass (avoids the duplicate Counter in _vote_confidence).
            self._vote_number(track_id, candidate)
            confirmed, conf = self._vote_status(track_id)

            if confirmed is not None and tracker is not None:
                tracker.update_jersey(track_id, confirmed)

            # Team classification (fast, uses full player crop)
            team = self._classify_team(player_crop, track_id)
            team_hsv = self._team_colors.get(team, []) if team else []

            results.append({
                "track_id":       track_id,
                "jersey_number":  confirmed,
                "confidence":     conf,
                "team":           team,
                "team_color_hsv": team_hsv,
                "blur_skipped":   False,
            })

        return {"jersey_results": results, "team_colors": self._team_colors}

    # ------------------------------------------------------------------
    # Layer 1 — number localisation
    # ------------------------------------------------------------------

    def _detect_number_bbox(
        self, player_crop: np.ndarray
    ) -> Optional[tuple[int, int, int, int]]:
        """
        Returns (y1, y2, x1, x2) of the number region inside player_crop.

        Uses heuristic torso crop directly — jersey_no.pt is bypassed because
        in practice the two-stage approach (detect then OCR) is less reliable
        than sending a consistent torso region straight to PaddleOCR.
        """
        h, w = player_crop.shape[:2]
        if h < 20 or w < 10:
            return None
        y1, y2, x1, x2 = _back_crop_heuristic(h, w)
        return y1, y2, x1, x2

    # ------------------------------------------------------------------
    # Layer 2 — OCR
    # ------------------------------------------------------------------

    def _run_ocr(self, number_crop: np.ndarray) -> Optional[int]:
        """
        Preprocess crop → PaddleOCR → parse first numeric result in [0-99].
        Tries normal and inverted image (light number on dark jersey and vice versa).
        Returns None if no valid number found or confidence below OCR_MIN_SCORE.
        """
        if self._ocr is None:
            return None
        if number_crop.size == 0:
            return None

        processed = _preprocess_for_ocr(number_crop)
        otsu      = _otsu_variant(number_crop)
        adaptive  = _adaptive_variant(number_crop)

        # Four variants: CLAHE, inverted CLAHE, Otsu binary, adaptive binary.
        # Adaptive handles light jerseys (dark numbers on white) without needing
        # a separate bright-crop branch — THRESH_BINARY_INV normalises both cases.
        # Early exit: once any variant produces a high-confidence result
        # (≥ HIGH_CONF_EARLY_EXIT) there is no benefit in trying the remaining ones.
        candidates: list[tuple[int, float]] = []
        for img in [processed, cv2.bitwise_not(processed), otsu, adaptive]:
            try:
                results = self._ocr.predict(img)
            except Exception as exc:
                logger.debug("PaddleOCR predict error: %s", exc)
                continue

            if not results:
                continue

            # PaddleOCR 3.5.0: predict() returns list[OCRResult]
            # Each OCRResult is dict-like: res['rec_texts'], res['rec_scores']
            for res in results:
                try:
                    texts  = res["rec_texts"]
                    scores = res["rec_scores"]
                except (TypeError, KeyError):
                    continue

                for text, score in zip(texts or [], scores or []):
                    # Filter low-confidence results
                    if score < OCR_MIN_SCORE:
                        logger.debug(
                            "OCR: '%s' rejected (score=%.2f < %.2f)",
                            text, score, OCR_MIN_SCORE,
                        )
                        continue
                    number = self._parse_jersey_number(text)
                    if number is not None:
                        candidates.append((number, score))
                        logger.debug(
                            "OCR: '%s' (score=%.2f) → jersey %d", text, score, number
                        )

            # High-confidence early exit — skip remaining variants.
            if candidates and max(c[1] for c in candidates) >= HIGH_CONF_EARLY_EXIT:
                break

        if not candidates:
            return None

        # Return highest-confidence candidate
        best_number, best_score = max(candidates, key=lambda x: x[1])
        logger.debug("OCR best: jersey %d (score=%.2f)", best_number, best_score)
        return best_number

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
        - Returns confirmed jersey number (mode) once >= VOTE_THRESHOLD votes exist,
          else None.
        """
        if candidate is not None:
            if track_id not in self._votes:
                self._votes[track_id] = deque(maxlen=MAX_VOTE_HISTORY)
            self._votes[track_id].append(candidate)

        return self._current_confirmed(track_id)

    def _current_confirmed(self, track_id: int) -> Optional[int]:
        if track_id not in self._votes or not self._votes[track_id]:
            return None
        counter = Counter(self._votes[track_id])
        most_common, count = counter.most_common(1)[0]
        return most_common if count >= VOTE_THRESHOLD else None

    def _vote_confidence(self, track_id: int) -> float:
        """Fraction of votes for the leading candidate [0.0, 1.0]."""
        if track_id not in self._votes or not self._votes[track_id]:
            return 0.0
        counter = Counter(self._votes[track_id])
        top_count = counter.most_common(1)[0][1]
        return top_count / len(self._votes[track_id])

    def _vote_status(self, track_id: int) -> tuple[Optional[int], float]:
        """Return (confirmed_number, confidence) in a single Counter pass.

        Use this wherever confirmed and confidence are both needed to avoid
        building Counter twice for the same track in the same call.
        """
        if track_id not in self._votes or not self._votes[track_id]:
            return None, 0.0
        votes = self._votes[track_id]
        counter = Counter(votes)
        most_common, count = counter.most_common(1)[0]
        confirmed = most_common if count >= VOTE_THRESHOLD else None
        confidence = count / len(votes)
        return confirmed, confidence

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
        self._frame_counter = 0
        self._last_ocr_frame.clear()
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

    def _classify_team(
        self, player_crop: np.ndarray, track_id: int
    ) -> Optional[str]:
        """
        Assign track to "A" or "B" based on nearest team color reference.
        Returns None if team colors have not been calibrated yet.
        """
        if track_id in self._track_team:
            return self._track_team[track_id]

        if "A" not in self._team_colors or "B" not in self._team_colors:
            return None

        dominant_hsv = self._dominant_hsv(player_crop)
        if dominant_hsv is None:
            return None

        dist_a = self._hsv_distance(dominant_hsv, self._team_colors["A"])
        dist_b = self._hsv_distance(dominant_hsv, self._team_colors["B"])
        team = "A" if dist_a <= dist_b else "B"
        self._track_team[track_id] = team
        return team

    def calibrate_teams(
        self, frame: np.ndarray, tracked_players: list[dict]
    ) -> dict[str, list[float]]:
        """
        Collect torso pixel samples from all visible players, run K-Means (K=2),
        and store the two cluster centers as team color references.

        Call once per game start (or after significant lighting change).
        Returns {"A": [h,s,v], "B": [h,s,v]}.
        """
        if frame is None or not tracked_players:
            return self._team_colors

        h_frame, w_frame = frame.shape[:2]
        all_hsv_samples: list[np.ndarray] = []

        for player in tracked_players:
            x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(w_frame, x2); y2c = min(h_frame, y2)
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue
            # Sample torso region only
            th, tw = crop.shape[:2]
            torso = crop[int(th * 0.15):int(th * 0.65), int(tw * 0.1):int(tw * 0.9)]
            if torso.size == 0:
                continue
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
            # Sub-sample to limit compute
            step = max(1, len(hsv) // 100)
            all_hsv_samples.append(hsv[::step])

        if len(all_hsv_samples) < 2:
            logger.warning("calibrate_teams: need ≥ 2 players, got %d", len(all_hsv_samples))
            return self._team_colors

        samples = np.vstack(all_hsv_samples)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, _, centers = cv2.kmeans(
            samples, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        self._team_colors["A"] = centers[0].tolist()
        self._team_colors["B"] = centers[1].tolist()
        self._track_team.clear()   # re-classify all players

        logger.info(
            "Team colors calibrated: A=%s  B=%s",
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

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_hsv(crop: np.ndarray) -> Optional[list[float]]:
        """Return mean HSV of the torso region of a player crop."""
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return None
        torso = crop[int(h * 0.15):int(h * 0.65), int(w * 0.1):int(w * 0.9)]
        if torso.size == 0:
            return None
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        return hsv.reshape(-1, 3).astype(np.float32).mean(axis=0).tolist()

    @staticmethod
    def _hsv_distance(a: list[float], b: list[float]) -> float:
        """Hue-aware HSV distance. Hue is circular [0, 180]."""
        dh = abs(a[0] - b[0])
        dh = min(dh, 180.0 - dh)   # circular
        ds = abs(a[1] - b[1])
        dv = abs(a[2] - b[2])
        return float(dh * 2 + ds + dv * 0.5)

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
    """Create, load model, and load OCR in one call."""
    model_file = os.path.join(models_path or MODELS_DIR, MODEL_FILENAME)
    ocr = JerseyOCR(model_path=model_file, device=device)
    ocr.load_model()
    ocr.load_ocr()
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

    # 2 votes for 7 → not confirmed
    jerseyocr._vote_number(tid, 7)
    jerseyocr._vote_number(tid, 7)
    assert jerseyocr._vote_number(tid, 3) is None, "Should not confirm with 2 votes"
    print("  2 votes for 7, 1 for 3 → not confirmed ✓")

    # 3rd vote for 7 → confirmed
    result = jerseyocr._vote_number(tid, 7)
    assert result == 7, f"Expected 7, got {result}"
    print(f"  3 votes for 7 → confirmed: {result} ✓")

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
    assert processed.shape[0] == OCR_HEIGHT_PX, f"Expected height={OCR_HEIGHT_PX}"
    assert processed.shape[2] == 3, "Expected 3 channels"
    print(f"  Input (80×60) → output {processed.shape[:2]} (height={OCR_HEIGHT_PX}px) ✓")

    tiny = np.zeros((2, 1, 3), dtype=np.uint8)
    result_tiny = _preprocess_for_ocr(tiny)
    assert result_tiny.shape[0] == OCR_HEIGHT_PX
    print("  Tiny 2×1 crop → safe output ✓")

    # ── 4. _back_crop_heuristic ──────────────────────────────────────
    print("\n--- 4. _back_crop_heuristic ---")
    y1, y2, x1, x2 = _back_crop_heuristic(200, 100)
    assert y1 < y2 and x1 < x2
    assert y1 == 30 and y2 == 130  # 15% of 200 and 65% of 200
    assert x1 == 20 and x2 == 80   # 20% and 80% of 100 (center 60%)
    print(f"  200×100 player → number bbox: y=[{y1},{y2}] x=[{x1},{x2}] ✓")

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
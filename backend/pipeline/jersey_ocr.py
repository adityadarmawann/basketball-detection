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
DIGIT_CONF_THRESHOLD   = 0.12   # minimum per-digit detection confidence. LOW on purpose:
                                 # the digit model detects only ONE digit in ~60% of crops
                                 # at 0.20, so 2-digit jerseys (11,16,88) read as a single
                                 # ambiguous digit. Measured on real frames: 0.20→40% two-box
                                 # reads, 0.12→53%, 0.08→62%. Confidence-weighted voting
                                 # (see _resolve_votes) down-weights the weak reads this lets
                                 # in, so recall rises without confirming noise. 0.12 balances
                                 # recall vs noise; 0.08 recovers more but adds spurious reads.

VOTE_THRESHOLD        = 2    # OCR reads required to confirm a jersey number
                             # at ocr_interval=2 (0.13s), 2 votes ≈ 0.27s — faster confirmation;
                             # roster whitelist + MAX_VOTE_HISTORY absorb stray misreads
# Confidence-upgrade flush thresholds. Relaxed from 0.60/0.25 → 0.50/0.10 after a
# ground-truth sweep (19 hand-labelled tracks, real process() replay): 0.50/0.10
# lifted mid-track number accuracy 57.7%→61.2% and final accuracy 57.9%→63.2% while
# flip-rate barely moved (3.84→3.95/track). Env-overridable. NOTE: lowering DISAGREE_N
# to 2 was ALSO tested and REJECTED — it thrashed (acc 57.7%→52.9%, flips +47%).
CONF_UPGRADE_MIN      = float(os.getenv("CONF_UPGRADE_MIN",   "0.50"))  # digit-conf floor to flush
CONF_UPGRADE_DELTA    = float(os.getenv("CONF_UPGRADE_DELTA", "0.10"))  # margin over prior best to flush
MAX_VOTE_HISTORY      = 20   # rolling window — 20 reads per player (OCR_SAMPLE_EVERY=1)
                             # smaller window lets wrong reads be forgotten faster
TRANSFER_MIN_LONG_VOTES = 3  # 2-digit candidate needs this many reads before 1-digit votes
                             # are reinterpreted as partial reads; raised to 3 so "10→1"
                             # partial reads accumulate enough evidence before reassigning
OCR_SAMPLE_EVERY      = 1    # unconfirmed players: OCR on every process() call
                             # roster whitelist removes false positives; voting absorbs noise
CONFIRMED_OCR_EVERY   = 3    # confirmed players: re-OCR every N calls (Fix #4). Kept low
                             # so a WRONG confirmation (ID swap / early misread) is re-read
                             # and corrected fast — the disagreement flush below needs fresh
                             # reads. Was 6; 3 halves the stale-lock lag for a little more GPU.
# Stale-lock correction: if the last DISAGREE_N reads for a track all agree on a
# number DIFFERENT from its confirmed one, the underlying player likely changed
# (ID swap) or the first confirm was wrong — flush the stale votes so the new
# number takes over in ~1-2 frames instead of waiting out MAX_VOTE_HISTORY inertia.
DISAGREE_WINDOW       = 5    # keep the last N reads per track for disagreement checks
DISAGREE_N            = 3    # consecutive agreeing reads that override a stale confirmation

# ── Team-color SOFT lock / split (workflow fix C) ────────────────────────────
# Team is the roster key, so a team flip mislabels the name. Once a track has
# strong consistent color evidence its team LOCKS (a pan/occlusion can't flip it).
# The lock breaks only on TEAM_SPLIT_DISAGREE consecutive CONFIDENT disagreements
# = the physical player under this track changed (ID drift) -> split the whole
# identity. This gives a SECOND, color-based split trigger complementing the
# number-based disagreement flush (covers interleaved-read cases the number flush
# misses). Soft (not hard) + split-sensitive so an early mislock self-heals.
TEAM_LOCK_MIN_VOTES   = 5     # votes before a track's team may lock
TEAM_LOCK_MAJORITY    = 0.7   # majority fraction required to lock
TEAM_SPLIT_DISAGREE   = 4     # consecutive CONFIDENT disagreements that split identity
TEAM_CONFIDENT_MARGIN = 25.0  # min |dist_a - dist_b| for a frame's color vote to count

# ── NOT-A-TEAM rejection (referees, coaches — anyone wearing neither team colour) ──
# The A→B axis forces a BINARY decision: a referee's dark shirt is projected onto the
# line and lands on one side, so referees always get a team. Reject them by DISTANCE
# to the nearest anchor, measured on the RAW chest median — NOT on _dominant_lab,
# which is anchor-filtered and therefore drags ANY colour toward a team (that is why
# an earlier residual test failed).
# Per crop the distributions overlap (players p90=3.60 vs referees p10=3.45), but the
# MEDIAN over a few crops separates cleanly. Measured (84 hand-labelled players vs 274
# detector-class referee crops), threshold 3.55:
#     1 crop  → 83% players kept / 88% referees rejected   (overlaps)
#     5 crops → 97.2% / 97.4%      10 crops → 99.6% / 99.7%      20 crops → 100% / 100%
# DEFAULT OFF — the absolute threshold below is ASYMMETRIC in practice: the uploaded
# maroon anchor is vivid while the rendered maroon on court is washed out, so team B
# players sit systematically FARTHER from their anchor than team A players
# (measured p50: A=2.54 vs B=3.19) and a single global threshold rejects ~20% of B
# and ~5% of A. Worse, my calibration set was stratified by a*, which over-sampled
# VIVID maroon and hid this. Enabling it in a real run collapsed team B to ~0.
# A correct version must normalise per team (or use an adaptive/outlier threshold),
# not one absolute distance. Kept behind a flag until that is measured properly.
TEAM_REJECT_ENABLED     = os.getenv("TEAM_REJECT_ENABLED", "0") == "1"
TEAM_REJECT_DIST        = float(os.getenv("TEAM_REJECT_DIST", "3.55"))
TEAM_REJECT_MIN_SAMPLES = int(os.getenv("TEAM_REJECT_MIN_SAMPLES", "5"))
# Team votes are WEIGHTED (mirrors the jersey-number vote, which is already weighted by
# read_conf × blur_q). weight = decision-margin × jersey-likeness, so a blurred or
# half-occluded crop no longer counts the same as a clean one.
TEAM_VOTE_WEIGHT_FLOOR  = float(os.getenv("TEAM_VOTE_WEIGHT_FLOOR", "0.15"))
# Weight of the brightness (V) term in team-colour distance. Raised 0.25->0.5:
# measured on real UNESA(white,bright ~V220) vs UBAYA(maroon,dark ~V130) frames,
# brightness is a strong camera-robust discriminator that 0.25 under-used, so a
# warm-lit white player (elevated S) mis-flipped to red. 0.5 fixed every measured
# borderline (6/6) with no regressions. Env-gated for A/B testing.
TEAM_VALUE_WEIGHT     = float(os.getenv("TEAM_VALUE_WEIGHT", "0.25"))  # legacy metric only; 0.5 hurt red (measured), reverted

# ── Team-colour matching: UPLOAD-AXIS PROJECTION (default) ───────────────────
# Classify by projecting a player's chest colour onto the UPLOAD-defined A→B axis
# in LAB, then thresholding. Camera-invariant (LAB separates luminance from colour)
# and GENERAL to ANY team colours: the axis auto-selects whichever channels separate
# the two uploaded colours (red/white, blue/black, green/yellow, …). Orientation is
# fixed by the axis (A at t=0, B at t>0) so A/B can NEVER globally flip. The boundary
# sits partway along the A→B distance because jerseys render MUTED vs the vivid upload
# (rendered colours compress toward the middle). "distance" = legacy HSV metric.
# FRAC tuned on 84 HAND-LABELLED crops (UNESA white vs UBAYA muted-maroon, real run):
# 0.35 = 91% balanced (A 95% / B 87%), best of {0.25:75%, 0.30:86%, 0.40:87%} and
# ties/beats legacy distance (86%). NOTE: single match, one LAB scale — re-validate on
# the next match (different colours) before trusting the exact value.
TEAM_MATCH_MODE       = os.getenv("TEAM_MATCH_MODE", "axis")             # "axis" | "distance"
TEAM_AXIS_FRAC        = float(os.getenv("TEAM_AXIS_FRAC", "0.35"))       # threshold along A→B
TEAM_AXIS_CONF_FRAC   = float(os.getenv("TEAM_AXIS_CONF_FRAC", "0.15"))  # confident-vote margin
LAB_SCALE_MIN_SAMPLES = int(os.getenv("LAB_SCALE_MIN_SAMPLES", "50"))    # crops before scale freezes
# Per-channel LAB normalisation for the team axis. FIXED — deliberately NOT measured
# from the video. Measuring it was a LOTTERY: the std of the first 50 chest medians
# mixes sensor noise with BETWEEN-TEAM variation, so if that window happens to be
# dominated by one team the a* spread collapses and the axis stops separating the
# teams. Measured across 20 calibration windows on a real match (79 hand-labelled
# crops):
#     measured-then-frozen : 82-97% balanced (spread 16), team B as low as 64%
#     rolling (whole video):    93% flat,                  team B 86%
#     FIXED (this)         :    97% flat (spread 0),       team B 97%
# The scale describes the COLOUR SPACE (L varies far more than a*/b* under stadium
# lighting), not the teams — the teams enter through the uploaded anchors. So a
# constant is both more robust AND still general to any pair of jersey colours.
_LAB_SCALE_DEFAULT    = (40.0, 8.0, 8.0)
LAB_SCALE_MEASURE     = os.getenv("LAB_SCALE_MEASURE", "0") == "1"   # opt-in, not advised
# Jersey-colour extraction: keep the chest pixels CLOSEST to either team anchor
# (drops skin / court / white-number / an overlapping neighbour's jersey) before
# the median. On 84 hand-labelled crops this lifted the RAW classifier 91% → 98%
# balanced AND made it scale-robust (median-only swung 79-95% across LAB scales;
# anchor-filtered stayed 96-98%). Fraction of chest pixels kept:
TEAM_JERSEY_PIXEL_FRAC = float(os.getenv("TEAM_JERSEY_PIXEL_FRAC", "0.60"))
HIGH_CONF_EARLY_EXIT  = 0.85 # stop trying OCR variants once any one exceeds this score
DIGIT_DEDUP_OVERLAP = 0.5  # x-overlap (fraction of narrower box) above which two
                           # digit detections are treated as the SAME physical digit
                           # and the lower-confidence one is dropped (Fix #5)
CONF_THRESHOLD    = 0.30  # jersey_no.pt detection confidence (lowered from 0.40 — distant/small numbers)
OCR_HEIGHT_PX     = 160   # larger resize → OCR reads small/distant jersey numbers better
OCR_MIN_SCORE     = 0.35  # slightly more permissive — voting absorbs noise
BLUR_THRESHOLD    = 0.0   # legacy hard-skip disabled — replaced by soft blur
                          # down-weighting in voting (Fix #3), which keeps every frame

# ── Fix #3: confidence-weighted voting ───────────────────────────────────────
# Votes are stored as (candidate, weight) where weight = digit-model confidence
# scaled by a blur-quality factor. A sharp, high-confidence read outvotes several
# blurry/low-confidence ones instead of every read counting equally. This kills
# the "two 0.15-confidence garbage reads confirm a wrong number" failure while
# still letting genuine reads accumulate across a fast play.
MIN_VOTE_WEIGHT    = 0.05  # floor per weighted vote (keeps a read from vanishing)
MIN_CONFIRM_WEIGHT = 0.6   # leading candidate's summed weight required to confirm
                           # (with VOTE_THRESHOLD reads also required); ~two 0.3 reads
# Blur → vote-weight mapping (Laplacian-variance sharpness; see _blur_score):
BLUR_SHARP_FULL    = 200.0 # >= this = fully sharp, weight 1.0
BLUR_SHARP_MIN     = 40.0  # <= this = heavily motion-blurred, weight BLUR_MIN_WEIGHT
BLUR_MIN_WEIGHT    = 0.3    # a very blurry read still votes, but at 0.3× weight


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


def _blur_quality(blur: float) -> float:
    """Map Laplacian-variance sharpness to a [BLUR_MIN_WEIGHT, 1.0] vote weight.

    Sharp reads (>= BLUR_SHARP_FULL) count fully; motion-smeared reads
    (<= BLUR_SHARP_MIN) still vote but at reduced weight — never dropped (Fix #3).
    """
    if blur >= BLUR_SHARP_FULL:
        return 1.0
    if blur <= BLUR_SHARP_MIN:
        return BLUR_MIN_WEIGHT
    frac = (blur - BLUR_SHARP_MIN) / (BLUR_SHARP_FULL - BLUR_SHARP_MIN)
    return BLUR_MIN_WEIGHT + frac * (1.0 - BLUR_MIN_WEIGHT)


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
        # Recent raw reads per track (post-whitelist) for the stale-lock flush.
        self._recent_reads: dict[int, deque] = {}
        # Team-color soft lock (fix C): track_id → locked team, and consecutive
        # confident-disagreement counter used to detect a player change under the id.
        self._team_lock:     dict[int, str] = {}
        self._team_disagree: dict[int, int] = {}
        # Rolling RAW colour distance to the nearest team anchor, per track. A referee /
        # coach sits far from BOTH anchors; the median over a few crops rejects them
        # (see TEAM_REJECT_DIST). Also feeds the team-vote weight.
        self._track_color_dist: dict[int, deque] = {}
        # Tracks judged NOT-A-TEAM (referee/coach). Distinct from "no vote this frame":
        # video_processor must PURGE its sticky team for these, not hold the last one.
        self._team_rejected: set[int] = set()
        # Upload-axis team matching (default): LAB anchors + a per-channel LAB scale
        # measured once from early crops, and a fingerprinted axis cache.
        self._team_colors_lab: dict[str, list[float]] = {}   # "A"/"B" → [L, a, b]
        self._lab_scale = (None if LAB_SCALE_MEASURE
                           else np.array(_LAB_SCALE_DEFAULT, dtype=np.float64))
        self._lab_samples: list = []      # LAB medians collected until scale freezes
        self._team_axis = None            # cache: (fingerprint, As, axis, scale, tB, thr, cmargin)

        # Uniqueness guard: (team, jersey_number) → canonical track_id.
        # When two live tracks both read the same (team, number), only the first
        # claimant keeps the confirmed identity; duplicates are suppressed until
        # the owner track disappears (then the duplicate can claim ownership).
        self._team_number_owner: dict[tuple, int] = {}

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

        # Set of track IDs visible this frame — used by identity uniqueness guard.
        current_track_ids: set[int] = {
            p["track_id"] for p in tracked_players if p.get("track_id") is not None
        }

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

            # Per-track OCR schedule (Fix #4): a player whose number is NOT yet
            # confirmed is OCR'd on every call to grab the brief window where the
            # number is readable under fast camera motion; an already-confirmed
            # player is only re-read every CONFIRMED_OCR_EVERY calls (enough to
            # catch a zoom-in confidence upgrade) so GPU isn't spent re-reading
            # known numbers. Default seeds a first-appearance OCR regardless.
            frames_since_ocr = self._frame_counter - self._last_ocr_frame.get(
                track_id, self._frame_counter - CONFIRMED_OCR_EVERY
            )
            _is_confirmed = (
                self._current_confirmed(track_id) is not None
                or (tracker is not None
                    and tracker.get_jersey_number(track_id) is not None)
            )
            _interval = CONFIRMED_OCR_EVERY if _is_confirmed else OCR_SAMPLE_EVERY
            run_ocr = frames_since_ocr >= _interval

            # Fast path A: timer hasn't fired AND team already known
            if not run_ocr and track_id in self._track_team:
                confirmed_num, conf = self._vote_status(track_id)
                cached_team = self._track_team[track_id]
                # Uniqueness guard: suppress if another live track owns this identity
                if confirmed_num is not None and cached_team:
                    key   = (cached_team, confirmed_num)
                    owner = self._team_number_owner.get(key)
                    if owner is not None and owner != track_id and owner in current_track_ids:
                        confirmed_num = None
                        conf          = 0.0
                    else:
                        self._team_number_owner[key] = track_id
                results[i] = {
                    "track_id":       track_id,
                    "jersey_number":  confirmed_num,
                    "confidence":     conf,
                    "team":           cached_team,
                    "team_color_hsv": self._team_colors.get(cached_team, []),
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

            # Blur → soft vote-weight (Fix #3). We deliberately do NOT hard-skip
            # blurry crops — under fast camera motion that discards most readable
            # frames. Instead sharpness scales the vote weight (see Pass 2) so a
            # crisp read dominates a motion-smeared one while every frame still votes.
            _crop_gray = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY) \
                if number_crop.ndim == 3 else number_crop
            _blur_q = _blur_quality(_blur_score(_crop_gray))

            # Schedule for batch digit OCR in Pass 2
            pending[i] = (player, player_crop, number_crop, _blur_q)

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
                _, player_crop, _, _ = pending[slot_i]
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
                player, player_crop, _, _blur_q = pending[slot_i]
                track_id  = player["track_id"]

                # Unpack (number, mean_conf) from digit model; default to no read.
                raw_result = tight_cands[j]
                candidate  = None
                read_conf  = 0.0
                if raw_result is not None:
                    candidate, read_conf = raw_result

                # Roster whitelist: drop candidates not worn by any player.
                # When team is already known from color rolling-vote, only accept
                # numbers on that team's side of the roster — this blocks
                # cross-team number ambiguity (e.g. both teams wearing #7).
                if candidate is not None and roster:
                    js = str(candidate)
                    known_team = self._track_team.get(track_id)
                    if known_team:
                        team_key  = f"{js}_{known_team}"
                        in_roster = (team_key in roster or
                                     (js in roster and
                                      f"{js}_A" not in roster and
                                      f"{js}_B" not in roster))
                    else:
                        in_roster = (js in roster or
                                     f"{js}_A" in roster or
                                     f"{js}_B" in roster)
                    if not in_roster:
                        logger.debug(
                            "track %d: OCR candidate %d rejected — not in roster"
                            " (team=%s)",
                            track_id, candidate, known_team or "unknown",
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

                # ── Stale-lock correction ────────────────────────────────────
                # If the last DISAGREE_N reads all agree on a number different from
                # the currently-confirmed one, drop the stale votes so the new
                # number takes over immediately (ID swap / early misread recovery)
                # instead of waiting out the 20-deep deque inertia.
                if candidate is not None:
                    rq = self._recent_reads.setdefault(
                        track_id, deque(maxlen=DISAGREE_WINDOW))
                    rq.append(candidate)
                    if len(rq) >= DISAGREE_N:
                        last = list(rq)[-DISAGREE_N:]
                        cur  = self._current_confirmed(track_id)
                        if (cur is not None and last[0] != cur
                                and all(x == last[0] for x in last)):
                            self._votes.pop(track_id, None)
                            self._best_read_conf.pop(track_id, None)
                            logger.info(
                                "track %d: %d consecutive reads of %s disagree with "
                                "confirmed %s — flushing stale lock",
                                track_id, DISAGREE_N, last[0], cur,
                            )

                self._last_ocr_frame[track_id] = self._frame_counter
                # Vote weight = digit-model confidence scaled by blur quality (Fix #3).
                self._vote_number(track_id, candidate, read_conf * _blur_q)
                confirmed, conf = self._vote_status(track_id)

                logger.info(
                    "track %d: candidate=%s  conf=%.2f blur_q=%.2f  votes=%s  confirmed=%s",
                    track_id, candidate, read_conf, _blur_q,
                    dict(Counter(c for c, _w in self._votes.get(track_id, []))),
                    confirmed,
                )

                # Classify team first so uniqueness check can use (team, number) key.
                team     = self._classify_team(player_crop, track_id)
                team_hsv = self._team_colors.get(team, []) if team else []

                # Uniqueness guard: suppress identity if another live track is the
                # canonical owner of (team, jersey_number). Votes still accumulate
                # so this track can take over instantly when the owner disappears.
                if confirmed is not None and team:
                    key   = (team, confirmed)
                    owner = self._team_number_owner.get(key)
                    if owner is not None and owner != track_id and owner in current_track_ids:
                        logger.debug(
                            "track %d: identity (%s, #%d) already owned by active "
                            "track %d — suppressing duplicate",
                            track_id, team, confirmed, owner,
                        )
                        confirmed = None
                        conf      = 0.0
                    else:
                        self._team_number_owner[key] = track_id

                if confirmed is not None and tracker is not None:
                    tracker.update_jersey(track_id, confirmed)

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

        # Upscale small crops (mirrors _run_yolo_digit_ocr's resize step).
        # Also apply CLAHE on the LAB luminance channel — improves digit contrast
        # for dark jerseys and distant players without discarding color information.
        processed: list = []
        for crop in crops:
            if crop is None or crop.size == 0:
                processed.append(None)
                continue
            h, w = crop.shape[:2]
            if h < 8 or w < 4:
                processed.append(None)
                continue
            try:
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                tile_h = max(1, min(4, h // 2))
                tile_w = max(1, min(4, w // 2))
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile_w, tile_h))
                l = clahe.apply(l)
                crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            except Exception:
                pass  # fall through with raw crop if LAB conversion fails
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
        digit_boxes: list[tuple[float, float, str, float]] = []
        for box in det_result.boxes:
            cls        = int(box.cls[0])
            raw_name   = self._digit_model.names.get(cls, str(cls))
            digit_char = (raw_name
                          if (len(raw_name) == 1 and raw_name.isdigit())
                          else str(cls))
            x1   = float(box.xyxy[0][0])
            x2   = float(box.xyxy[0][2])
            conf = float(box.conf[0])
            digit_boxes.append((x1, x2, digit_char, conf))
            logger.debug("digit: '%s' x=%.0f-%.0f conf=%.2f", digit_char, x1, x2, conf)

        if not digit_boxes:
            return None

        # ── Fix #5: drop duplicate detections on the SAME physical digit ─────────
        # The detector sometimes fires two overlapping boxes for one digit; left
        # unchecked they concatenate to e.g. "11" from a single "1". Greedy NMS on
        # the x-axis keeps the higher-confidence box when two overlap heavily.
        digit_boxes.sort(key=lambda b: -b[3])   # highest confidence first
        kept: list[tuple[float, float, str, float]] = []
        for x1b, x2b, ch, cf in digit_boxes:
            is_dup = False
            for kx1, kx2, _kc, _kf in kept:
                ox = max(0.0, min(x2b, kx2) - max(x1b, kx1))
                narrower = max(1.0, min(x2b - x1b, kx2 - kx1))
                if ox / narrower >= DIGIT_DEDUP_OVERLAP:
                    is_dup = True
                    break
            if not is_dup:
                kept.append((x1b, x2b, ch, cf))

        kept = kept[:2]                         # jersey numbers are 0–99 (≤ 2 digits)
        kept.sort(key=lambda b: b[0])           # left-to-right reading order
        number_str  = "".join(b[2] for b in kept)
        mean_conf   = sum(b[3] for b in kept) / len(kept)

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

    def _vote_number(
        self, track_id: int, candidate: Optional[int], weight: float = 1.0
    ) -> Optional[int]:
        """
        Accumulate weighted candidate votes per track_id (Fix #3).

        - Each vote is (candidate, weight); weight = digit-model confidence scaled
          by blur quality. Sharp/high-confidence reads dominate blurry/weak ones.
        - None candidates are ignored (not added to history), preserving votes.
        - Returns confirmed jersey number (via prefix-aware resolution) once
          >= VOTE_THRESHOLD reads AND >= MIN_CONFIRM_WEIGHT summed weight exist.
        """
        if candidate is not None:
            if track_id not in self._votes:
                self._votes[track_id] = deque(maxlen=MAX_VOTE_HISTORY)
            self._votes[track_id].append((candidate, max(float(weight), MIN_VOTE_WEIGHT)))

        return self._current_confirmed(track_id)

    def _resolve_votes(self, votes) -> tuple[Optional[int], float]:
        """
        Confidence-weighted, prefix/suffix-aware vote resolution (Fix #3).

        `votes` is an iterable of (candidate, weight). Per candidate we track a
        read COUNT (evidence it is real) and a summed WEIGHT (how sharp/confident
        those reads were).

        OCR often produces partial digit reads under fast camera motion:
          "10" → "1"  (right digit clipped when camera pans left)
          "23" → "3"  (left digit clipped when camera pans right)
        Rule: if a 1-digit candidate AND a 2-digit candidate share a digit in the
        same position (prefix OR suffix), AND the 2-digit candidate has
        >= TRANSFER_MIN_LONG_VOTES reads, the 1-digit reads are reinterpreted as
        partial reads and their count+weight transferred to the 2-digit number.
        True single-digit players (#1–#9) are protected when no such 2-digit
        candidate has enough evidence.

        The leading candidate has the highest summed weight; it is confirmed only
        when it also has >= VOTE_THRESHOLD reads AND weight >= MIN_CONFIRM_WEIGHT.
        """
        count:  dict[int, int]   = {}
        weight: dict[int, float] = {}
        for item in votes:
            # Accept both (candidate, weight) tuples and bare-int votes so legacy
            # callers / tests that push plain numbers keep working.
            cand, w = item if isinstance(item, tuple) else (item, 1.0)
            if cand is None:
                continue
            count[cand]  = count.get(cand, 0) + 1
            weight[cand] = weight.get(cand, 0.0) + w
        if not count:
            return None, 0.0

        for short_num in [k for k in list(count) if 0 <= k <= 9]:
            short_str = str(short_num)
            # 2-digit candidates sharing this digit (prefix OR suffix) with enough evidence.
            matching_long = [
                (long_num, count[long_num])
                for long_num in count
                if 10 <= long_num <= 99
                and (str(long_num)[0] == short_str or str(long_num)[-1] == short_str)
                and count[long_num] >= TRANSFER_MIN_LONG_VOTES
            ]
            if not matching_long:
                continue  # keep the 1-digit reads as-is

            total_long = sum(c for _, c in matching_long)
            for long_num, long_count in matching_long:
                frac = long_count / total_long
                count[long_num]  += round(count[short_num] * frac)
                weight[long_num] += weight[short_num] * frac
            count[short_num]  = 0   # 1-digit reads fully absorbed
            weight[short_num] = 0.0

        valid = {k: weight[k] for k in weight if weight[k] > 0 and count.get(k, 0) > 0}
        if not valid:
            return None, 0.0

        best       = max(valid, key=valid.get)
        total_w    = sum(valid.values())
        confidence = valid[best] / total_w if total_w > 0 else 0.0
        confirmed  = (best if count[best] >= VOTE_THRESHOLD
                      and valid[best] >= MIN_CONFIRM_WEIGHT else None)
        return confirmed, confidence

    def _current_confirmed(self, track_id: int) -> Optional[int]:
        if track_id not in self._votes or not self._votes[track_id]:
            return None
        confirmed, _ = self._resolve_votes(self._votes[track_id])
        return confirmed

    def _vote_confidence(self, track_id: int) -> float:
        """Weight share of the leading candidate [0.0, 1.0]."""
        if track_id not in self._votes or not self._votes[track_id]:
            return 0.0
        _, confidence = self._resolve_votes(self._votes[track_id])
        return confidence

    def _vote_status(self, track_id: int) -> tuple[Optional[int], float]:
        """Return (confirmed_number, confidence) using weighted vote resolution."""
        if track_id not in self._votes or not self._votes[track_id]:
            return None, 0.0
        return self._resolve_votes(self._votes[track_id])

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
        self._track_color_dist.clear()
        self._team_rejected.clear()
        self._frame_counter = 0
        self._last_ocr_frame.clear()
        self._best_read_conf.clear()
        self._recent_reads.clear()
        self._team_number_owner.clear()
        self._team_lock.clear()
        self._team_disagree.clear()
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

        Soft lock (fix C): once evidence is strong the team LOCKS so camera pan /
        occlusion can't flip it; the lock breaks only on TEAM_SPLIT_DISAGREE
        consecutive CONFIDENT disagreements (= the player under this id changed).

        Vote/confidence come from _team_vote_confident (upload-axis projection by
        default — camera-invariant and general to any team colours).
        """
        # ── (1) NOT-A-TEAM gate: referees/coaches wear neither team's colour ────
        # Accumulate the RAW distance to the nearest anchor. One crop is ambiguous
        # (the two distributions overlap), but the MEDIAN over a few crops separates
        # players from referees almost perfectly — so we only reject once we have
        # TEAM_REJECT_MIN_SAMPLES of evidence, never on a single frame.
        dist = self._anchor_distance(player_crop) if TEAM_REJECT_ENABLED else None
        if dist is not None:
            dq = self._track_color_dist.setdefault(
                track_id, deque(maxlen=self._TEAM_VOTE_HISTORY))
            dq.append(dist)
            if (len(dq) >= TEAM_REJECT_MIN_SAMPLES
                    and float(np.median(np.asarray(dq))) > TEAM_REJECT_DIST):
                # Neither team → emit NO team, so the UI shows a neutral box instead
                # of forcing a referee into A or B.
                self._track_team.pop(track_id, None)
                self._team_lock.pop(track_id, None)
                self._team_rejected.add(track_id)
                return None
            self._team_rejected.discard(track_id)

        vote, confident, conf = self._team_vote_confident(player_crop)
        if vote is None:
            # no calibration / crop failed — hold last known (prefer the lock)
            return self._team_lock.get(track_id) or self._track_team.get(track_id)

        locked = self._team_lock.get(track_id)
        if locked is not None:
            # Hold the lock; a sustained CONFIDENT disagreement = player changed.
            if vote != locked and confident:
                self._team_disagree[track_id] = self._team_disagree.get(track_id, 0) + 1
            else:
                self._team_disagree[track_id] = 0
            if self._team_disagree[track_id] >= TEAM_SPLIT_DISAGREE:
                self._split_track_identity(track_id, vote)   # re-locks to `vote`
                result = vote
            else:
                result = locked
            self._track_team[track_id] = result
            return result

        # ── (2) WEIGHTED rolling votes (mirrors the weighted jersey-number vote) ──
        # weight = decision margin × jersey-likeness. A borderline or contaminated
        # crop no longer counts as much as a clean, decisive one.
        likeness = 1.0
        if dist is not None and TEAM_REJECT_DIST > 1e-6:
            likeness = float(np.clip(1.0 - dist / TEAM_REJECT_DIST, 0.0, 1.0))
        weight = max(TEAM_VOTE_WEIGHT_FLOOR, conf * likeness)

        if track_id not in self._track_team_votes:
            self._track_team_votes[track_id] = deque(maxlen=self._TEAM_VOTE_HISTORY)
        self._track_team_votes[track_id].append((vote, weight))
        votes   = self._track_team_votes[track_id]
        w_a     = sum(w for v, w in votes if v == "A")
        w_b     = sum(w for v, w in votes if v == "B")
        total_w = w_a + w_b
        result  = "A" if w_a >= w_b else "B"
        if (len(votes) >= TEAM_LOCK_MIN_VOTES
                and max(w_a, w_b) / max(total_w, 1e-6) >= TEAM_LOCK_MAJORITY):
            self._team_lock[track_id] = result
            self._team_disagree[track_id] = 0

        # Update current-best cache (used by Fast Path A and inherit_team)
        self._track_team[track_id] = result
        return result

    def _split_track_identity(self, track_id: int, new_team: str) -> None:
        """The physical player under track_id changed (confirmed by a sustained,
        confident team-color disagreement). Drop the accumulated NUMBER identity so
        the new player is read fresh, release any (team,number) ownership, and
        re-anchor the team lock to the new team."""
        self._votes.pop(track_id, None)
        self._best_read_conf.pop(track_id, None)
        self._recent_reads.pop(track_id, None)
        for key in [k for k, v in self._team_number_owner.items() if v == track_id]:
            self._team_number_owner.pop(key, None)
        self._team_lock[track_id]     = new_team
        self._team_disagree[track_id] = 0
        self._track_team_votes[track_id] = deque([(new_team, 1.0)], maxlen=self._TEAM_VOTE_HISTORY)
        self._track_color_dist.pop(track_id, None)   # new person → re-judge not-a-team from scratch
        self._team_rejected.discard(track_id)
        logger.info("track %d: identity SPLIT — team→%s (player changed under id)",
                    track_id, new_team)

    @staticmethod
    def _hsv_to_lab(hsv: list) -> list:
        """Convert an OpenCV-HSV [H0-180,S0-255,V0-255] colour to LAB [L,a,b]."""
        px = np.uint8([[[int(hsv[0]) % 180,
                         int(np.clip(hsv[1], 0, 255)),
                         int(np.clip(hsv[2], 0, 255))]]])
        lab = cv2.cvtColor(cv2.cvtColor(px, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2LAB)[0, 0]
        return [float(lab[0]), float(lab[1]), float(lab[2])]

    @staticmethod
    def _chest_median_lab(crop: np.ndarray) -> Optional[list]:
        """Plain (unfiltered) median LAB of the chest strip.

        Used ONLY to measure the per-channel LAB scale. It must stay unfiltered:
        the anchor-filtered _dominant_lab polarises its outputs toward the anchors,
        so its spread reflects team separation, not sensor/lighting noise."""
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return None
        chest = crop[int(h * 0.20):int(h * 0.50), int(w * 0.15):int(w * 0.85)]
        if chest.size == 0:
            return None
        lab = cv2.cvtColor(chest, cv2.COLOR_BGR2LAB)
        return np.median(lab.reshape(-1, 3).astype(np.float32), axis=0).tolist()

    def _anchor_distance(self, crop: np.ndarray) -> Optional[float]:
        """Distance from the RAW chest median to the NEAREST team anchor (scaled LAB).

        Deliberately uses the RAW median, not _dominant_lab: _dominant_lab keeps only
        the pixels closest to an anchor, so it drags ANY shirt — including a referee's
        black one — toward a team colour and destroys exactly the signal we need here.
        Large distance ⇒ this person wears neither team's colour."""
        lab = self._chest_median_lab(crop)
        if lab is None:
            return None
        a = self._team_colors_lab.get("A")
        b = self._team_colors_lab.get("B")
        if a is None or b is None:
            return None
        scale = (self._lab_scale if self._lab_scale is not None
                 else np.array(_LAB_SCALE_DEFAULT, dtype=np.float64))
        v  = np.asarray(lab, dtype=np.float64)
        da = np.linalg.norm((v - np.asarray(a, dtype=np.float64)) / scale)
        db = np.linalg.norm((v - np.asarray(b, dtype=np.float64)) / scale)
        return float(min(da, db))

    def _dominant_lab(self, crop: np.ndarray) -> Optional[list]:
        """Jersey chest colour in LAB, robust to contamination.

        A plain median of the chest strip blends in skin (neck/arms), the white
        number print, shadows, and — when players overlap — a neighbour's jersey.
        Instead we keep only the chest pixels CLOSEST to either team anchor (the
        jersey-like majority) and median THOSE. Measured on 84 hand-labelled crops:
        raw classifier 91% → 98% balanced, and scale-robust (96-98% vs median's
        79-95% across LAB scales). Falls back to a plain median before the anchors
        exist (early frames / uncalibrated)."""
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return None
        chest = crop[int(h * 0.20):int(h * 0.50), int(w * 0.15):int(w * 0.85)]
        if chest.size == 0:
            return None
        px = cv2.cvtColor(chest, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

        a = self._team_colors_lab.get("A")
        b = self._team_colors_lab.get("B")
        if a is None or b is None or len(px) < 10:
            return np.median(px, axis=0).tolist()   # not calibrated yet → plain median

        scale = (self._lab_scale if self._lab_scale is not None
                 else np.array(_LAB_SCALE_DEFAULT, dtype=np.float64))
        a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
        da = np.linalg.norm((px - a) / scale, axis=1)
        db = np.linalg.norm((px - b) / scale, axis=1)
        d  = np.minimum(da, db)                      # distance to the NEAREST anchor
        thr = np.percentile(d, TEAM_JERSEY_PIXEL_FRAC * 100.0)
        sub = px[d <= thr]                           # keep the jersey-like fraction
        if len(sub) < 5:
            return np.median(px, axis=0).tolist()
        return np.median(sub, axis=0).tolist()

    def set_team_colors_lab(self) -> None:
        """(Re)derive LAB anchors from the HSV _team_colors and drop the axis cache.
        Call whenever _team_colors changes (upload set_team_reference or K-Means)."""
        for t in ("A", "B"):
            if t in self._team_colors:
                self._team_colors_lab[t] = self._hsv_to_lab(self._team_colors[t])
        self._team_axis = None

    def _ensure_team_axis(self):
        """Build & cache the UPLOAD-defined A→B discriminant axis in scaled LAB.
        Fingerprinted on anchors+scale so any change rebuilds it. Orientation is
        fixed here (A at t=0, B at t=+|axis|) → A/B can never flip at runtime."""
        if "A" not in self._team_colors_lab or "B" not in self._team_colors_lab:
            return None
        scale = (self._lab_scale if self._lab_scale is not None
                 else np.array(_LAB_SCALE_DEFAULT, dtype=np.float64))
        A = np.array(self._team_colors_lab["A"], dtype=np.float64)
        B = np.array(self._team_colors_lab["B"], dtype=np.float64)
        fp = (tuple(A), tuple(B), tuple(scale))
        if self._team_axis is not None and self._team_axis[0] == fp:
            return self._team_axis
        As = A / scale
        axis = B / scale - As
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:                        # degenerate (A≈B) → caller falls back
            self._team_axis = None
            return None
        axis = axis / norm
        tB = float((B / scale - As) @ axis)    # == norm; B at tB, A at 0
        self._team_axis = (fp, As, axis, scale, tB,
                           TEAM_AXIS_FRAC * tB, TEAM_AXIS_CONF_FRAC * tB)
        # LOUD, once per (anchors, scale): the complete configuration the classifier
        # uses. If team colours come out wrong, this single line says why.
        if fp != getattr(self, "_team_axis_logged", None):
            self._team_axis_logged = fp
            logger.warning(
                "TEAM-CFG  mode=%s  FRAC=%.2f  scale=%s  anchorHSV A=%s B=%s  anchorLAB A=%s B=%s",
                TEAM_MATCH_MODE, TEAM_AXIS_FRAC,
                [round(float(v), 1) for v in scale],
                [round(v, 1) for v in self._team_colors.get("A", [])],
                [round(v, 1) for v in self._team_colors.get("B", [])],
                [round(v) for v in self._team_colors_lab.get("A", [])],
                [round(v) for v in self._team_colors_lab.get("B", [])],
            )
        return self._team_axis

    def _team_vote_confident(self, player_crop: np.ndarray):
        """Return (vote, confident, conf) for one crop, or (None, False, 0.0).

        `conf` ∈ [0,1] is the CONTINUOUS decision margin (|t - threshold| / margin,
        clipped) — used as the team-vote weight so a decisive read outweighs a
        borderline one. `confident` stays the boolean the split logic uses.
        Default = UPLOAD-AXIS projection (camera-invariant, general, no A/B flip);
        legacy HSV nearest-anchor when TEAM_MATCH_MODE='distance'."""
        if "A" not in self._team_colors or "B" not in self._team_colors:
            return None, False, 0.0

        if TEAM_MATCH_MODE == "axis":
            lab = self._dominant_lab(player_crop)
            if lab is None:
                return None, False, 0.0
            # Measure the per-channel LAB scale ONCE from early crops, then freeze —
            # down-weights whatever varies most (usually luminance/lighting), so the
            # axis leans on the stable colour channels. Generalises to any lighting.
            #
            # CRITICAL: measure it from the RAW chest median, NOT from _dominant_lab.
            # _dominant_lab is anchor-filtered, so its outputs are POLARISED toward
            # whichever anchor each crop is nearest. Their std then measures the
            # A-vs-B SEPARATION rather than the per-channel noise — which would make
            # the axis down-weight exactly the channels that separate the teams.
            # (Measured: feeding _dominant_lab back in gave scale [59,15,7] and
            # collapsed A-recall to 73%; the raw median gives ~[33,9,8] → 98%.)
            if LAB_SCALE_MEASURE and self._lab_scale is None:
                raw = self._chest_median_lab(player_crop)
                if raw is not None:
                    self._lab_samples.append(raw)
                if len(self._lab_samples) >= LAB_SCALE_MIN_SAMPLES:
                    s = np.std(np.array(self._lab_samples, dtype=np.float64), axis=0)
                    s[s < 1e-3] = 1.0
                    self._lab_scale = s
                    self._team_axis = None     # rebuild axis with the measured scale
            ax = self._ensure_team_axis()
            if ax is None:
                return None, False, 0.0
            _fp, As, axis, scale, _tB, thr, cmargin = ax
            t = float((np.array(lab, dtype=np.float64) / scale - As) @ axis)
            margin = abs(t - thr)
            conf = min(1.0, margin / cmargin) if cmargin > 1e-6 else 1.0
            return ("A" if t < thr else "B"), (margin >= cmargin), conf

        # ── legacy HSV nearest-anchor ──
        hsv = self._dominant_hsv(player_crop)
        if hsv is None:
            return None, False, 0.0
        da = self._hsv_distance(hsv, self._team_colors["A"])
        db = self._hsv_distance(hsv, self._team_colors["B"])
        margin = abs(da - db)
        conf = (min(1.0, margin / TEAM_CONFIDENT_MARGIN)
                if TEAM_CONFIDENT_MARGIN > 1e-6 else 1.0)
        return ("A" if da <= db else "B"), (margin >= TEAM_CONFIDENT_MARGIN), conf

    def is_not_team(self, track_id: int) -> bool:
        """True when this track has enough colour evidence that it belongs to NEITHER
        team (referee / coach). Callers must clear any sticky team for it."""
        return track_id in self._team_rejected

    def raw_team_vote(self, player_crop: np.ndarray) -> Optional[str]:
        """Single-frame team with NO state/lock — for A/B orientation checks
        (video_processor disambiguation) that must see a fresh read."""
        vote, _, _ = self._team_vote_confident(player_crop)
        return vote

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
        self.set_team_colors_lab()       # keep LAB anchors + axis in sync (upload-axis)
        self._track_team.clear()
        self._track_team_votes.clear()
        self._track_color_dist.clear()
        self._team_rejected.clear()
        self._team_lock.clear()          # colors recomputed → invalidate locks (fix C)
        self._team_disagree.clear()

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
        self.set_team_colors_lab()       # derive LAB anchor + rebuild axis (upload-axis)
        self._track_team.clear()
        self._track_team_votes.clear()
        self._track_color_dist.clear()
        self._team_rejected.clear()
        self._team_lock.clear()          # anchor changed → invalidate locks (fix C)
        self._team_disagree.clear()

    def inherit_team(self, new_tid: int, old_tid: int) -> None:
        """Carry team AND accumulated OCR votes from old_tid to new_tid.

        Called when ByteTrack drops a track and re-assigns a new ID to the same
        physical player (frequent under fast camera pans / angle switches).

        Team transfer keeps the bbox colour stable. Vote transfer (Fix #2) is the
        important part: without it, a player who was mid-confirmation before a
        camera cut restarts number voting from zero on the new ID, so the 2-3
        good reads needed to cross VOTE_THRESHOLD scatter across fragments and a
        number is never confirmed. Pooling the votes preserves that momentum.
        """
        if old_tid in self._track_team:
            self._track_team[new_tid] = self._track_team[old_tid]
        if old_tid in self._track_team_votes:
            self._track_team_votes[new_tid] = deque(
                self._track_team_votes[old_tid],
                maxlen=self._TEAM_VOTE_HISTORY,
            )
        # Carry the not-a-team evidence too, else a referee whose id churns would keep
        # resetting to "unjudged" and get a team colour for the first few frames again.
        if old_tid in self._track_color_dist:
            self._track_color_dist[new_tid] = deque(
                self._track_color_dist[old_tid],
                maxlen=self._TEAM_VOTE_HISTORY,
            )
        if old_tid in self._team_rejected:
            self._team_rejected.add(new_tid)
        # Carry the soft team lock (same physical player); fresh disagreement counter.
        if old_tid in self._team_lock:
            self._team_lock[new_tid] = self._team_lock[old_tid]
        self._team_disagree.pop(new_tid, None)

        # ── Fix #2: pool OCR number votes across the fragment ────────────────
        old_votes = self._votes.get(old_tid)
        if old_votes:
            existing = self._votes.get(new_tid)
            if existing:
                # New ID already gathered some reads this life — merge old in
                # (deque maxlen keeps only the most recent MAX_VOTE_HISTORY).
                for v in old_votes:
                    existing.append(v)
            else:
                self._votes[new_tid] = deque(old_votes, maxlen=MAX_VOTE_HISTORY)
        # Carry the best digit-model confidence so the confidence-upgrade logic
        # doesn't treat the re-ID'd player as brand new.
        if old_tid in self._best_read_conf:
            self._best_read_conf[new_tid] = max(
                self._best_read_conf.get(new_tid, 0.0),
                self._best_read_conf[old_tid],
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
        return float(hue_weight * dh + ds + dv * TEAM_VALUE_WEIGHT)

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
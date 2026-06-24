"""
Basketball object detection using YOLOv8 (best-object-basketball.pt).
Detects: player, ball, referee, hoop, backboard per frame.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Class names must match the training labels in best-object-basketball.pt.
# If the model was trained with different label ordering, update this list.
DEFAULT_CLASS_NAMES = ["player", "ball", "referee", "hoop", "backboard"]

MODELS_DIR = os.getenv("MODELS_PATH", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_FILENAME = "best-object-basketball.pt"


def _bbox_center(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [(x1 + x2) / 2, (y1 + y2) / 2]


class BasketballDetector:
    """
    YOLOv8 inference wrapper for basketball scene detection.
    Runs on GPU if available, falls back to CPU automatically.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        max_det: int = 30,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.model_path = model_path or os.path.join(MODELS_DIR, MODEL_FILENAME)
        self.device = device  # None → ultralytics auto-selects
        self.model = None
        self._class_names: list[str] = DEFAULT_CLASS_NAMES

    def load_model(self) -> None:
        """Load .pt weights. Raises FileNotFoundError if file is missing."""
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path.resolve()}\n"
                f"Place best-object-basketball.pt in backend/models/ or set MODELS_PATH env."
            )

        try:
            import torch
            from ultralytics import YOLO

            # Resolve device: prefer explicit arg, then CUDA, then CPU
            if self.device:
                target_device = self.device
            elif torch.cuda.is_available():
                target_device = "cuda"
                logger.info("GPU detected — running detector on CUDA")
            else:
                target_device = "cpu"
                logger.info("No GPU — running detector on CPU")

            self.model = YOLO(str(path))
            self.model.to(target_device)
            if target_device.startswith("cuda"):
                self.model.half()   # FP16 weights → Tensor Core acceleration
            self._use_half = target_device.startswith("cuda")

            # torch.compile: fuses CUDA ops at runtime, ~15-25% faster inference.
            # Mathematically identical output — no weight change, no quantization.
            # First call compiles (~15s one-time); all subsequent calls use the
            # compiled graph. fullgraph=False lets YOLO's custom ops fall back
            # to eager mode safely if they can't be traced.
            if target_device.startswith("cuda") and hasattr(torch, "compile"):
                try:
                    self.model.model = torch.compile(
                        self.model.model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    logger.info("torch.compile applied to detector model")
                except Exception as _tc:
                    logger.debug("torch.compile skipped for detector: %s", _tc)

            # Sync class name list from model metadata when available
            if hasattr(self.model, "names") and self.model.names:
                # model.names is a dict {int: str} in ultralytics
                self._class_names = [
                    self.model.names[i] for i in sorted(self.model.names.keys())
                ]

            logger.info("Loaded detector model: %s (device=%s, fp16=%s)",
                        path.name, target_device, self._use_half)

        except ImportError as exc:
            raise ImportError(
                "ultralytics and torch are required. "
                "Install via: pip install ultralytics torch"
            ) from exc

        except RuntimeError as exc:
            # GPU OOM or CUDA init error → retry on CPU
            logger.warning("GPU init failed (%s) — falling back to CPU", exc)
            from ultralytics import YOLO

            self.model = YOLO(str(path))
            self.model.to("cpu")

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def warmup(self, height: int = 720, width: int = 1280) -> None:
        """One dummy inference to trigger CUDA JIT compilation before real frames arrive."""
        if not self.is_model_loaded():
            return
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        try:
            self.detect(dummy)
            logger.info("Detector warmup done (%dx%d dummy frame)", width, height)
        except Exception as e:
            logger.debug("Detector warmup error (non-fatal): %s", e)

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run inference on one BGR frame (from OpenCV cap.read()).

        Returns a dict with keys:
          players, ball, referees, hoops, backboards, raw_results
        """
        empty = self._empty_result()

        if frame is None or frame.size == 0:
            return empty

        if not self.is_model_loaded():
            logger.warning("detect() called before load_model() — returning empty result")
            return empty

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False,
            half=getattr(self, "_use_half", False),
        )

        parsed = self._parse_results(results)
        parsed["raw_results"] = results
        return parsed

    def _parse_results(self, results) -> dict:
        """Convert ultralytics Results into the pipeline dict format."""
        output = self._empty_result()
        output.pop("raw_results", None)  # filled by caller

        if not results or len(results) == 0:
            return output

        result = results[0]  # single-image inference → first result
        if result.boxes is None:
            return output

        boxes = result.boxes
        # boxes.xyxy  → (N, 4) float32 tensor [x1, y1, x2, y2]
        # boxes.conf  → (N,)   float32
        # boxes.cls   → (N,)   float32 (class index)

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i, (coords, conf, cls_idx) in enumerate(zip(xyxy, confs, classes)):
            x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
            bbox = [x1, y1, x2, y2]
            class_name = (
                self._class_names[cls_idx]
                if cls_idx < len(self._class_names)
                else f"class_{cls_idx}"
            )

            if class_name == "player":
                output["players"].append({"bbox": bbox, "confidence": float(conf), "class": "player"})

            elif class_name == "ball":
                # Keep only the highest-confidence ball detection
                center = _bbox_center(x1, y1, x2, y2)
                candidate = {"bbox": bbox, "confidence": float(conf), "center": center}
                if output["ball"] is None or float(conf) > output["ball"]["confidence"]:
                    output["ball"] = candidate

            elif class_name == "referee":
                output["referees"].append({"bbox": bbox, "confidence": float(conf), "class": "referee"})

            elif class_name == "hoop":
                center = _bbox_center(x1, y1, x2, y2)
                output["hoops"].append({"bbox": bbox, "confidence": float(conf), "center": center})

            elif class_name == "backboard":
                center = _bbox_center(x1, y1, x2, y2)
                output["backboards"].append({"bbox": bbox, "confidence": float(conf), "center": center})

        return output

    @staticmethod
    def get_ball_center(ball_bbox: list[float]) -> list[float]:
        """Compute center point from [x1, y1, x2, y2] bbox."""
        x1, y1, x2, y2 = ball_bbox
        return _bbox_center(x1, y1, x2, y2)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "players": [],
            "ball": None,
            "referees": [],
            "hoops": [],
            "backboards": [],
            "raw_results": None,
        }


def create_detector(
    models_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
) -> BasketballDetector:
    """Factory — create and load a BasketballDetector in one call."""
    model_file = os.path.join(models_path or MODELS_DIR, MODEL_FILENAME)
    detector = BasketballDetector(
        model_path=model_file,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    detector.load_model()
    return detector


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== BasketballDetector smoke test ===")

    # Locate models relative to this script
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"

    detector = BasketballDetector(
        model_path=str(models_dir / MODEL_FILENAME),
        conf_threshold=0.5,
        iou_threshold=0.45,
    )

    print(f"Model path  : {detector.model_path}")
    print(f"Model loaded: {detector.is_model_loaded()}")

    try:
        detector.load_model()
        print(f"Model loaded: {detector.is_model_loaded()}")
        print(f"Class names : {detector._class_names}")
    except FileNotFoundError as e:
        print(f"[SKIP] Model file missing — {e}")
        sys.exit(0)
    except ImportError as e:
        print(f"[SKIP] Dependencies not installed — {e}")
        sys.exit(0)

    # Test with black dummy frame (1080p)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = detector.detect(dummy_frame)

    print("\n--- Detection on 1080p black frame ---")
    print(f"  Players   : {len(result['players'])}")
    print(f"  Ball      : {result['ball']}")
    print(f"  Referees  : {len(result['referees'])}")
    print(f"  Hoops     : {len(result['hoops'])}")
    print(f"  Backboards: {len(result['backboards'])}")
    print(f"  raw_results type: {type(result['raw_results'])}")

    # Edge case: empty frame
    result_empty = detector.detect(None)
    assert result_empty["players"] == []
    assert result_empty["ball"] is None
    print("\n[PASS] Empty/None frame returns empty result without crash")

    # Ball center helper
    center = BasketballDetector.get_ball_center([100.0, 200.0, 140.0, 240.0])
    assert center == [120.0, 220.0], f"Unexpected center: {center}"
    print("[PASS] get_ball_center returns correct midpoint")

    print("\n=== All tests passed ===")

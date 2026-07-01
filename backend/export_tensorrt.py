"""
TensorRT engine export — run ONCE on paul-higo before starting the backend.

  conda activate cl
  cd ~/Documents/vision-computer/.../smart-vision-cl
  python backend/export_tensorrt.py

Exports .pt → .engine for the three YOLO models used in the pipeline.
Engines are device-specific (RTX 3060 Ti); re-run after driver/TRT updates.
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

# Models to export: (filename, workspace_GB)
# basketball detector needs more workspace for larger input resolution
TARGETS = [
    ("jersey_no.pt",              2),
    ("best-detect-num-v2.pt",    2),
    ("best-object-basketball.pt", 4),
]


def check_tensorrt() -> bool:
    try:
        import tensorrt as trt  # noqa: F401
        print(f"TensorRT found: {trt.__version__}")
        return True
    except ImportError:
        print("TensorRT not installed. Run:")
        print("  pip install tensorrt")
        print("  # or: pip install nvidia-tensorrt")
        return False


def export_model(name: str, workspace_gb: int) -> bool:
    pt_path   = MODELS_DIR / name
    eng_path  = MODELS_DIR / name.replace(".pt", ".engine")

    if not pt_path.exists():
        print(f"  SKIP (not found): {pt_path}")
        return False

    if eng_path.exists():
        size_mb = eng_path.stat().st_size / 1e6
        print(f"  SKIP (already exists, {size_mb:.1f} MB): {eng_path.name}")
        return True

    print(f"  Exporting {name} → {eng_path.name} ...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        result = model.export(
            format="engine",
            half=True,          # FP16 — matches runtime inference mode
            device=0,           # GPU 0 (RTX 3060 Ti)
            workspace=workspace_gb,
            verbose=False,
        )
        size_mb = Path(str(result)).stat().st_size / 1e6
        print(f"  ✓  {eng_path.name}  ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  ✗  Export failed: {exc}")
        return False


def main():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — must run on GPU machine.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Models dir: {MODELS_DIR.resolve()}\n")

    if not check_tensorrt():
        sys.exit(1)

    print()
    ok = 0
    for name, ws in TARGETS:
        print(f"[{name}]")
        ok += export_model(name, ws)

    print(f"\nDone: {ok}/{len(TARGETS)} models exported.")
    if ok == len(TARGETS):
        print("Restart the backend — .engine files will be loaded automatically.")


if __name__ == "__main__":
    main()

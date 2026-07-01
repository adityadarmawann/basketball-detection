"""
TensorRT engine export — run ONCE on paul-higo before starting the backend.

  conda activate cl-gpu
  cd ~/computer-vision/.../smart-vision-cl/basketball-detection
  python backend/export_tensorrt.py

Exports .pt → .onnx → .engine for the three YOLO models used in the pipeline.
Engines are device-specific (RTX 3060 Ti); re-run after driver/TRT updates.

Compatible with TensorRT 10 and 11 (EXPLICIT_BATCH flag removed in TRT 10+).
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

# Models to export: (filename, workspace_GB)
TARGETS = [
    ("jersey_no.pt",              2),
    ("best-detect-num-v2.pt",    2),
    ("best-object-basketball.pt", 4),
]


def check_tensorrt():
    try:
        import tensorrt as trt
        print(f"TensorRT found: {trt.__version__}")
        return trt
    except ImportError:
        print("TensorRT not installed. Run:  pip install tensorrt")
        return None


def export_pt_to_onnx(pt_path: Path, onnx_path: Path) -> bool:
    """Export .pt → .onnx using ultralytics (only the ONNX step, no TRT)."""
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1e6
        print(f"    ONNX already exists ({size_mb:.1f} MB), skipping export")
        return True

    print(f"    Exporting {pt_path.name} → {onnx_path.name} ...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(pt_path))
        result = model.export(format="onnx", half=False, device="cpu", verbose=False)
        exported = Path(str(result))
        if not exported.exists():
            print(f"    ✗  ONNX not found at expected path: {exported}")
            return False
        if exported != onnx_path:
            exported.rename(onnx_path)
        size_mb = onnx_path.stat().st_size / 1e6
        print(f"    ✓  {onnx_path.name}  ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"    ✗  ONNX export failed: {exc}")
        return False


def export_onnx_to_engine(trt, onnx_path: Path, engine_path: Path, workspace_gb: int) -> bool:
    """Convert .onnx → .engine using TensorRT Python API (TRT 10/11 compatible)."""
    print(f"    Building TRT engine from {onnx_path.name} ...")
    try:
        logger  = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        # TRT 10+: EXPLICIT_BATCH is always on, no flag needed
        network = builder.create_network()

        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            data = f.read()
        if not parser.parse(data):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            print(f"    ✗  ONNX parse failed: {errors}")
            return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_gb * (1 << 30),
        )

        # BuilderFlag.FP16 was reorganised in TRT 10/11 — check before setting
        fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
        if fp16_flag is not None:
            config.set_flag(fp16_flag)
            precision_label = "FP16"
        else:
            precision_label = "FP32 (FP16 flag removed in this TRT version)"

        print(f"    Compiling engine ({precision_label}, workspace={workspace_gb}GB) — may take a few minutes ...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print("    ✗  Engine build returned None")
            return False

        with open(engine_path, "wb") as f:
            f.write(serialized)

        size_mb = engine_path.stat().st_size / 1e6
        print(f"    ✓  {engine_path.name}  ({size_mb:.1f} MB)")
        return True

    except Exception as exc:
        print(f"    ✗  TRT engine build failed: {exc}")
        return False


def export_model(trt, name: str, workspace_gb: int) -> bool:
    pt_path     = MODELS_DIR / name
    onnx_path   = MODELS_DIR / name.replace(".pt", ".onnx")
    engine_path = MODELS_DIR / name.replace(".pt", ".engine")

    if engine_path.exists():
        size_mb = engine_path.stat().st_size / 1e6
        print(f"  SKIP (engine already exists, {size_mb:.1f} MB): {engine_path.name}")
        return True

    if not pt_path.exists():
        print(f"  SKIP (model not found): {pt_path}")
        return False

    print(f"  [{name}]")

    # Step 1: .pt → .onnx
    if not export_pt_to_onnx(pt_path, onnx_path):
        return False

    # Step 2: .onnx → .engine  (TRT 10/11 Python API — no EXPLICIT_BATCH)
    ok = export_onnx_to_engine(trt, onnx_path, engine_path, workspace_gb)

    # Clean up intermediate .onnx to save disk space
    if ok and onnx_path.exists():
        onnx_path.unlink()
        print(f"    Removed intermediate {onnx_path.name}")

    return ok


def main():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — must run on GPU machine.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Models dir: {MODELS_DIR.resolve()}\n")

    trt = check_tensorrt()
    if trt is None:
        sys.exit(1)

    print()
    ok = 0
    for name, ws in TARGETS:
        ok += export_model(trt, name, ws)

    print(f"\nDone: {ok}/{len(TARGETS)} models exported.")
    if ok == len(TARGETS):
        print("Restart the backend — .engine files will be loaded automatically.")


if __name__ == "__main__":
    main()

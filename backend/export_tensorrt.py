"""
TensorRT engine export — run ONCE on paul-higo before starting the backend.

  conda activate cl-gpu
  python backend/export_tensorrt.py

Exports .pt → .engine for the three YOLO models used in the pipeline.
Uses ultralytics' built-in export so that metadata (task, imgsz, names) is
embedded inside the engine — required for Ultralytics to cache the predictor
across calls (prevents per-frame engine reload).
"""

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

# (filename, imgsz, workspace_GB)
TARGETS = [
    ("jersey_no.pt",               640, 2),
    ("best-detect-num-v2.pt",     640, 2),
    ("best-object-basketball.pt",  640, 4),
]


def export_model(name: str, imgsz: int, workspace_gb: int) -> bool:
    """
    Export .pt → .engine via ultralytics' built-in exporter.

    Using ultralytics export (not raw TRT API) is required because:
    - ultralytics embeds metadata (task, imgsz, names) into the engine
    - without metadata, AutoBackend cannot cache the predictor
    - result: engine reloads on every model.predict() call (per-frame spam)
    """
    pt_path     = MODELS_DIR / name
    engine_path = MODELS_DIR / name.replace(".pt", ".engine")

    if engine_path.exists():
        size_mb = engine_path.stat().st_size / 1e6
        print(f"  SKIP (engine exists, {size_mb:.1f} MB): {engine_path.name}")
        return True

    if not pt_path.exists():
        print(f"  SKIP (model not found): {pt_path}")
        return False

    print(f"  [{name}]  imgsz={imgsz}  workspace={workspace_gb}GB")

    from ultralytics import YOLO
    model = YOLO(str(pt_path))

    # Try FP16 first (faster inference); fall back to FP32 if TRT version
    # doesn't expose BuilderFlag.FP16 (removed in some TRT 11 builds).
    for half in (True, False):
        precision = "FP16" if half else "FP32"
        print(f"    Exporting {precision}...")
        try:
            result = model.export(
                format="engine",
                half=half,
                device=0,
                imgsz=imgsz,
                workspace=workspace_gb,
                verbose=False,
            )
            exported = Path(str(result))
            if not exported.exists():
                print(f"    ✗  Engine not found at: {exported}")
                continue

            if exported != engine_path:
                exported.rename(engine_path)

            size_mb = engine_path.stat().st_size / 1e6
            print(f"    ✓  {engine_path.name}  ({size_mb:.1f} MB, {precision})")
            return True

        except Exception as exc:
            print(f"    ✗  {precision} failed: {exc}")
            if half:
                print("        Retrying with FP32...")

    print(f"  ✗  Export failed: {name}")
    return False


def main():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — must run on GPU machine (paul-higo).")
        sys.exit(1)

    try:
        import tensorrt as trt
        trt_ver = trt.__version__
    except ImportError:
        trt_ver = "not found"

    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"TensorRT   : {trt_ver}")
    print(f"Models dir : {MODELS_DIR.resolve()}\n")

    # Remove old engines built via raw TRT API — they have no embedded metadata
    # so Ultralytics recreates AutoBackend on every predict() call.
    old = list(MODELS_DIR.glob("*.engine"))
    if old:
        print("Removing old engines (no embedded metadata — caused per-frame reload):")
        for f in old:
            f.unlink()
            print(f"  deleted {f.name}")
        print()

    ok = 0
    for name, imgsz, ws in TARGETS:
        ok += export_model(name, imgsz, ws)
        print()

    print(f"Done: {ok}/{len(TARGETS)} models exported.")
    if ok == len(TARGETS):
        print("Restart the backend — .engine files will be loaded automatically.")
    else:
        print("Some exports failed — backend will use .pt (PyTorch, no TRT speedup).")


if __name__ == "__main__":
    main()

"""
Structured diagnostics for the two things that keep going wrong:
  1. TEAM colour (A vs B from the jersey), and
  2. JERSEY NUMBER OCR.

Prose logs are useless for this — you cannot score a decision you cannot replay.
So when DIAG_DUMP=1 this writes machine-readable JSONL plus a sample of the actual
crops, which is everything needed to re-score a run offline and to HAND-LABEL it
without going back to the video.

Output (per match, under DIAG_DIR):
    <match_id>/config.json   one-shot: anchors, scale, FRAC, mode, roster
    <match_id>/team.jsonl    one row per player-frame that was colour-classified
    <match_id>/ocr.jsonl     one row per OCR read
    <match_id>/crops/        every DIAG_CROP_EVERY-th crop, named f{frame}_t{tid}.jpg

Everything is off unless DIAG_DUMP=1, and every call is wrapped so a diagnostics
bug can never take the pipeline down.

Env:
    DIAG_DUMP=1            enable
    DIAG_DIR=./diag        output root
    DIAG_CROP_EVERY=150    save 1 in N crops (0 = none)
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional

import cv2
import numpy as np

ENABLED    = os.getenv("DIAG_DUMP", "0") == "1"
DIAG_DIR   = os.getenv("DIAG_DIR", "./diag")
CROP_EVERY = int(os.getenv("DIAG_CROP_EVERY", "150"))

_lock = threading.Lock()


class Diag:
    """One diagnostics sink per match. Cheap no-op when disabled."""

    def __init__(self, match_id: str):
        self.on = ENABLED
        self.match_id = match_id
        self._n_team = 0
        self._n_ocr = 0
        if not self.on:
            return
        try:
            self.root = os.path.join(DIAG_DIR, str(match_id))
            self.crops = os.path.join(self.root, "crops")
            os.makedirs(self.crops, exist_ok=True)
            self._team_f = open(os.path.join(self.root, "team.jsonl"), "w", buffering=1)
            self._ocr_f = open(os.path.join(self.root, "ocr.jsonl"), "w", buffering=1)
        except Exception:
            self.on = False

    # ── one-shot: the exact configuration the classifier runs with ──────────
    def config(self, **kw: Any) -> None:
        if not self.on:
            return
        try:
            with open(os.path.join(self.root, "config.json"), "w") as f:
                json.dump(_clean(kw), f, indent=1)
        except Exception:
            pass

    # ── per player-frame: everything behind one team decision ───────────────
    def team(self, crop: Optional[np.ndarray] = None, **row: Any) -> None:
        if not self.on:
            return
        try:
            with _lock:
                self._n_team += 1
                n = self._n_team
                self._team_f.write(json.dumps(_clean(row), separators=(",", ":")) + "\n")
            if crop is not None and CROP_EVERY > 0 and n % CROP_EVERY == 0 and crop.size:
                f, t = row.get("f", n), row.get("tid", 0)
                cv2.imwrite(os.path.join(self.crops, f"f{f}_t{t}.jpg"), crop)
        except Exception:
            pass

    # ── per OCR read: candidate, weight, votes, confirmation, flushes ───────
    def ocr(self, **row: Any) -> None:
        if not self.on:
            return
        try:
            with _lock:
                self._n_ocr += 1
                self._ocr_f.write(json.dumps(_clean(row), separators=(",", ":")) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        if not self.on:
            return
        for f in ("_team_f", "_ocr_f"):
            try:
                getattr(self, f).close()
            except Exception:
                pass


def _clean(d: dict) -> dict:
    """numpy/py types → JSON, rounded so the files stay small and diffable."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = round(float(v), 3)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, np.ndarray):
            out[k] = [round(float(x), 2) for x in v.ravel()]
        elif isinstance(v, (list, tuple)):
            out[k] = [round(float(x), 2) if isinstance(x, (float, np.floating)) else x
                      for x in v]
        elif isinstance(v, dict):
            out[k] = {str(kk): (round(float(vv), 3)
                                if isinstance(vv, (float, np.floating)) else vv)
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


_NULL = Diag.__new__(Diag)
_NULL.on = False


def null() -> Diag:
    """Sink used before a match is known / when disabled."""
    return _NULL

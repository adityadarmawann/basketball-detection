"""Algoritma pipeline basket Roboflow yang diadopsi ke smart-vision-cl.

Ditulis ulang SETIA pada sumbernya, repositori `roboflow/sports` cabang
`feat/basketball` (lisensi MIT), berkas:
    sports/common/path.py      -> clean_paths
    sports/common/temporal.py  -> ConsecutiveValueTracker
    sports/basketball/tools.py -> ShotEventTracker
ditambah KeyPointsSmoother dari notebook make-or-miss.

Ditulis ulang di sini, bukan dipasang sebagai paket, karena `sports` menarik
`umap-learn`, `supervision`, dan `inference` yang belum ada di lingkungan ini.
Yang dipakai hanya numpy dan scipy, keduanya sudah ada.

Masalah smart-vision-cl yang ditargetkan tiap kelas:

    clean_paths             galat jarak tempuh 18 sampai 28 persen; akselerasi
                            median 47 m/s2 (4,8 g) yang mustahil secara fisika
    KeyPointsSmoother       hanya 13 dari 33 keypoint lapangan lolos di footage CL
    ConsecutiveValueTracker akurasi nomor punggung 63 persen, nomor sering berganti
    ShotEventTracker        MISSED_FG kini butuh bukti, bukan sekadar waktu habis
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:                                    # pragma: no cover
    _HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════════
# 1. clean_paths  (sports/common/path.py)
# ═══════════════════════════════════════════════════════════════════
def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation berskala Gauss.

    Dikali 1,4826 supaya setara simpangan baku bila datanya normal. Dipakai
    karena median dan MAD tidak tertarik oleh pencilan, sedangkan rata rata
    dan simpangan baku justru ditarik oleh teleport yang mau kita buang.
    """
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _runs(bool_arr: np.ndarray) -> List[Tuple[int, int]]:
    """Kelompokkan True yang berurutan jadi pasangan (awal, akhir) inklusif."""
    idx = np.flatnonzero(bool_arr)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(g[0], g[-1]) for g in np.split(idx, splits)]


def _linear_interp_1d(y: np.ndarray) -> np.ndarray:
    """Isi NaN dengan interpolasi linear; ujung diisi nilai valid terdekat."""
    n = y.shape[0]
    out = y.copy()
    isnan = np.isnan(out)
    if isnan.all():
        return out
    valid = np.flatnonzero(~isnan)
    if valid.size:
        out[: valid[0]] = out[valid[0]]
        out[valid[-1] + 1 :] = out[valid[-1]]
    isnan = np.isnan(out)
    if isnan.any():
        x = np.arange(n)
        out[isnan] = np.interp(x[isnan], x[~isnan], out[~isnan])
    return out


def _savgol_safe(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Savitzky-Golay dengan jaring pengaman rata rata bergerak untuk deret pendek."""
    y = y.astype(float)
    n = y.shape[0]
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < poly + 2 or not _HAS_SCIPY:
        k = min(5, n)
        if k < 2:
            return y
        pad = k // 2
        return np.convolve(np.pad(y, (pad, pad), mode="edge"), np.ones(k) / k,
                           mode="valid")[:n]
    out = savgol_filter(y, window_length=window, polyorder=poly, mode="interp")

    # TAMBAHAN khusus smart-vision-cl, tidak ada di Roboflow.
    # mode="interp" mencocokkan polinomial ke jendela tepi lalu mengevaluasinya,
    # dan itu bisa melesat jauh di luar data aslinya. Pada data nyata q1 UBAYA
    # UNESA, pemain yang hampir diam mendapat langkah palsu 0,93 m di sampel
    # pertama hanya karena artefak tepi ini. Jadi keluaran di daerah tepi
    # dikurung ke rentang nilai mentah di jendelanya, sehingga penghalusan tidak
    # pernah mengarang posisi yang tidak pernah ada. Bagian tengah tidak
    # disentuh sama sekali.
    tepi = window // 2
    if tepi > 0 and n > tepi:
        for i in list(range(tepi)) + list(range(n - tepi, n)):
            lo = max(0, i - tepi)
            hi = min(n, i + tepi + 1)
            out[i] = np.clip(out[i], y[lo:hi].min(), y[lo:hi].max())
    return out


def clean_paths(
    video_xy: np.ndarray,
    jump_sigma: float = 5.0,
    min_jump_dist: float = 0.7,
    max_jump_run: int = 6,
    pad_around_runs: int = 1,
    smooth_window: int = 9,
    smooth_poly: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bersihkan lintasan pemain dari lompatan posisi yang mustahil.

    Empat tahap:
        1. deteksi lompatan memakai median + MAD, bukan rata rata
        2. buang hanya rentetan PENDEK, karena rentetan panjang berarti
           pemainnya memang berpindah
        3. tambal lubang dengan interpolasi linear
        4. haluskan dengan Savitzky-Golay yang menjaga bentuk gerakan

    Args:
        video_xy: (T, P, 2) koordinat lapangan, T frame, P pemain, satuan meter.
        jump_sigma: berapa MAD di atas median baru disebut lompatan. Turunkan
            untuk lebih galak. Nilai resmi Roboflow 5,0; notebook memakai 3,5.
        min_jump_dist: lompatan di bawah ini (meter) diabaikan walau lolos sigma.
        max_jump_run: rentetan lebih panjang dari ini DIPERTAHANKAN.
        pad_around_runs: ikut buang sekian frame di kiri kanan rentetan buruk.
        smooth_window: panjang jendela Savitzky-Golay, harus ganjil.
        smooth_poly: orde polinomial.

    Returns:
        cleaned: (T, P, 2) lintasan bersih
        edited:  (T, P) bool, True pada frame yang diubah
    """
    video_xy = np.asarray(video_xy, dtype=float)
    if video_xy.ndim != 3 or video_xy.shape[2] != 2:
        raise ValueError(f"video_xy harus (T, P, 2), diterima {video_xy.shape}")
    T, P, _ = video_xy.shape
    cleaned = video_xy.copy()
    edited = np.zeros((T, P), dtype=bool)

    for p in range(P):
        traj = cleaned[:, p, :]
        speed = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        finite = np.isfinite(speed)
        if speed.size == 0 or not finite.any():
            continue

        med = np.median(speed[finite])
        scale = max(_mad(speed[finite]), 1e-6)
        jump_mask = (speed > med + jump_sigma * scale) & (speed > min_jump_dist)

        remove = np.zeros(T, dtype=bool)
        for s, e in _runs(jump_mask):
            if (e - s + 1) <= max_jump_run:
                # speed[i] adalah perpindahan frame i -> i+1, jadi akhir rentetan
                # menyentuh frame e+1
                rs = max(0, s - pad_around_runs)
                re = min(T - 1, e + 1 + pad_around_runs)
                remove[rs : re + 1] = True

        if not remove.any():
            for d in range(2):
                traj[:, d] = _savgol_safe(traj[:, d], smooth_window, smooth_poly)
            cleaned[:, p, :] = traj
            continue

        edited[:, p] |= remove
        traj_nan = traj.copy()
        traj_nan[remove, :] = np.nan
        for d in range(2):
            traj_nan[:, d] = _linear_interp_1d(traj_nan[:, d])
            traj_nan[:, d] = _savgol_safe(traj_nan[:, d], smooth_window, smooth_poly)
        cleaned[:, p, :] = traj_nan

    return cleaned, edited


def path_distance(xy: np.ndarray, max_step: Optional[float] = None) -> np.ndarray:
    """Jarak tempuh kumulatif per pemain, satuan mengikuti masukan (meter).

    Args:
        xy: (T, P, 2)
        max_step: bila diisi, langkah lebih besar dari ini diabaikan. Pengaman
            terakhir setelah clean_paths. Untuk basket 30 fps, batas wajar
            sekitar 0,4 m per frame (sprint 12 m/s).
    """
    xy = np.asarray(xy, dtype=float)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=2)
    d = np.nan_to_num(d, nan=0.0)
    if max_step is not None:
        d[d > max_step] = 0.0
    return d.sum(axis=0)


# ═══════════════════════════════════════════════════════════════════
# 2. KeyPointsSmoother  (notebook make-or-miss)
# ═══════════════════════════════════════════════════════════════════
class KeyPointsSmoother:
    """Rata rata berjalan untuk keypoint lapangan sebelum homografi dihitung.

    Keypoint yang keyakinannya di bawah ambang dijadikan NaN, bukan nol, supaya
    tidak menarik rata rata ke titik asal. Hasilnya homografi tidak lagi
    bergetar tiap frame.
    """

    def __init__(self, length: int = 3):
        if length < 1:
            raise ValueError("length minimal 1")
        self.length = length
        self.buffer: deque = deque(maxlen=length)

    def update(
        self,
        xy: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        conf_threshold: float = 0.0,
    ) -> np.ndarray:
        """Masukkan keypoint frame ini, kembalikan versi halus. (K, 2) masuk, (K, 2) keluar."""
        arr = np.asarray(xy, dtype=float).copy()
        if confidence is not None:
            arr[np.asarray(confidence, dtype=float) < conf_threshold] = np.nan
        self.buffer.append(arr)
        stacked = np.stack(self.buffer, axis=0)
        kosong = np.isnan(stacked).all(axis=0)
        out = np.full(arr.shape, np.nan, dtype=float)
        if (~kosong).any():
            aman = np.where(kosong, 0.0, stacked)       # cegah "Mean of empty slice"
            jml = np.sum(np.where(np.isnan(stacked), 0.0, aman), axis=0)
            cnt = np.sum(~np.isnan(stacked), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                terisi = jml / np.maximum(cnt, 1)
            out[~kosong] = terisi[~kosong]
        return out

    def reset(self) -> None:
        self.buffer.clear()


# ═══════════════════════════════════════════════════════════════════
# 3. ConsecutiveValueTracker  (sports/common/temporal.py)
# ═══════════════════════════════════════════════════════════════════
Value = Union[int, str, None]


class ConsecutiveValueTracker:
    """Kunci sebuah nilai ke track setelah N bacaan sama berturut turut.

    Setelah terkunci, nilainya TIDAK PERNAH berubah lagi sampai di-reset.
    Inilah bedanya dengan smart-vision-cl, di mana nomor punggung bisa
    berganti ganti sepanjang pertandingan.
    """

    def __init__(self, n_consecutive: int = 3):
        if n_consecutive < 1:
            raise ValueError("n_consecutive minimal 1")
        self.n = n_consecutive
        self._streak: Dict[int, int] = {}
        self._last: Dict[int, Optional[str]] = {}
        self._validated: Dict[int, Optional[str]] = {}

    @staticmethod
    def _normalize(value: Value) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        return s or None

    def update(
        self, tracker_ids: Sequence[int], values: Sequence[Value]
    ) -> List[Optional[str]]:
        if len(tracker_ids) != len(values):
            raise ValueError("tracker_ids dan values harus sama panjang")
        out: List[Optional[str]] = []
        for tid, raw in zip(tracker_ids, values):
            tid = int(tid)
            if self._validated.get(tid) is not None:
                out.append(self._validated[tid])          # sudah terkunci
                continue
            val = self._normalize(raw)
            if val is None:
                self._streak.setdefault(tid, 0)
                self._last.setdefault(tid, None)
                self._validated.setdefault(tid, None)
                out.append(None)
                continue
            if self._last.get(tid) == val:
                self._streak[tid] = self._streak.get(tid, 0) + 1
            else:
                self._streak[tid] = 1
                self._last[tid] = val
            if self._streak[tid] >= self.n:
                self._validated[tid] = val
            out.append(self._validated.get(tid))
        return out

    def get_validated(
        self, tracker_ids: Union[int, Iterable[int]]
    ) -> Union[Optional[str], List[Optional[str]]]:
        if isinstance(tracker_ids, (int, np.integer)):
            return self._validated.get(int(tracker_ids))
        return [self._validated.get(int(t)) for t in tracker_ids]

    def validated_dict(self) -> Dict[int, str]:
        return {t: v for t, v in self._validated.items() if v is not None}

    # ── tambahan khusus smart-vision-cl, tidak ada di Roboflow ──
    def transfer(self, from_id: int, to_id: int) -> bool:
        """Pindahkan nomor terkunci saat track pecah lalu lahir dengan id baru.

        ByteTrack di smart-vision-cl menghasilkan 98 sampai 3222 track_id untuk
        10 pemain. Ini jembatan sementara sampai ReID betulan dibangun: begitu
        track lama hilang dan track baru muncul di tempat yang sama, nomornya
        ikut pindah alih alih hilang.
        """
        from_id, to_id = int(from_id), int(to_id)
        if self._validated.get(from_id) is not None and \
           self._validated.get(to_id) is None:
            self._validated[to_id] = self._validated[from_id]
            return True
        return False

    def reset_id(self, tracker_id: int) -> None:
        tid = int(tracker_id)
        self._streak.pop(tid, None)
        self._last.pop(tid, None)
        self._validated.pop(tid, None)

    def reset_all(self) -> None:
        self._streak.clear()
        self._last.clear()
        self._validated.clear()


# ═══════════════════════════════════════════════════════════════════
# 4. ShotEventTracker  (sports/basketball/tools.py)
# ═══════════════════════════════════════════════════════════════════
BALL_IN_BASKET_MIN_CONSECUTIVE_FRAMES = 2
JUMP_SHOT_MIN_CONSECUTIVE_FRAMES = 3
LAYUP_DUNK_MIN_CONSECUTIVE_FRAMES = 3


class ShotType(Enum):
    NONE = "NONE"
    JUMP = "JUMP"
    LAYUP = "LAYUP"


class ShotEvent(Enum):
    START = "START"
    MADE = "MADE"
    MISSED = "MISSED"


@dataclass
class ShotEventTracker:
    """Mesin status tembakan berbasis kelas deteksi, bukan timeout buta.

    Bedanya dengan smart-vision-cl: MADE di sini DIBUKTIKAN oleh kelas deteksi
    `ball-in-basket` selama 2 frame berturut turut. Smart-vision-cl menyatakan
    MISSED_FG hanya karena zona ring tidak pernah dimasuki sampai waktu habis,
    tanpa bukti positif apa pun, sehingga dokumen 05 melabelinya ESTIMASI RENDAH.

    PRASYARAT: detektor objek harus dilatih ulang dengan kelas
    `ball-in-basket`, `player-jump-shot`, dan `player-layup-dunk`. Sebelum itu,
    tracker ini tidak akan menghasilkan apa apa karena masukannya selalu False.
    """

    reset_time_frames: int
    minimum_frames_between_starts: int
    cooldown_frames_after_made: int

    shot_in_progress: bool = False
    shot_type: ShotType = ShotType.NONE
    shot_start_frame: Optional[int] = None
    shot_deadline_frame: Optional[int] = None
    frames_since_start: int = 0
    consecutive_jump_shot_frames: int = 0
    consecutive_layup_frames: int = 0
    consecutive_ball_in_basket_frames: int = 0
    last_made_frame: Optional[int] = None

    @classmethod
    def from_fps(cls, fps: float) -> "ShotEventTracker":
        """Parameter mengikuti Roboflow: 1,7 detik tenggat, 0,5 detik jeda."""
        return cls(
            reset_time_frames=int(fps * 1.7),
            minimum_frames_between_starts=int(fps * 0.5),
            cooldown_frames_after_made=int(fps * 0.5),
        )

    def update(
        self,
        frame_index: int,
        has_jump_shot: bool,
        has_layup_dunk: bool,
        has_ball_in_basket: bool,
    ) -> List[dict]:
        events: List[dict] = []

        self.consecutive_jump_shot_frames = (
            self.consecutive_jump_shot_frames + 1 if has_jump_shot else 0)
        self.consecutive_layup_frames = (
            self.consecutive_layup_frames + 1 if has_layup_dunk else 0)
        self.consecutive_ball_in_basket_frames = (
            self.consecutive_ball_in_basket_frames + 1 if has_ball_in_basket else 0)

        capai_jump = self.consecutive_jump_shot_frames == JUMP_SHOT_MIN_CONSECUTIVE_FRAMES
        capai_layup = self.consecutive_layup_frames == LAYUP_DUNK_MIN_CONSECUTIVE_FRAMES
        mulai = capai_jump or capai_layup

        if mulai and self._dalam_cooldown(frame_index):
            mulai = False

        if mulai:
            if self.shot_in_progress:
                if self.frames_since_start >= self.minimum_frames_between_starts:
                    events.append(self._ev(ShotEvent.MISSED, frame_index))
                    self._reset_shot()
                else:
                    mulai = False
            if mulai:
                self.shot_in_progress = True
                self.shot_type = ShotType.JUMP if capai_jump else ShotType.LAYUP
                self.shot_start_frame = frame_index
                self.shot_deadline_frame = frame_index + self.reset_time_frames
                self.frames_since_start = 0
                self.consecutive_ball_in_basket_frames = 0
                events.append(self._ev(ShotEvent.START, frame_index))

        if self.shot_in_progress:
            self.frames_since_start += 1
            if self.consecutive_ball_in_basket_frames >= BALL_IN_BASKET_MIN_CONSECUTIVE_FRAMES:
                events.append(self._ev(ShotEvent.MADE, frame_index))
                self.last_made_frame = frame_index
                self._reset_shot()
                return events
            if (self.shot_deadline_frame is not None
                    and frame_index >= self.shot_deadline_frame):
                events.append(self._ev(ShotEvent.MISSED, frame_index))
                self._reset_shot()
                return events

        return events

    def _dalam_cooldown(self, frame_index: int) -> bool:
        if self.last_made_frame is None:
            return False
        return (frame_index - self.last_made_frame) < self.cooldown_frames_after_made

    def _ev(self, event: ShotEvent, frame_index: int) -> dict:
        return {"event": event.value, "frame": frame_index, "type": self.shot_type.value}

    def _reset_shot(self) -> None:
        self.shot_in_progress = False
        self.shot_type = ShotType.NONE
        self.shot_start_frame = None
        self.shot_deadline_frame = None
        self.frames_since_start = 0
        self.consecutive_jump_shot_frames = 0
        self.consecutive_layup_frames = 0
        self.consecutive_ball_in_basket_frames = 0

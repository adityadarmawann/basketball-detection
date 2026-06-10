"""
Statistics calculator for basketball analytics — Smart Vision Campus League.

Accumulates box-score stats from event_engine output and computes physical
performance metrics (MPI) from tracking / pose data.

Call update() every N frames or on each new event batch.
Call get_live_stats() for the full output structure.
"""

import logging
import math
import os
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FPS_DEFAULT     = 30
SMOOTH_WINDOW   = 5          # frames for speed low-pass
JUMP_PX_TO_CM   = 0.08       # approx pixel→cm (camera/court specific)
MIN_SPEED_KMH   = 0.1        # noise floor — speeds below this are ignored
DIR_CHANGE_DEG  = 45.0       # degrees threshold for direction-change count
MAX_POS_HISTORY = 2000       # position samples kept per player

MPI_WEIGHTS = {
    "power":      0.25,
    "agility":    0.20,
    "endurance":  0.20,
    "efficiency": 0.20,
    "cognitive":  0.15,
}

MVP_EFF_W = 0.6
MVP_MPI_W = 0.4


# ── Box-score helpers (module-level so smoke test can access them) ─────────────

def _empty_stats() -> dict:
    return {
        "pts": 0, "fgm": 0, "fga": 0,
        "three_pm": 0, "three_pa": 0,
        "ftm": 0, "fta": 0,
        "oreb": 0, "dreb": 0, "reb": 0,
        "ast": 0, "stl": 0, "blk": 0,
        "tov": 0, "pf": 0,
        "plus_minus": 0, "min": 0.0,
        "fg_pct": 0.0, "three_pct": 0.0, "eff": 0.0,
    }


def _compute_derived(s: dict) -> None:
    """In-place: FG%, 3P%, EFF from accumulated counters."""
    s["fg_pct"]    = round(s["fgm"] / s["fga"],         3) if s["fga"]    > 0 else 0.0
    s["three_pct"] = round(s["three_pm"] / s["three_pa"], 3) if s["three_pa"] > 0 else 0.0
    s["eff"]       = float(
        (s["pts"] + s["reb"] + s["ast"] + s["stl"] + s["blk"])
        - (s["fga"] - s["fgm"] + s["tov"])
    )


# ── Main class ────────────────────────────────────────────────────────────────

class StatsCalculator:
    """
    Per-player, per-quarter box score + MPI metrics + MVP ranking.

    tracking_snapshot format (one frame):
        {track_id: (x_meters, y_meters, timestamp_ms)}

    pose_snapshot format: {"poses": [{track_id, jump_height_px, ...}]}

    roster format: {track_id: {name, team, jersey_number}}
    """

    def __init__(self, fps: int = FPS_DEFAULT, jump_px_to_cm: float = JUMP_PX_TO_CM):
        self.fps           = fps
        self.jump_px_to_cm = jump_px_to_cm
        self.current_quarter = 1

        self._roster:  dict = {}   # track_id → {name, team, jersey_number}
        self._all_tids: set = set()

        # Box score: track_id → {quarter → stats_dict}
        self._qstats:  dict = {}

        # ── Tracking state ──
        self._prev_pos:     dict = {}   # track_id → (x, y, ts_ms)
        self._smooth_buf:   dict = {}   # track_id → deque(maxlen=SMOOTH_WINDOW)
        self._total_dist:   dict = {}   # track_id → total meters
        self._all_speeds:   dict = {}   # track_id → deque of km/h values
        self._speed_ms_buf: dict = {}   # track_id → deque of m/s values (for accel)
        self._positions:    dict = {}   # track_id → deque of (x, y, ts)
        self._playing_frames: dict = {} # track_id → int
        self._quarter_speeds: dict = {} # track_id → {quarter → list[float]}
        self._quarter_dist:   dict = {} # track_id → {quarter → float}

        # Jump heights: track_id → list[float px]
        self._jump_px: dict = {}

        # Possession tracking (from POSSESSION_CHANGE events)
        self._possession_frames: dict = {"A": 0, "B": 0}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        events:            Optional[list] = None,
        tracking_snapshot: Optional[dict] = None,
        pose_snapshot:     Optional[dict] = None,
        roster:            Optional[dict] = None,
    ) -> dict:
        """
        Process new events and one frame of tracking/pose data.
        Returns get_live_stats() output.
        """
        if roster:
            self._roster.update(roster)
            self._all_tids.update(roster.keys())

        for event in (events or []):
            self._update_box_score(event)

        if tracking_snapshot:
            self._update_tracking_stats(tracking_snapshot)

        if pose_snapshot:
            self._update_pose_stats(pose_snapshot)

        return self.get_live_stats()

    def get_live_stats(self) -> dict:
        """Build and return full output structure."""
        total_all   = {tid: self._get_total_stats(tid) for tid in self._all_tids}
        raw_mpi_all = {tid: self._calc_mpi_raw(tid)     for tid in self._all_tids}

        norm = self._build_norm_ranges(total_all, raw_mpi_all)

        player_stats = {}
        for tid in self._all_tids:
            info = self._roster.get(tid, {})
            ts   = total_all[tid]
            rm   = raw_mpi_all[tid]
            mpi  = self._apply_normalization(tid, ts, rm, norm)

            player_stats[tid] = {
                "name":          info.get("name", f"Player_{tid}"),
                "jersey_number": info.get("jersey_number"),
                "team":          info.get("team", ""),
                "quarter_stats": self._get_quarter_stats_all(tid),
                "total_stats":   ts,
                "mpi":           mpi,
            }

        return {
            "player_stats":   player_stats,
            "team_stats":     self._compute_team_stats(total_all),
            "mvp_ranking":    self._rank_players(total_all, player_stats),
            "possession_pct": self._compute_possession_pct(),
        }

    def get_player_stats(self, track_id: int) -> Optional[dict]:
        return self.get_live_stats()["player_stats"].get(track_id)

    def get_mvp_ranking(self) -> list:
        return self.get_live_stats()["mvp_ranking"]

    def reset_quarter(self, quarter: int) -> None:
        """Advance the current quarter (per-quarter state is lazy-initialised)."""
        self.current_quarter = quarter
        logger.info("Quarter %d started", quarter)

    def reset_match(self) -> None:
        self.__init__(fps=self.fps, jump_px_to_cm=self.jump_px_to_cm)

    # ── Box-score accumulation ────────────────────────────────────────────────

    def _update_box_score(self, event: dict) -> None:
        etype   = str(event.get("type", "")).upper()
        tid     = event.get("track_id")
        quarter = int(event.get("quarter", self.current_quarter))
        # Fallback: if the event carries no team (OCR not yet confirmed), try the
        # roster cache — team may have been registered by an earlier event for this tid.
        team    = event.get("team", "") or self._roster.get(tid, {}).get("team", "")

        # Handle possession-change before tid guard (tid may be None for these)
        if etype == "POSSESSION_CHANGE":
            poss_team = event.get("possession_team") or team
            if poss_team in ("A", "B"):
                self._possession_frames[poss_team] += 1
            return

        if tid is None:
            return

        self._all_tids.add(tid)
        s = self._get_qstats_d(tid, quarter)

        # Auto-register team from event so _compute_team_stats works even without OCR.
        # Once any scoring/stat event carries team info, that mapping is retained.
        if team in ("A", "B"):
            rec = self._roster.setdefault(tid, {})
            if not rec.get("team"):
                rec["team"] = team
                rec.setdefault("name", f"Player_{tid}")

        is_three = bool(event.get("is_three", etype in ("MADE_3", "MISSED_3")))

        if etype in ("MADE_FG", "MADE_3", "SHOT_MADE", "MADE_SHOT"):
            s["fgm"] += 1
            s["fga"] += 1
            pts = 3 if is_three else 2
            s["pts"] += pts
            if is_three:
                s["three_pm"] += 1
                s["three_pa"] += 1
            if team:
                self._apply_plus_minus(quarter, team, pts)
            logger.info(
                "STATS_SCORED  +%dpts  tid=%s  team=%s  Q%d  total_pts=%d",
                pts, tid, team or "?", quarter, s["pts"],
            )

        elif etype in ("MISSED_FG", "MISSED_3", "SHOT_MISSED", "MISSED_SHOT"):
            s["fga"] += 1
            if is_three:
                s["three_pa"] += 1

        elif etype in ("MADE_FT", "FT_MADE"):
            s["ftm"] += 1
            s["fta"] += 1
            s["pts"] += 1
            if team:
                self._apply_plus_minus(quarter, team, 1)
            logger.info(
                "STATS_SCORED  +1pt(FT)  tid=%s  team=%s  Q%d  total_pts=%d",
                tid, team or "?", quarter, s["pts"],
            )

        elif etype in ("MISSED_FT", "FT_MISSED"):
            s["fta"] += 1

        elif etype in ("OREB", "REBOUND_OFF"):
            s["oreb"] += 1; s["reb"] += 1

        elif etype in ("DREB", "REBOUND_DEF"):
            s["dreb"] += 1; s["reb"] += 1

        elif etype == "REBOUND":
            sub = str(event.get("sub_type", "")).upper()
            if sub == "OFF":
                s["oreb"] += 1
            else:
                s["dreb"] += 1
            s["reb"] += 1

        elif etype in ("ASSIST", "AST"):
            s["ast"] += 1

        elif etype in ("STEAL", "STL"):
            s["stl"] += 1

        elif etype in ("BLOCK", "BLK"):
            s["blk"] += 1

        elif etype in ("TURNOVER", "TOV"):
            s["tov"] += 1

        elif etype in ("PERSONAL_FOUL", "FOUL", "PF"):
            s["pf"] += 1

    def _apply_plus_minus(self, quarter: int, scoring_team: str, points: int) -> None:
        for tid, info in self._roster.items():
            team = info.get("team", "")
            if not team:
                continue
            pm = self._get_qstats_d(tid, quarter)
            if team == scoring_team:
                pm["plus_minus"] += points
            else:
                pm["plus_minus"] -= points

    # ── Tracking stats ────────────────────────────────────────────────────────

    def _update_tracking_stats(self, snapshot: dict) -> None:
        """
        Process one frame of court positions.
        snapshot: {track_id: (x_m, y_m, ts_ms)}
        """
        q = self.current_quarter

        for tid, pos in snapshot.items():
            if pos is None or len(pos) < 3:
                continue

            x, y, ts = float(pos[0]), float(pos[1]), float(pos[2])
            self._all_tids.add(tid)
            self._init_tracking_buffers(tid)

            self._positions[tid].append((x, y, ts))
            self._playing_frames[tid] += 1

            if tid in self._prev_pos:
                px, py, pts_ms = self._prev_pos[tid]
                dt_ms = ts - pts_ms
                if dt_ms > 0:
                    dist_m    = math.hypot(x - px, y - py)
                    dt_s      = dt_ms / 1000.0
                    speed_ms  = dist_m / dt_s
                    speed_kmh = speed_ms * 3.6

                    # Low-pass smooth
                    self._smooth_buf[tid].append(speed_kmh)
                    smoothed = sum(self._smooth_buf[tid]) / len(self._smooth_buf[tid])

                    if smoothed >= MIN_SPEED_KMH:
                        self._all_speeds[tid].append(smoothed)
                        self._speed_ms_buf[tid].append(speed_ms)
                        self._total_dist[tid] += dist_m
                        self._quarter_speeds[tid][q].append(smoothed)
                        self._quarter_dist[tid][q] += dist_m

            self._prev_pos[tid] = (x, y, ts)

    def _init_tracking_buffers(self, tid: int) -> None:
        if tid not in self._smooth_buf:
            self._smooth_buf[tid]   = deque(maxlen=SMOOTH_WINDOW)
            self._all_speeds[tid]   = deque(maxlen=MAX_POS_HISTORY)
            self._speed_ms_buf[tid] = deque(maxlen=MAX_POS_HISTORY)
            self._positions[tid]    = deque(maxlen=MAX_POS_HISTORY)
            self._total_dist[tid]   = 0.0
            self._playing_frames[tid]  = 0
            self._quarter_speeds[tid]  = defaultdict(list)
            self._quarter_dist[tid]    = defaultdict(float)

    def _update_pose_stats(self, pose_snapshot: dict) -> None:
        for pose in pose_snapshot.get("poses", []):
            tid     = pose.get("track_id")
            jump_px = float(pose.get("jump_height_px", 0.0))
            if tid is None or jump_px <= 0:
                continue
            self._all_tids.add(tid)
            if tid not in self._jump_px:
                self._jump_px[tid] = []
            self._jump_px[tid].append(jump_px)

    # ── MPI computation ───────────────────────────────────────────────────────

    def _calc_mpi_raw(self, tid: int) -> dict:
        """Raw (un-normalised) MPI metrics for one player."""
        speeds = list(self._all_speeds.get(tid, deque()))
        dist   = self._total_dist.get(tid, 0.0)

        avg_spd = (sum(speeds) / len(speeds)) if speeds else 0.0
        max_spd = max(speeds, default=0.0)

        jump_samples = self._jump_px.get(tid, [])
        jump_cm = (sum(jump_samples) / len(jump_samples) * self.jump_px_to_cm
                   if jump_samples else 0.0)

        positions   = list(self._positions.get(tid, deque()))
        dir_changes = self._calc_direction_changes(positions)
        play_min    = self._playing_frames.get(tid, 0) / (self.fps * 60)
        dir_per_min = dir_changes / play_min if play_min > 0 else 0.0

        endurance_raw = (avg_spd / max_spd * 100.0) if max_spd > 0 else 0.0
        fatigue       = self._calc_fatigue(tid)
        accel         = self._calc_acceleration(tid)

        return {
            "distance_m":       round(dist, 3),
            "avg_speed_kmh":    round(avg_spd, 3),
            "max_speed_kmh":    round(max_spd, 3),
            "jump_height_cm":   round(jump_cm, 3),
            "acceleration_ms2": round(accel, 3),
            "agility_raw":      round(dir_per_min, 3),
            "endurance_raw":    round(endurance_raw, 3),
            "fatigue_score":    round(fatigue, 3),
        }

    def _apply_normalization(
        self, tid: int, total_stats: dict, raw_mpi: dict, norm: dict
    ) -> dict:
        N = self._normalize

        jump   = N(raw_mpi["jump_height_cm"], *norm["jump_height_cm"])
        mspeed = N(raw_mpi["max_speed_kmh"],  *norm["max_speed_kmh"])
        power  = (jump + mspeed) / 2.0

        agility   = N(raw_mpi["agility_raw"],   *norm["agility_raw"])
        endurance = N(raw_mpi["endurance_raw"], *norm["endurance_raw"])
        efficiency = N(total_stats["eff"],      *norm["eff"])
        cognitive  = N(
            total_stats["ast"] + total_stats["stl"] - total_stats["tov"],
            *norm["cognitive"],
        )

        mpi_composite = self._mpi_composite(power, agility, endurance, efficiency, cognitive)

        play_min = self._playing_frames.get(tid, 0) / (self.fps * 60)
        total_stats["min"] = round(play_min, 2)

        return {
            "distance_m":       raw_mpi["distance_m"],
            "avg_speed_kmh":    raw_mpi["avg_speed_kmh"],
            "max_speed_kmh":    raw_mpi["max_speed_kmh"],
            "jump_height_cm":   raw_mpi["jump_height_cm"],
            "acceleration_ms2": raw_mpi["acceleration_ms2"],
            "agility_score":    round(agility, 2),
            "endurance_score":  round(endurance, 2),
            "fatigue_score":    raw_mpi["fatigue_score"],
            "mpi_composite":    round(mpi_composite, 2),
        }

    def _build_norm_ranges(self, total_all: dict, raw_all: dict) -> dict:
        """Compute (min, max) for each metric across all active players."""

        def r(vals: list) -> tuple:
            if not vals:
                return (0.0, 1.0)
            mn, mx = min(vals), max(vals)
            return (mn, mx) if mn != mx else (mn - 1.0, mn + 1.0)

        raw_metrics = [
            "jump_height_cm", "max_speed_kmh", "avg_speed_kmh",
            "agility_raw", "endurance_raw", "distance_m", "acceleration_ms2",
        ]
        norm = {m: r([raw_all[tid].get(m, 0.0) for tid in raw_all])
                for m in raw_metrics}

        norm["eff"]       = r([total_all[tid]["eff"] for tid in total_all])
        norm["cognitive"] = r([
            total_all[tid]["ast"] + total_all[tid]["stl"] - total_all[tid]["tov"]
            for tid in total_all
        ])
        return norm

    # ── Stats aggregation ─────────────────────────────────────────────────────

    def _get_total_stats(self, tid: int) -> dict:
        total = _empty_stats()
        for qs in self._qstats.get(tid, {}).values():
            for key in ("pts", "fgm", "fga", "three_pm", "three_pa",
                        "ftm", "fta", "oreb", "dreb", "reb",
                        "ast", "stl", "blk", "tov", "pf", "plus_minus"):
                total[key] += qs.get(key, 0)
        _compute_derived(total)
        return total

    def _get_quarter_stats_all(self, tid: int) -> dict:
        result = {}
        for q, qs in self._qstats.get(tid, {}).items():
            out = dict(qs)
            _compute_derived(out)
            result[q] = out
        return result

    def _compute_team_stats(self, total_all: dict) -> dict:
        teams = {
            "A": {"pts": 0, "reb": 0, "ast": 0, "stl": 0,
                  "blk": 0, "tov": 0, "fgm": 0, "fga": 0, "fg_pct": 0.0},
            "B": {"pts": 0, "reb": 0, "ast": 0, "stl": 0,
                  "blk": 0, "tov": 0, "fgm": 0, "fga": 0, "fg_pct": 0.0},
        }
        for tid, ts in total_all.items():
            team = self._roster.get(tid, {}).get("team", "")
            if team not in teams:
                continue
            t = teams[team]
            for k in ("pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga"):
                t[k] += ts.get(k, 0)
        for t in teams.values():
            t["fg_pct"] = round(t["fgm"] / t["fga"], 3) if t["fga"] > 0 else 0.0
        return teams

    def _rank_players(self, total_all: dict, player_stats: dict) -> list:
        ranked = []
        for tid, ts in total_all.items():
            mpi_c = player_stats[tid]["mpi"]["mpi_composite"]
            eff   = ts["eff"]
            score = MVP_EFF_W * eff + MVP_MPI_W * mpi_c
            ranked.append({
                "rank":     0,
                "track_id": tid,
                "name":     player_stats[tid]["name"],
                "eff":      round(eff, 2),
                "mpi":      round(mpi_c, 2),
                "score":    round(score, 3),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
        return ranked

    def _compute_possession_pct(self) -> dict:
        total = sum(self._possession_frames.values())
        if total == 0:
            return {"A": 50.0, "B": 50.0}
        return {team: round(v / total * 100, 1)
                for team, v in self._possession_frames.items()}

    # ── Physical helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize(value: float, min_v: float, max_v: float) -> float:
        """
        Min-max normalise to [0, 100].
        Returns 50.0 when min_v == max_v (all-equal field).
        Clamps values outside [min_v, max_v].
        """
        if min_v == max_v:
            return 50.0
        norm = (value - min_v) / (max_v - min_v) * 100.0
        return float(max(0.0, min(100.0, norm)))

    @staticmethod
    def _calc_direction_changes(positions: list) -> int:
        """
        Count how many times the velocity vector rotates > DIR_CHANGE_DEG degrees.
        positions: list of (x, y, ...) tuples in court metres.
        Skips zero-velocity segments (standing still).
        """
        if len(positions) < 3:
            return 0
        changes = 0
        for i in range(1, len(positions) - 1):
            x0, y0 = positions[i - 1][:2]
            x1, y1 = positions[i][:2]
            x2, y2 = positions[i + 1][:2]
            v1 = (x1 - x0, y1 - y0)
            v2 = (x2 - x1, y2 - y1)
            m1 = math.hypot(*v1)
            m2 = math.hypot(*v2)
            if m1 < 1e-6 or m2 < 1e-6:
                continue
            cos_t = (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)
            cos_t = max(-1.0, min(1.0, cos_t))
            if math.degrees(math.acos(cos_t)) > DIR_CHANGE_DEG:
                changes += 1
        return changes

    def _calc_fatigue(self, tid: int) -> float:
        """
        Speed decline from Q1 to Q4, expressed as 0–100.
        Higher = more fatigued.  Returns 0 if Q1 or Q4 data absent.
        """
        qs  = self._quarter_speeds.get(tid, {})
        q1  = qs.get(1, [])
        q4  = qs.get(4, [])
        if not q1 or not q4:
            return 0.0
        q1_avg = sum(q1) / len(q1)
        q4_avg = sum(q4) / len(q4)
        if q1_avg < 0.01:
            return 0.0
        return round(max(0.0, (q1_avg - q4_avg) / q1_avg * 100.0), 2)

    def _calc_acceleration(self, tid: int) -> float:
        """Mean absolute acceleration in m/s² from raw speed history."""
        buf = list(self._speed_ms_buf.get(tid, deque()))
        if len(buf) < 2:
            return 0.0
        dt     = 1.0 / self.fps
        accels = [abs(buf[i] - buf[i - 1]) / dt for i in range(1, len(buf))]
        return round(sum(accels) / len(accels), 3) if accels else 0.0

    @staticmethod
    def _mpi_composite(
        power: float, agility: float, endurance: float,
        efficiency: float, cognitive: float,
    ) -> float:
        """
        MPI composite from 0–100 normalised components.
        Weights: power=0.25, agility=0.20, endurance=0.20,
                 efficiency=0.20, cognitive=0.15  (sum=1.00)
        """
        return (
            MPI_WEIGHTS["power"]      * power      +
            MPI_WEIGHTS["agility"]    * agility     +
            MPI_WEIGHTS["endurance"]  * endurance   +
            MPI_WEIGHTS["efficiency"] * efficiency  +
            MPI_WEIGHTS["cognitive"]  * cognitive
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_qstats_d(self, tid: int, quarter: int) -> dict:
        """Return (creating if absent) the stats dict for track_id × quarter."""
        if tid not in self._qstats:
            self._qstats[tid] = {}
        if quarter not in self._qstats[tid]:
            self._qstats[tid][quarter] = _empty_stats()
        return self._qstats[tid][quarter]


# ── Factory ───────────────────────────────────────────────────────────────────

def create_stats_calculator(fps: int = FPS_DEFAULT) -> StatsCalculator:
    return StatsCalculator(fps=fps)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    print("=== StatsCalculator smoke test ===\n")

    sc = StatsCalculator()

    # ── 1. _normalize() ───────────────────────────────────────────────────────
    print("--- 1. _normalize() ---")
    cases = [
        (50,   0,  100,  50.0,  "midpoint"),
        (0,    0,  100,   0.0,  "at min"),
        (100,  0,  100, 100.0,  "at max"),
        (-10,  0,  100,   0.0,  "below min → clamp 0"),
        (110,  0,  100, 100.0,  "above max → clamp 100"),
        (5,    5,    5,  50.0,  "min==max → 50"),
        (0,  -10,   10,  50.0,  "zero in negative range"),
        (25,   0,  100,  25.0,  "quarter point"),
    ]
    for val, mn, mx, expected, label in cases:
        got = sc._normalize(val, mn, mx)
        ok  = abs(got - expected) < 0.001
        print(f"  normalize({val},{mn},{mx}) = {got:.1f}  [{label}]  {'✓' if ok else '✗ FAIL'}")
        assert ok, f"Expected {expected}, got {got}"

    # ── 2. _calc_direction_changes() ──────────────────────────────────────────
    print("\n--- 2. _calc_direction_changes() ---")

    # Straight path → 0 changes
    straight = [(i, 0, i * 100) for i in range(6)]
    dc_str = sc._calc_direction_changes(straight)
    assert dc_str == 0, f"Straight path: expected 0, got {dc_str}"
    print(f"  Straight path (6 pts)  → {dc_str} changes ✓")

    # U-turn path: (0,0)→(1,0)→(1,1)→(0,1)→(0,0) — each turn ~90°
    zigzag = [
        (0, 0, 0), (2, 0, 100), (2, 2, 200),
        (0, 2, 300), (0, 0, 400),
    ]
    dc_zz = sc._calc_direction_changes(zigzag)
    assert dc_zz >= 2, f"Zigzag: expected ≥2, got {dc_zz}"
    print(f"  Square-loop zigzag     → {dc_zz} changes ✓")

    # 180° reversal
    reversal = [(0, 0, 0), (3, 0, 100), (0, 0, 200)]
    dc_rev = sc._calc_direction_changes(reversal)
    assert dc_rev == 1, f"Reversal: expected 1, got {dc_rev}"
    print(f"  180° reversal          → {dc_rev} change  ✓")

    # Stationary segments → skipped
    stationary = [(0, 0, 0), (0, 0, 100), (0, 0, 200), (1, 0, 300)]
    dc_stat = sc._calc_direction_changes(stationary)
    assert dc_stat == 0, f"Stationary+move: expected 0, got {dc_stat}"
    print(f"  Stationary then move   → {dc_stat} changes ✓")

    # Short list
    assert sc._calc_direction_changes([]) == 0
    assert sc._calc_direction_changes([(0, 0)]) == 0
    assert sc._calc_direction_changes([(0, 0), (1, 0)]) == 0
    print("  len<3 → 0 ✓")

    # ── 3. EFF formula ────────────────────────────────────────────────────────
    print("\n--- 3. EFF formula ---")
    s = _empty_stats()
    s["pts"] = 20; s["reb"] = 5; s["ast"] = 3; s["stl"] = 2; s["blk"] = 1
    s["fga"] = 15; s["fgm"] = 8; s["three_pm"] = 2; s["three_pa"] = 5; s["tov"] = 3
    _compute_derived(s)
    # EFF = (20+5+3+2+1) - (15-8+3) = 31 - 10 = 21
    assert s["eff"] == 21.0, f"Expected EFF=21, got {s['eff']}"
    assert s["fg_pct"]    == round(8 / 15, 3)
    assert s["three_pct"] == round(2 / 5, 3)
    print(f"  EFF = {s['eff']}  (expected 21)  ✓")
    print(f"  FG% = {s['fg_pct']}  (expected {round(8/15,3)})  ✓")
    print(f"  3P% = {s['three_pct']}  (expected {round(2/5,3)})  ✓")

    # Zero-attempts edge cases
    s0 = _empty_stats()
    _compute_derived(s0)
    assert s0["fg_pct"] == 0.0 and s0["three_pct"] == 0.0 and s0["eff"] == 0.0
    print("  All-zero stats → FG%=0, 3P%=0, EFF=0  ✓")

    # Negative EFF (many misses, turnovers)
    s_neg = _empty_stats()
    s_neg["pts"] = 2; s_neg["fga"] = 10; s_neg["fgm"] = 1; s_neg["tov"] = 5
    _compute_derived(s_neg)
    # EFF = (2+0+0+0+0) - (10-1+5) = 2 - 14 = -12
    assert s_neg["eff"] == -12.0
    print(f"  Negative EFF = {s_neg['eff']}  (expected -12)  ✓")

    # ── 4. MPI composite formula ──────────────────────────────────────────────
    print("\n--- 4. MPI composite ---")
    # power=80, agility=60, endurance=70, efficiency=75, cognitive=50
    # = 0.25*80 + 0.20*60 + 0.20*70 + 0.20*75 + 0.15*50
    # = 20 + 12 + 14 + 15 + 7.5 = 68.5
    expected_mpi = 0.25 * 80 + 0.20 * 60 + 0.20 * 70 + 0.20 * 75 + 0.15 * 50
    got_mpi = StatsCalculator._mpi_composite(80, 60, 70, 75, 50)
    assert abs(got_mpi - expected_mpi) < 0.001, f"Expected {expected_mpi}, got {got_mpi}"
    print(f"  _mpi_composite(80,60,70,75,50) = {got_mpi}  (expected {expected_mpi})  ✓")

    # All zero → 0
    assert StatsCalculator._mpi_composite(0, 0, 0, 0, 0) == 0.0
    print("  All-zero → 0  ✓")

    # All 100 → 100 (weights sum to 1.0)
    assert abs(StatsCalculator._mpi_composite(100, 100, 100, 100, 100) - 100.0) < 0.001
    print("  All-100 → 100 (weights sum to 1)  ✓")

    assert abs(sum(MPI_WEIGHTS.values()) - 1.0) < 1e-9
    print(f"  MPI_WEIGHTS sum = {sum(MPI_WEIGHTS.values())}  ✓")

    # ── 5. MVP ranking sort ───────────────────────────────────────────────────
    print("\n--- 5. MVP ranking ---")
    sc5 = StatsCalculator()
    roster5 = {
        1: {"name": "Alpha", "team": "A", "jersey_number": 10},
        2: {"name": "Beta",  "team": "B", "jersey_number":  5},
        3: {"name": "Gamma", "team": "A", "jersey_number": 23},
    }

    # Player 1 (Alpha): 10 FGM, 3 AST → EFF = 20+3-0 = 23
    for _ in range(10):
        sc5._update_box_score(
            {"type": "MADE_FG", "track_id": 1, "quarter": 1, "team": "A", "is_three": False}
        )
    for _ in range(3):
        sc5._update_box_score({"type": "ASSIST", "track_id": 1, "quarter": 1, "team": "A"})

    # Player 2 (Beta): 5 FGM → EFF = 10
    for _ in range(5):
        sc5._update_box_score(
            {"type": "MADE_FG", "track_id": 2, "quarter": 1, "team": "B", "is_three": False}
        )

    # Player 3 (Gamma): 0 FGM, 3 misses, 2 TOV → EFF = -(3+2) = -5
    for _ in range(3):
        sc5._update_box_score(
            {"type": "MISSED_FG", "track_id": 3, "quarter": 1, "team": "A", "is_three": False}
        )
    for _ in range(2):
        sc5._update_box_score({"type": "TURNOVER", "track_id": 3, "quarter": 1, "team": "A"})

    sc5._roster.update(roster5)
    sc5._all_tids.update(roster5.keys())

    ranking = sc5.get_mvp_ranking()
    assert len(ranking) == 3, f"Expected 3 players, got {len(ranking)}"

    # Verify sorted descending by score
    for i in range(len(ranking) - 1):
        assert ranking[i]["score"] >= ranking[i + 1]["score"], (
            f"Not sorted: {ranking[i]['score']} < {ranking[i+1]['score']}"
        )

    # Alpha should be rank 1 (highest EFF=23 dominates since no tracking data)
    assert ranking[0]["track_id"] == 1, (
        f"Expected Alpha (tid=1) at rank 1, got {ranking[0]['name']}"
    )
    assert ranking[0]["rank"] == 1
    assert ranking[1]["track_id"] == 2
    assert ranking[2]["track_id"] == 3

    print(f"  Rank 1: {ranking[0]['name']}  EFF={ranking[0]['eff']}  score={ranking[0]['score']}  ✓")
    print(f"  Rank 2: {ranking[1]['name']}  EFF={ranking[1]['eff']}  score={ranking[1]['score']}  ✓")
    print(f"  Rank 3: {ranking[2]['name']}  EFF={ranking[2]['eff']}  score={ranking[2]['score']}  ✓")

    # Verify EFF values
    assert ranking[0]["eff"] == 23.0
    assert ranking[1]["eff"] == 10.0
    assert ranking[2]["eff"] == -5.0
    print("  EFF values match ✓")

    # ── 6. update() with empty / None inputs ──────────────────────────────────
    print("\n--- 6. update() edge cases ---")
    sc6 = StatsCalculator()

    out = sc6.update([], {}, {}, {})
    assert "player_stats"   in out
    assert "team_stats"     in out
    assert "mvp_ranking"    in out
    assert "possession_pct" in out
    assert out["mvp_ranking"] == []
    assert out["possession_pct"] == {"A": 50.0, "B": 50.0}
    print("  update([],{},{},{}) → valid empty structure  ✓")

    out_none = sc6.update(None, None, None, None)
    assert out_none["mvp_ranking"] == []
    print("  update(None*4) → valid empty structure  ✓")

    # ── 7. Distance & speed accumulation ─────────────────────────────────────
    print("\n--- 7. Distance & speed from tracking ---")
    sc7 = StatsCalculator(fps=10)
    roster7 = {1: {"name": "Runner", "team": "A", "jersey_number": 7}}

    # Frame 0: position (0, 0) at t=0
    sc7.update([], {1: (0.0, 0.0, 0)},    {}, roster7)
    # Frame 1: position (3, 4) at t=1000ms → 5m in 1s = 18 km/h
    sc7.update([], {1: (3.0, 4.0, 1000)}, {}, {})

    ps7 = sc7.get_player_stats(1)
    mpi7 = ps7["mpi"]
    assert abs(mpi7["distance_m"] - 5.0) < 0.01, f"Expected 5m, got {mpi7['distance_m']}"
    assert abs(mpi7["max_speed_kmh"] - 18.0) < 0.5, f"Expected ~18 kmh, got {mpi7['max_speed_kmh']}"
    print(f"  distance = {mpi7['distance_m']} m  (expected 5.0)  ✓")
    print(f"  max_speed = {mpi7['max_speed_kmh']} km/h  (expected 18.0)  ✓")

    # Frame 2: no movement → distance unchanged
    sc7.update([], {1: (3.0, 4.0, 2000)}, {}, {})
    ps7b = sc7.get_player_stats(1)
    assert abs(ps7b["mpi"]["distance_m"] - 5.0) < 0.01
    print("  No-move frame → distance unchanged  ✓")

    # ── 8. _calc_fatigue() ───────────────────────────────────────────────────
    print("\n--- 8. _calc_fatigue() ---")
    sc8 = StatsCalculator()
    sc8._quarter_speeds[1] = defaultdict(list)
    sc8._quarter_speeds[1][1] = [20.0, 22.0, 21.0]   # Q1 avg = 21
    sc8._quarter_speeds[1][4] = [12.0, 10.0, 11.0]   # Q4 avg = 11

    fatigue = sc8._calc_fatigue(1)
    # decline = (21 - 11) / 21 × 100 ≈ 47.6
    expected_fatigue = (21 - 11) / 21 * 100
    assert abs(fatigue - expected_fatigue) < 0.5, f"Expected ~{expected_fatigue:.1f}, got {fatigue}"
    print(f"  fatigue = {fatigue:.2f}%  (expected ~{expected_fatigue:.1f}%)  ✓")

    sc8._quarter_speeds[2] = {1: [10.0], 4: []}
    assert sc8._calc_fatigue(2) == 0.0   # no Q4 data
    print("  Missing Q4 → 0  ✓")

    # ── 9. Acceleration ──────────────────────────────────────────────────────
    print("\n--- 9. _calc_acceleration() ---")
    sc9 = StatsCalculator(fps=10)   # dt = 0.1s
    sc9._speed_ms_buf[1] = deque([0.0, 5.0, 10.0])
    accel9 = sc9._calc_acceleration(1)
    # |5-0|/0.1 = 50, |10-5|/0.1 = 50 → avg = 50
    assert abs(accel9 - 50.0) < 0.01, f"Expected 50.0, got {accel9}"
    print(f"  accel = {accel9} m/s²  (expected 50.0)  ✓")

    sc9._speed_ms_buf[2] = deque([5.0])   # only 1 value → 0
    assert sc9._calc_acceleration(2) == 0.0
    print("  Single speed sample → 0  ✓")

    # ── 10. reset_match() ────────────────────────────────────────────────────
    print("\n--- 10. reset_match() ---")
    sc5.reset_match()
    assert len(sc5._all_tids) == 0
    assert sc5._qstats == {}
    assert sc5.get_mvp_ranking() == []
    print("  State cleared after reset_match()  ✓")

    # ── 11. Possession percentage ─────────────────────────────────────────────
    print("\n--- 11. possession_pct ---")
    sc11 = StatsCalculator()
    for _ in range(3):
        sc11._update_box_score(
            {"type": "POSSESSION_CHANGE", "track_id": None, "possession_team": "A", "quarter": 1}
        )
    for _ in range(1):
        sc11._update_box_score(
            {"type": "POSSESSION_CHANGE", "track_id": None, "possession_team": "B", "quarter": 1}
        )
    poss11 = sc11._compute_possession_pct()
    assert poss11["A"] == 75.0, f"Expected 75%, got {poss11['A']}"
    assert poss11["B"] == 25.0
    print(f"  A={poss11['A']}%  B={poss11['B']}%  (3:1 ratio)  ✓")

    # ── 12. team_stats aggregation ───────────────────────────────────────────
    print("\n--- 12. team_stats ---")
    sc12 = StatsCalculator()
    sc12._roster = {
        1: {"name": "P1", "team": "A", "jersey_number": 1},
        2: {"name": "P2", "team": "A", "jersey_number": 2},
        3: {"name": "P3", "team": "B", "jersey_number": 3},
    }
    sc12._all_tids = {1, 2, 3}

    for _ in range(4):
        sc12._update_box_score({"type": "MADE_FG", "track_id": 1, "quarter": 1, "team": "A", "is_three": False})
    for _ in range(2):
        sc12._update_box_score({"type": "MADE_FG", "track_id": 2, "quarter": 1, "team": "A", "is_three": False})
    for _ in range(3):
        sc12._update_box_score({"type": "MADE_FG", "track_id": 3, "quarter": 1, "team": "B", "is_three": False})

    out12 = sc12.get_live_stats()
    ts = out12["team_stats"]
    assert ts["A"]["pts"] == 12, f"Expected 12, got {ts['A']['pts']}"   # (4+2)*2
    assert ts["B"]["pts"] == 6,  f"Expected 6, got {ts['B']['pts']}"    # 3*2
    assert ts["A"]["fg_pct"] == round(6 / 6, 3)
    print(f"  Team A: {ts['A']['pts']} pts  FG%={ts['A']['fg_pct']}  ✓")
    print(f"  Team B: {ts['B']['pts']} pts  FG%={ts['B']['fg_pct']}  ✓")

    print("\n=== All tests passed ===")

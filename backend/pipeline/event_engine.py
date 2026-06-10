"""
Event detection engine — Smart Vision Campus League basketball analytics.

Detects game events from per-frame pipeline data:
  MADE_FG / MISSED_FG   field goal made / missed
  REBOUND               offensive (OFF) or defensive (DEF)
  AST                   assist (pass → FGM within window)
  STL / TOV             steal and turnover
  BLK                   block
  FOUL                  personal foul candidate
  POSSESSION_CHANGE     team possession change

Returns a *list* of event dicts so video_processor can do:
    frame_data["events"] = engine.process(frame_data) or []
Score and possession state are also accessible as instance attributes.
"""

import logging
import math
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# ── FIBA constants (mirror court.py — avoid circular import) ──────────────────
COURT_W              = 28.0
COURT_H              = 15.0
HOOP_LEFT            = [1.575,  7.5]    # FIBA: backboard inner face 1.2m + ring 0.375m
HOOP_RIGHT           = [26.425, 7.5]   # 28.0 - 1.575 = 26.425
THREE_PT_RADIUS      = 6.75
THREE_PT_STRAIGHT_X  = 2.99    # x where corner straight meets arc (left basket)

# ── Tunable thresholds ────────────────────────────────────────────────────────
POSSESSION_DIST_PX     = 80     # pixels  — ball "owned" if player inside this
POSSESSION_DIST_M      = 1.2    # meters
POSSESSION_CONFIRM_F   = 3      # consecutive frames to confirm new possession

FG_HOOP_RADIUS_PX      = 60     # pixels  — "ball in hoop zone"
FG_HOOP_RADIUS_M       = 0.9    # meters
FG_COOLDOWN_F          = 45     # frames before another FG can fire
FG_ATTEMPT_WINDOW_F    = 90     # frames after SHOOT action to detect MADE_FG
FG_MISS_TIMEOUT_F      = 100    # frames after SHOOT with no MADE → MISSED_FG

REBOUND_HOOP_RADIUS_PX = 90
REBOUND_HOOP_RADIUS_M  = 1.5
REBOUND_COOLDOWN_F     = 20

BLOCK_DIST_PX          = 60
BLOCK_DIST_M           = 1.0

STEAL_DIST_PX          = 150    # generous — needs 2-player proximity
STEAL_DIST_M           = 2.5

ASSIST_WINDOW_F        = 150    # ~5 s at 30 fps

FOUL_DIST_PX           = 40
FOUL_CONFIRM_F         = 5      # consecutive contact frames


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _dist(a, b) -> float:
    if a is None or b is None:
        return float("inf")
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _bbox_center(bbox) -> Optional[list]:
    if not bbox or len(bbox) < 4:
        return None
    return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]


# ── Main class ────────────────────────────────────────────────────────────────

class EventEngine:
    """
    Stateful per-frame event detector.

    Call process(frame_data) every frame; it returns a list[dict] of events
    detected in that frame (empty list when nothing happened).

    Public attributes updated each frame:
        score       : {"team_a": int, "team_b": int}
        possession  : {"team": str|None, "player_id": int|None, "duration_frames": int}
    """

    def __init__(self) -> None:
        # ── Public state ──────────────────────────────────────────────────
        self.score:           dict  = {"team_a": 0, "team_b": 0}
        self.possession:      dict  = {"team": None, "player_id": None,
                                       "duration_frames": 0}
        self.last_passer:     Optional[int]  = None
        self.last_shot_pos:   Optional[list] = None
        self.events_history:  deque = deque(maxlen=300)
        self.ball_trajectory: deque = deque(maxlen=30)
        self.frame_count:     int   = 0

        # ── Possession confirmation ───────────────────────────────────────
        self._poss_candidate: Optional[int] = None
        self._poss_frames:    int            = 0

        # ── Shot / FG state ───────────────────────────────────────────────
        self._last_shooter_id:    Optional[int]  = None
        self._shot_attempt_frame: int             = -999
        self._fg_cooldown:        int             = 0
        self._ball_in_hoop_prev:  bool            = False
        self._fg_miss_emitted:    bool            = False  # one miss per attempt

        # ── Rebound state ─────────────────────────────────────────────────
        self._rebound_cooldown: int             = 0
        self._miss_frame:       int             = -999
        self._last_shooter_team: Optional[str]  = None

        # ── Assist state ──────────────────────────────────────────────────
        self._pass_frame: int = -999

        # ── Block state ───────────────────────────────────────────────────
        self._shot_defender: Optional[int] = None

        # ── Steal / TOV state ─────────────────────────────────────────────
        self._prev_poss_player: Optional[int] = None
        self._prev_poss_team:   Optional[str] = None

        # ── Foul state ────────────────────────────────────────────────────
        self._contact_frames: dict = {}   # (id_a, id_b) → consecutive frames

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame_data: dict) -> list:
        """
        Analyse one frame of pipeline data.
        Returns list[dict] — may be empty.  Each dict is an event record
        compatible with StatsCalculator._update_box_score().
        """
        if not frame_data:
            return []

        self.frame_count += 1

        tracking   = frame_data.get("tracking",   {})
        dets       = frame_data.get("detections", {})
        court_info = frame_data.get("court",      {})
        actions_d  = frame_data.get("actions",    {})

        players    = tracking.get("tracked_players", [])
        ball       = tracking.get("tracked_ball")
        hoops      = dets.get("hoops", [])
        backboards = dets.get("backboards", [])
        actions    = actions_d.get("actions", [])

        is_court = bool(court_info.get("is_calibrated", False))
        quarter  = int(frame_data.get("quarter", 1))
        ts_ms    = int(frame_data.get("timestamp_ms", 0))

        events: list = []

        # ── Tick cooldowns ────────────────────────────────────────────────
        self._fg_cooldown      = max(0, self._fg_cooldown - 1)
        self._rebound_cooldown = max(0, self._rebound_cooldown - 1)

        # ── Record ball trajectory ────────────────────────────────────────
        ball_pos = self._ball_pos(ball, is_court)
        if ball_pos is not None:
            self.ball_trajectory.append(ball_pos)

        # ── Possession tracking ───────────────────────────────────────────
        poss_evt = self._track_possession(ball, players, is_court, quarter, ts_ms)
        if poss_evt:
            events.append(poss_evt)

        # ── Index action results ──────────────────────────────────────────
        action_map = {a["track_id"]: a.get("action", "") for a in actions}

        # ── Detect SHOOT action → record shot state ───────────────────────
        for act in actions:
            if act.get("action", "").upper() in ("SHOOT", "SHOOTING"):
                tid = act["track_id"]
                p   = next((x for x in players if x["track_id"] == tid), None)
                if p:
                    self._last_shooter_id    = tid
                    self._shot_attempt_frame = self.frame_count
                    self._fg_miss_emitted    = False
                    self._last_shooter_team  = p.get("team", "")
                    self.last_shot_pos       = self._player_pos(p, is_court)

        # ── Field goal (made) ─────────────────────────────────────────────
        if self._fg_cooldown == 0:
            fg_evts = self._detect_field_goal(
                ball, hoops, is_court, players, quarter, ts_ms
            )
            events.extend(fg_evts)

        # ── Missed FG timeout ─────────────────────────────────────────────
        miss_evt = self._check_fg_miss_timeout(players, is_court, quarter, ts_ms)
        if miss_evt:
            events.append(miss_evt)

        # ── Rebound ───────────────────────────────────────────────────────
        if self._rebound_cooldown == 0:
            reb = self._detect_rebound(
                ball, players, hoops, backboards, is_court, quarter, ts_ms
            )
            if reb:
                events.append(reb)

        # ── Assist ────────────────────────────────────────────────────────
        ast = self._detect_assist(events, quarter, ts_ms)
        if ast:
            events.append(ast)

        # ── Steal / turnover ──────────────────────────────────────────────
        steal_tov = self._detect_steal_turnover(
            players, ball, is_court, quarter, ts_ms
        )
        events.extend(steal_tov)

        # ── Block ─────────────────────────────────────────────────────────
        blk = self._detect_block(players, ball, actions, is_court, quarter, ts_ms)
        if blk:
            events.append(blk)

        # ── Foul candidate ────────────────────────────────────────────────
        foul_evts = self._detect_foul_candidate(players, quarter, ts_ms)
        events.extend(foul_evts)

        # ── Update score & history ────────────────────────────────────────
        for evt in events:
            self._update_score(evt)
            self.events_history.append(evt)

        return events

    # ── Field goal detection ──────────────────────────────────────────────────

    def _detect_field_goal(
        self,
        ball:     Optional[dict],
        hoops:    list,
        is_court: bool,
        players:  list,
        quarter:  int,
        ts_ms:    int,
    ) -> list:
        """
        Emit MADE_FG when the ball enters the hoop zone after a SHOOT action.
        Uses zone-entry transition (outside → inside) to fire exactly once.
        """
        events = []

        ball_pos = self._ball_pos(ball, is_court)
        if ball_pos is None:
            self._ball_in_hoop_prev = False
            return events

        # Distance to nearest detected or known hoop
        nearest_d, nearest_h = self._nearest_hoop(ball_pos, hoops, is_court)
        if nearest_h is None:
            self._ball_in_hoop_prev = False
            return events

        threshold  = FG_HOOP_RADIUS_M if is_court else FG_HOOP_RADIUS_PX
        in_zone    = nearest_d < threshold
        zone_entry = in_zone and not self._ball_in_hoop_prev
        self._ball_in_hoop_prev = in_zone

        if not zone_entry:
            return events

        # Only emit if a SHOOT action fired recently
        frames_since_shot = self.frame_count - self._shot_attempt_frame
        if frames_since_shot > FG_ATTEMPT_WINDOW_F:
            return events

        # Identify shooter
        shooter = None
        if self._last_shooter_id is not None:
            shooter = next(
                (p for p in players if p["track_id"] == self._last_shooter_id),
                None,
            )
        if shooter is None:
            shooter = self._closest_to(ball_pos, players, is_court)
        if shooter is None:
            return events

        shot_pos = self.last_shot_pos or self._player_pos(shooter, is_court)
        is_three = self._is_three(shot_pos) if shot_pos else False

        events.append({
            "type":       "MADE_FG",
            "track_id":   shooter["track_id"],
            "quarter":    quarter,
            "team":       shooter.get("team", ""),
            "is_three":   is_three,
            "court_pos":  shot_pos,
            "timestamp_ms": ts_ms,
        })

        self._fg_cooldown     = FG_COOLDOWN_F
        self._miss_frame      = -999         # made → no rebound window
        self._last_shooter_id = None
        self.last_shot_pos    = None
        self._fg_miss_emitted = True

        logger.debug(
            "EventEngine MADE_FG  3PT=%s  shooter=%s  team=%s",
            is_three, shooter["track_id"], shooter.get("team"),
        )
        return events

    def _check_fg_miss_timeout(
        self,
        players:  list,
        is_court: bool,
        quarter:  int,
        ts_ms:    int,
    ) -> Optional[dict]:
        """Emit MISSED_FG once when SHOOT was detected but hoop zone never entered."""
        if self._fg_miss_emitted:
            return None
        if self._last_shooter_id is None:
            return None

        frames_since_shot = self.frame_count - self._shot_attempt_frame
        if frames_since_shot < FG_MISS_TIMEOUT_F:
            return None

        shooter = next(
            (p for p in players if p["track_id"] == self._last_shooter_id),
            None,
        )
        team     = self._last_shooter_team or (shooter.get("team", "") if shooter else "")
        shot_pos = self.last_shot_pos
        is_three = self._is_three(shot_pos) if shot_pos else False

        self._fg_miss_emitted    = True
        self._miss_frame         = self.frame_count
        self._last_shooter_id    = None
        self.last_shot_pos       = None

        tid = shooter["track_id"] if shooter else 0

        return {
            "type":       "MISSED_FG",
            "track_id":   tid,
            "quarter":    quarter,
            "team":       team,
            "is_three":   is_three,
            "court_pos":  shot_pos,
            "timestamp_ms": ts_ms,
        }

    # ── Rebound detection ─────────────────────────────────────────────────────

    def _detect_rebound(
        self,
        ball:       Optional[dict],
        players:    list,
        hoops:      list,
        backboards: list,
        is_court:   bool,
        quarter:    int,
        ts_ms:      int,
    ) -> Optional[dict]:
        """
        Detect rebound after a miss.  Fires when:
          1. A MISSED_FG was recorded recently.
          2. Ball is within rebound zone (near hoop / backboard).
          3. A player gains possession of the ball.
        """
        frames_since_miss = self.frame_count - self._miss_frame
        if frames_since_miss > REBOUND_COOLDOWN_F * 3:
            return None

        ball_pos = self._ball_pos(ball, is_court)
        if ball_pos is None:
            return None

        # Check ball is in rebound zone
        reb_th    = REBOUND_HOOP_RADIUS_M if is_court else REBOUND_HOOP_RADIUS_PX
        in_zone   = False

        for h in hoops:
            h_c = h.get("center") or _bbox_center(h.get("bbox"))
            if h_c and _dist(ball_pos, h_c) < reb_th:
                in_zone = True
                break
        if not in_zone and is_court:
            for h_pos in (HOOP_LEFT, HOOP_RIGHT):
                if _dist(ball_pos, h_pos) < reb_th:
                    in_zone = True
                    break
        if not in_zone:
            for bb in backboards:
                bb_c = bb.get("center") or _bbox_center(bb.get("bbox"))
                if bb_c and _dist(ball_pos, bb_c) < reb_th:
                    in_zone = True
                    break

        if not in_zone:
            return None

        # Find player gaining possession
        rebounder = self._closest_to(ball_pos, players, is_court)
        if rebounder is None:
            return None

        poss_th = POSSESSION_DIST_M if is_court else POSSESSION_DIST_PX
        if _dist(ball_pos, self._player_pos(rebounder, is_court)) > poss_th:
            return None

        reb_team   = rebounder.get("team", "")
        sub_type   = "DEF"
        if self._last_shooter_team and reb_team:
            sub_type = "OFF" if reb_team == self._last_shooter_team else "DEF"

        self._rebound_cooldown = REBOUND_COOLDOWN_F
        self._miss_frame       = -999

        return {
            "type":      "REBOUND",
            "track_id":  rebounder["track_id"],
            "quarter":   quarter,
            "team":      reb_team,
            "sub_type":  sub_type,
            "timestamp_ms": ts_ms,
        }

    # ── Assist detection ──────────────────────────────────────────────────────

    def _detect_assist(
        self,
        current_events: list,
        quarter: int,
        ts_ms:   int,
    ) -> Optional[dict]:
        """
        Emit AST when: last_passer is set (same-team possession change)
        AND a MADE_FG appears in current_events within the assist window.
        """
        if self.last_passer is None:
            return None

        fgm = next(
            (e for e in current_events
             if e.get("type") in ("MADE_FG", "MADE_3", "SHOT_MADE")),
            None,
        )
        if fgm is None:
            return None

        if self.frame_count - self._pass_frame > ASSIST_WINDOW_F:
            self.last_passer = None
            return None

        if self.last_passer == fgm.get("track_id"):
            self.last_passer = None
            return None

        passer_id   = self.last_passer
        self.last_passer = None

        return {
            "type":      "AST",
            "track_id":  passer_id,
            "quarter":   quarter,
            "team":      fgm.get("team", ""),
            "timestamp_ms": ts_ms,
        }

    # ── Steal / turnover ──────────────────────────────────────────────────────

    def _detect_steal_turnover(
        self,
        players:  list,
        ball:     Optional[dict],
        is_court: bool,
        quarter:  int,
        ts_ms:    int,
    ) -> list:
        """
        Emit STL + TOV when possession switches to the opposing team and the
        two players were physically close (adversarial contact).
        """
        events = []

        ball_pos = self._ball_pos(ball, is_court)
        if ball_pos is None:
            return events

        cur_owner = self._closest_to(ball_pos, players, is_court)
        if cur_owner is None:
            return events

        poss_th = POSSESSION_DIST_M if is_court else POSSESSION_DIST_PX
        if _dist(ball_pos, self._player_pos(cur_owner, is_court)) > poss_th:
            self._prev_poss_player = None
            self._prev_poss_team   = None
            return events

        cur_id   = cur_owner["track_id"]
        cur_team = cur_owner.get("team", "")

        prev_id   = self._prev_poss_player
        prev_team = self._prev_poss_team

        if (
            prev_team and cur_team
            and prev_team != cur_team
            and prev_id is not None
            and prev_id != cur_id
        ):
            prev_player = next((p for p in players if p["track_id"] == prev_id), None)
            if prev_player:
                sep_th = STEAL_DIST_M if is_court else STEAL_DIST_PX
                sep    = _dist(
                    self._player_pos(cur_owner, is_court),
                    self._player_pos(prev_player, is_court),
                )
                if sep < sep_th:
                    events.append({
                        "type":      "STL",
                        "track_id":  cur_id,
                        "quarter":   quarter,
                        "team":      cur_team,
                        "timestamp_ms": ts_ms,
                    })
                    events.append({
                        "type":      "TOV",
                        "track_id":  prev_id,
                        "quarter":   quarter,
                        "team":      prev_team,
                        "timestamp_ms": ts_ms,
                    })
                    self.last_passer = None  # forced turnover ≠ pass

        self._prev_poss_player = cur_id
        self._prev_poss_team   = cur_team
        return events

    # ── Block detection ───────────────────────────────────────────────────────

    def _detect_block(
        self,
        players:  list,
        ball:     Optional[dict],
        actions:  list,
        is_court: bool,
        quarter:  int,
        ts_ms:    int,
    ) -> Optional[dict]:
        """
        Two paths:
          1. Action classifier returns BLOCK label directly.
          2. A SHOOT fires and an opposing player is within block-distance.
        """
        # Path 1 — direct from classifier
        for act in actions:
            if act.get("action", "").upper() == "BLOCK":
                tid = act["track_id"]
                p   = next((x for x in players if x["track_id"] == tid), None)
                if p:
                    return {
                        "type":      "BLK",
                        "track_id":  tid,
                        "quarter":   quarter,
                        "team":      p.get("team", ""),
                        "timestamp_ms": ts_ms,
                    }

        # Path 2 — proximity at shot moment
        if (
            self._last_shooter_id is not None
            and self.frame_count - self._shot_attempt_frame <= 3
        ):
            shooter = next(
                (p for p in players if p["track_id"] == self._last_shooter_id),
                None,
            )
            if shooter is None:
                return None

            s_pos  = self._player_pos(shooter, is_court)
            s_team = shooter.get("team", "")
            blk_th = BLOCK_DIST_M if is_court else BLOCK_DIST_PX

            for p in players:
                if p.get("team") == s_team:
                    continue
                if _dist(s_pos, self._player_pos(p, is_court)) < blk_th:
                    self._shot_defender = p["track_id"]
                    # Don't emit yet — wait for miss to confirm block
                    break

        return None

    # ── Foul candidate ────────────────────────────────────────────────────────

    def _detect_foul_candidate(
        self,
        players:  list,
        quarter:  int,
        ts_ms:    int,
    ) -> list:
        """
        Conservative: two opposing players stay within FOUL_DIST_PX for
        FOUL_CONFIRM_F consecutive frames while one has the ball.
        Emits at most one event per contact episode.
        """
        events = []

        if self._fg_cooldown > 0:
            return events

        ball_carrier = self.possession.get("player_id")

        for i, p1 in enumerate(players):
            for p2 in players[i + 1:]:
                if p1.get("team") == p2.get("team"):
                    continue
                c1 = p1.get("center") or _bbox_center(p1.get("bbox"))
                c2 = p2.get("center") or _bbox_center(p2.get("bbox"))
                if c1 is None or c2 is None:
                    continue

                key = (
                    min(p1["track_id"], p2["track_id"]),
                    max(p1["track_id"], p2["track_id"]),
                )
                if _dist(c1, c2) < FOUL_DIST_PX:
                    self._contact_frames[key] = self._contact_frames.get(key, 0) + 1
                else:
                    self._contact_frames[key] = 0
                    continue

                if self._contact_frames[key] == FOUL_CONFIRM_F:
                    self._contact_frames[key] = 0
                    # Fouler = player without ball
                    if ball_carrier in (p1["track_id"], p2["track_id"]):
                        fouler = (
                            p2 if p1["track_id"] == ball_carrier else p1
                        )
                        events.append({
                            "type":      "FOUL",
                            "track_id":  fouler["track_id"],
                            "quarter":   quarter,
                            "team":      fouler.get("team", ""),
                            "timestamp_ms": ts_ms,
                        })

        return events

    # ── Possession tracking ───────────────────────────────────────────────────

    def _track_possession(
        self,
        ball:     Optional[dict],
        players:  list,
        is_court: bool,
        quarter:  int,
        ts_ms:    int,
    ) -> Optional[dict]:
        """
        Update self.possession.  Returns POSSESSION_CHANGE event dict
        when possession switches teams, else None.
        """
        if ball is None or not players:
            return None

        ball_pos = self._ball_pos(ball, is_court)
        if ball_pos is None:
            return None

        closest = self._closest_to(ball_pos, players, is_court)
        if closest is None:
            return None

        poss_th = POSSESSION_DIST_M if is_court else POSSESSION_DIST_PX
        if _dist(ball_pos, self._player_pos(closest, is_court)) > poss_th:
            self._poss_candidate = None
            self._poss_frames    = 0
            return None

        cand_id = closest["track_id"]
        if cand_id == self._poss_candidate:
            self._poss_frames += 1
        else:
            self._poss_candidate = cand_id
            self._poss_frames    = 1

        if self._poss_frames < POSSESSION_CONFIRM_F:
            return None

        new_team = closest.get("team", "")
        old_id   = self.possession.get("player_id")
        old_team = self.possession.get("team")

        if cand_id == old_id:
            self.possession["duration_frames"] += 1
            return None

        # Possession confirmed to new player
        self.possession = {
            "team":            new_team,
            "player_id":       cand_id,
            "duration_frames": self._poss_frames,
        }

        # Same-team transfer → potential pass for assist
        if old_id is not None and old_team == new_team:
            self.last_passer = old_id
            self._pass_frame = self.frame_count

        # Cross-team → POSSESSION_CHANGE event
        if old_team and new_team and old_team != new_team:
            return {
                "type":            "POSSESSION_CHANGE",
                "track_id":        cand_id,
                "quarter":         quarter,
                "team":            new_team,
                "possession_team": new_team,
                "timestamp_ms":    ts_ms,
            }

        return None

    # ── Score update ──────────────────────────────────────────────────────────

    def _update_score(self, event: dict) -> None:
        """Increment self.score for scoring events."""
        etype = str(event.get("type", "")).upper()
        team  = event.get("team", "")
        if team not in ("A", "B"):
            return

        key = "team_a" if team == "A" else "team_b"

        if etype in ("MADE_FG", "SHOT_MADE"):
            self.score[key] += 3 if event.get("is_three") else 2
        elif etype == "MADE_3":
            self.score[key] += 3
        elif etype in ("MADE_FT", "FT_MADE"):
            self.score[key] += 1

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state.  Call between quarters or matches."""
        self.score            = {"team_a": 0, "team_b": 0}
        self.possession       = {"team": None, "player_id": None, "duration_frames": 0}
        self.last_passer      = None
        self.last_shot_pos    = None
        self.events_history   = deque(maxlen=300)
        self.ball_trajectory  = deque(maxlen=30)
        self.frame_count      = 0

        self._poss_candidate     = None
        self._poss_frames        = 0
        self._last_shooter_id    = None
        self._shot_attempt_frame = -999
        self._fg_cooldown        = 0
        self._ball_in_hoop_prev  = False
        self._fg_miss_emitted    = False
        self._rebound_cooldown   = 0
        self._miss_frame         = -999
        self._last_shooter_team  = None
        self._pass_frame         = -999
        self._shot_defender      = None
        self._prev_poss_player   = None
        self._prev_poss_team     = None
        self._contact_frames     = {}

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ball_pos(ball: Optional[dict], is_court: bool) -> Optional[list]:
        if ball is None:
            return None
        if is_court:
            cp = ball.get("court_pos")
            if cp is not None:
                return list(cp)
        c = ball.get("center")
        return list(c) if c is not None else None

    @staticmethod
    def _player_pos(player: dict, is_court: bool) -> Optional[list]:
        if is_court:
            cp = player.get("court_pos")
            if cp is not None:
                return list(cp)
        c = player.get("center")
        if c is not None:
            return list(c)
        return _bbox_center(player.get("bbox"))

    def _closest_to(
        self,
        pos:      list,
        players:  list,
        is_court: bool,
    ) -> Optional[dict]:
        best_d, best_p = float("inf"), None
        for p in players:
            d = _dist(pos, self._player_pos(p, is_court))
            if d < best_d:
                best_d, best_p = d, p
        return best_p

    @staticmethod
    def _nearest_hoop(
        ball_pos: list,
        hoops:    list,
        is_court: bool,
    ):
        """Return (distance, position) of the nearest hoop to ball_pos."""
        best_d, best_h = float("inf"), None

        for h in hoops:
            h_c = h.get("center") or _bbox_center(h.get("bbox"))
            if h_c:
                d = _dist(ball_pos, h_c)
                if d < best_d:
                    best_d, best_h = d, h_c

        # In calibrated mode also check known FIBA positions
        if is_court:
            for h_pos in (HOOP_LEFT, HOOP_RIGHT):
                d = _dist(ball_pos, h_pos)
                if d < best_d:
                    best_d, best_h = d, h_pos

        return best_d, best_h

    @staticmethod
    def _is_three(shot_pos: Optional[list]) -> bool:
        """True when shot_pos is beyond the FIBA 3PT line (court coordinates)."""
        if shot_pos is None or len(shot_pos) < 2:
            return False
        x, y    = float(shot_pos[0]), float(shot_pos[1])
        dist_l  = math.hypot(x - HOOP_LEFT[0],  y - HOOP_LEFT[1])
        dist_r  = math.hypot(x - HOOP_RIGHT[0], y - HOOP_RIGHT[1])
        in_corner = y < 0.9 or y > 14.1
        if in_corner:
            return (x >= THREE_PT_STRAIGHT_X
                    if dist_l <= dist_r
                    else x <= COURT_W - THREE_PT_STRAIGHT_X)
        return min(dist_l, dist_r) > THREE_PT_RADIUS


# ── Factory function ──────────────────────────────────────────────────────────

def create_event_engine() -> EventEngine:
    """Factory used by video_processor via _rel_or_abs()."""
    return EventEngine()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=== EventEngine smoke test ===\n")

    # 1 — instantiate
    engine = create_event_engine()
    assert engine.score == {"team_a": 0, "team_b": 0}
    assert engine.possession["team"] is None
    print("✅  1. create_event_engine() OK")

    # 2 — process empty frame_data returns list
    result = engine.process({})
    assert result == []
    print("✅  2. process({}) → []")

    # 3 — process minimal frame_data
    fd = {
        "frame_id": 0, "timestamp_ms": 0, "quarter": 1,
        "tracking": {
            "tracked_players": [
                {"track_id": 1, "bbox": [100,200,150,350],
                 "center": [125,275], "team": "A", "confidence": 0.9},
                {"track_id": 2, "bbox": [300,200,350,350],
                 "center": [325,275], "team": "B", "confidence": 0.9},
            ],
            "tracked_ball": {"bbox": [120,260,140,280],
                             "center": [130,270], "confidence": 0.95},
            "tracked_referees": [],
        },
        "detections": {
            "hoops": [{"bbox": [500,100,560,130],
                       "center": [530,115], "confidence": 0.99}],
            "backboards": [],
        },
        "court":   {"is_calibrated": False},
        "actions": {"actions": []},
        "pose":    {"poses": []},
        "events":  [],
    }
    result = engine.process(fd)
    assert isinstance(result, list)
    print(f"✅  3. process(minimal) → {len(result)} events")

    # 4 — _update_score accumulation
    engine.reset()
    engine._update_score({"type": "MADE_FG", "team": "A", "is_three": False})  # +2
    engine._update_score({"type": "MADE_FG", "team": "A", "is_three": False})  # +2
    engine._update_score({"type": "MADE_FG", "team": "A", "is_three": True})   # +3
    engine._update_score({"type": "MADE_FG", "team": "B", "is_three": False})  # +2
    assert engine.score == {"team_a": 7, "team_b": 2}, engine.score
    engine._update_score({"type": "MADE_FT", "team": "B"})                     # +1
    assert engine.score["team_b"] == 3
    print(f"✅  4. _update_score() → {engine.score}")

    # 5 — _track_possession confirms after 3 frames
    engine.reset()
    ball_d   = {"center": [130, 270]}
    p_a      = {"track_id": 1, "center": [125, 268], "team": "A",
                "bbox": [100,240,150,300]}
    p_b      = {"track_id": 2, "center": [325, 275], "team": "B",
                "bbox": [300,240,350,300]}
    plrs     = [p_a, p_b]
    for _ in range(POSSESSION_CONFIRM_F):
        engine._track_possession(ball_d, plrs, False, 1, 0)
    assert engine.possession["player_id"] == 1
    assert engine.possession["team"]      == "A"
    print(f"✅  5. possession confirmed: {engine.possession}")

    # 6 — possession change emitted when ball moves to opposing player
    ball_b = {"center": [320, 274]}
    evt    = None
    for _ in range(POSSESSION_CONFIRM_F):
        evt = engine._track_possession(ball_b, plrs, False, 1, 0)
    assert engine.possession["team"] == "B", engine.possession
    print(f"✅  6. possession switched: {engine.possession}")

    # 7 — reset clears all state
    engine.reset()
    assert engine.score         == {"team_a": 0, "team_b": 0}
    assert engine.possession    == {"team": None, "player_id": None, "duration_frames": 0}
    assert engine.last_passer   is None
    assert engine.frame_count   == 0
    print("✅  7. reset() clears state")

    # 8 — _is_three geometry
    # HOOP_LEFT=[1.575,7.5], 3PT arc radius=6.75 m → arc tangent at x=8.325
    assert EventEngine._is_three([14.0, 7.5]) is True    # half-court
    assert EventEngine._is_three([8.5,  7.5]) is True    # just outside left arc (8.5-1.575=6.925 > 6.75)
    assert EventEngine._is_three([7.0,  7.5]) is False   # inside left arc (7.0-1.575=5.425 < 6.75)
    assert EventEngine._is_three([1.0,  7.5]) is False   # inside left paint
    assert EventEngine._is_three([0.0,  7.5]) is False   # baseline
    print("✅  8. _is_three() geometry OK")

    # 9 — JSON-serialisable output dict (WS state)
    engine.reset()
    engine._update_score({"type": "MADE_FG", "team": "A", "is_three": False})
    ws = json.dumps({
        "events":     list(engine.events_history),
        "score":      engine.score,
        "possession": engine.possession,
    })
    assert len(ws) > 0
    print(f"✅  9. WS-format serialisable: {len(ws)} bytes")

    # 10 — process() with SHOOT action records shot state
    engine.reset()
    fd_shoot = dict(fd)
    fd_shoot["actions"] = {"actions": [
        {"track_id": 1, "action": "Shoot", "confidence": 0.92, "source": "model"}
    ]}
    engine.process(fd_shoot)
    assert engine._last_shooter_id    == 1
    assert engine._shot_attempt_frame == engine.frame_count
    print("✅  10. SHOOT action recorded → shot state updated")

    print("\n=== All smoke tests passed ✅ ===")

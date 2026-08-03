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
# Reference processed fps the frame-windows below were tuned & validated at.
# EventEngine(fps=…)/set_fps() scale every window by fps/REFERENCE_FPS so event
# timing stays constant wall-clock time at any processed rate (e.g. 45-frame FG
# cooldown = 3 s @15fps stays 3 s @30fps → 90 frames). At 15fps scale=1.0 → the
# behaviour is byte-identical to before this change.
REFERENCE_FPS          = 15.0

POSSESSION_CONFIRM_F   = 3      # consecutive frames to confirm new possession

FG_HOOP_RADIUS_PX      = 70     # pixels  — ball inside hoop zone (wider → fewer missed baskets)
FG_HOOP_RADIUS_M       = 0.45   # meters  — tighter in court space (no action gate)
FG_COOLDOWN_F          = 45     # frames before another FG can fire
FG_ABOVE_MARGIN_PX     = 10     # ball must be ≥ this many px above hoop_y to arm
FG_ABOVE_WINDOW_F      = 20     # frames to look back for "was above" check (~1.3 s @ 15 eff fps)
FG_ATTEMPT_WINDOW_F    = 150    # kept for MISSED_FG accounting only (not scoring gate)
FG_MISS_TIMEOUT_F      = 100    # frames after SHOOT with no MADE → MISSED_FG
FG_BALL_LOST_RESET_F   = 3      # consecutive ball-absent frames before resetting zone state
FG_ZONE_DWELL_F        = 2      # ball must stay in zone N frames before MADE_FG fires
FG_HOOP_CACHE_TTL      = 90     # reject stale hoop cache after this many frames (~6 s)
FG_SHOOTER_TTL_F       = 120    # frames after shot attempt before shooter attribution expires
FG_MIN_DOWNWARD_DY     = 2      # ball must be moving downward by at least this many px at entry

REBOUND_HOOP_RADIUS_PX = 90
REBOUND_HOOP_RADIUS_M  = 1.5
REBOUND_COOLDOWN_F     = 20
REBOUND_MIN_F          = 8      # min frames after the shot before a rebound can count
                                # (ball needs time to reach the rim first)

BLOCK_DIST_PX          = 60
BLOCK_DIST_M           = 1.0
BLK_COOLDOWN_F         = 45     # frames before another BLK can fire (same as FG)
BLK_SHOT_WINDOW_F      = 20     # Path-1 BLK only fires when shot attempt ≤ 20 frames ago
BLK_PATH2_WINDOW_F     = 12     # Path-2 proximity check extends to 12 frames after shot

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


def _fmt_ts(ts_ms: int) -> str:
    """Format millisecond video-timestamp as MM:SS.mmm for log readability."""
    total_s, ms = divmod(int(ts_ms), 1000)
    m, s = divmod(total_s, 60)
    return f"{m:02d}:{s:02d}.{ms:03d}"


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

    def __init__(self, fps: float = REFERENCE_FPS) -> None:
        # Scale all frame-windows to the actual processed fps (see REFERENCE_FPS).
        # Must run first: _ball_px_hist below uses a scaled maxlen.
        self._configure_fps(fps)

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
        self._last_shooter_id:     Optional[int]  = None
        self._shot_attempt_frame:  int             = -999
        self._fg_cooldown:         int             = 0
        self._ball_in_hoop_frames: int             = 0    # consecutive in-zone frames
        self._ball_lost_frames:    int             = 0    # consecutive ball-absent frames
        self._hoop_cache_age:      int             = 0    # frames since last hoop detection
        self._fg_miss_emitted:     bool            = False  # one miss per attempt

        # ── Rebound state ─────────────────────────────────────────────────
        self._rebound_cooldown: int             = 0
        self._miss_frame:       int             = -999
        self._last_shooter_team: Optional[str]  = None

        # ── Assist state ──────────────────────────────────────────────────
        self._pass_frame: int = -999

        # ── Block state ───────────────────────────────────────────────────
        self._shot_defender: Optional[int] = None
        self._blk_cooldown:  int           = 0

        # ── Steal / TOV state ─────────────────────────────────────────────
        self._prev_poss_player: Optional[int] = None
        self._prev_poss_team:   Optional[str] = None

        # ── Foul state ────────────────────────────────────────────────────
        self._contact_frames: dict = {}   # (id_a, id_b) → consecutive frames

        # ── Hoop pixel-position cache (for pixel-space scoring fallback) ───
        self._last_hoops_px: list = []    # last detected hoop dicts in pixel space

        # ── Trajectory-based scoring — pixel-space ball history ───────────
        # Stores recent (ball_x, ball_y) in pixel coords regardless of court calib.
        # Used to check ball came from ABOVE the hoop before counting a score.
        self._ball_px_hist: deque = deque(maxlen=self._fg_above_window_f)

        # ── Per-player court-position history for 2PT/3PT estimation ─────
        # When action model doesn't fire (no last_shot_pos), we look back in
        # a player's recent court positions and pick the one furthest from the
        # hoop — that's the best proxy for where they actually released the ball.
        # Only populated when is_court=True (court calibrated).
        self._player_court_hist: dict = {}   # track_id → deque[(x_m, y_m)]

    def _configure_fps(self, fps: float) -> None:
        """(Re)compute every frame-window for the given processed fps.

        Module constants hold the values validated at REFERENCE_FPS; each is
        scaled by fps/REFERENCE_FPS and floored at 1 frame. Pure-integer windows
        so downstream frame comparisons are unchanged in kind, only in span.
        """
        self.fps = float(fps)
        _s = self.fps / REFERENCE_FPS

        def _sc(v: int) -> int:
            return max(1, int(round(v * _s)))

        self._possession_confirm_f = _sc(POSSESSION_CONFIRM_F)
        self._fg_cooldown_f        = _sc(FG_COOLDOWN_F)
        self._fg_above_window_f    = _sc(FG_ABOVE_WINDOW_F)
        self._fg_miss_timeout_f    = _sc(FG_MISS_TIMEOUT_F)
        self._fg_ball_lost_reset_f = _sc(FG_BALL_LOST_RESET_F)
        self._fg_zone_dwell_f      = _sc(FG_ZONE_DWELL_F)
        self._fg_hoop_cache_ttl    = _sc(FG_HOOP_CACHE_TTL)
        self._fg_shooter_ttl_f     = _sc(FG_SHOOTER_TTL_F)
        self._rebound_cooldown_f   = _sc(REBOUND_COOLDOWN_F)
        self._reb_min_f            = _sc(REBOUND_MIN_F)
        self._blk_cooldown_f       = _sc(BLK_COOLDOWN_F)
        self._blk_shot_window_f    = _sc(BLK_SHOT_WINDOW_F)
        self._blk_path2_window_f   = _sc(BLK_PATH2_WINDOW_F)
        self._assist_window_f      = _sc(ASSIST_WINDOW_F)
        self._foul_confirm_f       = _sc(FOUL_CONFIRM_F)

    def set_fps(self, fps: float) -> None:
        """Reconfigure frame-windows for the actual processed fps. Call once
        before processing (after the source fps is known). Safe to call on a
        fresh engine — _ball_px_hist is empty at that point."""
        self._configure_fps(fps)
        # maxlen depends on fps → rebuild the ring buffer (preserve any contents).
        self._ball_px_hist = deque(self._ball_px_hist, maxlen=self._fg_above_window_f)

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

        # Keep pixel hoop cache fresh whenever hoops are detected; track staleness.
        if hoops:
            self._last_hoops_px = hoops
            self._hoop_cache_age = 0
        else:
            self._hoop_cache_age += 1

        # Maintain pixel-space ball history for from-above scoring check.
        # ball.center is always pixel coords regardless of court calibration.
        ball_px_center = ball.get("center") if ball else None
        if ball_px_center:
            self._ball_px_hist.append(ball_px_center)

        events: list = []

        # ── Tick cooldowns ────────────────────────────────────────────────
        self._fg_cooldown      = max(0, self._fg_cooldown - 1)
        self._rebound_cooldown = max(0, self._rebound_cooldown - 1)
        self._blk_cooldown     = max(0, self._blk_cooldown - 1)

        # ── Record ball trajectory ────────────────────────────────────────
        ball_pos = self._ball_pos(ball, is_court)
        if ball_pos is not None:
            self.ball_trajectory.append(ball_pos)

        # ── Record player court positions for 2PT/3PT estimation ─────────
        # Only when calibrated — pixel positions can't be used for 3PT geometry.
        if is_court:
            for p in players:
                tid   = p["track_id"]
                c_pos = p.get("court_pos")
                if c_pos:
                    if tid not in self._player_court_hist:
                        self._player_court_hist[tid] = deque(maxlen=90)
                    self._player_court_hist[tid].append(c_pos)

        # ── Possession tracking ───────────────────────────────────────────
        poss_evt = self._track_possession(ball, players, is_court, quarter, ts_ms)
        if poss_evt:
            events.append(poss_evt)

        # ── Index action results ──────────────────────────────────────────
        action_map = {a["track_id"]: a.get("action", "") for a in actions}

        # ── Detect SHOOT action → record shot state ───────────────────────
        # Also treat JUMP near basket as shot attempt — action model often
        # classifies shooting as jump_action instead of Shoot.
        SHOT_AREA_M  = 5.5   # metres from hoop to count a jump as shot attempt
        SHOT_AREA_PX = 200   # pixels from hoop (pixel mode)
        # Pixel-space hoop candidates: detected this frame + last-seen cache.
        # Apply same staleness gate as FG detection so dead-camera hoop positions
        # don't silently inflate JUMP→SHOOT promotions after a camera pan.
        px_hoops = hoops or (
            self._last_hoops_px if self._hoop_cache_age <= self._fg_hoop_cache_ttl else []
        )
        for act in actions:
            raw_act = act.get("action", "").upper()
            is_shoot = raw_act in ("SHOOT", "SHOOTING")
            tid = act["track_id"]
            p   = next((x for x in players if x["track_id"] == tid), None)
            if not p:
                continue
            if not is_shoot and raw_act == "JUMP":
                # Promote JUMP→shot if player is close to any known hoop position
                p_pos = self._player_pos(p, is_court)
                if p_pos is not None:
                    th = SHOT_AREA_M if is_court else SHOT_AREA_PX
                    near = False
                    if is_court:
                        # Court mode: compare against FIBA fixed positions
                        for h_pos in (HOOP_LEFT, HOOP_RIGHT):
                            if _dist(p_pos, h_pos) < th:
                                near = True
                                break
                    else:
                        # Pixel mode: compare against detected/cached hoop centers
                        for h in px_hoops:
                            h_c = h.get("center") or _bbox_center(h.get("bbox"))
                            if h_c and _dist(p_pos, h_c) < th:
                                near = True
                                break
                    if near:
                        is_shoot = True
            if is_shoot:
                # Don't overwrite an already-active shot within its TTL window.
                # Two players jumping near the basket in the same frame (e.g. tip-off,
                # contested rebound) would otherwise mis-attribute the FG to the second
                # player processed in the loop instead of the actual shooter.
                active_shot = (
                    self._last_shooter_id is not None
                    and self.frame_count - self._shot_attempt_frame <= self._fg_shooter_ttl_f
                )
                if active_shot:
                    continue
                self._last_shooter_id    = tid
                self._shot_attempt_frame = self.frame_count
                self._fg_miss_emitted    = False
                self._last_shooter_team  = p.get("team", "")
                self.last_shot_pos       = self._player_pos(p, is_court)
                trigger = "JUMP→SHOOT" if raw_act == "JUMP" else raw_act
                logger.info(
                    "SHOT_ATTEMPT  trigger=%s  shooter=%s  team=%s  frame=%d  video=%s",
                    trigger, tid, p.get("team", "?"), self.frame_count, _fmt_ts(ts_ms),
                )

        # Expire shooter attribution after TTL (prevents stale attribution when
        # neither MADE_FG nor MISSED_FG fires — e.g. loose ball after a pass).
        if (self._last_shooter_id is not None and
                self.frame_count - self._shot_attempt_frame > self._fg_shooter_ttl_f):
            self._last_shooter_id = None
            self.last_shot_pos    = None

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
        Emit MADE_FG using ball-trajectory detection — no action model required.

        Scoring condition (both must be true):
          1. Zone entry: ball transitions from outside → inside the hoop zone.
             Always checked in pixel space — the ball is airborne at shot time,
             so floor-plane homography introduces perspective error at hoop height
             (~3 m above floor).  Pixel-space is accurate: ball near hoop bbox
             in the image = ball physically near the hoop.
          2. From above: ball's pixel-y was above the hoop's pixel-y (by at least
             FG_ABOVE_MARGIN_PX) at some point in the last FG_ABOVE_WINDOW_F frames.
             This is the broadcast-camera equivalent of "ball came down through ring"
             and filters out dribbles, rolling balls, and side-entry false positives.

        Action model (SHOOT/JUMP) is still used for MISSED_FG accounting and
        action labels, but is NOT a gate for MADE_FG.
        """
        events = []

        # ── 1. Zone detection — always pixel space ────────────────────────
        # court_pos from homography is unreliable for an airborne ball: the
        # floor-plane projection displaces the ball by 0.5–2 m at hoop height
        # depending on camera angle, making the 0.45 m FIBA threshold fail.
        ball_px = ball.get("center") if ball else None
        if ball_px is None:
            # Small grace window before resetting zone state — 1–2 missing frames
            # are common for fast-moving balls and shouldn't cancel a detection.
            self._ball_lost_frames += 1
            if self._ball_lost_frames >= self._fg_ball_lost_reset_f:
                self._ball_in_hoop_frames = 0
            return events
        self._ball_lost_frames = 0

        # Reject stale hoop cache — hoops from many seconds ago are unreliable.
        if self._hoop_cache_age > self._fg_hoop_cache_ttl:
            self._ball_in_hoop_frames = 0
            return events

        # _nearest_hoop_px_center uses self._last_hoops_px (already updated
        # in process() before this call, so it reflects the current frame).
        hoop_ref_px = self._nearest_hoop_px_center(ball_px)
        if hoop_ref_px is None:
            # No hoop reference yet — cannot confirm zone entry
            self._ball_in_hoop_frames = 0
            return events

        in_zone = _dist(ball_px, hoop_ref_px) < FG_HOOP_RADIUS_PX

        # Count consecutive in-zone frames; reset immediately on exit.
        if in_zone:
            self._ball_in_hoop_frames += 1
        else:
            self._ball_in_hoop_frames = 0

        # Require ball to dwell in the zone for FG_ZONE_DWELL_F consecutive frames
        # before scoring.  This eliminates single-frame false positives from
        # dribbles, deflections, and ball-near-rim but not-through-rim moments.
        # Fire exactly on the Nth frame (== not >=) so cooldown prevents re-fire.
        if self._ball_in_hoop_frames != self._fg_zone_dwell_f:
            return events

        # ── 2. "From above" check — pixel space ──────────────────────────
        # Pixel y increases downward; "above hoop" means smaller y value.
        # Skip check when ball history is empty (first frames) — allow scoring.
        if self._ball_px_hist:
            hoop_y = float(hoop_ref_px[1])
            hoop_x = float(hoop_ref_px[0])
            # Ball must have been above hoop AND within a tighter horizontal window —
            # prevents cross-court or dribble history from satisfying this check.
            horiz_window = FG_HOOP_RADIUS_PX * 1.5
            ball_was_above = any(
                float(pos[1]) < hoop_y - FG_ABOVE_MARGIN_PX
                and abs(float(pos[0]) - hoop_x) < horiz_window
                for pos in self._ball_px_hist
            )
            if not ball_was_above:
                logger.debug(
                    "FG zone entry rejected — ball did not come from above hoop "
                    "(hoop_y=%.0f, recent ball_y range=[%.0f-%.0f])",
                    hoop_y,
                    min(p[1] for p in self._ball_px_hist),
                    max(p[1] for p in self._ball_px_hist),
                )
                self._ball_in_hoop_frames = 0
                return events

            # Ball must be moving clearly downward (pixel y increasing) at zone
            # entry.  Upward or sideways motion means bounce, pass, or dribble.
            if len(self._ball_px_hist) >= 3:
                recent_ys = [float(p[1]) for p in list(self._ball_px_hist)[-3:]]
                dy = recent_ys[-1] - recent_ys[0]
                if dy < FG_MIN_DOWNWARD_DY:
                    logger.debug(
                        "FG zone entry rejected — ball not moving downward (dy=%.1f)",
                        dy,
                    )
                    self._ball_in_hoop_frames = 0
                    return events

        # ── 3. Identify shooter + the SCORING (offensive) team ────────────
        # Team-level correctness holds even when the exact player/number is
        # unknown: a basket belongs to the ATTACKING team. Prefer the team
        # captured at the shot attempt (the real shooter, on offense), then the
        # confirmed possession team; the resolved shooter's colour is last resort.
        # This stops a contesting DEFENDER near the rim (which the closest-to-ball
        # fallback can pick) from crediting the WRONG team.
        ball_pos = self._ball_pos(ball, is_court)   # court or pixel — for closest() search
        offense_team = ""
        if self._last_shooter_id is not None and self._last_shooter_team:
            offense_team = self._last_shooter_team
        elif self.possession.get("team"):
            offense_team = self.possession["team"]

        shooter = None
        if self._last_shooter_id is not None:
            shooter = next(
                (p for p in players if p["track_id"] == self._last_shooter_id),
                None,
            )
        if shooter is None and offense_team:
            # closest player ON THE OFFENSIVE TEAM — never credit a contesting defender
            offensive = [p for p in players if p.get("team") == offense_team]
            if offensive:
                shooter = self._closest_to(
                    ball_pos if ball_pos is not None else ball_px,
                    offensive,
                    is_court and ball_pos is not None,
                )
        if shooter is None:
            shooter = self._closest_to(
                ball_pos if ball_pos is not None else ball_px,
                players,
                is_court and ball_pos is not None,
            )
        if shooter is None:
            return events

        # Scoring team: offense signal first (robust without a jersey number),
        # then the resolved shooter's colour as last resort.
        made_team = offense_team or shooter.get("team", "")

        # ── 4. Determine shot position & 2PT/3PT ────────────────────────
        # Priority: (a) last_shot_pos from action model (most accurate — recorded
        # at the moment of jump), (b) furthest recent court position from the hoop
        # (best proxy for release point when action model didn't fire), (c) current
        # player position (last resort).
        # 3PT geometry only works in court coordinates — pixel space can't be used.
        shot_pos = (
            self.last_shot_pos
            or self._estimate_shot_pos(shooter["track_id"], is_court)
        )
        is_three = self._is_three(shot_pos) if (shot_pos and is_court) else False

        # ── 5. Emit event ────────────────────────────────────────────────
        events.append({
            "type":         "MADE_FG",
            "track_id":     shooter["track_id"],
            "quarter":      quarter,
            "team":         made_team,
            "is_three":     is_three,
            "court_pos":    shot_pos,
            "timestamp_ms": ts_ms,
        })

        self._fg_cooldown          = self._fg_cooldown_f
        self._miss_frame           = -999
        self._last_shooter_id      = None
        self.last_shot_pos         = None
        self._fg_miss_emitted      = True
        self._ball_in_hoop_frames  = 0
        # Clear trajectory history so the just-scored ball position can't
        # trigger another zone entry check during cooldown.
        self._ball_px_hist.clear()

        pts = 3 if is_three else 2
        logger.info(
            "MADE_FG  +%dPTS  shooter=%s  team=%s  3pt=%s  Q%d  video=%s",
            pts, shooter["track_id"], made_team or "?",
            is_three, quarter, _fmt_ts(ts_ms),
        )
        return events

    def _nearest_hoop_px_center(self, ball_px) -> Optional[list]:
        """Return pixel-space center of the hoop nearest to ball_px."""
        candidates = self._last_hoops_px
        best_d, best_c = float("inf"), None
        if ball_px:
            for h in candidates:
                c = h.get("center") or _bbox_center(h.get("bbox"))
                if c:
                    d = _dist(ball_px, c)
                    if d < best_d:
                        best_d, best_c = d, c
        return best_c

    def _estimate_shot_pos(self, track_id: int, is_court: bool) -> Optional[list]:
        """
        Estimate where the player released the ball.

        When the action model fires (last_shot_pos is set) that's more accurate.
        This method is the fallback: scan the player's recent court-position history
        and return the point FURTHEST from the nearest hoop — that's the best proxy
        for the launch position (player was still at their shooting spot).

        Returns None when is_court=False (pixel coords can't determine 2/3PT).
        """
        if not is_court:
            return None
        hist = self._player_court_hist.get(track_id)
        if not hist:
            return None
        # Use only the most recent ~2 seconds to avoid picking a position
        # from many seconds before the actual shot.
        recent = list(hist)[-30:]
        best_pos, best_d = None, -1.0
        for pos in recent:
            d_l = _dist(pos, HOOP_LEFT)
            d_r = _dist(pos, HOOP_RIGHT)
            d   = min(d_l, d_r)
            if d > best_d:
                best_d, best_pos = d, pos
        return best_pos

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
        if frames_since_shot < self._fg_miss_timeout_f:
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

        logger.info(
            "MISSED_FG  shooter=%s  team=%s  3pt=%s  Q%d  video=%s",
            tid, team, is_three, quarter, _fmt_ts(ts_ms),
        )
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
        # A rebound follows a live, UNMADE shot. The physical rebound happens
        # ~1-2 s after the release — so anchor the window to the SHOT attempt,
        # NOT to the MISSED_FG accounting event (which only fires
        # FG_MISS_TIMEOUT_F frames later, long after the ball was rebounded, so
        # the old `frames_since_miss` window opened too late and rebounds were
        # almost never credited). Two entry paths:
        #   • live shot: SHOOT fired, not yet made (a MADE clears _last_shooter_id
        #     BEFORE this runs — FG detection precedes rebound in process()), past
        #     the minimum ball-flight time.
        #   • post-miss: the air-ball MISSED_FG just fired — keep a short window
        #     after it so those rebounds are still caught.
        live_shot = (
            self._last_shooter_id is not None
            and (self.frame_count - self._shot_attempt_frame) >= self._reb_min_f
            and self._ball_in_hoop_frames == 0   # not currently dwelling for a make
        )
        post_miss = (
            self._miss_frame > 0
            and (self.frame_count - self._miss_frame) <= self._rebound_cooldown_f * 3
        )
        if not (live_shot or post_miss):
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

        # End the shot: the ball has been secured, so no further rebound (or a
        # late MISSED_FG) should fire for this attempt.
        self._rebound_cooldown = self._rebound_cooldown_f
        self._miss_frame       = -999
        self._last_shooter_id  = None
        self._fg_miss_emitted  = True

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

        if self.frame_count - self._pass_frame > self._assist_window_f:
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
        # Path 1 — action classifier says "Block".
        # Gate: cooldown + active shot attempt within BLK_SHOT_WINDOW_F frames.
        # Without the gate, any dribble direction-change + 5% bbox overlap fires a BLK.
        if self._blk_cooldown == 0:
            frames_since_shot = self.frame_count - self._shot_attempt_frame
            if self._last_shooter_id is not None and frames_since_shot <= self._blk_shot_window_f:
                for act in actions:
                    if act.get("action", "").upper() == "BLOCK":
                        tid = act["track_id"]
                        p   = next((x for x in players if x["track_id"] == tid), None)
                        if p and p.get("team") != self._last_shooter_team:
                            self._blk_cooldown = self._blk_cooldown_f
                            return {
                                "type":      "BLK",
                                "track_id":  tid,
                                "quarter":   quarter,
                                "team":      p.get("team", ""),
                                "timestamp_ms": ts_ms,
                            }

        # Path 2 — proximity at shot moment (extended to BLK_PATH2_WINDOW_F frames).
        if (
            self._last_shooter_id is not None
            and self.frame_count - self._shot_attempt_frame <= self._blk_path2_window_f
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

                if self._contact_frames[key] == self._foul_confirm_f:
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

        if self._poss_frames < self._possession_confirm_f:
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
        # Reset the REAL ball/hoop dwell counters (the old `_ball_in_hoop_prev`
        # here was a ghost attribute — defined nowhere, read nowhere — so these
        # leaked across quarters and could trigger a phantom MADE_FG on the first
        # frames of the next quarter).
        self._ball_in_hoop_frames = 0
        self._ball_lost_frames    = 0
        self._hoop_cache_age      = 0
        self._fg_miss_emitted    = False
        self._rebound_cooldown   = 0
        self._miss_frame         = -999
        self._last_shooter_team  = None
        self._pass_frame         = -999
        self._shot_defender      = None
        self._blk_cooldown       = 0
        self._prev_poss_player   = None
        self._prev_poss_team     = None
        self._contact_frames     = {}
        self._last_hoops_px      = []
        self._ball_px_hist       = deque(maxlen=self._fg_above_window_f)
        self._player_court_hist  = {}

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
        ball_pos:       list,
        hoops:          list,
        is_court:       bool,
        fallback_hoops: list = [],
    ):
        """Return (distance, position) of the nearest hoop to ball_pos.

        fallback_hoops: cached pixel-space hoops used when hoops is empty and
        is_court is False — keeps scoring alive when hoop leaves frame.
        """
        best_d, best_h = float("inf"), None

        effective_hoops = hoops if hoops else fallback_hoops
        for h in effective_hoops:
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

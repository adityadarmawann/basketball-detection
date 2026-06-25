export interface Player {
  trackId: number;
  jerseyNumber: number | null;  // null until OCR confirms jersey
  name: string;
  team: 'A' | 'B' | '';
  bbox: [number, number, number, number];
  courtPos: [number, number] | null;
  action: ActionLabel;
  speedKmh: number;
  keypoints: Array<[number, number, number]>;
}

export type ActionLabel =
  | 'SHOOT' | 'DRIBBLE' | 'PASS' | 'CATCH'
  | 'BLOCK' | 'REBOUND' | 'STEAL' | 'JUMP'
  | 'STAND';

export interface FrameUpdate {
  type: 'frame_update';
  timestamp: number;
  quarter: number;
  gameClock: string;
  score: { teamA: number; teamB: number };
  possession: { teamA: number; teamB: number };
  players: Player[];
  ball: { bbox: number[]; courtPos: [number, number]; trajectory: Array<[number, number]> };
  event: GameEvent | null;
  fps?: number;
  gpuMetrics?: { gpu: number; vramUsed: number; vramTotal: number };
}

export interface GameEvent {
  type: 'event';
  eventType: 'FGM' | 'FGA' | 'REB' | 'AST' | 'STL' | 'BLK' | 'TOV' | 'FOUL';
  trackId?: number;       // raw ByteTrack ID — stable anchor for retroactive patches
  playerId: number;       // jersey_number once confirmed, else trackId
  playerName: string;
  team: 'A' | 'B' | '';
  points?: number;
  quarter: number;
  gameClock: string;
  courtPos: [number, number];
}

export interface JerseyConfirmedMessage {
  type: 'jersey_confirmed';
  trackId: number;
  jerseyNumber: number;
  playerName: string;
  team: 'A' | 'B' | '';
}

export interface RosterPlayer {
  jerseyNumber: number;
  name: string;
  team: 'A' | 'B';
  photoUrl?: string;
}

export type SponsorCategory = 'mvp' | 'score' | 'speed' | 'endurance'

export interface Sponsor {
  id: string
  category: SponsorCategory
  companyName: string
  logoUrl: string | null
}

export interface PlayerStats {
  playerId: number;
  name: string;
  jerseyNumber: number | null;
  team: 'A' | 'B' | '';
  minutes: number;
  pts: number;
  twoPointMade: number;
  twoPointAtt: number;
  threePointMade: number;
  threePointAtt: number;
  ftMade: number;
  ftAtt: number;
  fgPercent: number;
  offReb: number;
  defReb: number;
  totReb: number;
  ast: number;
  stl: number;
  tov: number;
  fouls: number;
  blocks: number;
  plusMinus: number;
  eff: number;
}

export interface MpiMetrics {
  playerId: number;
  quarter: number;
  distanceCoveredM: number;
  avgSpeedKmh: number;
  maxSpeedKmh: number;
  jumpHeightCm: number;
  accelerationMs2: number;
  agility: number;
  endurance: number;
  fatigue: number;
  mpiComposite: number;
}

export interface BallState {
  bbox: number[];
  courtPos: [number, number];
  trajectory: Array<[number, number]>;
}

/**
 * Compact per-frame bbox snapshot written by VideoProcessor and served via
 * GET /api/upload/frames/{match_id}.
 *
 * Short field names (i/j/t/b/bl) keep the JSON file small.
 *   ts  — video timestamp in milliseconds
 *   p   — player array
 *     i — trackId
 *     j — jerseyNumber (null until OCR confirms)
 *     t — team ('A' | 'B' | '')
 *     b — normalized bbox [x1, y1, x2, y2] ∈ [0,1]
 *   bl  — ball (null if not detected), same b format
 */
export interface FrameBboxEntry {
  ts: number
  p: Array<{
    i: number
    j: number | null
    t: string
    b: [number, number, number, number]
    a: string        // action UPPER e.g. "DRIBBLE" — empty string if unknown
    s: number        // speedKmh — 0 if not yet computed
    cp: [number, number] | null  // court position [x_m, y_m] for 2D overlay
  }>
  bl: { b: [number, number, number, number]; cp?: [number, number] | null } | null
  sc?: [number, number]    // [scoreA, scoreB] cumulative at this frame
  q_sc?: [number, number]  // [scoreA, scoreB] within current quarter only
  q?: number               // quarter at this frame
}

export interface MatchState {
  matchId: string;
  region: string;
  teamA: { name: string; score: number };
  teamB: { name: string; score: number };
  quarter: number;
  gameClock: string;
  shotClock: number;
  possession: { teamA: number; teamB: number };
  isLive: boolean;
  isVideoMode: boolean;
  players: Player[];
  playersVideoTs: number;
  ball: BallState | null;
  events: GameEvent[];
  stats: Record<number, PlayerStats>;
  mpi: Record<number, MpiMetrics>;
  roster: RosterPlayer[];
  sponsors: Sponsor[];
  videoUrl: string;
  matchStep: string;
  pipelineFps: number;
  gpuMetrics: { gpu: number; vramUsed: number; vramTotal: number };
  quarterScoreA: number;
  quarterScoreB: number;
  jerseyColorA: string;
  jerseyColorB: string;
}

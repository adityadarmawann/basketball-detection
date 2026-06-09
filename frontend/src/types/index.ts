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
}

export interface GameEvent {
  type: 'event';
  eventType: 'FGM' | 'FGA' | 'REB' | 'AST' | 'STL' | 'BLK' | 'TOV' | 'FOUL';
  playerId: number;
  playerName: string;
  team: 'A' | 'B';
  points?: number;
  quarter: number;
  gameClock: string;
  courtPos: [number, number];
}

export interface RosterPlayer {
  jerseyNumber: number;
  name: string;
  team: 'A' | 'B';
}

export interface PlayerStats {
  playerId: number;
  name: string;
  jerseyNumber: number;
  team: 'A' | 'B';
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
  players: Player[];
  playersVideoTs: number;
  ball: BallState | null;
  events: GameEvent[];
  stats: Record<number, PlayerStats>;
  mpi: Record<number, MpiMetrics>;
  roster: RosterPlayer[];
}

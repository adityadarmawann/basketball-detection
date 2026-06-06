/**
 * MPI Formula and calculations
 * Metrics Performance Index for basketball players
 */

export interface MPIComponents {
  power: number      // 0-100, based on jump height and acceleration
  agility: number    // 0-100, based on direction changes and deceleration
  endurance: number  // 0-100, based on distance covered and fatigue
  efficiency: number // 0-100, based on stats (FG%, assist ratio, etc)
  cognitive: number  // 0-100, based on decision making (estimated from events)
}

/**
 * Calculate MPI Composite Score
 * Formula: 0.25×Power + 0.20×Agility + 0.20×Endurance + 0.20×Efficiency + 0.15×Cognitive
 */
export const calculateMPI = (components: MPIComponents): number => {
  return (
    0.25 * components.power +
    0.20 * components.agility +
    0.20 * components.endurance +
    0.20 * components.efficiency +
    0.15 * components.cognitive
  )
}

/**
 * Estimate jump height from max vertical velocity
 * Physics: h = v^2 / (2 * g), g = 9.81 m/s^2
 */
export const estimateJumpHeight = (maxVerticalVelocity: number): number => {
  const g = 9.81
  return (Math.pow(maxVerticalVelocity, 2) / (2 * g)) * 100 // Convert to cm
}

/**
 * Estimate power score from jump height
 */
export const estimatePowerScore = (jumpHeightCm: number): number => {
  // Normalize to 0-100 (max ~80cm for elite players)
  return Math.min(100, (jumpHeightCm / 80) * 100)
}

/**
 * Estimate agility from acceleration/deceleration ratio
 */
export const estimateAgilityScore = (
  avgAcceleration: number,
  avgDeceleration: number
): number => {
  // Lower ratio = better agility (can decelerate quickly)
  const ratio = avgAcceleration / (avgDeceleration + 0.1)
  return Math.max(0, Math.min(100, 100 - ratio * 20))
}

/**
 * Estimate endurance from distance covered and fatigue factor
 */
export const estimateEnduranceScore = (
  distanceCoveredKm: number,
  fatigueIndex: number
): number => {
  // Normalize distance (assume full game = 5km)
  const distanceScore = Math.min(100, (distanceCoveredKm / 5) * 100)
  // Fatigue reduces endurance score
  return distanceScore * (1 - fatigueIndex / 100)
}

/**
 * Estimate efficiency from shooting % and assist rate
 */
export const estimateEfficiencyScore = (
  fgPercent: number,
  astRate: number,
  tovRate: number
): number => {
  return (fgPercent * 0.5 + astRate * 30 - tovRate * 10) / 2
}

/**
 * Estimate cognitive score from assist/turnover ratio and decision making
 */
export const estimateCognitiveScore = (
  assists: number,
  turnovers: number,
  steals: number
): number => {
  const astTovRatio = assists / (turnovers + 1)
  const scoreFromRatio = Math.min(100, astTovRatio * 20)
  const scoreFromSteals = Math.min(20, steals * 5)
  return Math.min(100, scoreFromRatio + scoreFromSteals)
}

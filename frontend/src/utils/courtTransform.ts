/**
 * Court transformation utilities
 * Converts between pixel coordinates (video) and real court coordinates (meters)
 */

/**
 * SVG path data for one FIBA half-court (x: 0=baseline → 14=center, y: 0–15m).
 * The same elements are drawn twice (mirrored) for the full court in CourtMap2D.
 * Coordinates are in meters; D3 applies the pixel-scale transform.
 */
export const courtElements: Array<{ path: string }> = [
  // Outer boundary: baseline + both sidelines (center line drawn separately)
  { path: 'M 0 0 L 0 15 M 0 0 L 14 0 M 0 15 L 14 15' },
  // Paint area (key box): 5.8m deep × 4.9m wide, centred at y=7.5
  { path: 'M 0 5.05 L 5.8 5.05 M 5.8 5.05 L 5.8 9.95 M 5.8 9.95 L 0 9.95' },
  // Free throw circle: centre (5.8, 7.5), r=1.8m
  { path: 'M 4.0 7.5 A 1.8 1.8 0 1 0 7.6 7.5 A 1.8 1.8 0 1 0 4.0 7.5' },
  // 3-point line: corner straights + arc (r=6.75m from basket at 1.575, 7.5)
  { path: 'M 0 0.9 L 2.99 0.9 A 6.75 6.75 0 1 0 2.99 14.1 L 0 14.1' },
  // Centre-circle half (only the half facing inward, x ≤ 14)
  { path: 'M 14 5.7 A 1.8 1.8 0 0 0 14 9.3' },
  // Restricted area (no-charge semicircle): r=1.25m, curves away from baseline
  { path: 'M 1.575 6.25 A 1.25 1.25 0 0 1 1.575 8.75' },
]

export interface HomographyMatrix {
  h: number[][]  // 3x3 matrix
}

/**
 * Apply homography transformation
 * court_pos = H * pixel_pos
 */
export const transformPixelToCourt = (
  pixelPos: [number, number],
  homographyMatrix: HomographyMatrix
): [number, number] => {
  const [px, py] = pixelPos
  const h = homographyMatrix.h

  // Homogeneous coordinates
  const denom = h[2][0] * px + h[2][1] * py + h[2][2]
  const x = (h[0][0] * px + h[0][1] * py + h[0][2]) / denom
  const y = (h[1][0] * px + h[1][1] * py + h[1][2]) / denom

  return [x, y]
}

/**
 * Get court dimensions (FIBA standard)
 */
export const courtDimensions = {
  length: 28,  // meters
  width: 15,   // meters
  threePointDistance: 7.24,  // meters
}

/**
 * Standard basketball court coordinates in meters
 */
export const courtCoordinates = {
  baseline: 0,
  courtEnd: 28,
  sideline: 0,
  courtSideline: 15,
  centerLine: 14,
  threePointLeft: [7.24, 0],
  threePointRight: [7.24, 15],
  freethrowLeft: [5.8, 3.675],
  freethrowRight: [5.8, 11.325],
  hoop1: [1.575, 7.5],  // Left basket
  hoop2: [26.425, 7.5], // Right basket
}

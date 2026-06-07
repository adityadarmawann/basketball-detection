import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { useMatchStore } from '../../store/matchStore'
import { courtElements } from '../../utils/courtTransform'

const COURT_WIDTH = 15
const COURT_LENGTH = 28

export default function CourtMap2D() {
  const svgRef = useRef<SVGSVGElement>(null)
  const players = useMatchStore((s) => s.players)
  const ball = useMatchStore((s) => s.ball)

  // Draw static court lines once on mount
  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)
    const width = 300
    const height = (width / COURT_LENGTH) * COURT_WIDTH

    svg.attr('viewBox', `0 0 ${width} ${height}`)

    // Court background
    svg
      .selectAll<SVGRectElement, unknown>('.court-bg')
      .data([0])
      .join('rect')
      .attr('class', 'court-bg')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', '#C8A96E')

    // Helper to draw one half-court
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const drawHalfCourt = (group: d3.Selection<SVGGElement, any, any, any>, transform: string) => {
      const half = group.append('g').attr('transform', transform)
      half
        .selectAll<SVGPathElement, { path: string }>('path')
        .data(courtElements)
        .join('path')
        .attr('d', (d) => d.path)
        .attr('fill', 'none')
        .attr('stroke', '#FFFFFF')
        .attr('stroke-width', 0.5)
    }

    const sx = width / COURT_LENGTH
    const sy = height / COURT_WIDTH

    const courtGroup = svg.append('g').attr('class', 'court-lines')
    drawHalfCourt(courtGroup, `translate(0,${height}) scale(${sx},${-sy})`)
    drawHalfCourt(courtGroup, `translate(${width},${height}) scale(${-sx},${-sy})`)

    // Center line
    svg
      .append('line')
      .attr('x1', width / 2).attr('y1', 0)
      .attr('x2', width / 2).attr('y2', height)
      .attr('stroke', '#FFFFFF')
      .attr('stroke-width', 0.5)
  }, [])

  // Update player dots and ball on every frame
  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)
    const width = 300
    const height = (width / COURT_LENGTH) * COURT_WIDTH

    const xScale = d3.scaleLinear().domain([0, COURT_LENGTH]).range([0, width])
    const yScale = d3.scaleLinear().domain([0, COURT_WIDTH]).range([height, 0])

    svg
      .selectAll<SVGCircleElement, typeof players[number]>('.player-dot')
      .data(players, (d) => d.trackId)
      .join(
        (enter) =>
          enter
            .append('circle')
            .attr('class', 'player-dot')
            .attr('r', 4)
            .attr('stroke', 'white')
            .attr('stroke-width', 1),
        (update) => update,
        (exit) => exit.remove()
      )
      .attr('cx', (d) => xScale(d.courtPos[0]))
      .attr('cy', (d) => yScale(d.courtPos[1]))
      .attr('fill', (d) => (d.team === 'A' ? '#00BCD4' : '#1A1A2E'))

    if (ball?.courtPos) {
      svg
        .selectAll<SVGCircleElement, typeof ball>('.ball-dot')
        .data([ball])
        .join('circle')
        .attr('class', 'ball-dot')
        .attr('r', 3)
        .attr('fill', '#F59E0B')
        .attr('cx', xScale(ball.courtPos[0]))
        .attr('cy', yScale(ball.courtPos[1]))
    }
  }, [players, ball])

  return (
    <div className="bg-black/70 backdrop-blur-sm border border-white/20 rounded-lg p-2">
      <svg ref={svgRef} className="w-full" />
    </div>
  )
}

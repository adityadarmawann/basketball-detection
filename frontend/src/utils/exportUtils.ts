import jsPDF from 'jspdf'
import Papa from 'papaparse'
import { PlayerStats } from '../types'

/**
 * Export match statistics to CSV file
 */
export const exportToCSV = (
  data: PlayerStats[],
  filename: string = 'match-stats.csv'
) => {
  const csv = Papa.unparse({
    fields: [
      '#',
      'Name',
      'Team',
      'MIN',
      'PTS',
      '2P',
      '3P',
      'FT',
      'FG%',
      'OFF',
      'DEF',
      'TOT',
      'AST',
      'STL',
      'TOV',
      'C',
      'R',
      'M',
      '+/-',
      'EFF',
    ],
    data: data.map((stat) => [
      stat.jerseyNumber,
      stat.name,
      stat.team,
      (stat.minutes / 60).toFixed(1),
      stat.pts,
      `${stat.twoPointMade}/${stat.twoPointAtt}`,
      `${stat.threePointMade}/${stat.threePointAtt}`,
      `${stat.ftMade}/${stat.ftAtt}`,
      stat.fgPercent.toFixed(1),
      stat.offReb,
      stat.defReb,
      stat.totReb,
      stat.ast,
      stat.stl,
      stat.tov,
      stat.fouls,
      stat.blocks,
      0, // Not in current schema
      stat.plusMinus,
      stat.eff.toFixed(1),
    ]),
  })

  const link = document.createElement('a')
  link.href = `data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`
  link.download = filename
  link.click()
}

/**
 * Export match statistics to PDF file
 */
export const exportToPDF = (
  teamA: string,
  teamB: string,
  scoreA: number,
  scoreB: number,
  _data: PlayerStats[],
  filename: string = 'match-stats.pdf'
) => {
  const doc = new jsPDF()
  const pageWidth = doc.internal.pageSize.getWidth()

  // Title
  doc.setFontSize(18)
  doc.text('Match Statistics Report', pageWidth / 2, 15, { align: 'center' })

  // Match info
  doc.setFontSize(12)
  doc.text(`${teamA} ${scoreA} - ${scoreB} ${teamB}`, pageWidth / 2, 25, { align: 'center' })

  // Auto table would be nice here but keeping it simple
  doc.setFontSize(10)
  doc.text('Detailed stats table to be rendered...', 10, 40)

  doc.save(filename)
}

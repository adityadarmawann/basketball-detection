import { useMemo, useState } from 'react'
import { PlayerStats } from '../../types'
import { ArrowDown, ArrowUp } from 'lucide-react'
import clsx from 'clsx'

type SortableKey = keyof PlayerStats | '2P' | '3P' | 'FT'

interface SortConfig {
  key: SortableKey
  direction: 'ascending' | 'descending'
}

interface PlayerStatsTableProps {
  players: PlayerStats[]
  teamName: string
}

const useSortableData = (items: PlayerStats[], config: SortConfig | null = null) => {
  const [sortConfig, setSortConfig] = useState(config)

  const sortedItems = useMemo(() => {
    let sortableItems = [...items]
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        const aVal = a[sortConfig.key as keyof PlayerStats]
        const bVal = b[sortConfig.key as keyof PlayerStats]

        if (typeof aVal === 'number' && typeof bVal === 'number') {
          if (aVal < bVal) {
            return sortConfig.direction === 'ascending' ? -1 : 1
          }
          if (aVal > bVal) {
            return sortConfig.direction === 'ascending' ? 1 : -1
          }
        }
        return 0
      })
    }
    return sortableItems
  }, [items, sortConfig])

  const requestSort = (key: SortableKey) => {
    let direction: 'ascending' | 'descending' = 'descending'
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'descending') {
      direction = 'ascending'
    }
    setSortConfig({ key, direction })
  }

  return { items: sortedItems, requestSort, sortConfig }
}

const columns: { label: string; key: SortableKey; sortable: boolean }[] = [
  { label: '#', key: 'jerseyNumber', sortable: true },
  { label: 'Name', key: 'name', sortable: false },
  { label: 'MIN', key: 'minutes', sortable: true },
  { label: 'PTS', key: 'pts', sortable: true },
  { label: '2P', key: '2P', sortable: false },
  { label: '3P', key: '3P', sortable: false },
  { label: 'FT', key: 'ftMade', sortable: true },
  { label: 'FG%', key: 'fgPercent', sortable: true },
  { label: 'REB', key: 'totReb', sortable: true },
  { label: 'AST', key: 'ast', sortable: true },
  { label: 'STL', key: 'stl', sortable: true },
  { label: 'BLK', key: 'blocks', sortable: true },
  { label: 'TOV', key: 'tov', sortable: true },
  { label: '+/-', key: 'plusMinus', sortable: true },
  { label: 'EFF', key: 'eff', sortable: true },
]

export default function PlayerStatsTable({ players, teamName }: PlayerStatsTableProps) {
  const { items, requestSort, sortConfig } = useSortableData(players, { key: 'pts', direction: 'descending' });
  const highestEff = useMemo(() => Math.max(...players.map(p => p.eff)), [players]);

  const getSortIcon = (key: SortableKey) => {
    if (!sortConfig || sortConfig.key !== key) {
      return null
    }
    if (sortConfig.direction === 'ascending') {
      return <ArrowUp size={12} className="inline ml-1" />
    }
    return <ArrowDown size={12} className="inline ml-1" />
  }

  return (
    <div className="bg-surface rounded-lg p-4 shadow-sm overflow-x-auto">
      <h3 className="font-display text-xl font-bold mb-4 px-2">{teamName}</h3>
      <table className="w-full text-sm text-left">
        <thead className="border-b-2 border-gray-300 text-xs text-text-secondary uppercase">
          <tr>
            {columns.map((col) => (
              <th key={col.key} className="py-3 px-2">
                {col.sortable ? (
                  <button onClick={() => requestSort(col.key)} className="font-bold">
                    {col.label}
                    {getSortIcon(col.key)}
                  </button>
                ) : (
                  col.label
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((p) => (
            <tr
              key={p.playerId}
              className={clsx(
                'border-b border-gray-200 hover:bg-gray-50',
                p.eff === highestEff && highestEff > 0 && 'bg-accent-gold/10'
              )}
            >
              <td className="py-3 px-2 font-bold">{p.jerseyNumber}</td>
              <td className="py-3 px-2 font-bold text-text-primary">{p.name}</td>
              <td className="py-3 px-2 tabular-nums">{p.minutes.toFixed(1)}</td>
              <td className="py-3 px-2 tabular-nums font-bold">{p.pts}</td>
              <td className="py-3 px-2 tabular-nums">{`${p.twoPointMade}/${p.twoPointAtt}`}</td>
              <td className="py-3 px-2 tabular-nums">{`${p.threePointMade}/${p.threePointAtt}`}</td>
              <td className="py-3 px-2 tabular-nums">{`${p.ftMade}/${p.ftAtt}`}</td>
              <td className="py-3 px-2 tabular-nums">{p.fgPercent.toFixed(1)}%</td>
              <td className="py-3 px-2 tabular-nums">{p.totReb}</td>
              <td className="py-3 px-2 tabular-nums">{p.ast}</td>
              <td className="py-3 px-2 tabular-nums">{p.stl}</td>
              <td className="py-3 px-2 tabular-nums">{p.blocks}</td>
              <td className="py-3 px-2 tabular-nums">{p.tov}</td>
              <td
                className={clsx('py-3 px-2 tabular-nums font-bold', {
                  'text-success': p.plusMinus > 0,
                  'text-danger': p.plusMinus < 0,
                })}
              >
                {p.plusMinus > 0 ? `+${p.plusMinus}` : p.plusMinus}
              </td>
              <td className="py-3 px-2 tabular-nums font-bold">{p.eff}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
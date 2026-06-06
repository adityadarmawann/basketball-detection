import clsx from 'clsx'
import { useState } from 'react'

interface QuarterFilterProps {
  onFilterChange: (quarter: number) => void
}

const QUARTERS = [
  { label: 'All', value: 0 },
  { label: 'Q1', value: 1 },
  { label: 'Q2', value: 2 },
  { label: 'Q3', value: 3 },
  { label: 'Q4', value: 4 },
]

export default function QuarterFilter({ onFilterChange }: QuarterFilterProps) {
  const [selected, setSelected] = useState(0)

  const handleSelect = (value: number) => {
    setSelected(value)
    onFilterChange(value)
  }

  return (
    <div className="flex gap-2">
      {QUARTERS.map((q) => (
        <button
          key={q.value}
          onClick={() => handleSelect(q.value)}
          className={clsx(
            'px-4 py-2 rounded-full text-sm font-bold transition-smooth',
            selected === q.value
              ? 'bg-primary text-white'
              : 'bg-surface text-text-secondary border border-gray-200 hover:border-primary hover:text-primary'
          )}
        >
          {q.label}
        </button>
      ))}
    </div>
  )
}

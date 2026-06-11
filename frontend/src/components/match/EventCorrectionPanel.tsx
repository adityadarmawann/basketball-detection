import { useState, useMemo } from 'react'
import axios from 'axios'
import { AlertTriangle, CheckCircle, Loader } from 'lucide-react'
import { useMatchStore } from '../../store/matchStore'

interface CorrectionForm {
  team: 'A' | 'B' | ''
  jerseyNumber: number | ''
}

interface SubmitState {
  trackId: number
  status: 'loading' | 'ok' | 'error'
  message?: string
}

export default function EventCorrectionPanel() {
  const { stats, roster, teamA, teamB, matchId, patchEventsByTrackId } = useMatchStore()

  const [forms, setForms]       = useState<Record<number, CorrectionForm>>({})
  const [submits, setSubmits]   = useState<Record<number, SubmitState>>({})

  // Unidentified players with at least 1 point or field goal attempt
  const unidentified = useMemo(() =>
    Object.values(stats).filter(
      (p) => p.jerseyNumber == null && (p.pts > 0 || p.twoPointAtt > 0 || p.threePointAtt > 0)
    ),
  [stats])

  const rosterByTeam = useMemo(() => ({
    A: roster.filter((r) => r.team === 'A'),
    B: roster.filter((r) => r.team === 'B'),
  }), [roster])

  const setField = (trackId: number, field: keyof CorrectionForm, value: string | number) => {
    setForms((prev) => ({
      ...prev,
      [trackId]: {
        ...{ team: '', jerseyNumber: '' },
        ...prev[trackId],
        [field]: value,
        // reset player when team changes
        ...(field === 'team' ? { jerseyNumber: '' } : {}),
      },
    }))
  }

  const handleSubmit = async (trackId: number) => {
    const form = forms[trackId]
    if (!form?.team || form.jerseyNumber === '') return

    const player = roster.find(
      (r) => r.jerseyNumber === Number(form.jerseyNumber) && r.team === form.team
    )
    if (!player) return

    setSubmits((prev) => ({ ...prev, [trackId]: { trackId, status: 'loading' } }))

    try {
      await axios.post(`${import.meta.env.VITE_API_URL}/api/events/correct`, {
        match_id:      matchId,
        track_id:      trackId,
        team:          form.team,
        jersey_number: player.jerseyNumber,
        player_name:   player.name,
      })

      // Retroactively patch event log in store
      patchEventsByTrackId(trackId, player.jerseyNumber, player.name, form.team)

      setSubmits((prev) => ({
        ...prev,
        [trackId]: { trackId, status: 'ok', message: `Dikonfirmasi: #${player.jerseyNumber} ${player.name}` },
      }))
    } catch (e) {
      setSubmits((prev) => ({
        ...prev,
        [trackId]: { trackId, status: 'error', message: 'Gagal menyimpan koreksi' },
      }))
    }
  }

  if (unidentified.length === 0) {
    return (
      <div className="p-8 text-center text-text-secondary">
        <CheckCircle className="mx-auto mb-3 text-success" size={32} />
        <p className="font-bold">Semua scoring event sudah teridentifikasi</p>
        <p className="text-xs mt-1">Tidak ada event yang membutuhkan koreksi manual</p>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-start gap-2 p-3 bg-amber-50 border border-amber-200 rounded-lg">
        <AlertTriangle className="text-amber-500 flex-shrink-0 mt-0.5" size={16} />
        <div className="text-xs text-amber-800">
          <span className="font-bold">{unidentified.length} player</span> memiliki scoring event
          yang belum teridentifikasi nomornya oleh AI. Pilih tim dan player yang sesuai untuk
          setiap entry.
        </div>
      </div>

      {unidentified.map((player) => {
        const trackId   = player.playerId
        const form      = forms[trackId] ?? { team: '', jerseyNumber: '' }
        const submitSt  = submits[trackId]
        const isDone    = submitSt?.status === 'ok'
        const isLoading = submitSt?.status === 'loading'
        const teamPlayers = form.team ? rosterByTeam[form.team] : []

        return (
          <div
            key={trackId}
            className={`border rounded-lg overflow-hidden transition-all ${
              isDone ? 'border-success bg-success/5' : 'border-gray-200 bg-surface'
            }`}
          >
            {/* Header row */}
            <div className="px-4 py-3 bg-gray-50 border-b border-gray-100 flex items-center justify-between">
              <div>
                <span className="text-xs font-bold text-orange-500 uppercase tracking-wide">
                  ?? Unidentified
                </span>
                <span className="ml-2 text-xs text-text-secondary">ID internal: {trackId}</span>
              </div>
              <div className="flex gap-3 text-xs font-bold">
                {player.pts > 0 && (
                  <span className="text-danger">{player.pts} PTS</span>
                )}
                {(player.twoPointAtt + player.threePointAtt) > 0 && (
                  <span className="text-text-secondary">
                    {player.twoPointMade + player.threePointMade}/
                    {player.twoPointAtt + player.threePointAtt} FG
                  </span>
                )}
                {player.totReb > 0 && (
                  <span className="text-text-secondary">{player.totReb} REB</span>
                )}
              </div>
            </div>

            {isDone ? (
              <div className="px-4 py-3 flex items-center gap-2 text-sm text-success font-bold">
                <CheckCircle size={16} />
                {submitSt.message}
              </div>
            ) : (
              <div className="px-4 py-3 space-y-3">
                {/* Team selector */}
                <div className="flex gap-2">
                  {(['A', 'B'] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => setField(trackId, 'team', t)}
                      className={`flex-1 py-2 rounded-lg text-sm font-bold border transition-all ${
                        form.team === t
                          ? t === 'A'
                            ? 'bg-team-a text-white border-team-a'
                            : 'bg-team-b text-white border-team-b'
                          : 'border-gray-200 text-text-secondary hover:border-primary hover:text-primary'
                      }`}
                    >
                      {t === 'A' ? (teamA.name || 'Tim A') : (teamB.name || 'Tim B')}
                    </button>
                  ))}
                </div>

                {/* Player selector */}
                <select
                  value={form.jerseyNumber}
                  onChange={(e) => setField(trackId, 'jerseyNumber', Number(e.target.value))}
                  disabled={!form.team}
                  className="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg
                             focus:outline-none focus:border-primary disabled:opacity-40
                             disabled:bg-gray-50 bg-white"
                >
                  <option value="">
                    {form.team ? 'Pilih player...' : 'Pilih tim dulu'}
                  </option>
                  {teamPlayers.map((r) => (
                    <option key={r.jerseyNumber} value={r.jerseyNumber}>
                      #{r.jerseyNumber} — {r.name}
                    </option>
                  ))}
                </select>

                {/* Error message */}
                {submitSt?.status === 'error' && (
                  <p className="text-xs text-danger">{submitSt.message}</p>
                )}

                {/* Submit */}
                <button
                  onClick={() => handleSubmit(trackId)}
                  disabled={!form.team || form.jerseyNumber === '' || isLoading}
                  className="w-full py-2 bg-primary text-white text-sm font-bold rounded-lg
                             hover:bg-primary-dark transition-smooth
                             disabled:opacity-40 disabled:cursor-not-allowed
                             flex items-center justify-center gap-2"
                >
                  {isLoading
                    ? <><Loader size={14} className="animate-spin" /> Menyimpan...</>
                    : 'Simpan Koreksi'
                  }
                </button>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

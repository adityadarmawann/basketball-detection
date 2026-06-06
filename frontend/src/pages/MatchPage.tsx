import { useState, useMemo, useEffect } from 'react'
import { useMatchStore } from '../store/matchStore'
import FormSetupMatch from '../components/match/FormSetupMatch'
import RosterManager from '../components/match/RosterManager'
import MatchHeader from '../components/match/MatchHeader'
import VideoUpload from '../components/video/VideoUpload'
import VideoPlayer from '../components/video/VideoPlayer'
import Scoreboard from '../components/dashboard/Scoreboard'
import MvpRanking from '../components/dashboard/MvpRanking'
import LiveEventFeed from '../components/dashboard/LiveEventFeed'
import TabsStatsLineup from '../components/match/TabsStatsLineup'
import { useWebSocket } from '../hooks/useWebSocket'

type MatchStep = 'setup' | 'roster' | 'upload' | 'live'

export default function MatchPage() {
  const [step, setStep] = useState<MatchStep>('setup')
  const [videoUrl, setVideoUrl] = useState<string>('')

  const store = useMatchStore()
  const { connect, disconnect } = useWebSocket()

  useEffect(() => {
    if (step === 'live') {
      connect()
    }
    return () => {
      disconnect()
    }
  }, [step, connect, disconnect])

  const handleUploadComplete = (result: { videoId: string; videoUrl: string }) => {
    setVideoUrl(result.videoUrl)
    setStep('live')
  }

  const mvpPlayers = useMemo(() => {
    return Object.values(store.stats)
      .map((player) => ({
        jerseyNumber: player.jerseyNumber,
        name: player.name,
        eff: player.eff,
        mpi: store.mpi[player.playerId]?.mpiComposite || 0,
      }))
      .sort((a, b) => b.eff - a.eff)
  }, [store.stats, store.mpi, store.events])

  return (
    <div className="container mx-auto py-8 px-4">
      {/* Hero Banner */}
      <div className="bg-gradient-to-r from-primary-dark to-primary rounded-lg p-8 mb-8 text-white">
        <h1 className="font-display text-4xl font-bold">Basketball</h1>
        <p className="text-white/70 mt-2">Campus League Regional Analytics</p>
      </div>

      {step === 'setup' && (
        <FormSetupMatch onComplete={() => setStep('roster')} />
      )}

      {step === 'roster' && store.matchId && (
        <RosterManager
          matchId={store.matchId}
          teamA={store.teamA.name}
          teamB={store.teamB.name}
          onComplete={() => setStep('upload')}
        />
      )}

      {step === 'upload' && store.matchId && (
        <VideoUpload onUploadComplete={handleUploadComplete} />
      )}

      {step === 'live' && (
        <div className="space-y-6">
          {/* Match Header */}
          <MatchHeader
            teamA={store.teamA.name}
            teamB={store.teamB.name}
            scoreA={store.teamA.score}
            scoreB={store.teamB.score}
            quarter={store.quarter}
            gameClock={store.gameClock}
            shotClock={store.shotClock}
            isLive={store.isLive}
          />

          {/* Video Player */}
          {videoUrl && <VideoPlayer videoUrl={videoUrl} showCourtMap={true} />}

          {/* Dashboard Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Scoreboard
              teamA={store.teamA.name}
              teamB={store.teamB.name}
              scoreA={store.teamA.score}
              scoreB={store.teamB.score}
              quarter={store.quarter}
              gameClock={store.gameClock}
              shotClock={store.shotClock}
            />

            <MvpRanking players={mvpPlayers} />

            <LiveEventFeed
              events={store.events}
              teamA={store.teamA.name}
              teamB={store.teamB.name}
            />
          </div>

          {/* Stats Tabs */}
          <TabsStatsLineup />
        </div>
      )}
    </div>
  )
}

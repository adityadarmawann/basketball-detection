import { useCallback } from 'react'
import { useMatchStore } from '../store/matchStore'
import { FrameBboxEntry } from '../types'

/**
 * Menyalurkan posisi pemutaran video ke store: jam, skor, skor per kuarter,
 * dan nomor kuarter.
 *
 * KENAPA JADI HOOK
 * Logika ini dulu ditulis DUA KALI: di MatchPage untuk hasil analisis baru, dan
 * disalin ulang di ViewMatchPage untuk hasil yang dibuka dari riwayat. Salinan
 * di riwayat menulis ke state LOKAL, bukan ke store, dan mengabaikan `q_sc`
 * sama sekali. Akibatnya angka di riwayat berbeda dari angka setelah analisis,
 * dan tab statistik per kuarter ikut salah.
 *
 * Satu hook dipakai kedua halaman, jadi mustahil keduanya berbeda lagi.
 *
 * Skor sengaja mengikuti POSISI VIDEO, bukan nilai akhir. Menjeda video
 * membekukan skor di titik itu, dan itu memang yang diinginkan: angkanya harus
 * cocok dengan apa yang sedang terlihat di layar.
 */
export function useWaktuVideoKeStore(frameData: FrameBboxEntry[]) {
  // Ambil setter-nya saja, bukan seluruh store. Setter di zustand stabil, jadi
  // penangan ini tidak dibuat ulang tiap store berubah.
  const setGameClock    = useMatchStore((s) => s.setGameClock)
  const setScore        = useMatchStore((s) => s.setScore)
  const setQuarterScore = useMatchStore((s) => s.setQuarterScore)
  const setQuarter      = useMatchStore((s) => s.setQuarter)

  return useCallback((currentTime: number) => {
    // Jam mengikuti posisi pemutaran (MM:SS berjalan), bukan hitung mundur
    // FIBA, karena video mentah memuat jeda dan timeout sehingga 1 detik video
    // tidak sama dengan 1 detik permainan.
    const mm = String(Math.floor(currentTime / 60)).padStart(2, '0')
    const ss = String(Math.floor(currentTime % 60)).padStart(2, '0')
    setGameClock(`${mm}:${ss}`)

    if (frameData.length === 0) return

    // Cari biner frame terdekat dengan posisi video sekarang.
    const videoMs = currentTime * 1000
    let lo = 0, hi = frameData.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (frameData[mid].ts < videoMs) lo = mid + 1
      else hi = mid
    }
    const a = frameData[lo]
    const b = lo > 0 ? frameData[lo - 1] : null
    const snap = b && Math.abs(b.ts - videoMs) < Math.abs(a.ts - videoMs) ? b : a

    if (snap.sc) setScore(snap.sc[0], snap.sc[1])
    if (snap.q_sc) setQuarterScore(snap.q_sc[0], snap.q_sc[1])
    if (snap.q != null) setQuarter(snap.q)
  }, [frameData, setGameClock, setScore, setQuarterScore, setQuarter])
}

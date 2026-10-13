# Alur Penyajian Hasil Pertandingan

Ditulis 13 Oktober 2026 setelah audit menyeluruh. Dokumen ini ada karena hasil
pertandingan pernah tampil BERBEDA antara "baru selesai dianalisis" dan "dibuka
lagi dari riwayat", dan bedanya hanya ketahuan satu per satu lewat laporan
pengguna. Sebabnya bukan satu bug, melainkan **dua jalur kode yang mengerjakan
hal yang sama**. Dokumen ini menjelaskan susunan yang sekarang, supaya tidak
ada lagi jalur kedua yang diam diam menyimpang.

---

## 1. Dua pintu, satu ruangan

Ada dua cara pengguna sampai ke halaman hasil:

| Pintu | Rute | Kapan |
|---|---|---|
| Selesai menganalisis | `/match` tahap `live` | setelah video selesai diproses |
| Riwayat | `/view/{match_id}` | membuka hasil lama dari lencana pengguna |

Keduanya merender komponen yang **sama persis**:

```
HasilPertandingan.tsx
  ├── Video Original            (tag <video> biasa)
  ├── MatchHeader               papan skor dan jam
  ├── Video Analisis AI
  │     └── VideoPlayer
  │           ├── CourtOverlay  kotak pemain, digambar dari frameData
  │           └── CourtMap2D    denah lapangan
  ├── Scoreboard
  ├── RewardBoxes               kartu MVP, Top Scorer, Top Speed, Endurance
  ├── TabsStatsLineup           tab STATS dan LINEUPS, tombol All dan Q1-Q4
  └── LiveEventFeed             RIWAYAT EVENT
```

Dua tab lain di navigasi atas adalah halaman terpisah yang membaca store yang
sama: `LineupPage` (`/match/lineups`) dan `MpiPage` (`/match/mpi`).

**Aturan: jangan pernah menulis tata letak hasil di halaman.** Tambahkan di
`HasilPertandingan.tsx` saja, supaya kedua pintu berubah bersamaan.

---

## 2. Satu pengisi keadaan

Keduanya memanggil fungsi yang sama untuk mengisi store:

```
hooks/muatHasil.ts  →  muatHasilKeStore(matchId, { videoUrlAda })
```

Yang dikerjakannya:

```
GET /api/match/{id}            → nama tim, wilayah
GET /api/upload/frames/{id}    → frameData, lalu tanamDariFrameTerakhir()
GET /api/roster/{id}           → roster (jersey_number → jerseyNumber, foto diberi awalan API)
GET /api/sponsors/{id}         → sponsor (company_name → companyName, logo diberi awalan API)
```

lalu **membersihkan** bidang yang hanya bermakna selama analisis berjalan.

Stats dan MPI tidak diambil di sini. Keduanya diisi `useMatchStats(matchId)`,
hook yang sama di kedua pintu:

```
GET /api/stats/live?match_id=       → store.stats
GET /api/mpi/team?match_id=         → store.mpi
GET /api/stats/quarter/{q}?match_id= → saat tab kuarter dipilih
```

---

## 3. Bidang store: mana hasil, mana sementara

Audit menemukan **19 bidang** store dibaca pohon hasil. Pembagiannya:

### Bidang HASIL, diisi dari server

| Bidang | Sumber |
|---|---|
| `matchId` | alamat atau formulir |
| `teamA.name`, `teamB.name`, `region` | `/api/match/{id}` |
| `teamA.score`, `teamB.score` | frame TERAKHIR (`sc`) |
| `quarterScoreA`, `quarterScoreB` | frame TERAKHIR (`q_sc`) |
| `quarter` | frame TERAKHIR (`q`) |
| `gameClock` | waktu frame TERAKHIR, lalu ikut posisi video |
| `shotClock` | selalu 24, tidak pernah diubah siapa pun |
| `roster` | `/api/roster/{id}` |
| `sponsors` | `/api/sponsors/{id}` |
| `stats`, `mpi` | `useMatchStats` |
| `videoUrl` | blob unggahan kalau masih ada, selain itu `/api/video/{id}` |
| `matchStep` | selalu `live` |

### Bidang SEMENTARA, WAJIB dibersihkan saat analisis selesai

| Bidang | Kalau tidak dibersihkan |
|---|---|
| `isLive`, `wsStatus` | lencana "Live" tertinggal menyala |
| `pipelineFps`, `gpuMetrics` | lencana "9.7 fps" dan "GPU 85%" tertinggal di atas video |
| `players`, `playersVideoTs`, `ball` | sisa frame WebSocket terakhir |
| `possession` | sisa nilai terakhir |
| `events` | daftar sementara; `LiveEventFeed` mengambil sendiri dari MongoDB saat `isLive` bernilai false |

**Aturan: setiap bidang store baru harus masuk salah satu daftar di atas.**
Kalau dia bagian hasil, isi di `muatHasilKeStore`. Kalau dia hanya berlaku
selama analisis, bersihkan di sana juga.

---

## 4. Dari mana angkanya, dan berapa lama bertahan

`/api/stats/live` memakai rantai berlapis, berhenti di yang pertama ada:

| Urutan | Sumber | Masa berlaku |
|---|---|---|
| 1 | Redis `stats:{id}` (gabungan) | 1 hari |
| 2 | Redis `stats:{id}:q{n}` | 7 hari |
| 3 | Mongo `matches.ringkasan.gabungan` | **permanen** |
| 4 | bentuk kosong | tidak pernah melempar galat |

Dua jenis ringkasan, jangan tertukar:

- `ringkasan.q{n}` statistik **mentah** per kuarter, satu baris per track_id
- `ringkasan.gabungan` hasil `_merge_all_quarters`, disatukan per (nomor
  punggung, tim). **Inilah yang ditampilkan.**

Bedanya besar dan terukur: 1292 baris mentah menjadi 488 pemain, 3357 menjadi
1034, 3358 menjadi 1024. Kalau rantai di atas mengembalikan yang mentah,
angka riwayat berubah dari yang baru saja dilihat pengguna.

MPI diambil dari `stats:{id}` yang berlaku hanya **satu hari**, jadi itu yang
paling cepat kosong. Cadangannya `ringkasan.gabungan.player_stats`, karena
bagian `mpi` tersimpan di dalam tiap pemain.

### Statistik per kuarter

`/api/stats/quarter/{q}` **tidak** membaca kunci `stats:{id}:q{q}` apa adanya.
Alasannya cacat penamaan di `video_processor.py`:

```python
q_key = f"stats:{match_id}:q{self._start_quarter}"
```

Kunci dinamai memakai kuarter AWAL unggahan, bukan kuarter yang sedang
berjalan. Jadi satu video yang melintasi Q1 sampai Q4 menumpuk seluruh
statistiknya di `:q1`, dan tab Q2, Q3, Q4 tampil kosong. Itu gejala "cuma tab
All yang terisi".

Angka per kuarter sebenarnya sudah dicatat `stats_calculator` di dalam tiap
pemain sebagai `quarter_stats`, bentuk bidangnya sama persis dengan
`total_stats`. Endpoint membacanya dari sana, jadi benar untuk unggahan per
klip maupun satu video penuh.

Kuarter yang memang tidak pernah diunggah dikembalikan **kosong**, bukan diisi
angka kuarter lain.

---

## 5. Satu beda yang disengaja

Hanya satu bagian yang berbeda antara dua pintu: **baris kendali di atas
hasil**, yaitu tombol "Video & Analisis", penanda sambungan langsung, dan
lencana Q1 sampai Q4 untuk mengunggah kuarter berikutnya. Semuanya terikat
alur unggah dan tidak bermakna saat membuka hasil lama, jadi halaman riwayat
tidak mengirimkannya.

Dikirim sebagai prop `kendali` dan `statusAnalisis`, bukan ditulis di dalam
komponen hasil.

---

## 6. Cara memeriksa, jangan percaya kode saja

Membaca kode tidak membuktikan apa yang sampai ke layar. Dua kesalahan ini
nyata dan pernah lolos pemeriksaan kode:

- Menguji muatan MongoDB dengan data yang diambil dari Redis selalu **lolos
  palsu**, karena JSON sudah mengubah kunci angka jadi teks lebih dulu.
- Log yang digerepe ternyata berkas yang sudah mati, sementara proses yang
  hidup menulis ke berkas lain.

Maka ada pemeriksa yang merender halamannya sungguhan:

```bash
cd basketball-dev/frontend
python scripts/periksa_tampilan.py <match_id> [http://127.0.0.1:5174]
```

Ia membuka `/view/{id}`, `/match/lineups`, dan `/match/mpi` di Chrome tanpa
tampilan, memastikan delapan bagian wajib benar benar muncul, menyimpan
potret layar, dan keluar dengan kode 1 kalau ada yang hilang.

Untuk angka, bandingkan jalur Redis dengan jalur Mongo memakai match tiruan
yang tidak punya kunci Redis sama sekali. Untuk mengisi ulang ringkasan
analisis lama:

```bash
cd basketball-dev/backend
python scripts/isi_ringkasan_dari_redis.py          # lihat dulu
python scripts/isi_ringkasan_dari_redis.py --tulis  # baru tulis
```

---

## 7. Daftar periksa sebelum menyentuh halaman hasil

1. Tata letak hanya di `HasilPertandingan.tsx`, bukan di halaman.
2. Bidang store baru: masuk daftar HASIL atau daftar SEMENTARA di bagian 3.
3. Pengambilan data baru: taruh di `muatHasilKeStore`, bukan di salah satu halaman.
4. Perhitungan dari frame: taruh di `useWaktuVideoKeStore.ts`, bukan disalin.
5. Endpoint baru yang dipakai hasil: beri cadangan Mongo, jangan hanya Redis.
6. Jalankan `scripts/periksa_tampilan.py` sebelum menyatakan beres.

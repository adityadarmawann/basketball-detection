# Perubahan Kecepatan Analisis

Catatan ini ada supaya setiap perubahan demi kecepatan bisa **dikembalikan**
atau **diperbaiki** kalau ternyata merusak hasil. Tiap entri memuat apa yang
diubah, kenapa, buktinya, cara membatalkan, dan apa yang harus dicurigai kalau
hasil analisis jadi aneh.

Aturan yang dipakai: percepatan hanya boleh menghapus pekerjaan yang hasilnya
TIDAK dipakai. Kalau sebuah perubahan mengurangi apa yang dihitung atau
ditampilkan, itu bukan percepatan, itu penukaran akurasi, dan harus diukur
lebih dulu.

---

## Keadaan awal, sebagai pembanding

Video `Q1-UNESA-UBAYA-CUT-2-17.mp4`: 1280x720, 60 fps, 36078 frame, 601 detik.
`PROCESS_STRIDE=2` membuat 18039 frame diproses, yaitu 30 frame per detik
efektif.

| Sisi | Waktu | Kecepatan | Frame terbuang |
|---|---|---|---|
| prod 20261012_225512 | 2:02:47 | 4,9 fps | 4 |
| dev 20261012_201407 | 2:37:20 | 3,8 fps | 121 |

Rincian waktu per tahap, dari baris TIMING di log, rata rata per frame:

| Tahap | PROD | DEV | Jalan di berapa % frame | Sekali jalan (dev) |
|---|---|---|---|---|
| sts | 129 ms | **210 ms** | 99,9% | 211 ms |
| act | 65 ms | 166 ms | 24,6% | 673 ms |
| ocr | 71 ms | 153 ms | 68,7% | 223 ms |
| pose | 46 ms | 55 ms | 25,0% | 219 ms |
| court | 46 ms | 39 ms | 43,1% | 89 ms |
| trk | 18 ms | 13 ms | 100% | 13 ms |
| **TOTAL** | **376 ms** | **636 ms** | | |

---

## 1. Melewati penyusunan statistik yang dibuang (13 Oktober 2026)

**Commit** `b66ffdb`

**Masalahnya.** `StatsCalculator.update()` diakhiri `return self.get_live_stats()`.
`get_live_stats()` menyusun ulang struktur lengkap untuk SEMUA track_id
(1292 pada rekaman nyata) beserta MPI tiap pemain dan peringkat MVP. Sementara
satu satunya pemanggil di pipeline, di `video_processor.py`, **membuang** nilai
kembaliannya. Jadi 211 ms per frame dikerjakan lalu dibuang, di 99,9 persen
dari 18039 frame.

**Yang diubah.** `update()` menerima `bangun_keluaran`. Saat `False`, ia tetap
mencatat semuanya tapi tidak menyusun hasil bacanya dan mengembalikan `None`.
Pemanggil di pipeline memakai `False`.

**Buktinya tidak merusak.** Dua kalkulator dijalankan bersisian dengan masukan
identik, satu menyusun keluaran tiap frame dan satu tidak. `get_live_stats()`
di akhir keduanya **identik persis**. Uji terkendali pada 1292 track_id dengan
pembersihan lintasan menyala: hemat 28 sampai 48 ms per frame, tergantung
kedalaman riwayat.

**MEMBATALKAN.** Di `backend/.env`:

```
STATS_SKIP_BUILD=0
```

lalu nyalakan ulang backend. Tanpa setelan itu, bawaannya `1` yaitu cepat.

**Curigai ini kalau.** Statistik pemain, MPI, atau peringkat MVP tampil kosong
atau basi selama analisis berjalan. Hasil akhirnya seharusnya tidak mungkin
terpengaruh, karena yang dilewati hanya penyusunan, bukan pencatatan.

---

## 2. Laju proses diseragamkan antar video (13 Oktober 2026)

**Masalahnya.** Langkah tetap `PROCESS_STRIDE=2` membuat tiap video diproses
pada laju yang berbeda beda:

| Video | fps sumber | Efektif SEBELUM |
|---|---|---|
| q1-UBAYA-UNESA.mp4 | 25 | **12 fps** |
| UNESA-UBAYA-Q1-a.mp4 | 25 | **12 fps** |
| Q1-UNESA-UBAYA-CUT-2-17.mp4 | 60 | 30 fps |
| Quarter-1_UNESA-UBAYA.mp4 | 60 | 30 fps |

Beda 2,5 kali. Sepuluh ambang yang satuannya JUMLAH FRAME karena itu berarti
lain di tiap video. `FG_ZONE_DWELL_F=2` yang berarti 67 ms di video 60 fps,
jadi 167 ms di video 25 fps, sehingga bola masuk yang asli bisa terlewat bukan
karena modelnya, melainkan karena satuannya bergeser.

**Yang diubah.** Langkah pemilihan frame jadi milik tiap video (`self._stride`,
dulu konstanta global), dan ditentukan dari laju sasaran:

```
langkah = max(1, floor(fps_sumber / TARGET_PROCESS_FPS))
```

Dipilih `floor`, bukan `round`, supaya tidak ada video yang jatuh DI BAWAH
sasaran. Hasilnya semua video mendarat di 25 sampai 30 fps efektif; sebarannya
turun dari 2,5 kali jadi 1,2 kali.

**Langkahnya tetap bilangan bulat dan seragam.** Itu disengaja. Enam tahap
digerbangi `frame_id % ...` (court, pose, ocr, action, dan dua log), dan
nilainya terjalin dengan langkah. Pemilihan berbasis cap waktu akan membuat
jarak antar frame tidak seragam dan membuat gerbang gerbang itu meleset.

**Ini TIDAK menyentuh dekode.** NVDEC tetap membaca berkas aslinya apa adanya.
Yang diatur hanya frame mana yang diteruskan ke pipeline. Tidak ada penyandian
ulang, tidak ada perubahan resolusi, tidak ada penurunan kualitas gambar.

**Buktinya, diuji pada potongan 20 detik, pipeline penuh:**

```
video 25 fps  tanpa sasaran : langkah=2  eff_fps=12
video 25 fps  sasaran 25    : langkah=1  eff_fps=25
video 60 fps  sasaran 25    : langkah=2  eff_fps=30   (sama dengan produksi)
```

**Dampaknya.** Video 60 fps tidak berubah sama sekali, jadi materi demo aman.
Video 25 fps memproses 2x lebih banyak frame: lebih lambat, tapi ambangnya
akhirnya benar.

**MEMBATALKAN.** Di `backend/.env`, setel `TARGET_PROCESS_FPS=0` atau hapus
barisnya, lalu nyalakan ulang backend. Salinan `.env` sebelum perubahan ada di
`backend/.env.bak-*`.

**Catatan.** Saat sasaran aktif, `MAX_FPS_PROCESS` DILEWATI. Keduanya mengatur
hal yang sama dengan cara berbeda, dan memakai keduanya sekaligus membingungkan.

**Curigai ini kalau.** Bola masuk yang terdeteksi di video 25 fps berubah
drastis. Itu memang diharapkan membaik, tapi kalau justru memburuk, ambang
berbasis frame perlu disetel ulang untuk laju yang baru.

---

## 3. Pecahan track digabung jadi identitas pemain (13 Oktober 2026)

**Masalahnya.** Pelacak memberi `track_id` BARU setiap kali pelacakan putus.
Terukur pada rekaman nyata: **1292 track untuk sekitar 10 pemain**, umur tengah
**0,6 detik**, 53 persen mati sebelum 1 detik, dan yang terpanjang pun hanya
40,8 detik dari video 601 detik.

Sebabnya dua, dan porsinya diukur dari data frame:

| Pola kelahiran track | Jumlah | Porsi | Artinya |
|---|---|---|---|
| Lahir berbarengan (>=4 sekaligus) di 40 frame | 252 | **20%** | kamera berpindah sudut |
| Lahir sendiri sendiri (1 per frame) | ~908 | **71%** | pemain tertutup / deteksi hilang |

Catatan: 43 gerakan kamera yang berbeda terdeteksi (148 deteksi mentah, tapi
satu gerakan panjang terbaca sampai 23 kali). Angka 40 frame kelahiran
berbarengan cocok dengan 43 gerakan itu. Jadi perpindahan kamera memang nyata
sebagai penyebab, TAPI hanya seperlima. Perbaikan khusus kamera saja tidak
cukup; penggabungan lewat nomor punggung menangani kedua sebab sekaligus.

**Kenapa penggabungan ini layak.** 846 track yang bernomor ternyata hanya
**42 identitas unik**, dan di sanalah seluruh angka yang berarti berada:

| Statistik | Track bernomor | Tanpa nomor |
|---|---|---|
| Poin | 18 | **0** |
| Rebound | 50 | **0** |
| Assist | 3 | **0** |
| Steal | 306 | 16 (5%) |

**Yang diubah.** `get_live_stats()` menggabungkan baris hasil dengan kunci
`(nomor punggung, tim)`, kunci yang SAMA dengan `_merge_all_quarters`. Pemain
yang nomornya belum terbaca memakai namanya sendiri sebagai kunci, jadi tidak
saling menelan.

**Yang TIDAK diubah, dan ini yang menjaga agar tidak merusak.** Pencatatan
tidak disentuh sama sekali. Penggabungan hanya saat MEMBACA. Penyangga
kesinambungan (`_prev_pos`, `_positions`, `_smooth_buf`) sengaja dibiarkan per
track mentah, karena pecahan baru memang TIDAK bersinambung dengan pecahan
lama; mewariskannya akan menciptakan lonjakan jarak raksasa, persis cacat yang
sudah diperbaiki sebelumnya.

**Aturan per bidang, sengaja eksplisit:**

| Bidang | Aturan | Alasan |
|---|---|---|
| pts, reb, ast, stl, blk, tov, min | dijumlah | akumulasi |
| fg_pct, three_pct, eff | dihitung ULANG | turunan, tidak boleh dijumlah |
| distance_m | dijumlah | pemain memang menempuh semuanya |
| max_speed, jump_height, acceleration | diambil TERBESAR | puncak, bukan akumulasi |
| avg_speed, agility, endurance, mpi_composite | rata rata BERBOBOT menit | pecahan 0,6 detik tidak boleh seberat pecahan 40 detik |

**Buktinya, diuji pada 1292 baris data NYATA:**

```
baris : 1292 -> 488
pts 18->18   reb 50->50   ast 3->3   stl 322->322   blk 23->23
tov 322->322  fga 47->47  fgm 9->9   min 69.20->69.20     semua SAMA
jarak total 12115.8 -> 12115.8 m     SAMA (dijumlah benar)
laju maks 36.0 -> 36.0 km/j          SAMA (diambil terbesar, bukan dijumlah)
```

Dan akhirnya ada jarak per PEMAIN, bukan per pecahan:

```
#1  Haviere Noah Richel Kester  tim B  1136.8 m
#4  Rio Julian Ramadhani        tim A  1032.9 m
#11 Albert Richard              tim B   945.8 m
```

**Uji regresi.** Dengan saklar MATI, keluaran `get_live_stats()` terbukti
**identik** dengan kode sebelum diubah, diuji dengan memuat versi lama sebagai
modul sebelah supaya impor relatifnya tetap jalan. (Memuatnya di luar paket
membuat `PATH_CLEAN_METHOD` dipaksa `none` dan hasilnya menyesatkan; itu
jebakan yang sempat menipu saya.)

**MEMBATALKAN.** Di `backend/.env`: `IDENTITAS_GABUNG=0`, lalu nyalakan ulang.

**Curigai ini kalau.** Dua pemain tergabung jadi satu, yaitu statistik satu
nomor punggung tiba tiba terlihat seperti milik dua orang. Penyebabnya nomor
salah baca. Penjaganya sudah ada: `DIGIT_CLS_MIN_CONF=0.70`,
`NUMBER_VALIDATOR=weighted`, `NUMBER_CONSECUTIVE_N=3`, dan kuncinya memuat tim
sehingga nomor sama di dua tim tidak saling menelan.

**Yang TIDAK diperbaiki oleh ini.** Deteksi bola masuk tidak tersentuh sama
sekali; ini memperbaiki ANGKA ITU MILIK SIAPA, bukan apakah bolanya masuk.
446 pecahan tanpa nomor tetap terpisah, tapi mereka membawa nol persen poin,
rebound, dan assist.

---

## Yang DIPERIKSA lalu TIDAK diubah, beserta alasannya

Supaya tidak ada yang mengulang pemeriksaan yang sama:

| Kandidat | Temuan | Keputusan |
|---|---|---|
| `get_live_stats()` tiap 5 frame (baris 1030) | hasilnya DIPAKAI: `_build_ws_message` membaca player_stats, team_stats, possession_pct; `_set_redis_stats` menulis seluruh blob yang dibaca `/api/stats/live` selama analisis | bukan pemborosan, tidak diubah |
| Klasifikasi aksi | sudah menyatukan semua pemain jadi satu tensor (`batch_np`) | sudah optimal |
| Pose | sudah "single-batch GPU inference for all valid crops" | sudah optimal |
| OCR nomor punggung | sudah punya tahap batch (Pass 1 bucket, Pass 2 batch) | sudah optimal |
| `stats_roster` dibangun ulang tiap frame | hanya 0,86 ms pada 1292 track_id, setara 0,3 menit | tidak sepadan dengan risikonya |
| `_clean_buf` | dibatasi `PATH_CLEAN_CHUNK=450`, bukan tumbuh tanpa batas | bukan kebocoran |
| Perangkat model | RF-DETR, court, pose, digit detect, digit classify, SlowFast semuanya cuda fp16 | tidak ada yang tertinggal di CPU |
| Dekode video | sudah NVDEC GPU (`ffmpegcv h264_cuvid`) | sudah optimal |
| `TEAM_SIGLIP_DEVICE=cpu` | SigLIP tidak pernah dipakai karena `TEAM_METHOD=lab` | setelan mati, bukan beban |

Kesimpulannya: jalur model dan GPU sudah habis dioptimalkan. Sisa pengungkitnya
semua berupa penukaran akurasi, dan itu wajib diukur, bukan diasumsikan.

---

## Kenapa menurunkan fps TIDAK gratis

Mekanismenya sudah ada: `MAX_FPS_PROCESS` membuat `_fps_skip`, lalu
`_eff_fps = source_fps / PROCESS_STRIDE` disebarkan ke tracker dan statistik.
Penyangga aksi pun diukur dalam DETIK, bukan jumlah frame, jadi modelnya tetap
melihat durasi yang sama. Rancangannya benar.

Masalahnya di tempat lain: ada sekitar **10 setelan yang satuannya jumlah
frame**, dan artinya ikut berubah kalau fps efektif berubah.

| Setelan | Nilai | Arti di 30 fps | Arti di 15 fps |
|---|---|---|---|
| `FG_ZONE_DWELL_F` | 2 | 67 ms | 133 ms |
| `FG_ZONE_DWELL_MAX_F` | 4 | 133 ms | 267 ms |
| `FG_CONFIRM_FRAMES` | 10 | 333 ms | 667 ms |
| `HOOP_ROI_HIST_F` | 15 | 500 ms | 1000 ms |
| `BALL_COAST_F` | 3 | 100 ms | 200 ms |
| `HELD_POS_MAX_AGE_F` | 30 | 1000 ms | 2000 ms |
| `SHOT_POS_LOOKBACK_F` | 15 | 500 ms | 1000 ms |
| `TRACK_COAST_FRAMES` | 15 | 500 ms | 1000 ms |
| `NUMBER_CONSECUTIVE_N` | 3 | 100 ms | 200 ms |
| `PATH_CLEAN_WINDOW` | 15 | 500 ms | 1000 ms |

Yang paling rawan adalah gerbang bola masuk. Bola melintas ring dalam waktu
sangat singkat; menuntutnya bertahan 133 ms di zona ring, bukan 67 ms, bisa
membuat bola masuk yang asli tidak terhitung. Gerbang itu sudah jadi bagian
paling lemah sistem, dan menyentuhnya tanpa mengukur beresiko memperburuk.

Catatan: komentar pada `SHOT_POS_LOOKBACK_F` menulis "~1 detik @15fps", padahal
pipeline berjalan 30 fps efektif. Jadi sebagian setelan memang disetel dengan
anggapan fps yang berbeda beda. Ini perlu dirapikan terpisah.

**Cara yang benar menurunkan fps**, kalau nanti diputuskan: ubah setelan
berbasis frame itu jadi berbasis milidetik lalu dikonversi memakai `_eff_fps`
yang sudah tersedia. Setelah itu fps bisa diubah tanpa diam diam mengubah
sepuluh ambang sekaligus.

**Cara mengujinya sebelum diputuskan**: `eksperimen/17_ukur_kejadian.py` pada
potongan 240 detik, membandingkan bola masuk yang terdeteksi dengan papan skor
sungguhan. Hanya skrip itu yang memutuskan; proksi lain pernah meramal arah
yang SALAH.

---

## Urutan rekomendasi berikutnya

1. **Perbaiki pecahnya track.** 1292 track_id untuk 10 pemain. Ini satu satunya
   yang mempercepat DAN memperbaiki akurasi sekaligus: setiap struktur per
   frame dibangun di atas seluruh track_id itu.
2. **Ukur fps 15 dengan skrip 17** sebelum mengubah apa pun. Kalau bola masuk
   yang terdeteksi tidak turun, waktunya terpangkas separuh.
3. **Rapikan ambang berbasis frame jadi berbasis milidetik.** Prasyarat supaya
   fps bisa diubah dengan aman di kemudian hari.
4. `COURT_EVERY_N` dari 3 ke 5 (bawaan kode) menghemat sekitar 16 ms per frame,
   kira kira 5 menit. Kecil, dan mempengaruhi kalibrasi lapangan saat kamera
   berpindah, jadi dahulukan tiga langkah di atas.

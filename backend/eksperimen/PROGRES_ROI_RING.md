# PROGRES: POTONGAN RING DAN GERBANG TEPI RIM

Ditulis 10 Oktober 2026. Berkas ini supaya pekerjaan tidak hilang kalau sesi
terputus di tengah uji pipeline yang lamanya 16 sampai 20 menit per jalan.

Penafsir WAJIB `/home/paul/miniconda3/envs/cl-gpu/bin/python`. Video uji WAJIB
`/home/paul/q1_4menit.mp4` (240 detik). Kebenaran 5 bola masuk, 11 poin.

## ACUAN YANG HARUS DIKALAHKAN

`17_ukur_kejadian.py`, ambang produksi apa adanya, log
`/home/paul/hasil_uji_acuan_produksi.log`:

    asli ditemukan 2/5, nilai 2/3 benar 2, palsu 2, poin 9, waktu 16,1 menit
    ditemukan : 0:58 (2,5 m) dan 1:57 (4,4 m)
    terlewat  : 0:19, 1:30, 2:26
    palsu     : 1:38 (0,5 m dari ring) dan 3:52 (10,2 m, dihitung 3 angka)

Laju lompatan posisi bola acuan: 3,0 persen pada deteksi kuat (skrip 20).

## YANG SUDAH TERPASANG, DUA DUANYA BAWAAN MATI

### Tugas A, `HOOP_ROI_ENABLE` di `pipeline/detector_rfdetr.py`
Saat rim terdeteksi, potongan persegi di sekitar rim diambil dari frame
RESOLUSI ASLI lalu detektor dijalankan pada potongan itu. Koordinatnya digeser
kembali ke frame penuh. HANYA kelas `ball` yang diambil dari potongan, jadi
tidak ada benda lain yang bisa terhitung dua kali.

Saklarnya: `HOOP_ROI_ENABLE` (0), `HOOP_ROI_SIZE` (448), `HOOP_ROI_NEAR_PX`
(300), `HOOP_ROI_HIST_F` (15), `HOOP_ROI_CONF` (0 = ikut ambang bola),
`HOOP_ROI_OVERLAP_PX` (60), `HOOP_ROI_EDGE_PX` (2).

Aturan gabung, supaya satu bola tidak jadi dua. Keluaran `ball` satu slot jadi
berganda mustahil secara bentuk; yang diatur SIAPA YANG MENANG:
1. potongan kosong -> frame penuh dipakai, tidak ada yang berubah
2. frame penuh kosong -> hasil potongan MENGISI celah
3. keduanya ada dan pusatnya dalam 60 px -> potongan MENGGANTI
4. keduanya ada tapi jauh -> frame penuh DIPERTAHANKAN (sengaja konservatif)

### Tugas B, `FG_RIM_GATE` di `pipeline/event_engine.py`
Daftar `rims` sekarang DIBACA (sebelumnya nol kali). Zona bola masuk memakai
rentang mendatar kotak rim, bukan lingkaran 45 px. Garis "sudah di bawah"
diambil dari tepi bawah kotak rim. Syarat "wajib deteksi NYATA" TIDAK disentuh.
Mundur sendiri ke lingkaran kalau kotak rim tidak ada (YOLO, ring sintetis).

Saklarnya: `FG_RIM_GATE` (0), `FG_RIM_X_MARGIN_PX` (0), `FG_RIM_Y_BAND_PX`
(= `FG_HOOP_RADIUS_PX`, jadi jangkauan TEGAK tidak berubah dan yang berbeda
hanya ketajaman mendatar).

## HASIL UKUR YANG SUDAH ADA

### Peta koordinat potongan, DIPERIKSA bukan diasumsikan
Kotak yang dikembalikan rfdetr berada di ruang koordinat GAMBAR MASUKAN, jadi
untuk potongan cukup ditambah (x0, y0). Diuji pada potongan 320, 448, dan 576;
rim hasil potongan sepakat dengan rim frame penuh dalam 1,0 sampai 3,6 px.
Potongan berukuran lain TIDAK ditolak pustaka selama `shape=(576,576)` dikirim.

### Keyakinan memang naik, dugaan brief TERBUKTI di titik ini
Skrip 21, 4.290 frame, jendela detik 72 sampai 215, saat KEDUANYA menemukan
bola (n=813):

    keyakinan median : frame penuh 0,820   potongan 0,869
    potongan lebih yakin pada 77,9 persen frame
    jarak pusat      : median 1,5 px  p90 10,1 px  maks 956 px
    sepakat dalam 10 px : 89,9 persen

Jadi memperbesar bola memang menaikkan keyakinan. TAPI posisinya praktis sama
(median 1,5 px lawan gerbang 45 px), jadi di frame yang frame penuh sudah
berhasil, potongan tidak menambah apa apa.

### Cakupan dan ketepatan posisi
    frame yang punya rim          61,1 %
    frame penuh menemukan bola    75,0 %
    potongan menemukan bola       23,1 %
    HANYA potongan                 4,2 %   (179 frame, ini satu satunya tambahan)
    potongan jalan tapi kosong    1.631 dari 2.623 potongan

    lintasan acuan dari deteksi KUAT frame penuh, lompat = > 60 px:
      frame penuh, kuat                 MELOMPAT  3,8 %   n=3197
      potongan, kuat                    MELOMPAT  9,2 %   n=954
      potongan, MENGISI celah penuh     MELOMPAT 24,7 %   n=146  p90 225 px

    gabungan (isi celah + ganti kalau sepakat 60 px): MELOMPAT 4,7 %

### UJI CERMIN, dan ini MEMBANTAH cara ukurnya sendiri
Kalau lintasan acuan dibalik, dibangun dari deteksi kuat POTONGAN:

      potongan, kuat (diri sendiri)     MELOMPAT  1,2 %   n=969
      frame penuh, kuat                 MELOMPAT 18,3 %   n=1089

Arahnya TERBALIK. Sebabnya bias kepadatan: sumber yang jarang hanya punya
tetangga di petak waktu saat dia memang beruntun, yaitu bagian yang mulus,
sehingga kekonsistenan dirinya selalu tampak bagus. Jadi baris "diri sendiri"
TIDAK BOLEH dipakai memutuskan, untuk kedua sumber.

Yang TIDAK terkena bias itu baris "MENGISI celah penuh": titik yang dinilai
bukan bagian dari acuan, dan acuannya padat di sekitarnya karena frame penuh
punya deteksi kuat di f-1 sampai f-3 dan f+1 sampai f+3. **Angka 24,7 persen
itulah satu satunya angka ketepatan posisi yang sahih di sini.**

### Biaya waktu
    detect frame penuh   16,4 ms/frame
    predict potongan     12,0 ms/potongan
    potongan di SEMUA frame: +73 persen waktu detektor
    gerbang NEAR_PX=300 : potongan jalan di 28,6 persen frame, 34 dari 179
                          pengisian celah hilang

### `ball-in-basket` pada potongan: 2 deteksi, bukan 0
Pada frame penuh kelas ini terukur NOL (skrip 18). Pada potongan 448 dia
menyala 2 kali dari 2.623 potongan. Masih praktis tidak terpakai, tapi bukan
nol lagi. Tidak digabung ke keluaran, hanya dicatat di `roi_stat`.

## YANG MASIH HARUS DIJALANKAN

1. `17_ukur_kejadian.py` dengan `HOOP_ROI_ENABLE=1 HOOP_ROI_SIZE=448
   HOOP_ROI_NEAR_PX=300`. Log `/home/paul/hasil_uji_roi_448_near300.log`.
2. `17_ukur_kejadian.py` dengan `FG_RIM_GATE=1`.
   Log `/home/paul/hasil_uji_rim_gate.log`.

Uji SATU perubahan pada satu waktu. Menjalankan A dan B sekaligus
menyembunyikan sebab akibat.

## CARA MENGULANG

    # rekam potongan lawan frame penuh, 1,8 menit, butuh GPU
    MODE=rekam DETECTOR_METHOD=rfdetr HOOP_ROI_NEAR_PX=0 HOOP_ROI_SIZE=448 \
      REKAM=/home/paul/rekam_roi/roi_448.json VIDEO=/home/paul/q1_4menit.mp4 \
      /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/21_ukur_potongan_ring.py

    # nilai ulang di ambang mana pun, tanpa GPU
    MODE=nilai REKAM=/home/paul/rekam_roi/roi_448.json \
      /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/21_ukur_potongan_ring.py

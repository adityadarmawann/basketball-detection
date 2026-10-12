# PENDEKATAN YANG SUDAH DIUJI DAN DITOLAK

Berkas ini ada supaya pendekatan yang SUDAH TERBUKTI merugikan tidak diuji ulang.
Semua angka di bawah hasil ukur nyata, bukan perkiraan. Setiap baris menyebut
skrip pengukurnya supaya bisa diulang kalau memang ada yang berubah.

**Aturan pakai berkas ini:** sebelum menyalakan saklar apa pun atau mengusulkan
metode baru, cari dulu di sini. Kalau sudah tercatat DITOLAK, jangan diuji ulang
kecuali syarat "boleh dicoba lagi kalau" di barisnya sudah terpenuhi.

Video acuan sepanjang catatan ini:
`test-video/Q1-UNESA-UBAYA-CUT-2-17.mp4` detik 60 sampai 300 (4 menit, 1280x720,
60 fps sumber, diproses 30 fps efektif). Kebenaran dari papan skor: 5 bola masuk,
11 poin.

---

## 1. PELACAKAN

### GMC (kompensasi gerak kamera), `GMC_ENABLED=1`
**DITOLAK.** `pipeline/tracker.py`. Diukur 10 Oktober 2026, RF-DETR, skrip 16.

    konfigurasi        track   umur median   >=5s   >=10s
    gmc=0 mt=0.5        1697      0,53 s      113     27
    gmc=1 mt=0.5        8985      0,10 s       15      0
    gmc=0 mt=0.7         540      1,60 s      141     66
    gmc=1 mt=0.7        4261      0,17 s       62      4

Lima kali lebih buruk, dan SEMUA track yang bertahan 10 detik hilang. Pengaman di
kelas `_GMC` (minimal 20 inlier, skala 0,9 sampai 1,1, geseran maks 30 persen
lebar) tidak cukup menahan kerusakan.

Sebabnya: GMC memperkirakan satu matriks affine antar frame berurutan. Itu sahih
untuk geseran dan zoom halus, tidak sahih saat gambar berganti sudut. Di rekaman
ini perpindahan keras terjadi 1 per 7,1 detik.

Boleh dicoba lagi kalau: sumber videonya kamera statis atau satu sudut tetap.

### SAM2, `TRACKER_METHOD=sam2`
**DITOLAK.** `pipeline/tracker.py`. Diukur 10 Oktober 2026, RF-DETR, skrip 16.

                            ByteTrack 0.7    SAM2
    track                       540           170
    umur median                1,60 s        1,92 s
    bertahan >=5 detik          141            36
    bertahan >=10 detik          66            26
    lompatan posisi p95        38 px        634 px
    KEMURNIAN, lompat warna>40  1,6 %        28,8 %
    waktu proses              5,3 menit    32,0 menit
    memori GPU                 wajar    7,7 dari 8 GB, KEHABISAN

JANGAN tertipu "track 170 lawan 540" dan "umur median 1,92 lawan 1,60 detik".
Dua angka itu terlihat lebih baik tapi menyesatkan:
- Lompatan posisi p95 634 piksel berarti track berpindah melintasi layar.
- Kemurnian 28,8 persen lawan 1,6 persen berarti satu track berpindah orang
  18 KALI LEBIH SERING.
Jadi tracknya sedikit dan panjang justru karena beberapa pemain DILEBUR jadi satu
identitas. Itu lebih merusak statistik per pemain daripada track yang pecah.
Ditambah `CUDA out of memory` berulang sehingga banyak pemain gagal didaftarkan,
jadi sebagian dari "sedikit" itu memang gagal daftar.

Catatan: notebook Roboflow memakai SAM2 dan berhasil, tapi klip demo mereka
pendek dan menerus tanpa ganti sudut, dan mereka memberi prompt hanya di frame
pertama tanpa pendaftaran ulang. Konteksnya berbeda.

Boleh dicoba lagi kalau: ada kartu grafis dengan memori jauh lebih besar DAN
masalah perpindahan sudut sudah diatasi lebih dulu.

### POLA UMUM, PALING PENTING
GMC dan SAM2 sama sama bersandar pada KESINAMBUNGAN PEMANDANGAN antar frame.
Rekaman ini berpindah sudut keras 1 per 7,1 detik dan bergerak menonjol tiap
1,0 detik, jadi landasan keduanya tidak ada. **Jangan mengusulkan pendekatan
berbasis kesinambungan pemandangan lagi sebelum perpindahan sudutnya ditangani.**
User sudah memperingatkan hal ini dua kali, dan dua kali pengukuran membuktikan
user benar.

### Jarak Jaccard himpunan keypoint untuk mendeteksi ganti sudut
**DITOLAK.** `pipeline/court.py`. Diukur 8 Oktober 2026, skrip 06.
Saat sisi lapangan TETAP, persentil 90 Jaccard tetap mencapai 1,000 karena
deteksi keypoint berkedip. Ambang 0,4 menangkap 15 dari 20 pindah sisi tapi
memicu 85 kali palsu. Yang dipakai sekarang: `COURT_CUT_PX=120` ditambah
`COURT_LOST_REFRESH_S=3.0`, dua sinyal bersama.
Boleh dicoba lagi kalau: model keypoint sudah dilatih ulang dan berhenti berkedip.

### Ambang pencocokan ByteTrack, bukan ditolak tapi SUDAH DISAPU
Jangan sapu ulang. Diukur 10 Oktober 2026, GMC mati, `track_buffer` 90, skrip 16.

    mt   track  umur median  >=5s  >=10s  lompat posisi p95
    0.5   1697     0,53 s     113    27        25 px
    0.6    853     0,90 s     143    49        32 px
    0.7    540     1,60 s     141    66        38 px
    0.8    429     1,97 s     134    71        46 px

Kemurnian TIDAK turun: lompatan warna>40 tetap 1,6 persen di 0.5 maupun 0.8.
Dipilih **0.7** karena jumlah track yang bertahan 5 detik paling banyak di situ,
dan jendela 1 sampai 2 detik dari percobaan tembakan sampai bola masuk itulah
yang dibutuhkan untuk menentukan penembak.

Peringatan lama di kode ("2,38 jersey per track pada mt 0.8") berasal dari
konfigurasi yang JUGA memakai `track_buffer` 450. Penyebabnya buffer 15 detik
itu, bukan ambang pencocokannya. Dulu keduanya diubah bersamaan sehingga tertukar.

---

## 2. SKOR DAN KEJADIAN

### `SHOT_POS_METHOD=lookback`
**TIDAK BERPENGARUH, jangan diuji lagi.** `pipeline/event_engine.py`.
Diukur 10 Oktober 2026, skrip 17, DUA KALI, hasilnya IDENTIK kedua kali.
Saya sempat menyimpulkan `_estimate_shot_pos` ("titik terjauh") menyebabkan 2 dari
3 galat. Pengujian membuktikan jalur itu TIDAK PERNAH terpakai pada klip ini, jadi
mengganti metodenya tidak mengubah apa pun. Perbaiki yang lain dulu.

### Melonggarkan gerbang bola masuk supaya menerima posisi PERKIRAAN
**DITOLAK atas dasar prinsip, bukan cuma angka.** `pipeline/event_engine.py`.
Setelah `PelacakBola` dipasang, cakupan bola naik 72,8 ke 90,9 persen, tapi skor
HAMPIR TIDAK BERGERAK (RF-DETR 9 tetap 9, YOLO 8 jadi 9) dan daftar kejadiannya
identik. Sebabnya gerbang SENGAJA menuntut deteksi NYATA untuk keputusan
terakhir; posisi perkiraan hanya menjaga kesinambungan penghitung.
Aturan itu DIPERTAHANKAN. Melonggarkannya berarti mengarang poin dari posisi yang
ditebak model kecepatan tetap. Nilai `PelacakBola` ada di lintasan, penguasaan,
dan jarak tempuh, bukan di skor.

### `FG_RECALL_BOOST=1`
**DITOLAK untuk skor.** `pipeline/event_engine.py`. Diukur 10 Oktober 2026, skrip 17.

    detektor   boost   asli ditemukan   palsu   poin
    RF-DETR    mati         2/5           1       9
    RF-DETR    hidup        2/5           2      10
    YOLO       mati         2/5           1       8
    YOLO       hidup        2/5           2       9

Satu bola masuk asli TIDAK bertambah, tapi deteksi palsunya bertambah. Untuk
ketepatan skor itu merugikan, maka MATI secara bawaan. Nyalakan hanya kalau yang
dikejar kelengkapan daftar kejadian, bukan ketepatan skor.

### Menurunkan ambang keyakinan BOLA demi cakupan
**DITOLAK, dan ini membantah dugaan saya sendiri.** Diukur 10 Oktober 2026,
potongan 240 detik (`/home/paul/q1_4menit.mp4`), skrip 17 dan 19 dan 20.

Saklar `RFDETR_CONF_BALL` sengaja dibuat supaya ambang bola bisa diturunkan
TANPA melonggarkan pemain, hakim, nomor, dan ring. Cakupan bolanya memang naik:

    ambang bola   cakupan bola   asli ditemukan   palsu   poin
    0,25 produksi     69,4 %          2/5            2      9
    0,20              72,8 %          2/5            2      9
    0,10              83,6 %          1/5            2      7

Cakupan naik 14 titik persen, skor justru TURUN. Dugaan saya bahwa cakupan
bolalah penghambatnya TERBANTAH.

**Sebabnya, terukur dengan skrip 20 dari rekaman skrip 19:**

                              melompat dari lintasan   p90 jarak
    deteksi kuat (c >= 0,25)          3,0 %              17,0 px
    deteksi tambahan (0,10-0,25)     33,3 %             379,8 px

Sepertiga deteksi tambahan itu bola yang ditaruh di tempat SALAH, 11 kali lebih
sering melompat daripada deteksi kuat. Mekanismenya: detektor memilih SATU bola
per frame, yang keyakinannya tertinggi. Di ambang 0,25 frame yang cuma punya
kandidat lemah menghasilkan TIDAK ADA bola, dan itu celah yang jujur. Di ambang
0,10 frame yang sama menghasilkan bola palsu di posisi salah. Gerbang bola masuk
menuntut rantai "datang dari ATAS rim lalu dikonfirmasi di BAWAH rim", dan satu
posisi salah di tengah rantai itu memutusnya.

**Pelajarannya: untuk bola, yang dibutuhkan POSISI BENAR pada saat menentukan,
bukan cakupan tinggi.** Jangan lagi mengejar cakupan bola sebagai sasaran.

Saklar `RFDETR_CONF_BALL` TETAP ADA karena bawaannya 0 yang berarti ikut ambang
biasa, jadi tidak mengubah apa pun kalau tidak diisi. JANGAN diisi di `.env`.

### Menilai POTONGAN RING dari laju lompatan posisi saja
**CARA UKUR YANG MENYESATKAN, dan ini membantah sasaran yang saya sendiri
pakai.** Diukur 10 Oktober 2026, skrip 21 dan 17, `/home/paul/q1_4menit.mp4`.

Setelah uji ambang bola membuktikan cakupan salah sasaran, sasaran diganti jadi
KETEPATAN POSISI dan diukur dengan laju lompatan dari lintasan (pola skrip 20).
Untuk potongan ring, proksi itu memberi angka yang jelek:

    laju lompatan (> 60 px dari lintasan kuat frame penuh):
      frame penuh, kuat                3,8 %   n=3197
      potongan, kuat                   9,2 %   n=954
      potongan, MENGISI celah penuh   24,7 %   n=146
      GABUNGAN (isi + ganti 60 px)     4,7 %

Kalau diputuskan dari situ, potongan ring DITOLAK. Pipeline membuktikan
sebaliknya: asli ditemukan 2/5 jadi **3/5**. Jadi proksinya meramalkan arah
yang SALAH.

Dua sebab, dua duanya terukur:
1. **Bias kepadatan pada uji cermin.** Kalau lintasan acuan dibalik, dibangun
   dari deteksi kuat POTONGAN, angkanya terbalik: potongan 1,2 persen, frame
   penuh 18,3 persen. Sumber yang jarang hanya menemukan tetangga di petak
   waktu saat dia memang beruntun, yaitu bagian yang mulus. Jadi baris
   "kekonsistenan diri sendiri" tidak sahih untuk KEDUA sumber.
2. **Jumlah mutlak kalah penting dari mutu rantai.** Dari 179 pengisian celah,
   114 posisinya benar, 23 ditolak `PelacakBola` (jatah 90 px per frame jeda),
   dan hanya 30 yang lolos dengan posisi salah. Tiga puluh posisi salah
   tersebar di 4 menit tidak cukup mengarang kejadian, sementara 114 posisi
   benar cukup menyambung satu rantai bukti yang tadinya putus.

**Yang boleh dipakai memutuskan tetap skrip 17.** Laju lompatan tetap berguna
untuk MEMAHAMI sebab, dan baris "mengisi celah" adalah satu satunya baris yang
bebas bias kepadatan (titik yang dinilai bukan bagian dari acuan). Tapi jangan
menerima atau menolak apa pun hanya dari angka itu.

### Menggabungkan deteksi bola dari dua detektor
**DITOLAK, tidak sepadan.** Diukur 10 Oktober 2026, skrip 19, jendela 143 detik
yang memuat kelima bola masuk, 4.290 frame.

    cakupan bola per ambang:
      ambang    RF-DETR    YOLO lama
        0,10     83,6 %      46,0 %
        0,25     69,4 %      39,3 %
        0,35     64,0 %      36,3 %

    silang RF-DETR 0,35 lawan YOLO 0,25:
      dua duanya menemukan   1539   35,9 %
      HANYA RF-DETR          1206   28,1 %
      HANYA YOLO              149    3,5 %
      tidak satu pun         1396   32,5 %
      gabungan               2894   67,5 %  (+3,5 titik atas RF-DETR saja)

YOLO cuma menambah 3,5 titik persen, dan harganya menjalankan model kedua di
setiap frame. Sebagai pembanding, menurunkan ambang bola RF-DETR menambah 14
titik persen tanpa model kedua, DAN ITU PUN DITOLAK karena tambahannya posisi
salah. Jadi menambah model untuk 3,5 titik jelas tidak sepadan.

Catatan yang berguna: saat keduanya menemukan bola, titik tengahnya sepakat
95,0 persen dalam 25 piksel (median beda 1,7 px). Jadi dua model ini TIDAK
saling bertentangan, cuma YOLO lebih sering tidak menemukan apa apa.

### Menilai akurasi skor dari TOTAL POIN
**CARA UKUR YANG SALAH, jangan diulang.** Pernah terjadi: dua konfigurasi sama
sama memberi 12 poin lawan kebenaran 11, dan saya menyimpulkan "5 dari 5 bola
masuk benar". Setelah dicocokkan PER WAKTU, ternyata cuma 2 dari 5, dan deteksi
palsu dengan yang terlewat SALING MENIADAKAN sehingga totalnya kebetulan mirip.
Klaim 5/5 itu sudah DICABUT. Selalu pakai skrip 17 yang mencocokkan tiap kejadian
`MADE_FG` ke papan skor menurut waktunya.

### Menurunkan ambang MINIMUM dwell gerbang bola masuk, `FG_ZONE_DWELL_F=2`
**TIDAK BERPENGARUH, jangan diuji lagi.** `pipeline/event_engine.py`.
Diukur 12 Oktober 2026, skrip 17, `/home/paul/q1_4menit.mp4`, RF-DETR.

    setelan                 asli /5   palsu   nilai 2/3   poin
    acuan (dwell 3, ef 6)      3        2         3        11
    dwell 2 (efektif 4)        3        2         3        11

Daftar kejadiannya SAMA PERSIS, detik demi detik. Sebabnya terukur dengan
telemetri skrip 23: kelima kejadian di acuan sudah menyala pada penghitung
zona 6, 6, 6, 10, dan 14, jadi ambang minimum 6 TIDAK PERNAH menjadi
penghalang bagi satu pun dari mereka. Dua bola masuk yang terlewat juga bukan
karena ambang ini: 2:26 sudah mencapai dwell 8, dan 1:30 hanya mencapai 3
sehingga ambang 4 pun masih terlalu tinggi.

### Menolak bola MENGGANTUNG dari LAMA berada di zona ring, `FG_HANG_REJECT=1` SENDIRIAN
**DITOLAK sebagai aturan yang berdiri sendiri.** `pipeline/event_engine.py`.
Diukur 12 Oktober 2026, skrip 17 dan 23.

    setelan                      asli /5   palsu   ketepatan   poin
    acuan                           3        2        60 %      11
    tolak menggantung saja          2        1        67 %       6

Mekanismenya BEKERJA seperti rancangannya: palsu 3:52 yang menyala pada dwell
10 memang terbuang. Masalahnya bola masuk ASLI 0:59 menyala pada dwell **14**,
lebih lama daripada palsu yang mau dibuang, jadi batas atas apa pun yang
membuang 3:52 juga membuang satu bola masuk asli. Di klip ini kejadian dengan
dwell TERPANJANG justru bola masuk yang BENAR.

Perpindahan tegak minimum (`FG_HANG_MIN_DROP_PX`) juga terbantah dari data
yang sama: dua palsu turun 37,4 dan 40,1 px sedangkan bola masuk asli yang
paling sedikit turun hanya 33,2 px. Urutannya terbalik, jadi ambang apa pun
membuang yang asli lebih dulu. Dibiarkan 0.

Catatan cara membaca yang pernah menyesatkan: deretan jarak hampir tetap
28, 27, 27 px sebelum palsu 3:52 itu semuanya pada `dwell=0`, artinya gerbang
tepi rim SUDAH menolaknya secara mendatar dan itu bukan frame di dalam zona.
Rentetan dwell yang sebenarnya bergerak 27 -> 6 -> 18 px, yaitu pola MENEMBUS,
bukan menggantung. Jadi "bola menggantung belasan frame di dalam zona" tidak
pernah benar benar terjadi di klip ini.

Boleh dicoba lagi kalau: HANYA bersama `FG_RIM_CORE_CONFIRM`, yang membuat
kejadian 0:58 menyala pada dwell rendah sehingga batas atas tidak lagi
membuangnya (asli 4, palsu 1, ketepatan 80 persen). Keamanannya di gabungan
itu KEBETULAN dan harus diukur ulang setiap kali setelan C berubah.

### Pecahan inti cincin 0,15 untuk `FG_RIM_CORE_CONFIRM`
**DITOLAK, kelewat longgar.** Diukur 12 Oktober 2026, skrip 17.

    pecahan   asli /5   palsu   ketepatan
     0,15        4        3       57 %
     0,10        4        2       67 %

Pecahan 0,15 menyalakan rentetan 3:57 jadi palsu BARU. Telemetri skrip 23
sudah meramalkannya sebelum run jalan: jarak mendatar terkecil frame nyata di
rentetan itu 0,12 dari separuh lebar kotak rim, jadi lolos ambang 0,15 dan
tertahan ambang 0,10. Ramalan telemetri BENAR untuk kedua pecahan.
Bawaannya sekarang 0,10.

### Satu setelan gerbang untuk semua detektor
**DITOLAK.** YOLO dan RF-DETR memberi kotak ring dan bola dengan ukuran serta
keyakinan yang berbeda, jadi satu `FG_HOOP_RADIUS_PX` mustahil benar untuk
keduanya. Sekarang bawaannya per `DETECTOR_METHOD`: RF-DETR 45 px dengan diam 3
frame, YOLO 70 px dengan diam 2 frame.

### Kelas `ball-in-basket` sebagai penentu bola masuk
**BUTUH LATIH ULANG, tidak bisa dipakai sekarang.** Diukur 10 Oktober 2026,
skrip 18, jendela 16 detik yang MEMUAT satu bola masuk nyata (detik 79):

    ambang RFDETR_CONF 0.35 : 0 deteksi ball-in-basket dari 480 frame
    ambang RFDETR_CONF 0.10 : 0 deteksi ball-in-basket dari 480 frame

NOL, bahkan di ambang 0.10. Kelas ini ada di model tapi praktis tidak pernah
menyala, jadi menjadikannya penentu atau pemecah seri MUSTAHIL tanpa menambah
contoh latihnya. Jangan coba menurunkan ambang lagi, sudah dicoba.

### Menghaluskan posisi ring antar waktu
**TIDAK LAYAK, getarannya terlalu kecil untuk jadi masalah.** Diukur 10 Oktober
2026, skrip 18:

    geseran titik tengah ring antar frame: median 1,7 px   p90 10,8 px
    lebar kotak ring                     : median 38,0 px

Gerbangnya 45 px, jadi getaran median 1,7 px tidak mengubah keputusan. Lonjakan
maksimum 318 px itu perpindahan sudut kamera atau ring yang satunya, dan sudah
ditangani jalur lain. Saya sempat menduga ini salah satu penyebab; pengukuran
menunjukkan BUKAN. Jangan dikerjakan.

---

## 3. SAKLAR BERLABEL ROBOFLOW

Diukur 8 Oktober 2026, `q1-UBAYA-UNESA` 300 frame, pipeline PENUH, skrip 04,
tiap saklar diuji TERPISAH dari acuan yang sama, roster AKTIF.

Keempatnya awalnya dinyalakan karena permintaan "pakai mode roboflow". Tiga di
antaranya terbukti merugikan. **Jangan menyalakan sesuatu hanya karena berlabel
roboflow.**

| Saklar | Putusan | Bukti |
|---|---|---|
| `PATH_CLEAN_METHOD=roboflow` | **NYALA** | pengaruh ke deteksi, nomor, tim, aksi, kejadian NOL; percepatan median 45,3 jadi 19,4 m/s2; biaya +10 persen waktu |
| `KEYPOINT_SMOOTH_METHOD` selain `none` | **DITOLAK** | frame terkalibrasi TURUN 240 jadi 235, tidak ada metrik lain membaik. Model keypoint cuma memberi 13 dari 33 titik, jadi merata ratakan titik yang sedikit justru mengurangi yang lolos RANSAC |
| `NUMBER_VALIDATOR=consecutive` | **DITOLAK** | weighted 681 bacaan / 14 nomor unik / 0 palsu, consecutive 455 / 11 / 0. Presisi sama sama sempurna karena penyaring roster SUDAH memblokir nomor palsu, jadi keunggulan consecutive mubazir sementara 3 nomor pemain hilang |
| `TEAM_METHOD=siglip` | **DITOLAK** | +24 persen waktu tanpa perbaikan terukur |

Boleh dicoba lagi kalau: `KEYPOINT_SMOOTH_METHOD` setelah model keypoint dilatih
ulang; `NUMBER_VALIDATOR=consecutive` kalau suatu saat berjalan TANPA roster.

---

## 4. MODEL NOMOR PUNGGUNG

### Melatih deteksi digit dengan augmentasi bawaan Ultralytics
**DITOLAK, ini sebab model v3 kalah.** v3 Kaggle 47,8 persen lawan model lama
64,2 persen pada potongan sistem. Sebabnya CARA LATIH, bukan jurang domain.
Kesimpulan "jurang domain" yang sempat saya catat SALAH dan sudah dicabut.
Berurut dari paling merusak:

1. `fliplr` bawaan 0.5. **Digit itu kiral**, cerminan 2 bukan 2. Cara lama
   menyetelnya 0.0 dengan komentar "digit mirror tidak valid". WAJIB 0.0.
2. `mosaic` bawaan 1.0. Empat gambar dijahit jadi kanvas 640, digit menyusut
   sekitar 2 kali padahal sudah objek terkecil. Cara lama 0.0.
3. `optimizer=auto` dengan lebih dari 10.000 iterasi memilih MuSGD `lr0=0.01`.
   Cara lama AdamW `lr0=0.001` `cos_lr=True`, sepuluh kali lebih kecil.

Resep cara lama yang utuh ada di `dataset/number/output/digit_det/args.yaml`
(`md5sum` memastikan bobotnya identik dengan `best-detect-num-v2.pt`).

### `DIGIT_CLS_PAD` kecil
**DITOLAK.** `pipeline/jersey_ocr.py`. Diukur 9 Oktober 2026, skrip 12.
Ketepatan tahap classify pada kotak dari detektor: pad 0,2 memberi 65,7 persen,
pad 0,8 memberi 97,0 persen. Model classify dilatih pada potongan longgar, jadi
potongan ketat tidak cocok dengan latihannya. **`DIGIT_CLS_PAD=0.8` wajib.**
Catatan metode: pernah tercatat 96,7 persen seolah itu kemampuan sistem.
Itu angka top-1 di data validasi classify, BUKAN pada kotak dari detektor.
Jangan mencampur keduanya.

### Menghitung kotak label pakai `wc -l`
**CARA UKUR YANG SALAH.** Berkas tanpa baris baru di akhir terhitung kurang satu.
Sebaran sebenarnya 49,8 persen satu digit dan 45,3 persen dua digit, tanpa gambar
kosong. Hitung ulang pakai Python, jangan `wc -l`.

---

## 5. YANG SUDAH TERBUKTI MEMBANTU, JANGAN DIBONGKAR

| Perubahan | Hasil terukur |
|---|---|
| `BYTETRACK_MATCH_THRESH=0.7` | track 1697 jadi 540, umur 0,53 jadi 1,60 detik, 31 persen lebih cepat, kemurnian tetap |
| `PelacakBola` (`BALL_TRACK=1`) | cakupan bola 72,8 jadi 90,9 persen, 400 lompatan palsu ditolak |
| 7 perbaikan gerbang `MADE_FG` | termasuk `>= dwell` yang dulu `== dwell` dan benar cuma karena kebetulan |
| Gerbang per `DETECTOR_METHOD` | 45/3 untuk RF-DETR, 70/2 untuk YOLO |
| Tahap kedua classify digit | bacaan satu frame 22 jadi 37,5 persen |
| SlowFast ganti VideoMAE | 0,792 lawan 0,549, kelas identik, bobot sudah ada |
| `PATH_CLEAN_METHOD=roboflow` | percepatan mustahil 53 jadi 4 persen |
| `HOOP_ROI_ENABLE=1` potongan 448 near 300 | asli ditemukan 2/5 jadi 3/5, palsu 2 jadi 3, nilai 2/3 benar 2 jadi 3, waktu +5 persen |
| `FG_RIM_GATE=1` gerbang tepi kotak rim | palsu 2 jadi 1 tanpa asli berkurang; yang dibuang detik 1:38 yang lewat 0,5 m DI SAMPING cincin, waktu +2 persen |
| KEDUANYA bersama | asli 2/5 jadi 3/5 dengan palsu TETAP 2; pengaruhnya menjumlah, tidak berinteraksi |
| `FG_RIM_CORE_CONFIRM=1` inti 0,10 | asli 3/5 jadi 4/5 dengan palsu TETAP 2, ketepatan 60 jadi 67 persen; yang ditemukan 2:26 yang tak pernah terdeteksi setelan mana pun; biaya waktu nol; TAPI nilai 2/3 benar turun 3 jadi 2 |

---

## 6. PENGHAMBAT YANG MASIH TERBUKA

DIPERBARUI 12 Oktober 2026. Dengan `HOOP_ROI_ENABLE`, `FG_RIM_GATE`, dan
`FG_RIM_CORE_CONFIRM` (inti 0,10) menyala, per kejadian **4 dari 5** bola masuk
dikenali dengan 2 palsu, ketepatan 67 persen. Acuan sebelumnya 3 dari 5 dengan
2 palsu, 60 persen.

Yang MASIH terbuka:

1. **Bola masuk 1:30 TIDAK BISA dicapai dari sisi gerbang, terukur.** Frame
   nyata terdekat ke pusat cincin 24,3 px, dan tiga frame yang lebih dalam
   (14 px, 9 px) semuanya posisi PERKIRAAN pelacak yang dilarang dipakai
   sebagai bukti terakhir. Penghitung dwell hanya mencapai 3, jadi ambang
   minimum apa pun yang masih sah tidak menolongnya. Jalannya memperbaiki
   DETEKSI bola di potongan dekat rim, bukan melonggarkan gerbang.
2. **Palsu 1:45 dan 3:52 bertahan.** 3:52 bisa dibuang batas atas dwell TAPI
   hanya bersama `FG_RIM_CORE_CONFIRM`, lihat bagian `FG_HANG_REJECT` di atas.
3. **NILAI 2 atau 3 angka sekarang jadi penghambat utama, bukan lagi geometri
   ring.** Dari 4 bola masuk yang dikenali, hanya 2 nilainya benar. Dua
   galatnya:

       0:58  2 angka dari 2,5 m  ->  3 angka dari 10,8 m   SALAH
       2:26  baru ditemukan      ->  2 angka, harusnya 3   SALAH

   Sebabnya FRAME saat kejadian menyala dipakai memperkirakan posisi penembak,
   dan gerbang inti cincin menyalakan kejadian lebih awal di dalam rentetan
   dwell. Palsu 3:52 juga dinilai 3 angka dari 10,2 m. Jadi tiga dari lima
   galat nilai di klip ini berasal dari POSISI PENEMBAK.
4. Latih ulang `ball-in-basket` PADA POTONGAN dekat rim. Lihat bagian kelas
   `ball-in-basket` di atas.

## CARA MENGUKUR ULANG

    # mutu pelacakan saja, sekitar 5 menit untuk video 4 menit
    DETECTOR_METHOD=rfdetr VIDEO=/path/klip.mp4 python eksperimen/16_ukur_pelacakan.py

    # pipeline penuh lalu cocokkan tiap MADE_FG ke papan skor per waktu
    DETECTOR_METHOD=rfdetr VIDEO=/path/klip.mp4 RENDER_OUTPUT_VIDEO=0 \
      python eksperimen/17_ukur_kejadian.py

Empat jebakan pengukuran yang pernah menyesatkan:
1. **Ukuran "lompatan dari lintasan" BERAT SEBELAH kalau tiap sumber dinilai
   terhadap tetangganya SENDIRI.** Sumber yang deteksinya jarang hanya menemukan
   tetangga di dalam rentetannya sendiri yang mulus, jadi lajunya terlihat
   rendah secara palsu. Terbukti 10 Oktober 2026: membandingkan potongan ring
   lawan frame penuh memberi satu urutan, lalu uji CERMIN (menukar mana yang
   jadi acuan) MEMBALIK hasilnya jadi 1,2 persen lawan 18,3 persen. Jadi tidak
   satu pun dari dua baris itu bisa dipakai memutuskan.
   **Cara yang benar: satu himpunan acuan yang SAMA untuk semua kelompok yang
   dibandingkan.** Itu yang dipakai `20_analisis_tambahan_bola.py`: acuannya
   selalu deteksi KUAT, dan hanya entri dirinya sendiri yang dibuang. Karena
   kerapatan pembandingnya identik untuk kedua kelompok, hasil skrip 20 TIDAK
   kena berat sebelah ini, dan kesimpulan ambang bola tetap berlaku. Kesimpulan
   itu juga dikuatkan bukti terpisah: skor di pipeline turun 9 jadi 7.
   Kalau cuma punya ukuran yang berat sebelah, pakai angka yang tidak
   bergantung kerapatan, misalnya khusus frame YANG DIISI sumber baru.
2. **Menjalankan skrip 17 pada VIDEO PENUH.** `Q1-UNESA-UBAYA-CUT-2-17.mp4`
   panjangnya **601 detik**, sedangkan `TRUTH` di skrip 17 cuma sampai detik
   146 dan disusun untuk POTONGAN yang dimulai detik 60. Memberi video penuh
   membuat 7,5 menit sisanya dinilai PALSU tanpa pembanding, dan bola masuk
   yang asli tidak pernah cocok karena waktunya bergeser 60 detik. Pernah
   terjadi dan menghasilkan "1 dari 5 asli, 5 palsu" yang sama sekali tidak
   berarti. SELALU pakai potongan 240 detik:
       ffmpeg -ss 60 -i <video penuh> -t 240 -c copy -y /home/paul/q1_4menit.mp4
3. **Kemurnian harus PER TRANSISI, bukan persen track yang memuat lompatan.**
   Setelan yang menghasilkan track lebih panjang otomatis punya lebih banyak
   transisi per track, jadi persentasenya naik tanpa kemurniannya benar benar
   turun. Kesalahan ini pernah hampir membuat kesimpulan terbalik.
4. **`env VAR=... python ...` TIDAK memuat `.env`.** Akibatnya `ACTION_EVERY_N`
   dan `COURT_EVERY_N` ikut berbeda, dan angka antar percobaan jadi tak
   sebanding. Skrip 16 dan 17 sudah memuat `.env` lebih dulu.

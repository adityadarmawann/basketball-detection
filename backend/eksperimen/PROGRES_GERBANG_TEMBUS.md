# PROGRES: gerbang TEMBUS cincin (saklar A, B, C)

Ditulis 12 Oktober 2026. Berkas ini dipakai supaya pekerjaan bisa dilanjut
tanpa menebak nebak kalau sesi terputus di tengah antrean.

Video uji: `/home/paul/q1_4menit.mp4` (240 detik, dipotong mulai detik 60).
Kebenaran papan skor: 5 bola masuk, 11 poin, daftarnya di `TRUTH` skrip 17.
Penafsir: `/home/paul/miniconda3/envs/cl-gpu/bin/python` (yang lain mundur
diam diam ke YOLO).

## YANG DIUJI

| Saklar | Isi | Cara menyalakan |
|---|---|---|
| A | ambang minimum dwell 3 jadi 2 (efektif 6 jadi 4 frame di 30 fps) | `FG_ZONE_DWELL_F=2` |
| B | tolak bola MENGGANTUNG: batas ATAS dwell + perpindahan tegak minimum | `FG_HANG_REJECT=1` |
| C | konfirmasi dari INTI cincin, tetap wajib deteksi NYATA | `FG_RIM_CORE_CONFIRM=1` |

Bawaan ketiganya MATI. B dan C ditambahkan di `pipeline/event_engine.py`,
A hanya mengisi env.

## HASIL

Diisi saat tiap run selesai. Lihat juga `/home/paul/hasil_*.log`.

| Run | setelan | asli /5 | palsu | ketepatan | nilai 2/3 benar | poin | waktu | putusan |
|---|---|---|---|---|---|---|---|---|
| acuan baru | `.env` apa adanya (DIGIT_CLS hidup) | 3 | 2 | 60 % | 3 | 11 | 18,1 mnt | acuan |
| A | `FG_ZONE_DWELL_F=2` | 3 | 2 | 60 % | 3 | 11 | 18,1 mnt | TIDAK BERPENGARUH |
| B | `FG_HANG_REJECT=1` | 2 | 1 | 67 % | 2 | 6 | 18,2 mnt | DITOLAK |
| C inti 0,15 | `FG_RIM_CORE_CONFIRM=1` | 4 | 3 | 57 % | 2 | 16 | 18,1 mnt | tawar menawar |
| C inti 0,10 | `+ FG_RIM_CORE_FRAC=0.10` | 4 | 2 | 67 % | 2 | 14 | 18,4 mnt | **DITERIMA, dinyalakan di `.env`** |
| B + C 0,10 | `FG_HANG_REJECT=1` + C 0,10 | 4 | 1 | **80 %** | 2 | 11 | 18,3 mnt | memenuhi syarat, TAPI rapuh |

Daftar kejadian tiap run:

    acuan      0:18 asli  0:58 asli  1:45 PALSU  1:57 asli              3:52 PALSU
    A          sama persis dengan acuan, detik demi detik
    B          0:18 asli             1:45 PALSU  1:57 asli
    C 0,15     0:18 asli  0:58 asli  1:45 PALSU  1:57 asli  2:24 asli   3:52 PALSU  3:57 PALSU
    C 0,10     0:18 asli  0:58 asli  1:45 PALSU  1:57 asli  2:24 asli   3:52 PALSU
    B + C      0:18 asli  0:58 asli  1:45 PALSU  1:57 asli  2:24 asli

Yang selalu terlewat di SEMUA setelan: 1:30. Yang selalu palsu: 1:45 dan 3:52.

Acuan baru PERSIS SAMA per kejadian dengan acuan sebelum `DIGIT_CLS_ENABLE`
dinyalakan: ditemukan 0:18, 0:58, 1:57; terlewat 1:30 dan 2:26; palsu 1:45 dan
3:52. Jadi tahap kedua classify digit TIDAK menyentuh skor sama sekali, sesuai
harapan, dan angka acuan lama tetap berlaku sebagai pembanding.

## TELEMETRI: SAKLAR B TERBANTAH SEBELUM DIJALANKAN

Diukur dengan skrip 23 lalu 24, satu kali jalan, 6.756 frame gerbang. Kolom
`dwell saat nyala` adalah nilai penghitung zona pada frame saat kejadian dibuat
(sudah termasuk penambahan frame itu sendiri).

    rentetan   isinya                      dwell maks  dwell saat nyala  turun px
    0:18,7     bola masuk ASLI                  6              6           36,4
    0:58,4     bola masuk ASLI                 14             14           52,0
    1:45,2     PALSU                            6              6           37,4
    1:57,0     bola masuk ASLI                  6              6           33,2
    3:51,9     PALSU                           10             10           40,1
    2:24,3     bola masuk ASLI, TERLEWAT        8         tidak nyala     186,0
    1:56,4     tidak pernah nyala               9         tidak nyala      20,9
    3:57,4     tidak pernah nyala               8         tidak nyala     146,8

**Batas ATAS dwell tidak bisa disetel.** Untuk membuang palsu 3:52 batasnya
harus di bawah 10, tapi bola masuk ASLI di 0:58 menyala pada dwell **14**, jadi
batas apa pun yang membuang 3:52 juga membuang satu bola masuk asli. Urutannya
malah TERBALIK dari dugaan: kejadian dengan dwell terpanjang di klip ini adalah
bola masuk yang BENAR, bukan yang palsu.

**Perpindahan tegak minimum juga tidak bisa disetel.** Dua palsu turun 37,4 dan
40,1 px, sedangkan bola masuk asli yang paling sedikit turun hanya 33,2 px. Jadi
ambang apa pun yang membuang palsu lebih dulu membuang bola masuk asli 1:57.
`FG_HANG_MIN_DROP_PX` tetap 0 (mati) karena itu.

Gambaran "bola menggantung di dalam zona belasan frame" sendiri tidak terbukti:
deretan jarak hampir tetap 28, 27, 27 px di 3:51,8 semuanya pada `dwell=0`,
artinya gerbang tepi rim sudah menolaknya secara mendatar dan itu BUKAN frame
di dalam zona. Rentetan dwell 3:51,9 yang sebenarnya bergerak 27 -> 6 -> 18 px,
yaitu pola menembus.

Saklar B tetap DIJALANKAN satu kali untuk membuktikan ramalan di atas, dengan
batas bawaan `FG_ZONE_DWELL_MAX_F=4` (efektif 8 frame). Ramalannya: asli turun
3 jadi 2 karena 0:58 ikut terbuang, palsu turun 2 jadi 1.

## TELEMETRI: SAKLAR C punya sasaran yang jelas

Jarak mendatar TERKECIL dari pusat kotak rim di antara frame NYATA, dibagi
separuh lebar kotak rim (lebar median 37,5 px):

    rentetan   isinya                      toleransi tegak 0 px   2 px
    0:18,7     bola masuk ASLI                   0,08            0,08
    0:58,4     bola masuk ASLI                   0,17            0,04
    2:24,3     bola masuk ASLI, TERLEWAT         3,68            0,02
    1:56,4     tidak pernah nyala               tak ada         tak ada
    3:51,9     PALSU, sudah nyala                0,27            0,27
    3:57,4     tidak pernah nyala                0,75            0,12

Dua hal penting:
1. **Toleransi tegak 0 px mustahil dipakai.** Di frame terdalam, titik tengah
   bola masih 0,1 sampai 1,8 px DI ATAS pusat kotak rim. Jadi bawaan yang
   semula saya tulis (`FG_RIM_CORE_UP_PX=0`) tidak akan meloloskan satu bola
   masuk pun. Diganti 2 px, masih jauh di dalam kotak rim yang separuh
   tingginya 9,4 px.
2. **Risikonya sudah terlihat sebelum dijalankan.** Dengan pecahan 0,15 dan
   toleransi 2 px, bola masuk terlewat 2:26 lolos (0,02) TAPI rentetan 3:57,4
   yang sekarang tidak pernah menyala juga ikut lolos (0,12). Kalau rentetan
   itu memang lolos syarat lain, hasilnya satu asli naik DAN satu palsu naik,
   yaitu TAWAR MENAWAR, bukan perbaikan bersih. Pecahan 0,10 memisahkan
   keduanya.

   Pecahan 0,15 DIJALANKAN lebih dulu justru supaya ramalan ini bisa diuji,
   bukan diakali. Hasilnya tepat seperti ramalannya (asli 4, palsu 3), lalu
   0,10 dijalankan dan juga tepat (asli 4, palsu 2). Setelah dua duanya
   terukur, bawaan di kode diubah jadi **0,10**. Perlu diingat ambang itu
   dipisahkan oleh satu kejadian saja (0,02 lawan 0,12), jadi kalau detektor
   atau resolusi potongan ring berubah, ukur ulang dengan skrip 17.

## TEMUAN YANG SUDAH PASTI SEBELUM RUN APA PUN

### Bola masuk 1:30 TIDAK BISA dicapai oleh A, B, maupun C

Dari `/home/paul/hasil_diagnosa_gerbang.log` garis waktu 1:25,2 (saat nyata
bolanya, papan skor telat):

    jarak ke pusat ring   24 px   14 px   9 px   272 px
    penghitung dwell      0->1    1->2    2->3   3->0
    sumber posisi         NYATA   kira    kira   NYATA

Tiga frame terdalam itu HANYA dua yang perkiraan pelacak, dan frame nyata
terdekat 24 px. Jadi:

1. Saklar A menurunkan ambang jadi 4 frame, sedangkan dwell di sini maksimum
   **3**. A saja secara aritmetika TIDAK CUKUP.
2. Menurunkan ambang sampai 2 frame (`FG_ZONE_DWELL_F=1`) memang akan lolos
   gerbang dwell, tapi frame yang dipakai sebagai bukti terakhir adalah posisi
   PERKIRAAN, dan aturan mutlaknya melarang itu. Jadi bukan jalan yang sah.
3. Saklar C menuntut deteksi NYATA di inti cincin. Frame nyata terdekat 24 px,
   jauh di luar inti.

**Kesimpulan: 1:30 bukan soal ambang gerbang, tapi soal tidak ada deteksi bola
nyata di dekat cincin.** Jalannya memperbaiki deteksi bola di potongan dekat
rim, bukan melonggarkan gerbang. Ini membantah harapan bahwa saklar A akan
mengembalikan 1:30.

### Bola masuk 2:26 ADALAH sasaran saklar C, bukan saklar A

Garis waktu 2:24,3 sampai 2:24,5:

    jarak ke pusat ring   39   32   12   5   3   0   1   3   162
    penghitung dwell       1    2    3   4   5   6   7   8   0
    sumber posisi          N   kira  N   N   N   N   N  kira  N

dwell mencapai 8 dari ambang 6, jadi gerbang dwell SUDAH lolos tanpa saklar A.
Yang menahannya syarat "deteksi nyata di bawah tepi bawah kotak rim", dan itu
tidak pernah terpenuhi karena bola melompat 162 px tepat setelah menembus.
Frame jarak 0, 1, dan 3 px itu deteksi NYATA, jadi inilah yang hendak dipakai
saklar C.

### Catatan cara membaca log diagnosa 3:52

Deretan "jarak hampir tidak berubah" 28, 28, 27, 27, 27 px di 3:51,8 sampai
3:51,9 itu semuanya pada `dwell=0`, artinya bola BELUM dianggap masuk zona
(gerbang tepi rim menolaknya secara mendatar). Rentetan dwell yang sebenarnya
baru mulai di 3:51,9 dan jaraknya bergerak 27 -> 6 -> 18 px, yaitu pola
menembus, bukan menggantung. Jadi gambaran "menggantung belasan frame di dalam
zona" tidak persis seperti itu; yang tersisa sebagai pembeda hanyalah PANJANG
rentetannya, 9 frame, sementara bola yang benar benar menembus cincin 0,45 m
dengan laju 4 sampai 6 m/s hanya perlu sekitar 3 frame di 30 fps.

## APAKAH MEKANISMENYA BEKERJA SEPERTI RANCANGANNYA

### A, ambang minimum dwell: TIDAK BERPENGARUH, dan itu bisa diramalkan
Daftar kejadiannya SAMA PERSIS dengan acuan, detik demi detik. Sebabnya terukur
di telemetri: kelima kejadian acuan sudah menyala pada dwell 6, 6, 6, 10, dan
14, jadi ambang minimum 6 tidak pernah menjadi penghalang bagi satu pun dari
mereka. Menurunkannya ke 4 hanya memperbolehkan penyalaan yang lebih awal, dan
yang menentukan tetap syarat di bawah cincin.
Dua yang terlewat tidak tertolong: 2:26 memang sudah mencapai dwell 8 jadi
bukan ambang yang menahannya, dan 1:30 hanya mencapai dwell 3 sehingga ambang
4 pun masih terlalu tinggi.

### B, tolak bola menggantung: BEKERJA tepat seperti rancangannya, dan justru
itu yang membuktikan rancangannya SALAH
Yang dibuang memang betul palsu 3:52, sesuai sasaran. Tapi yang ikut terbuang
bola masuk ASLI 0:59, persis seperti yang diramalkan dari telemetri, karena
kejadian itu menyala pada dwell 14, lebih lama daripada palsu yang mau dibuang.
Jadi mekanismenya jalan, premisnya yang keliru: di klip ini LAMANYA bola berada
di dalam zona cincin TIDAK memisahkan bola masuk dari bukan bola masuk.

### C, konfirmasi dari inti cincin: BEKERJA, dan meloloskan 2:26 seperti sasaran
2:26 ditemukan di kedua pecahan yang diuji, 0,15 maupun 0,10. 1:30 tetap tidak
tertolong, dan itu memang sudah diramalkan: frame nyata terdekatnya 24 px dari
pusat cincin, jauh di luar inti, dan frame yang lebih dalam semuanya perkiraan
pelacak yang dilarang dipakai.
Pecahan 0,15 juga menyalakan rentetan 3:57 jadi palsu baru, tepat seperti
ramalan telemetri (0,12 lolos ambang 0,15). Pecahan 0,10 membuangnya kembali,
juga tepat seperti ramalan. Jadi telemetri skrip 23 meramalkan arah dengan
BENAR untuk kedua pecahan, dua duanya terbukti di pipeline.

### BIAYA C yang TIDAK saya duga: nilai 2 atau 3 angka jadi lebih buruk
Di acuan, ketiga kejadian asli nilainya benar. Dengan C, hanya dua dari empat.
Dua galatnya:

    0:58  acuan: 2 angka dari 2,5 m BENAR   ->  C: 3 angka dari 10,8 m SALAH
    2:26  tidak pernah ditemukan            ->  C: 2 angka dari 2,1 m, harusnya 3

Sebabnya jelas: gerbang C membuat kejadian menyala LEBIH AWAL di dalam rentetan
dwell, dan FRAME saat kejadian menyala itulah yang dipakai memperkirakan posisi
penembak. Pada 0:58 penyalaan yang lebih awal mengambil titik riwayat yang
salah dan jaraknya meledak 2,5 jadi 10,8 m.

Ini menguatkan lead nomor 1 di `LANJUTAN_BERIKUTNYA.md`: penghambat berikutnya
POSISI PENEMBAK, bukan lagi geometri ring. Dua dari tiga galat nilai sekarang
berasal dari situ, dan palsu 3:52 juga dinilai 3 angka dari 10,2 m.

### GABUNGAN B + C: hasil terbaik, tapi keamanannya KEBETULAN
Asli 4 dari 5, palsu 1, ketepatan 80 persen. Naik di KEDUA sumbu sekaligus
dibanding acuan (asli 3 jadi 4, palsu 2 jadi 1), jadi memenuhi syarat terima.

Mekanismenya sebuah INTERAKSI, bukan penjumlahan:
saklar C membuat kejadian 0:58 menyala JAUH LEBIH AWAL di dalam rentetan dwell.
Sendirian, saklar B membuang 0:58 justru karena kejadian itu menyala pada dwell
14, di atas batas 8. Begitu C menyalakannya pada dwell rendah, batas atas tidak
lagi menyentuhnya, sementara palsu 3:52 yang menyala pada dwell 10 tetap
terbuang.

**Itu sebabnya B TIDAK dinyalakan di `.env`.** Keamanannya bergantung pada satu
kejadian di satu klip. Kalau setelan C diubah atau dimatikan, B kembali
membuang bola masuk asli tanpa pemberitahuan apa pun. Keputusan menyalakannya
diserahkan ke user, dan kalau dinyalakan harus diukur ulang dengan skrip 17
setiap kali setelan C berubah.

Poin 11 lawan kebenaran 11 pada baris ini KEBETULAN: satu palsu bernilai 2
angka menutupi satu bola masuk yang terlewat, dan satu nilai 2/3 yang salah
menambal sisanya. Jangan dipakai sebagai bukti.

## YANG DINYALAKAN DI `.env` SETELAH PEKERJAAN INI

    FG_RIM_CORE_CONFIRM=1
    FG_RIM_CORE_FRAC=0.10
    FG_RIM_CORE_UP_PX=2

Alasan dan angkanya ada di `.env.saklar.example` yang TERLACAK git, karena
`.env` sendiri diabaikan git. `FG_HANG_REJECT` dan `FG_ZONE_DWELL_MAX_F`
ditulis di situ juga tapi DIKOMENTARI, beserta peringatannya.

## ANTREAN DAN CARA MELANJUTKAN

    cat /home/paul/antrean_gerbang_status.txt      # kemajuan antrean
    tail -20 /home/paul/hasil_<nama>.log           # hasil tiap run
    python eksperimen/24_analisis_telemetri.py     # ambang dari telemetri

Antrean dilepas dengan `setsid nohup` dan menjalankan pekerjaannya SINKRON di
dalam skripnya sendiri, jadi tidak butuh `pgrep` sama sekali.

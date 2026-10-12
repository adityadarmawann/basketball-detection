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

| Run | setelan | asli /5 | palsu | ketepatan | poin | waktu | putusan |
|---|---|---|---|---|---|---|---|
| acuan baru | `.env` apa adanya (DIGIT_CLS hidup) | 3 | 2 | 60 % | 11 | 18,1 mnt | acuan |
| A | `FG_ZONE_DWELL_F=2` | | | | | | |
| C | `FG_RIM_CORE_CONFIRM=1` | | | | | | |
| B | `FG_HANG_REJECT=1` | | | | | | |

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
   yaitu TAWAR MENAWAR, bukan perbaikan bersih. Pecahan 0,10 akan memisahkan
   keduanya, tapi itu menyetel ambang pas ke satu kejadian; sengaja TIDAK
   dipakai sebagai bawaan supaya tidak menipu diri sendiri.

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

## ANTREAN DAN CARA MELANJUTKAN

    cat /home/paul/antrean_gerbang_status.txt      # kemajuan antrean
    tail -20 /home/paul/hasil_<nama>.log           # hasil tiap run
    python eksperimen/24_analisis_telemetri.py     # ambang dari telemetri

Antrean dilepas dengan `setsid nohup` dan menjalankan pekerjaannya SINKRON di
dalam skripnya sendiri, jadi tidak butuh `pgrep` sama sekali.

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
| acuan baru | `.env` apa adanya (DIGIT_CLS hidup) | | | | | | |
| A | `FG_ZONE_DWELL_F=2` | | | | | | |
| B | `FG_HANG_REJECT=1` | | | | | | |
| C | `FG_RIM_CORE_CONFIRM=1` | | | | | | |

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

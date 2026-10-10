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

## HASIL PIPELINE TUGAS A, DITERIMA

`17_ukur_kejadian.py`, `HOOP_ROI_ENABLE=1 HOOP_ROI_SIZE=448
HOOP_ROI_NEAR_PX=300`, log `/home/paul/hasil_uji_roi_448_near300.log`:

    setelan                    asli   nilai   palsu   poin   waktu
    acuan produksi             2/5      2       2       9    16,1 menit
    potongan 448, near 300     3/5      3       3      13    16,9 menit

    acuan  : ditemukan 0:58 dan 1:57; terlewat 0:19, 1:30, 2:26
    dengan : ditemukan 0:18, 0:58, 1:57; terlewat 1:30, 2:26
             palsu 1:38 (0,5 m), 1:45 (3,2 m, BARU), 3:52 (10,2 m)

Satu bola masuk ASLI yang tadinya terlewat (0:19) sekarang ditemukan, dan
satu deteksi palsu baru ikut masuk (1:45). Jadi asli +1 dan palsu +1: palsu
TIDAK naik lebih banyak daripada asli, sehingga syarat penerimaan terpenuhi.

Ketepatan nilai 2/3 juga ikut: 3 dari 3 yang ditemukan dinilai BENAR.
Presisi TIDAK berubah, 2 dari 4 jadi 3 dari 6, dua duanya 50 persen.
Yang naik CAKUPAN KEJADIAN, 40 jadi 60 persen.

Biaya waktu +5 persen (16,1 jadi 16,9 menit), jauh di bawah dua kali lipat.
Jauh lebih murah daripada +73 persen waktu detektor yang terukur kalau
potongan dijalankan di setiap frame, karena gerbang NEAR_PX=300 memangkasnya
ke 28,6 persen frame DAN deteksi berjalan di utas terpisah bersamaan dengan
ByteTrack di CPU.

### Bukti jalur ini benar benar terpakai
Buktinya dari PERILAKU, bukan dari log: tiga hal berubah yang mustahil berubah
tanpa jalur potongan, karena tidak ada lain yang berbeda antar jalan itu.
0:18 muncul, 1:45 muncul, dan jarak tembakan 1:57 bergeser 4,4 jadi 3,9 m
karena posisi bolanya dipertajam. Angka 3,9 m itu muncul di KEDUA jalan yang
potongannya hidup dan 4,4 m di kedua jalan yang mati, jadi sebabnya pasti.

Catatan jujur soal usaha membuktikannya lewat log: versi pertama baris bukti
dipasang di tingkat `logger.info`, dan itu TIDAK PERNAH muncul karena skrip 17
tidak memanggil `logging.basicConfig` sehingga logger akarnya tetap di bawaan
WARNING. Dua jalan pipeline penuh sudah terpakai sebelum itu ketahuan.
Sekarang potongan PERTAMA dicatat di tingkat WARNING dan sudah diuji muncul:

    Potongan ring AKTIF: ukuran 448 px, gerbang jarak 300 px, ambang 0.25

### Yang MEMBANTAH dugaan: angka proksi tidak meramalkan hasil pipeline
Ukuran proksi di atas (laju lompatan gabungan 3,8 jadi 4,7 persen, pengisian
celah 24,7 persen salah posisi) mengarah ke KESIMPULAN SALAH kalau dipakai
sendiri. Pipeline justru membaik. Sebabnya: dari 179 pengisian celah, 114
posisinya benar dan cuma 30 yang melewati `PelacakBola` dengan posisi salah
lebih dari 60 px (23 lagi ditolak pelacak, jatah 90 px per frame jeda).
Tiga puluh posisi salah tersebar di seluruh 4 menit tidak cukup mengarang
kejadian baru, sementara 114 posisi benar cukup menyambung satu rantai bukti
bola masuk yang tadinya putus.

**Pelajarannya: laju lompatan bagus untuk MEMAHAMI, tapi tidak boleh jadi
satu satunya penentu terima atau tolak. Yang memutuskan tetap skrip 17.**

## HASIL PIPELINE TUGAS B, DITERIMA

`17_ukur_kejadian.py`, `FG_RIM_GATE=1` SENDIRI (potongan ring MATI), log
`/home/paul/hasil_uji_rim_saja.log`:

    setelan              asli   nilai   palsu   poin   waktu
    acuan produksi       2/5      2       2       9    16,1 menit
    gerbang tepi rim     2/5      2       1       7    16,4 menit

Palsu TURUN 2 jadi 1 dan yang asli TIDAK berkurang, jadi syarat penerimaan
terpenuhi lewat jalan kedua (palsu turun tanpa asli berkurang).

Yang dibuang tepat yang dirancang dibuang: deteksi palsu detik 1:38 yang
jaraknya 0,5 m dari ring. Itu bola yang lewat DI SAMPING cincin, memenuhi
lingkaran radius 45 px tapi di luar rentang mendatar kotak rim yang 38 px.
Palsu detik 3:52 (10,2 m, dihitung 3 angka) TETAP ADA, jadi sebab palsu itu
bukan geometri ring melainkan penentuan posisi penembak.

Dua bola masuk asli yang sudah ditemukan TIDAK hilang, padahal gerbangnya
lebih ketat. Itu yang paling penting: menajamkan bidang skor tidak memakan
yang sah. Biaya waktu +2 persen.

### Uji sintetis yang mendahului, membuktikan jalurnya memang membedakan
Tanpa GPU, bola disimulasikan turun melewati kotak rim 38,0 x 16,6 px:

    bola lurus di tengah cincin      : lingkaran MENYALA, tepi rim MENYALA
    bola 35 px DI SAMPING cincin     : lingkaran MENYALA, tepi rim MENOLAK

35 px itu lolos lingkaran 45 px tapi di luar separuh lebar cincin 19 px.
Jadi perbedaannya nyata dan terarah, bukan kebetulan setelan.

## GABUNGAN A DAN B, DITERIMA, DAN PENGARUHNYA MENJUMLAH

Diuji sebagai jalan KETIGA, setelah masing masing terbukti sendiri sendiri.
Log `/home/paul/hasil_uji_roi_dan_rim.log`.

    setelan                     asli   nilai   palsu   poin   waktu
    acuan produksi              2/5      2       2       9    16,1 menit
    potongan ring saja          3/5      3       3      13    16,9 menit
    gerbang tepi rim saja       2/5      2       1       7    16,4 menit
    KEDUANYA                    3/5      3       2      11    16,9 menit

    keduanya: ditemukan 0:18, 0:58, 1:57; terlewat 1:30, 2:26
              palsu 1:45 (3,2 m) dan 3:52 (10,2 m)

Harapannya asli 3/5 dan palsu 2, dan itu PERSIS yang keluar. Jadi keduanya
TIDAK berinteraksi: potongan ring menambah bukti deteksi bola, gerbang tepi
rim menyaring geometri, dan keduanya bekerja di tempat yang berbeda.

Dibanding acuan: asli 2/5 jadi 3/5, palsu TETAP 2. Jadi lebih baik di satu
sumbu tanpa memburuk di sumbu lain. Nilai 2/3 juga 2 jadi 3 benar.

**JANGAN memakai "poin 11 lawan kebenaran 11" sebagai bukti.** Total poin
sudah pernah menipu persis begini dan klaimnya sudah dicabut sekali: palsu
dan yang terlewat SALING MENIADAKAN. Di sini pun begitu, dua palsu bernilai
2 dan 3 angka menutupi dua bola masuk yang terlewat. Yang sahih angka per
kejadian: 3 dari 5 ditemukan, 2 palsu.

## YANG MASIH TERBUKA

1. `HOOP_ROI_OVERLAP_PX=0` (potongan HANYA mengisi celah, tidak pernah
   mengganti) untuk memisah sumbangan mengisi dari sumbangan mengganti.
2. Palsu 3:52 bertahan di SEMUA setelan. Jaraknya 10,2 m dan dinilai 3 angka,
   jadi sebabnya POSISI PENEMBAK, bukan gerbang ring. Itu jalur terpisah.
3. Dua bola masuk yang masih terlewat di semua setelan: 1:30 dan 2:26.

Uji SATU perubahan pada satu waktu. Gabungan hanya sah sebagai jalan KETIGA,
setelah masing masing terpisah.

## CARA MENGULANG

    # rekam potongan lawan frame penuh, 1,8 menit, butuh GPU
    MODE=rekam DETECTOR_METHOD=rfdetr HOOP_ROI_NEAR_PX=0 HOOP_ROI_SIZE=448 \
      REKAM=/home/paul/rekam_roi/roi_448.json VIDEO=/home/paul/q1_4menit.mp4 \
      /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/21_ukur_potongan_ring.py

    # nilai ulang di ambang mana pun, tanpa GPU
    MODE=nilai REKAM=/home/paul/rekam_roi/roi_448.json \
      /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/21_ukur_potongan_ring.py

## KEADAAN SAKLAR DI `.env`, PERIKSA INI SEBELUM MENILAI JALAN BERIKUTNYA

`.env` tidak dilacak git (ada di `.gitignore`), jadi perubahannya TIDAK terlihat
di riwayat commit. Saat tugas ini ditutup keadaannya:

    FG_RIM_GATE=1        <- DINYALAKAN di .env oleh sesi lain, 10 Okt 23:08:38
    HOOP_ROI_ENABLE      <- tidak ada di .env, jadi MATI (bawaan kode)

Bawaan di KODE tetap mati untuk keduanya, jadi `.env` yang menyalakan, bukan
kodenya. Pengukurannya memang mendukung `FG_RIM_GATE=1` (palsu 2 jadi 1 tanpa
asli berkurang), tapi keputusan menyalakannya di produksi milik user.

Empat jalan uji di tugas ini TIDAK tercemar oleh perubahan itu, dan ini
diperiksa bukan diasumsikan:

    acuan produksi      mulai 21:14:44
    potongan ring saja  mulai 22:24:14
    gerbang rim saja    mulai 22:45:14
    KEDUANYA            mulai 23:02:16
    .env diubah                23:08:38   <- sesudah semuanya mulai

`load_dotenv` jalan sekali saat proses MULAI, dan dia TIDAK menimpa env yang
sudah ada, sehingga saklar yang dikirim lewat `env VAR=... ` selalu menang.
Keempat jalan mengirim saklarnya eksplisit.

Untuk mematikan kembali: hapus atau nolkan baris `FG_RIM_GATE` di `.env`.

## CATATAN: ADA SESI LAIN YANG BEKERJA DI POHON KERJA YANG SAMA

Terjadi 10 Oktober 2026 dan perlu diketahui karena bisa merusak pengukuran:
- Skrip `/home/paul/antrean_rim.sh` muncul di tengah tugas dan menjalankan
  `17_ukur_kejadian.py` KEDUA bersamaan dengan jalan saya. Dua proses GPU
  sekaligus, dan dua duanya menulis ke berkas log YANG SAMA. Itu melanggar
  aturan satu proses GPU dan membuat waktu proses tidak berarti.
  Sudah dihentikan dan diarsipkan jadi `antrean_rim.sh.dihentikan`, lalu uji
  gerbang rim DIJALANKAN ULANG dari bersih supaya angka waktunya sahih.
- Commit `0c88de9` berisi tulisan dari pohon kerja saya yang belum saya commit,
  ditambah satu butir jebakan pengukuran tulisan sesi itu. Isinya benar dan
  tidak ganda, jadi dibiarkan; hanya nomornya dirapikan.

Kalau menjalankan uji GPU, periksa dulu:

    ps -eo pid,sid,etime,args | grep -E "17_ukur|21_ukur|antrean" | grep -v grep
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# TITIK LANJUT, ditulis 10 Oktober 2026 saat sesi dijeda

Sesi dijeda karena user berpindah tempat dan koneksi terputus. Berkas ini
supaya pekerjaan bisa dilanjut tanpa menebak nebak.

## PERIKSA INI PERTAMA SAAT LANJUT

1. **Apakah masih ada proses uji yang hidup?**

       ps -eo pid,etime,args | grep -E "eksperimen/1[0-9]_" | grep -v grep
       nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

   Kalau ada proses yang keluarannya masuk PIPA (bukan ke berkas log), hasilnya
   tidak bisa dibaca lagi. Hentikan saja dan jalankan ulang dengan log ke
   berkas. Jangan dibiarkan memegang GPU.

2. **Apakah ada berkas log hasil run terlepas?**

       ls -la /home/paul/hasil_uji_ambang_*.log 2>/dev/null

   Kalau ada, baca bagian akhirnya. Itu hasil yang diselamatkan sebelum jeda.

3. **Apakah agent meninggalkan berkas progres?**
   `eksperimen/PROGRES_UJI_AMBANG_BOLA.md`. Kalau ada, itu lebih rinci daripada
   berkas ini untuk bagian uji ambang bola.

4. **Apakah pohon kerja bersih?** `git status --short`. Kalau ada perubahan
   setengah jadi dari agent, periksa dulu sebelum dipakai, jangan langsung
   dicommit.

## PENAFSIR PYTHON, JANGAN SALAH

    /home/paul/miniconda3/envs/cl-gpu/bin/python

Yang lain TIDAK punya paket `rfdetr`, dan `create_detector()` akan MUNDUR DIAM
DIAM ke YOLO tanpa pesan kesalahan, lalu gagal memuat `.engine`. Pastikan selalu:
`create_detector()` harus mengembalikan `RFDetrDetector`, bukan
`BasketballDetector`. Ini sudah dua kali hampir menyesatkan.

## ANTREAN YANG BERJALAN TERLEPAS SAAT SESI DIJEDA

`/home/paul/antrean_uji.sh` dilepas dengan `setsid nohup`, induknya 1, jadi
TIDAK butuh sesi Claude maupun internet. Enam pekerjaan berurutan, satu
pekerjaan GPU pada satu waktu:

    1. rekam_rfdetr        deteksi bola RF-DETR, ambang bola 0,10
    2. rekam_yolo          deteksi bola YOLO lama, jendela sama persis
    3. gabung_bola         silangkan keduanya, tidak pakai GPU
    4. uji_acuan_produksi  pipeline penuh, ambang produksi apa adanya
    5. uji_ball_010        pipeline penuh, hanya ambang BOLA jadi 0,10
    6. uji_ball_020        pipeline penuh, hanya ambang BOLA jadi 0,20

Pantau:  `cat /home/paul/antrean_uji_status.txt`
Hasil :  `tail -25 /home/paul/hasil_<nama>.log`
Video :  `/home/paul/q1_4menit.mp4` (240 detik, dipotong mulai detik 60)

## KEADAAN SEKARANG, diperbarui 10 Oktober 2026

Pada ambang produksi apa adanya: per kejadian 2 dari 5 bola masuk dikenali,
2 palsu, 9 poin. Dengan `HOOP_ROI_ENABLE=1` dan `FG_RIM_GATE=1` bersama:
**3 dari 5 dikenali, 2 palsu**. Dua saklar itu masih MATI di `.env`.

Penghambatnya BUKAN pelacakan pemain (sudah dibuktikan: pelacakan diperbaiki
3 kali lebih baik, skor justru bergerak 10 jadi 9). Penghambatnya BUKTI
DETEKSI BOLA di dekat ring, dan itulah yang digarap tugas nomor 2.

Yang masih terbuka, sebabnya sudah terpisah pisah:
- Bola masuk 1:30 dan 2:26 terlewat di SEMUA setelan yang pernah diuji.
- Palsu 3:52 bertahan di SEMUA setelan. 10,2 m dan dinilai 3 angka, jadi
  sebabnya POSISI PENEMBAK, bukan geometri ring. Gerbang ring sudah
  dikesampingkan sebagai tersangka, terukur.

Angka kuncinya: bola selebar 21 px di frame 1280 menyusut jadi **9,5 px** setelah
frame diperkecil ke 576 untuk masuk detektor. Di jendela 16 detik yang memuat
bola masuk nyata, cakupan bola cuma 52,1 persen di ambang bawaan 0.35, dan naik
ke 71,5 persen di ambang 0.10.

## TUGAS

### Nomor 1, SELESAI 10 Oktober 2026, DITOLAK dengan angka
Sapu ambang bola. Hasilnya membantah dugaan: cakupan naik, skor turun.

    ambang bola   cakupan   asli   palsu   poin
    0,25 produksi  69,4 %    2/5     2       9
    0,20           72,8 %    2/5     2       9
    0,10           83,6 %    1/5     2       7

Sebabnya terukur: sepertiga deteksi tambahan menaruh bola di posisi SALAH
(melompat dari lintasan 33,3 persen lawan 3,0 persen pada deteksi kuat).
Gabungan dua detektor juga ditolak, YOLO cuma menambah 3,5 titik persen.
Rinciannya di `PENDEKATAN_DITOLAK.md` dan `KEKUATAN_MODEL_PER_KELAS.md`.

**SASARAN YANG BENAR, hasil pelajaran ini: POSISI BENAR pada saat menentukan,
BUKAN cakupan tinggi.** Jangan lagi mengejar cakupan bola.

### Nomor 2, SELESAI 10 Oktober 2026, DUA DUANYA DITERIMA
Potongan resolusi asli di sekitar ring (`HOOP_ROI_ENABLE`) dan gerbang tepi
kotak rim (`FG_RIM_GATE`). Diuji TERPISAH lalu bersama, skrip 17:

    setelan                     asli   nilai   palsu   poin   waktu
    acuan produksi              2/5      2       2       9    16,1 menit
    potongan ring saja          3/5      3       3      13    16,9 menit
    gerbang tepi rim saja       2/5      2       1       7    16,4 menit
    KEDUANYA                    3/5      3       2      11    16,9 menit

Gabungannya lebih baik dari acuan di satu sumbu tanpa memburuk di sumbu lain:
asli 2/5 jadi 3/5, palsu TETAP 2. Pengaruhnya MENJUMLAH, tidak berinteraksi.
JANGAN memakai "poin 11 lawan 11" sebagai bukti; palsu dan yang terlewat
saling meniadakan, dan klaim semacam itu sudah pernah dicabut sekali.

**DUA DUANYA MASIH MATI DI `.env`.** Menyalakannya keputusan user, bukan saya,
karena presisi tidak membaik (3 dari 5 deteksi benar lawan 2 dari 4), yang
membaik CAKUPAN kejadian. Untuk menyalakan:

    HOOP_ROI_ENABLE=1
    HOOP_ROI_SIZE=448
    HOOP_ROI_NEAR_PX=300
    FG_RIM_GATE=1

Rincian lengkap beserta angka dan sebab teknisnya:
`eksperimen/PROGRES_ROI_RING.md`.

**Temuan cara ukur yang paling berharga dari tugas ini:** sasaran "ketepatan
posisi bola" yang saya pakai sebagai penentu JUSTRU MERAMALKAN ARAH YANG SALAH
untuk potongan ring. Lihat `PENDEKATAN_DITOLAK.md` bagian "Menilai POTONGAN
RING dari laju lompatan posisi saja". Yang memutuskan tetap skrip 17.

### KEPUTUSAN AKHIR nomor 2, 10 Oktober 2026: KEDUANYA DINYALAKAN

    setelan              asli   palsu   ketepatan   poin   waktu
    acuan awal            2/5     2       50 %       9    16,1 mnt
    gerbang rim saja      2/5     1       67 %       7    16,4 mnt
    ROI saja              3/5     3       50 %      13    16,9 mnt
    ROI + gerbang rim     3/5     2       60 %      11    16,9 mnt

`.env` sekarang: `FG_RIM_GATE=1` dan `HOOP_ROI_ENABLE=1`. Bawaan di KODE tetap
mati, jadi lingkungan lain tidak ikut berubah. Alasan dan angkanya ada di
`.env.saklar.example` yang terlacak git, karena `.env` sendiri diabaikan git.

Poin 11 lawan kebenaran 11 itu KEBETULAN. Dua palsu bernilai 2 dan 3 angka
menutupi dua bola masuk yang terlewat. Jangan jadikan bukti.

### Nomor 4, SELESAI 12 Oktober 2026: tiga saklar gerbang TEMBUS cincin
Diuji satu saklar satu run, skrip 17, `/home/paul/q1_4menit.mp4`, RF-DETR.

    setelan                       asli /5  palsu  ketepatan  nilai 2/3  poin  waktu
    acuan (DIGIT_CLS hidup)          3       2       60 %        3       11   18,1 mnt
    A  FG_ZONE_DWELL_F=2             3       2       60 %        3       11   18,1 mnt
    B  FG_HANG_REJECT=1              2       1       67 %        2        6   18,2 mnt
    C  inti cincin 0,15              4       3       57 %        2       16   18,1 mnt
    C  inti cincin 0,10              4       2       67 %        2       14   18,4 mnt
    B + C inti 0,10                  4       1       80 %        2       11   18,3 mnt

**A TIDAK BERPENGARUH.** Daftar kejadiannya sama persis dengan acuan, detik
demi detik. Kelima kejadian acuan sudah menyala pada dwell 6, 6, 6, 10, dan 14,
jadi ambang minimum 6 tidak pernah menjadi penghalang.

**B DITOLAK sendirian.** Mekanismenya jalan, premisnya keliru: bola masuk ASLI
0:59 menyala pada dwell 14, lebih lama daripada palsu 3:52 yang menyala pada
dwell 10. Jadi batas atas apa pun yang membuang palsu juga membuang asli.

**C DITERIMA dan DINYALAKAN di `.env`** dengan pecahan 0,10. Bola masuk 2:26,
yang TIDAK PERNAH terdeteksi di setelan mana pun sebelum ini, sekarang
ditemukan, dan palsu tidak bertambah. Biaya waktu nol.

**B + C memberi ketepatan tertinggi 80 persen tapi TIDAK dinyalakan.**
Keamanannya kebetulan: saklar C membuat 0:58 menyala pada dwell rendah sehingga
batas atas tidak lagi membuangnya. Kalau setelan C berubah, B kembali membuang
bola masuk asli tanpa pemberitahuan. Keputusannya diserahkan ke user.

Rincian lengkap, telemetri, dan sebab teknisnya:
`eksperimen/PROGRES_GERBANG_TEMBUS.md` dan `eksperimen/PENDEKATAN_DITOLAK.md`.
Alat barunya `eksperimen/23_telemetri_gerbang.py` dan `24_analisis_telemetri.py`.

## BATAS YANG MASIH TERSISA, lead untuk berikutnya

DIPERBARUI 12 Oktober 2026, setelah `FG_RIM_CORE_CONFIRM` dinyalakan.

1. **NILAI 2 atau 3 angka sekarang penghambat UTAMA, bukan lagi geometri ring.**
   Dari 4 bola masuk yang dikenali, hanya 2 nilainya benar:

       0:58  2 angka dari 2,5 m  ->  3 angka dari 10,8 m   SALAH
       2:26  baru ditemukan      ->  2 angka, harusnya 3   SALAH

   Sebabnya FRAME saat kejadian menyala dipakai memperkirakan posisi penembak,
   dan gerbang inti cincin menyalakan kejadian lebih awal. Palsu 3:52 juga
   dinilai 3 angka dari 10,2 m. Jadi tiga galat nilai di klip ini SATU SEBAB:
   posisi penembak. Ini lead paling jelas sekarang.
2. **Bola masuk 1:30 tidak bisa dicapai dari sisi gerbang, TERUKUR.** Frame
   nyata terdekat ke pusat cincin 24,3 px, dan frame yang lebih dalam (14 px,
   9 px) semuanya posisi PERKIRAAN pelacak yang dilarang dipakai sebagai bukti
   terakhir. Jalannya memperbaiki DETEKSI bola di potongan dekat rim.
3. **Palsu 1:45 bertahan di SEMUA setelan**, menyala pada dwell 6 dengan pola
   geometri yang tidak terpisahkan dari bola masuk asli. Sama seperti 3:52,
   jaraknya yang salah (3,2 m) menunjuk posisi penembak juga.
4. **`ball-in-basket` layak dilatih ulang PADA POTONGAN dekat rim**, bukan pada
   frame penuh. 2 deteksi dari 2.623 potongan resolusi asli lawan NOL di frame
   diperkecil, jadi kelas itu tenggelam karena ukuran piksel.

### Nomor 3, DISETUJUI USER, BELUM DIMULAI
Tabel pemilik kelas: dua detektor jalan bersama, tiap kelas dimiliki SATU model
yang paling kuat, kelas yang sama di model lemah DIMATIKAN supaya tidak terhitung
dua kali. Rancangannya `OWNER_BALL`, `OWNER_HOOP`, `OWNER_PLAYER`,
`OWNER_REFEREE` lewat env, bawaannya persis perilaku sekarang.

Dokumennya sudah ada: `eksperimen/KEKUATAN_MODEL_PER_KELAS.md`. Baris yang masih
BELUM dan SEBAGIAN harus diisi dari jalan nyata.

**JEBAKAN yang wajib diingat:** label di `test-video/object/labels/` itu
PRA-LABEL buatan YOLO lama yang menunggu dikoreksi manusia. Memakainya untuk
menilai YOLO itu berputar dan YOLO otomatis menang. Kebenaran netral BELUM ADA.

## YANG SUDAH SELESAI DAN TERCATAT

- `eksperimen/PENDEKATAN_DITOLAK.md` — semua pendekatan yang terbukti merugikan
  beserta angkanya. BACA SEBELUM menguji apa pun.
- `eksperimen/KEKUATAN_MODEL_PER_KELAS.md` — tabel pemilik kelas.
- SAM2 diuji dan ditolak. GMC ditolak. Penghalusan posisi ring tidak perlu.
  Kelas `ball-in-basket` nol deteksi, butuh latih ulang.
- `backboard` buta di DUA DUANYA, jangan diandalkan.

## SATU PERBAIKAN YANG SUDAH JADI TAPI BELUM DINYALAKAN

`DIGIT_CLS_ENABLE` masih `0` di `.env`, padahal tahap kedua classify digit sudah
terpasang dan terukur menaikkan bacaan nomor satu frame dari 22 ke 37,5 persen.
`DIGIT_CLS_PAD=0.8` WAJIB, pad kecil menjatuhkan ketepatan ke 65,7 persen.
Menyalakannya perlu satu kali uji pipeline penuh untuk memastikan tidak merusak.

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

## KEADAAN SEKARANG

Skor sistem 9 lawan kebenaran 11 pada potongan 4 menit. Per kejadian, 2 dari 5
bola masuk yang dikenali. Penghambatnya BUKAN pelacakan pemain (sudah dibuktikan:
pelacakan diperbaiki 3 kali lebih baik, skor justru bergerak 10 jadi 9).
Penghambatnya BUKTI DETEKSI BOLA di dekat ring.

Angka kuncinya: bola selebar 21 px di frame 1280 menyusut jadi **9,5 px** setelah
frame diperkecil ke 576 untuk masuk detektor. Di jendela 16 detik yang memuat
bola masuk nyata, cakupan bola cuma 52,1 persen di ambang bawaan 0.35, dan naik
ke 71,5 persen di ambang 0.10.

## TUGAS YANG SEDANG BERJALAN SAAT DIJEDA

### Nomor 1, SEDANG DIKERJAKAN agent
Sapu ambang keyakinan bola lewat `17_ukur_kejadian.py`, membandingkan ambang
0.35 (acuan, sudah diketahui: 2 dari 5 asli, 1 palsu, 9 poin) dengan 0.20 dan
0.10. Pertanyaannya: apakah ambang lebih rendah menambah bola masuk PALSU lebih
cepat daripada yang asli.

Satu run ternyata lebih dari 1 jam, jauh di atas acuan 16 menit. Dua kemungkinan,
dan ini BELUM terjawab:
- wajar, karena ambang rendah menambah deteksi sehingga pose dan aksi ikut
  bertambah banyak
- atau ada bagian yang jatuh ke CPU, atau penafsirnya salah sehingga diam diam
  memakai YOLO

Saat dijeda, GPU terbaca 0 persen berulang kali padahal memorinya dipegang 4 GB.
PERIKSA INI DULU sebelum mempercayai angka apa pun dari run itu.

Agent juga diminta memeriksa apakah `RFDETR_CONF` berlaku untuk SEMUA kelas.
Kalau ya, menurunkannya juga menambah pemain dan hakim palsu, jadi perlu saklar
khusus bola, misal `RFDETR_CONF_BALL`, dengan bawaan sama dengan `RFDETR_CONF`
supaya perilaku lama tidak berubah.

### Nomor 2, BELUM DIMULAI
Deteksi ulang pada potongan di sekitar ring. Sekarang TIDAK ADA jalur ROI sama
sekali, detektor selalu melihat frame utuh yang diperkecil. Kalau potongan
sekitar 320x320 di sekitar ring dijalankan pada resolusi asli, bola kembali
sekitar 21 px bukan 9,5 px. Tidak butuh model baru.

Digabung ke tugas ini, dua sinyal yang sudah terdeteksi tapi dibuang:
- **Kelas `rims` RF-DETR NOL kali dibaca mesin kejadian.** Rim itu bidang skor
  sebenarnya, kotaknya 38 px lebar. Tepi kiri dan kanannya bisa dipakai untuk
  uji lintasan yang lebih tajam daripada lingkaran radius 45 px di sekitar titik
  tengah ring.
- Uji "datang dari atas" sekarang memakai jendela mendatar `radius x 1,5`.
  Dengan rim, jendelanya bisa dari kotak rim yang sebenarnya.

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

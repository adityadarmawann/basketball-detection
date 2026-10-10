# PROGRES UJI AMBANG BOLA RF-DETR

Ditulis 10 Oktober 2026 pukul 21.15, sesi dihentikan di tengah jalan karena
koneksi akan terputus. Berkas ini supaya pekerjaannya bisa dilanjutkan tanpa
mengulang apa pun.

Video uji:
`/home/paul/computer-vision/campus-league/basketball/smart-vision-cl/test-video/Q1-UNESA-UBAYA-CUT-2-17.mp4`

---

## 1. PERTANYAAN YANG SEDANG DIJAWAB

Menurunkan ambang keyakinan bola menambah cakupan bola, tapi apakah ia
menambah bola masuk PALSU lebih banyak daripada yang asli?

Kriteria putusan yang dipakai: ambang lebih rendah DITERIMA hanya kalau jumlah
bola masuk asli yang ditemukan NAIK dan jumlah palsu tidak naik lebih banyak
dari kenaikan yang asli.

---

## 2. TEMUAN PERTAMA, SEBELUM ANGKA APA PUN: AMBANG TIDAK BISA DIPISAH PER KELAS

Dibaca di `pipeline/detector_rfdetr.py`. `model.predict()` hanya menerima SATU
argumen `threshold` untuk SEMUA kelas. Jadi menurunkan `RFDETR_CONF` demi bola
ikut melonggarkan pemain, hakim, nomor, dan ring sekaligus.

Karena itu dibuat saklar baru `RFDETR_CONF_BALL`. Sudah SELESAI dan sudah
dicommit, lihat bagian 5.

---

## 3. TEMUAN KEDUA, PENTING: DUA ASUMSI DI TUGAS TERBANTAH ANGKA

### 3a. Bawaan `RFDETR_CONF` yang BERLAKU bukan 0.35, tapi 0.25

Diukur langsung, bukan dibaca dari kode:

    /home/paul/miniconda3/envs/cl-gpu/bin/python -c "..." -> RFDetrDetector, conf 0.25

Sebabnya `.env` TIDAK memuat `RFDETR_CONF` sama sekali (sudah diperiksa dengan
grep). Lalu di `create_rfdetr_detector()` ada cabang:

    if os.getenv("RFDETR_CONF") is not None: conf = CONF       # 0.35
    elif conf_threshold is not None:         conf = conf_threshold

Karena env-nya kosong, cabang pertama TIDAK diambil, dan yang dipakai adalah
`conf_threshold` dari pemanggil, yaitu `DETECTOR_CONF` bawaan **0.25**
(`pipeline/detector.py` baris 240). Jadi "bawaan sekarang 0.35" itu keliru:
bawaan yang sungguh berlaku 0.25. Angka 0.35 hanya berlaku kalau
`RFDETR_CONF=0.35` disebut eksplisit di env.

Semua run di bawah MENYEBUT `RFDETR_CONF=0.35` eksplisit, jadi angkanya
sebanding satu sama lain.

### 3b. Videonya 601 detik, bukan 4 menit, dan `TRUTH` hanya menutupi 2,5 menit pertama

    ffprobe -> duration=601.301

`17_ukur_kejadian.py` TIDAK punya batas jendela, jadi yang diproses seluruh
10 menit video. Itu sebabnya satu run butuh **64,9 menit**, bukan 16 menit.
Bukan kerusakan, bukan juga akibat saklar baru: run 0.35 memakai ambang bola
0.35 yang sama dengan ambang umum, jadi `predict` jalan di 0.35 persis.

Akibat yang LEBIH SERIUS untuk cara ukur: `TRUTH` di skrip 17 berhenti di
detik 146 (2:26), sedangkan pipeline terus berjalan sampai 10:00. Setiap
`MADE_FG` setelah 2:26 otomatis dihitung PALSU padahal kebenarannya di situ
TIDAK DIKETAHUI. Pada run 0.35, 4 dari 5 palsu ada di 4:52, 7:35, 9:17, dan
9:45, yaitu seluruhnya di wilayah tanpa kebenaran.

Konsekuensinya acuan yang dikutip di tugas (2 dari 5 asli, 1 palsu, 9 poin)
**TIDAK TEREPRODUKSI** pada setelan yang identik. Acuan itu hampir pasti
diukur pada masukan yang lebih pendek. Jadi **jangan bandingkan angka palsu
run mana pun di bawah dengan angka acuan itu**; bandingkan hanya antar run di
berkas ini, karena wilayah dan cara penilaiannya sama.

---

## 4. RUN

### 4a. SUDAH SELESAI: ambang bola 0.35 (acuan diukur ulang sendiri)

Perintah persis:

    cd /home/paul/computer-vision/campus-league/basketball/smart-vision-cl/basketball-dev/backend
    DETECTOR_METHOD=rfdetr RFDETR_CONF=0.35 RFDETR_CONF_BALL=0.35 \
      VIDEO=/home/paul/computer-vision/campus-league/basketball/smart-vision-cl/test-video/Q1-UNESA-UBAYA-CUT-2-17.mp4 \
      RENDER_OUTPUT_VIDEO=0 \
      /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/17_ukur_kejadian.py

Hasil, 6 kejadian `MADE_FG`, 64,9 menit:

       waktu     nilai   jarak ring   putusan
       1:58.0   2 angka     2,4 m     cocok 1:59 (beda 0 s), nilai BENAR
       2:57.0   2 angka     3,9 m     PALSU
       4:52.0   3 angka     9,8 m     PALSU
       7:35.0   2 angka     1,2 m     PALSU
       9:17.0   2 angka     2,7 m     PALSU
       9:45.0   2 angka     4,7 m     PALSU

    asli ditemukan 1/5   nilai 2/3 benar 1   palsu 5
    terlewat 4 (0:19, 0:59, 1:30, 2:26)
    poin sistem 13 lawan KEBENARAN 11
    skor mesin {'team_a': 11, 'team_b': 2}

Ingat bagian 3b: 4 dari 5 palsu itu di luar jangkauan `TRUTH`.

### 4b. DILEWATI SENGAJA: ambang bola 0.20

Dihentikan di menit pertama atas keputusan penghemat waktu, karena satu run
ternyata 65 menit, bukan 16 menit. Titik data tengah tidak dipakai. Tidak ada
hasil, jangan mencari angkanya.

### 4c. SEDANG JALAN: ambang bola 0.10

Dilepas dengan `nohup` supaya tetap hidup kalau sesi mati. Perintah persis:

    cd /home/paul/computer-vision/campus-league/basketball/smart-vision-cl/basketball-dev/backend
    nohup env DETECTOR_METHOD=rfdetr RFDETR_CONF=0.35 RFDETR_CONF_BALL=0.10 \
      RENDER_OUTPUT_VIDEO=0 \
      VIDEO=/home/paul/computer-vision/campus-league/basketball/smart-vision-cl/test-video/Q1-UNESA-UBAYA-CUT-2-17.mp4 \
      /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/17_ukur_kejadian.py \
      > /home/paul/hasil_uji_ambang_ball010.log 2>&1 < /dev/null &

**HASILNYA AKAN MUNCUL DI `/home/paul/hasil_uji_ambang_ball010.log`.**
Mulai jalan sekitar pukul 21.12, perkiraan selesai sekitar pukul 22.17.
Yang dibaca: blok terakhir log, baris `asli ditemukan`, `palsu`, `poin sistem`.

Cara memeriksa apakah masih hidup:

    ps -eo pid,etime,cmd | grep 17_ukur_kejadian | grep -v grep
    tail -30 /home/paul/hasil_uji_ambang_ball010.log

---

## 5. PERUBAHAN KODE

### SELESAI dan SUDAH DICOMMIT: `RFDETR_CONF_BALL`

`pipeline/detector_rfdetr.py`, commit `acf9102`. Empat sentuhan:

1. konstanta modul `CONF_BALL = float(os.getenv("RFDETR_CONF_BALL", "0"))`
2. `__init__` menerima `conf_ball=None`; kalau `CONF_BALL <= 0` nilainya
   mengikuti `conf_threshold`, jadi tidak mengisinya berarti perilaku lama
   persis seperti sebelumnya
3. `predict(threshold=min(self.conf_threshold, self.conf_ball))`
4. di dalam perulangan kotak, penyaringan per kelas: kelas `ball` memakai
   `self.conf_ball`, semua kelas lain tetap `self.conf_threshold`

Kenapa prediksi di ambang terendah lalu disaring lagi itu SAHIH PERSIS, bukan
perkiraan: RF-DETR tidak memakai NMS, ia memprediksi himpunan lewat pencocokan
bipartit, jadi kotak yang keluar tidak berubah karena ambang. Ambang hanya
memotong daftar di akhir.

Pemeriksaan sinkron yang sudah dilakukan (bukan cuma uji sintaks):

    {}                                  -> conf 0.25  conf_ball 0.25  predict 0.25
    RFDETR_CONF=0.35                    -> conf 0.35  conf_ball 0.35  predict 0.35
    RFDETR_CONF=0.35 CONF_BALL=0.20     -> conf 0.35  conf_ball 0.20  predict 0.20
    RFDETR_CONF=0.35 CONF_BALL=0.10     -> conf 0.35  conf_ball 0.10  predict 0.10
    RFDETR_CONF_BALL=0.10 saja          -> conf 0.25  conf_ball 0.10  predict 0.10

Jalur hilir juga diperiksa: `pipeline/event_engine.py` tidak punya gerbang
keyakinan bola sendiri, dan `PelacakBola` di `pipeline/tracker.py` juga tidak,
jadi deteksi bola tambahan benar benar sampai ke gerbang `MADE_FG`. Alurnya
tidak terputus.

### TIDAK ADA perubahan separuh jadi

Tidak ada berkas yang ditinggal rusak. `.env` TIDAK disentuh, karena belum ada
putusan bawaan baru. Pohon kerja bersih selain berkas yang dicommit.

### BARU, BELUM PERNAH DIJALANKAN: `eksperimen/19_banding_bola_detektor.py`

Skrip untuk Tugas B (apakah RF-DETR dan YOLO saling melengkapi di kelas bola).
Sudah ditulis penuh dan lolos uji sintaks, tapi **BELUM PERNAH DIJALANKAN
sekali pun**, jadi angkanya belum ada dan kebenaran jalannya belum terbukti.
Dicommit karena berkas baru yang berdiri sendiri, tidak diimpor modul mana
pun, jadi tidak mungkin merusak alur pipeline.

Rancangannya: dua tahap, satu proses GPU per detektor, lalu disilangkan dari
berkas JSON tanpa GPU. Dipisah begini karena kartunya 8 GB dan memuat RF-DETR
plus YOLO sekaligus sudah pernah membuat kehabisan memori pada percobaan SAM2.

---

## 6. YANG BELUM DIKERJAKAN

1. **Baca hasil run 0.10** dari `/home/paul/hasil_uji_ambang_ball010.log`, lalu
   bandingkan dengan tabel 4a memakai kriteria putusan di bagian 1. Kalau
   palsu naik lebih cepat daripada yang asli, TOLAK dan catat ke
   `eksperimen/PENDEKATAN_DITOLAK.md`.

2. **Tugas B belum dimulai sama sekali.** Jalankan BERURUTAN, jangan dua proses
   GPU sekaligus. Jendela yang dipilih detik 5 sampai 220 supaya menutupi
   kelima waktu kebenaran menurut penafsiran apa pun (lihat bagian 3b):

       MODE=rekam DETECTOR_METHOD=rfdetr RFDETR_CONF=0.35 RFDETR_CONF_BALL=0.10 \
         MULAI=5 DETIK=215 VIDEO=<video uji> \
         /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/19_banding_bola_detektor.py

       MODE=rekam DETECTOR_METHOD=yolo DETECTOR_CONF=0.10 \
         MULAI=5 DETIK=215 VIDEO=<video uji> \
         /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/19_banding_bola_detektor.py

       MODE=gabung /home/paul/miniconda3/envs/cl-gpu/bin/python eksperimen/19_banding_bola_detektor.py

   Rekamannya ditaruh di `/tmp/rekam_bola/`, hapus setelah dipakai.
   Detektor saja, jadi cepat, perkiraan beberapa menit per detektor.

3. **Perbaiki cara ukur skrip 17**, ini penghambat yang sebenarnya. Selama
   `TRUTH` cuma menutupi 2,5 menit sedangkan pipeline memproses 10 menit,
   angka palsu TIDAK BISA dipercaya. Dua pilihan: perpanjang `TRUTH` sampai
   akhir video, atau batasi pipeline pada jendela yang ada kebenarannya. Tanpa
   ini, sapuan ambang mana pun menilai 75 persen videonya dengan kebenaran
   yang tidak ada.

4. Tambahkan putusan akhir Tugas A dan Tugas B ke
   `eksperimen/PENDEKATAN_DITOLAK.md` lengkap dengan angka, sebab teknis,
   skrip pengukurnya, dan syarat "boleh dicoba lagi kalau".

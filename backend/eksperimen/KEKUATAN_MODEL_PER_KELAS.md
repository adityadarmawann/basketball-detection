# KEKUATAN MODEL PER KELAS, DAN SIAPA PEMILIK TIAP KELAS

Berkas ini menjawab satu pertanyaan: **untuk tiap kelas, model mana yang paling
kuat, dan model mana yang kelasnya dimatikan.**

Prinsipnya: **satu kelas, satu pemilik.** Dua model tidak boleh menyuarakan
kelas yang sama, karena itu membuat satu benda terhitung dua kali. Satu satunya
kecuali adalah `ball`, yang boleh digabung, TAPI dengan aturan penggabungan
tegas supaya satu bola tidak jadi dua.

Dua model yang dibandingkan:
- **YOLO lama**: `models/best-object-basketball.pt`, 5 kelas
  `ball, player, hoop, referee, backboard`
- **RF-DETR**: `models/best-object-basket-detect-v1-DETR.pt`, 10 kelas, memisah
  `rim` dari `hoop`, menambah `ball-in-basket`, `number`,
  `player-in-possession` dan turunan pemain lain, TIDAK punya `backboard`

---

## PERINGATAN CARA UKUR, BACA SEBELUM MEMAKAI ANGKA DI BAWAH

**1. Label di `test-video/object/labels/` TIDAK BOLEH dipakai menilai YOLO.**
1500 frame campus league itu berlabel, tapi isinya **PRA-LABEL buatan YOLO lama
yang menunggu dikoreksi manusia** (lihat `test-video/object/README.md`). Menilai
YOLO dengan label buatan YOLO sendiri itu berputar, dan YOLO otomatis menang.

**2. Dataset `dataset/02-objek-basketball` juga berat sebelah.** YOLO lama
dilatih di split `train` dataset itu, jadi ujian di split mana pun masih sedomain
dengan latihannya, sementara RF-DETR dilatih di dataset lain. Selisih yang muncul
akan bercampur antara "lebih mampu" dan "lebih kenal datanya".

**3. Angka dari video berbeda TIDAK SEBANDING.** `q1-UBAYA-UNESA.mp4` 25 fps,
`Q1-UNESA-UBAYA-CUT-2-17.mp4` 60 fps. Jumlah frame, ukuran objek, dan peluang
deteksi ikut berubah. Selalu tulis videonya.

**4. Cakupan BUKAN presisi.** "Model A menemukan lebih banyak" tidak berarti
lebih baik; bisa jadi lebih banyak yang palsu. Setiap klaim "lebih kuat" harus
menyebut KEDUANYA, atau mengaku kalau cuma salah satu yang terukur.

### Kebenaran yang netral: belum ada
Yang dibutuhkan adalah frame campus league berlabel MANUSIA yang tidak dipakai
melatih kedua model. Dua jalan:
- **Cara pengganti tanpa label**: kekonsistenan lintasan untuk bola, kestabilan
  kotak untuk ring, dan pola kesepakatan antar model. Murah, hasilnya perkiraan.
- **Koreksi manual sekitar 200 dari 1500 frame pra-label itu.** Mengoreksi jauh
  lebih cepat daripada melabeli dari nol, hasilnya kebenaran sungguhan.

---

## TABEL PEMILIK KELAS

Status: `TERUKUR` = ada angka dari jalan nyata. `SEBAGIAN` = baru satu sisi
(cakupan saja atau presisi saja). `BELUM` = belum diukur, jangan diputuskan.

| kelas | pemilik sekarang | status | dasar |
|---|---|---|---|
| `ball` | RF-DETR | SEBAGIAN | cakupan terukur, presisi belum, perbandingan ke YOLO sedang diukur |
| `hoop` + `rim` | RF-DETR | SEBAGIAN | kotak dan getaran terukur, cakupan lawan YOLO belum |
| `player` | RF-DETR | SEBAGIAN | cakupan jauh lebih tinggi, presisi belum |
| `referee` | RF-DETR | BELUM | belum pernah dibandingkan langsung |
| `backboard` | TIDAK ADA | TERUKUR | buta di dua duanya, lihat di bawah |
| `ball-in-basket` | TIDAK DIPAKAI | TERUKUR | nol deteksi, lihat di bawah |
| `number` | RF-DETR | BELUM | jalur nomor terpisah lewat `jersey_ocr` |

---

## RINCIAN PER KELAS

### `ball`
**RF-DETR, terukur 10 Oktober 2026, skrip 18**, `Q1-UNESA-UBAYA-CUT-2-17.mp4`
detik 72 sampai 88 (480 frame, memuat satu bola masuk nyata):

    ambang 0,25 (BAWAAN PRODUKSI) : terdeteksi 52,1 persen, kotak median 21,0 px
    ambang 0,10                   : terdeteksi 71,5 persen, kotak median 20,0 px

KOREKSI 10 Oktober 2026: dua baris di atas sempat saya labeli "ambang 0.35".
Itu SALAH. `RFDETR_CONF` hanya menang kalau diisi tegas (lihat
`create_rfdetr_detector`, baris `if os.getenv("RFDETR_CONF") is not None`).
Kalau tidak diisi, yang berlaku `DETECTOR_CONF` dengan bawaan **0,25**. Jadi
pengukuran tanpa mengisi RFDETR_CONF itu berjalan di 0,25, bukan 0,35.
Kesimpulannya tidak berubah (ambang lebih rendah menambah cakupan bola), tapi
angka ambangnya harus ditulis benar supaya run berikutnya sebanding.

Sepanjang 4 menit penuh, cakupan mentahnya 72,8 persen (skrip 16). Jadi di
jendela dekat bola masuk cakupannya JUSTRU LEBIH BURUK daripada rata rata.

Temuan penting soal ukuran: bola selebar 21 px di frame 1280, menyusut jadi
**9,5 px** setelah frame diperkecil ke 576 untuk masuk detektor. Itu sebab
utama bola hilang tepat saat menentukan.

**YOLO lama**: catatan `test-video/object/README.md` mencatat recall 46 persen
dengan keyakinan 0,75, diukur pada 56 frame sampel `q1-UBAYA-UNESA.mp4` (25 fps,
VIDEO BERBEDA, jadi tidak sebanding langsung).

Perbandingan langsung RF-DETR lawan YOLO di video dan jendela yang sama, termasuk
berapa frame yang hanya ketemu salah satu, SEDANG DIUKUR.

### `hoop` dan `rim`
**RF-DETR, terukur 10 Oktober 2026, skrip 18**, jendela sama:

    getaran titik tengah ring antar frame : median 1,7 px   p90 10,8 px
    lebar kotak ring                      : median 38,0 px  p90 40,2 px

Getaran 1,7 px itu kecil terhadap gerbang 45 px, jadi posisi ring TIDAK perlu
dihaluskan (sudah dicatat di `PENDEKATAN_DITOLAK.md`).

**Yang belum dipakai:** RF-DETR memisah `rim` dari `hoop`, tapi kode memasukkan
rim ke daftar `hoops` dan daftar `rims` terpisah itu NOL kali dibaca mesin
kejadian. Rim adalah bidang skor yang sebenarnya, dan kotaknya 38 px lebar, jadi
tepi kiri dan kanannya bisa dipakai untuk uji lintasan yang lebih tajam daripada
lingkaran radius 45 px di sekitar titik tengah.

**YOLO lama**: README mencatat recall 64 persen, sering terlewat (56 frame,
video berbeda).

### `player`
**RF-DETR 8,33 pemain per frame lawan YOLO 5,58** (terukur 9 Oktober 2026).
Ini CAKUPAN, bukan presisi. Di lapangan ada 10 pemain, jadi dua duanya masih
kurang, dan 8,33 lebih dekat ke kenyataan. Tapi tanpa label manusia belum bisa
dipastikan tambahan RF-DETR itu pemain sungguhan atau orang di luar lapangan.

RF-DETR punya 5 subkelas pemain, dan `detector_rfdetr._satukan_pemain()`
menyatukan kotak yang bertumpang tindih supaya satu orang tidak jadi beberapa
deteksi.

**YOLO lama**: README mencatat recall 84 persen dengan keyakinan 0,89 dan
catatan "bagus, jangan dirusak" (56 frame, video berbeda).

### `referee`
**BELUM pernah dibandingkan langsung.** README model lama mencatat recall 61
persen, dan masalah hulu yang penting: wasit yang tidak terdeteksi sebagai
`referee` bocor jadi `player`, lalu dapat tim dan **nomor palsu** dari papan
iklan. Jadi kelas ini berpengaruh ke mutu statistik, bukan cuma hitungan orang.

### `backboard`
**BUTA DI DUA DUANYA, jangan dipakai.**
- RF-DETR: tidak punya kelas ini sama sekali. `detector_rfdetr._empty_result()`
  mengembalikan `"backboards": []` dan diberi komentar selalu kosong.
- YOLO lama: README mencatat **recall 0 persen, tidak pernah terdeteksi**, dan
  dataset koreksinya belum dikerjakan.

Akibatnya mesin kejadian yang membaca `backboards` untuk zona rebound praktis
tidak pernah mendapat masukan itu. Saya sempat mengusulkan menambal backboard
dari YOLO; usulan itu SALAH dan dicabut, karena YOLO juga buta.
Perlu dipastikan ulang dengan ukuran langsung di video uji sekarang.

### `ball-in-basket`
**TIDAK BISA DIPAKAI tanpa latih ulang. Terukur, skrip 18:**

    ambang 0,25 (bawaan produksi) : 0 deteksi dari 480 frame
    ambang 0,10                   : 0 deteksi dari 480 frame

Nol, padahal jendelanya memuat bola masuk nyata. Dugaan sebabnya masuk akal:
begitu bola berada di dalam jaring, rupanya samar dan tertutup, jadi contoh
latihnya memang sulit dan jarang.

**Ini tidak merugikan skor,** karena gerbang bola masuk memang TIDAK pernah
bergantung pada kelas ini. Gerbangnya murni geometri bola dan ring: masuk radius
ring, bertahan beberapa frame, terbukti datang dari ATAS rim dalam jendela
mendatar sempit, lalu dikonfirmasi di BAWAH rim oleh deteksi NYATA.

---

## CARA MENGISI TABEL INI

Setiap baris yang masih `BELUM` atau `SEBAGIAN` harus diisi dari jalan nyata,
bukan dugaan. Skrip yang tersedia:

    eksperimen/18_ukur_bukti_ring.py        bukti di sekitar ring, detektor saja
    eksperimen/19_banding_bola_detektor.py  banding bola antar detektor
    eksperimen/16_ukur_pelacakan.py         cakupan bola sepanjang video
    eksperimen/17_ukur_kejadian.py          akibatnya ke skor, pipeline penuh

Saat menambahkan angka, WAJIB tulis: videonya, jendelanya, jumlah frame,
ambang keyakinannya, dan apakah yang diukur cakupan atau presisi. Tanpa itu
angkanya tidak bisa dibandingkan dengan baris lain.

Penafsir Python yang benar: `/home/paul/miniconda3/envs/cl-gpu/bin/python`.
Yang lain tidak punya paket `rfdetr` dan akan MUNDUR DIAM DIAM ke YOLO,
sehingga hasilnya tertukar tanpa peringatan.

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
| `ball` | **RF-DETR** | **TERUKUR** | RF-DETR 1,8x lebih baik dari YOLO di tiap ambang; gabungan cuma +3,5 titik, ditolak. Potongan resolusi asli di sekitar rim menaikkan keyakinan 0,820 jadi 0,869 dan asli ditemukan 2/5 jadi 3/5 |
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


### KOREKSI 10 Oktober 2026: `ball` SUDAH TERUKUR, pemiliknya RF-DETR

Diukur pada potongan 240 detik, jendela 143 detik yang memuat kelima bola masuk,
4.290 frame, skrip 19:

    ambang    RF-DETR    YOLO lama
      0,10     83,6 %      46,0 %
      0,20     72,8 %      41,5 %
      0,25     69,4 %      39,3 %
      0,35     64,0 %      36,3 %
      0,50     58,1 %      32,0 %

RF-DETR menang di SETIAP ambang dengan selisih sekitar 1,8 kali. Keyakinan
mediannya juga lebih tinggi, 0,841 lawan 0,752. Kotak YOLO sedikit lebih besar
(25,0 px lawan 21,5 px), tapi itu bukan keunggulan, cuma gaya kotaknya.

Gabungan dua detektor DITOLAK: YOLO hanya menambah 3,5 titik persen (149 frame
dari 4.290) dengan harga menjalankan model kedua di setiap frame. Rinciannya di
`PENDEKATAN_DITOLAK.md`.

**Yang lebih penting dari semua angka cakupan di atas:** mengejar cakupan bola
TERBUKTI SALAH SASARAN. Menaikkan cakupan 69,4 jadi 83,6 persen justru
MENURUNKAN skor dari 9 jadi 7 poin, karena sepertiga deteksi tambahan menaruh
bola di posisi salah. Yang dibutuhkan POSISI BENAR pada saat menentukan.

### `ball` pada POTONGAN RESOLUSI ASLI, bukan frame utuh
**Terukur 10 Oktober 2026, skrip 21**, `q1_4menit.mp4` detik 72 sampai 215,
4.290 frame, potongan 448 px di sekitar rim, ambang bola 0,25.

Ini BUKAN model lain, ini model yang SAMA diberi gambar yang lebih besar.
Frame utuh 1280x720 diperkecil ke 576x576 (skala x 0,45, y 0,80) sehingga bola
21 px terlihat 9,5 px. Potongan 448 diperbesar ke 576 sehingga bola yang sama
terlihat 27 px.

    saat KEDUANYA menemukan bola (n=813):
      keyakinan median   frame penuh 0,820   potongan 0,869
      potongan lebih yakin pada 77,9 persen frame
      jarak pusat        median 1,5 px   p90 10,1 px
      sepakat dalam 10 px   89,9 persen

    cakupan:
      frame penuh menemukan bola   75,0 %
      potongan menemukan bola      23,1 %   (hanya 61,1 % frame punya rim)
      HANYA potongan                4,2 %   = 179 frame, satu satunya tambahan

Jadi memperbesar bola MEMANG menaikkan keyakinan, terukur. Tapi di frame yang
frame penuh sudah berhasil, posisinya praktis sama (1,5 px lawan gerbang 45
px), jadi sumbangannya ada di 179 frame yang frame penuh KOSONG.

Akibatnya ke skor, skrip 17: asli ditemukan 2/5 jadi 3/5, palsu 2 jadi 3,
waktu +5 persen. DITERIMA, lihat `PENDEKATAN_DITOLAK.md` bagian cara ukur.

Biaya: satu predict potongan 12,0 ms, satu detect frame penuh 16,4 ms. Kalau
potongan dijalankan di SETIAP frame, waktu detektor +73 persen. Dengan gerbang
`HOOP_ROI_NEAR_PX=300` potongan jalan di 28,6 persen frame saja.

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

KOREKSI KECIL 10 Oktober 2026, skrip 21: pada POTONGAN 448 px resolusi asli
kelas ini menyala **2 kali** dari 2.623 potongan, jadi bukan nol mutlak,
melainkan nol karena objeknya terlalu kecil di frame yang diperkecil. Dua
deteksi masih jauh dari cukup untuk jadi penentu, jadi kesimpulan "butuh latih
ulang" TIDAK berubah. Tapi arah perbaikannya sekarang punya dua kemungkinan,
bukan satu: menambah contoh latih, ATAU melatihnya pada potongan dekat ring.
Hitungannya ada di `roi_stat["masuk_jaring"]`; tidak digabung ke keluaran.

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

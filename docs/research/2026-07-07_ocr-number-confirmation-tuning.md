# Tuning Konfirmasi Nomor Punggung (OCR Voting) — Eksperimen Ground-Truth

**Tanggal:** 2026-07-07
**Branch:** `dev` (worktree `basketball-dev`) — demo/produksi tidak terpengaruh
**Commit hasil:** `4a13709` — *ocr: relax confidence-upgrade flush 0.60/0.25 → 0.50/0.10*
**File yang diubah:** `backend/pipeline/jersey_ocr.py`

---

## TL;DR

Pertanyaan awal: *"meskipun ada cache, kalau OCR membaca nomor baru dengan akurasi lebih tinggi, apakah bisa langsung update?"* — lalu: *"buat agresif dan cek ground-truth."*

Hasil, diukur dari **kode `process()` asli** (bukan replika) atas 1500 frame padat dengan 19 track berlabel tangan:

- **"Agresif" ternyata dua tuas dengan efek berlawanan.**
  - **Relax confidence-upgrade flush** (`CONF_UPGRADE_MIN` 0.60→0.50, `CONF_UPGRADE_DELTA` 0.25→0.10): **akurasi naik** (mid-track 57.7%→61.2%, final 57.9%→63.2%), flip-rate nyaris tak berubah. ✅ **Diterapkan.**
  - **Turunkan `DISAGREE_N` 3→2**: **merusak** — *thrashing*, akurasi turun ke 52.9%, flip +47%. ❌ **Ditolak.**
- **Plafon sistem dibatasi MODEL digit, bukan voting.** ~40% error berasal dari model yang salah membaca angka aslinya (mis. #16→#8, #72→#12). Tidak ada tuning voting yang bisa mengonfirmasi angka yang tak pernah dibaca model.

---

## 1. Latar Belakang

### 1.1 Sistem yang diuji
`JerseyOCR.process()` mengonfirmasi nomor punggung per-track lewat **weighted voting** dengan beberapa lapisan *cache*/*flush*:

- **Voting**: tiap bacaan digit-model `(candidate, confidence)` masuk deque per-track (`maxlen=MAX_VOTE_HISTORY=20`), berbobot `confidence × blur_quality`. Nomor **confirmed** = kandidat dengan bobot tertinggi, syarat `count ≥ VOTE_THRESHOLD(2)` dan `weight ≥ MIN_CONFIRM_WEIGHT(0.6)`.
- **Transfer 1-digit→2-digit**: bacaan "1" bisa dipindah ke "10"/"11" bila kandidat 2-digit punya `≥ TRANSFER_MIN_LONG_VOTES(3)` bacaan (menangani digit terpotong saat kamera pan).
- **Dua mekanisme flush** (inti eksperimen ini):
  1. **Confidence-upgrade flush** — bacaan dengan `conf ≥ CONF_UPGRADE_MIN` yang mengalahkan best sebelumnya `> CONF_UPGRADE_DELTA` **dan** beda dari confirmed → buang deque, nomor baru ambil alih cepat.
  2. **Stale-lock / disagreement flush** — bila `DISAGREE_N` bacaan berturut sepakat pada angka ≠ confirmed → buang deque.
- **Roster whitelist**: kandidat di luar roster tim ditolak sebelum voting.

### 1.2 Nilai baseline
```
VOTE_THRESHOLD        = 2       MIN_CONFIRM_WEIGHT      = 0.6
MAX_VOTE_HISTORY      = 20      TRANSFER_MIN_LONG_VOTES = 3
CONF_UPGRADE_MIN      = 0.60    CONF_UPGRADE_DELTA      = 0.25
DISAGREE_WINDOW       = 5       DISAGREE_N              = 3
```

---

## 2. Metodologi ("bukan feeling")

### 2.1 Prinsip: pakai kode ASLI, bukan replika
Upaya pertama adalah mereplikasi logika voting di harness terpisah lalu men-sweep parameter di atasnya (cepat). **Ditinggalkan** setelah validasi: replika meleset **27/4548 (0.6%)** dari output asli.

- 26 dari 27 adalah *downstream uniqueness-suppression* (dedup identitas `(team, number)` lintas-track) — ortogonal terhadap tuning.
- Sisanya adalah **jebakan sesungguhnya**: logika **TIM** dapat mem-flush vote **NOMOR**. Saat klasifikasi tim mendeteksi pergantian pemain di bawah satu track (`_split_track_identity`, dipicu ketidaksepakatan warna berturut), ia memanggil `self._votes.pop(track_id)` — meng-couple dua subsistem yang tampak terpisah.

**Kesimpulan metodologis:** hanya menjalankan `process()` penuh yang menangkap semua coupling. Semua angka di bawah berasal dari `process()` asli (TensorRT engine sungguhan), bukan model tiruan.

### 2.2 Determinisme — diverifikasi
Karena membandingkan antar-setting, run harus deterministik. Baseline dijalankan **2×** atas window & input identik:

```
DETERMINISM: baseline vs dup  len 4548/4548  confirmed-diffs = 0   ✅
```

TensorRT digit-model deterministik → seluruh perbedaan antar-setting murni dari parameter.

### 2.3 Dataset
- **Video:** kuarter-1 UNESA vs UBAYA, 1280×720, 60fps, 351.8 s (21108 raw frame; diproses ~30fps → 10550 frame).
- **Window:** processed frame **500–2000** (1500 frame; raw 1008–4006) — dipilih dari blok terpadat (7+ pemain/frame, 6–7 confirmed-read/frame).
- **Bbox & track-id:** diambil dari `frames.json` output pipeline (`i`=track_id, `b`=bbox) → `process()` dijalankan ulang atas frame nyata.
- **Roster (dari MongoDB, tepat):**
  - Tim A (UNESA/putih): `4,5,6,7,8,9,10,11,12,13,14,16,18`
  - Tim B (UBAYA/merah): `0,1,2,3,4,5,7,8,9,10,11,12,15,17,18,20,23,32,72,88`
  - Nomor dipakai kedua tim (shared): `4,5,7,8,9,10,11,12,18` (9 dari 24 unik).
- **Statistik bacaan (baseline):** 4548 reads, **64% candidate≠None** (2910), `read_conf` median 0.37 / max 0.86; 201 track, 109 pernah confirmed.

### 2.4 Ground-truth
Sistem **tidak boleh** menilai dirinya sendiri. Aku ekspor crop ber-confidence tertinggi tiap track, lihat langsung gambarnya, dan **melabeli 19 track secara manual**:

| Nomor | Track (label tangan) |
|---|---|
| #8 (putih) | 330, 521, 793, 430, 541, 729, 511 |
| #7 (putih) | 176, 783 |
| #14 (putih) | 378 |
| **#16 (putih)** | **620, 307** — model sering baca #8 |
| #88 (merah) | 274, 633 |
| #23 (merah) | 247, 631, 786 |
| **#72 (merah)** | **305, 784** — model sering baca #12 |

Empat track tebal adalah **kasus emas**: model salah membaca angka aslinya → menguji apakah agresivitas memperbaiki atau memperburuk.

### 2.5 Definisi metrik
| Metrik | Arti |
|---|---|
| **frameAcc** | proporsi frame ber-confirmed di mana `confirmed == truth` (rata-rata atas track berlabel) — akurasi *sepanjang* track |
| **finalAcc** | apakah confirmed terakhir track == truth |
| **latency** | median jumlah bacaan dari awal track hingga pertama kali `confirmed == truth` |
| **adoptLat** | median bacaan dari *bacaan benar pertama* (`cand==truth`, `conf≥0.4`) hingga `confirmed==truth` |
| **flips/lbl** | rata-rata jumlah pergantian nilai confirmed per track berlabel (stabilitas) |
| **gFlip** | flips rata-rata atas **seluruh 201 track** (stabilitas global) |
| **gCov** | proporsi frame ber-confirmed atas seluruh track (cakupan global) |

---

## 3. Hasil

### 3.1 Sweep luas — baseline vs berbagai agresif
```
setting        frameAcc finalAcc latency adoptLat flips/lbl  gFlip  gCov
S0_baseline       57.7%    57.9%       6      2.0      3.84   1.45   48%
S1_flush          60.7%    57.9%       8      2.0      4.11   1.58   48%   (CUM .50 / CUD .15)
S2_disagree       52.9%    52.6%       9      2.0      5.63   2.08   42%   (DISAGREE_N 2)
S3_both           53.1%    52.6%       8      2.0      5.79   2.11   41%
S4_strong         54.5%    52.6%       8      2.0      5.84   2.24   41%   (+MAX_VOTE_HISTORY 12)
```
- Semua varian yang menyertakan **`DISAGREE_N=2`** (S2/S3/S4) **turun** akurasi & **naik** flip drastis.
- Hanya **S1 (flush relax)** yang naik.

### 3.2 Sweep fokus — hanya relax flush (DISAGREE_N tetap 3)
```
setting        frameAcc finalAcc latency adoptLat flips/lbl  gFlip  gCov
S0_baseline       57.7%    57.9%       6      2.0      3.84   1.45   48%
SA_50_15          60.7%    57.9%       8      2.0      4.11   1.58   48%
SB_50_10          61.2%    63.2%       8      2.0      3.95   1.57   47%   ← PEMENANG
SC_45_12          60.7%    63.2%       8      2.0      4.05   1.55   48%
SD_50_15_mcw5     59.4%    63.2%       8      1.0      4.47   1.67   49%   (+MIN_CONFIRM_WEIGHT 0.5)
SE_50_15_vh14     56.2%    52.6%       8      2.0      5.00   1.80   48%   (+MAX_VOTE_HISTORY 14)
```
- **SB (`CUM 0.50` / `CUD 0.10`)**: frameAcc **+3.5pt**, finalAcc **+5.3pt**, flip nyaris tak naik (3.84→3.95), coverage sama.
- **SE (memori pendek)** & **SD (confirm lebih cepat)** tidak lebih baik → **jangan** perpendek memori atau turunkan ambang confirm.

### 3.3 Kasus emas (truth vs confirmed-final per setting; sweep luas)
```
t620 (→16): base=16  flush=16  | DISAGREE2=7   | model reads {16:11, 8:7, 7:6, 9:2}
t307 (→16): base=6   flush=6   | DISAGREE2=8   | model reads {8:13, 16:12, 7:10, 18:7}
t305 (→72): base=1   flush=1   | DISAGREE2=0   | model reads {12:27, 2:17, 1:15, 7:8}
t784 (→72): base=12  flush=12  | DISAGREE2=7   | model reads {12:24, 7:7, 2:6, 72:5}
t378 (→14): base=14  flush=14  | DISAGREE2=11  | model reads {7:11, 8:8, 14:7, 18:6}
t247 (→23): base=1   flush=1   | DISAGREE2=7   | model reads {2:22, 23:16, 1:8, 0:7}
```
Dua pelajaran dari kolom "model reads":
1. **`DISAGREE_N=2` merusak yang tadinya benar** (t620: 16→7).
2. **Voting tak bisa melebihi model.** t307 (model baca 8 lebih sering dari 16), t305/t784 (model nyaris tak pernah baca 72), t247 (model baca 2 dominan) — **tak ada setting** yang benar, karena angka benar memang jarang/tak pernah masuk sebagai bacaan.

---

## 4. Analisis

### 4.1 Kenapa flush-relax menang
Melonggarkan ambang confidence-upgrade (`0.60→0.50`, `0.25→0.10`) membuat bacaan yang **cukup yakin dan lebih baik dari sebelumnya** langsung membuang deque lama → nomor benar diadopsi lebih cepat *sepanjang* track (naik frameAcc), tanpa cukup longgar untuk menerima noise (flip stabil). Ini persis jawaban untuk pertanyaan awal: *"kalau ada bacaan lebih akurat, update cepat"* — ya, dan aman.

### 4.2 Kenapa `DISAGREE_N=2` thrashing
Override "N bacaan berturut sepakat" pada N=2 terlalu mudah dipicu: dua bacaan noise beruntun (umum saat kamera bergerak) cukup untuk membuang confirmed yang benar. Efeknya berantai — flip naik 47%, coverage turun, dan kasus yang sebelumnya benar ikut rusak. N=3 memberi bukti yang cukup sebelum menyerah pada perubahan.

### 4.3 Plafon = model digit
`adoptLat` sudah 2 bacaan di semua setting → sistem **sudah cepat** mengadopsi bacaan benar; agresivitas tak menambah kecepatan, hanya menambah risiko. Sisa error didominasi *garbage-in*: crop blur/kecil/terpotong membuat model salah baca. Tuas nyata di sana bukan parameter vote, melainkan **kualitas crop nomor** (resolusi/kontras/deteksi bbox digit) sebelum masuk model.

---

## 5. Keputusan yang diterapkan

```python
# backend/pipeline/jersey_ocr.py  (env-overridable, default = nilai pemenang)
CONF_UPGRADE_MIN   = float(os.getenv("CONF_UPGRADE_MIN",   "0.50"))  # was 0.60
CONF_UPGRADE_DELTA = float(os.getenv("CONF_UPGRADE_DELTA", "0.10"))  # was 0.25
# DISAGREE_N tetap 3 (menurunkan ke 2 terbukti merusak)
```
- Keduanya **env-overridable** → bisa dikembalikan tanpa edit kode (`CONF_UPGRADE_MIN=0.60 CONF_UPGRADE_DELTA=0.25`).
- Kontrak output tak berubah; hanya kecepatan adopsi bacaan-lebih-baik yang naik.

---

## 6. Ancaman terhadap validitas (jujur)

- **Sampel ground-truth kecil (19 track).** finalAcc 57.9%→63.2% ≈ selisih ~1 track; sinyal yang lebih kuat adalah **frameAcc** (+3.5pt, agregat ribuan frame). Perbedaan finalAcc harus dibaca hati-hati.
- **Satu window, satu pertandingan** (merah/putih). Generalisasi ke warna/kamera lain belum diuji — walau parameter yang diubah tak spesifik-warna.
- **Confidence sebagai proksi.** Label tangan mengurangi ini, tapi crop ambigu (2 pemain overlap) sengaja di-*skip*, bukan dinilai.
- **Uniqueness-suppression** (dedup lintas-track) diperlakukan sebagai lapisan hilir; metrik memakai *vote-level confirmed* (keyakinan sistem terhadap track), bukan output ter-dedup.

---

## 7. Reproduksibilitas

Harness (di scratchpad sesi, bersifat sementara):
1. `cap_ocr.py` — jalankan `create_jersey_ocr()` (memuat TensorRT `jersey_no.engine` + `best-detect-num-v2.engine`) atas window, tangkap stream `(frame, track, candidate, read_conf, blur_q, confirmed)` via hook attr-gated di `process()` **(hook sudah direvert; tidak ikut commit)**.
2. `sweep_real.py` — untuk tiap setting: patch konstanta modul, buat instance segar, jalankan `process()` asli ulang atas window yang sama; rekam confirmed per read.
3. `analyze.py` — hitung metrik atas 19 track berlabel + stabilitas global.

Parameter yang di-sweep: `CONF_UPGRADE_MIN`, `CONF_UPGRADE_DELTA`, `DISAGREE_N`, `MAX_VOTE_HISTORY`, `MIN_CONFIRM_WEIGHT`.

---

## 8. Kerja lanjutan

1. **Kualitas crop nomor** (dampak terbesar untuk kasus model-limited): upscale + CLAHE/kontras + bbox digit lebih ketat sebelum model. Ukur dengan harness yang sama (frameAcc pada track model-limited seperti #72/#16).
2. **Court-homography sebagai indikator cut kamera** → re-ID posisi-court untuk menjaga cache tracker-ID/tim lintas-cut, dan reset EMA homografi saat cut (lihat catatan terpisah). Ukur ID-switch per-cut sebelum/sesudah.
3. **Validasi silang** pada window & pertandingan lain (warna berbeda) sebelum mengangkat perubahan ke `main`.

---

*Metode inti yang bisa dipakai ulang: **jalankan kode produksi asli secara offline atas frame nyata, verifikasi determinisme, nilai terhadap label tangan (bukan output sistem), dan pisahkan batas-model dari batas-algoritma.** Pola ini berlaku untuk tuning subsistem apa pun di pipeline ini.*

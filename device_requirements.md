# Device Requirements — Smart Vision Basketball

Pipeline yang berjalan: **YOLOv8n** (deteksi) → **BoxMOT** (tracking) → **SlowFast-R50** (action) → **PaddleOCR** (jersey OCR) → **Pose Estimation** → Stats Calculator

---

## Tier 1 — Minimum (Demo / Video Pendek)

> Cocok untuk demo kampus, video hasil rekaman, bukan live.

| Komponen | Spesifikasi |
|---|---|
| **GPU** | NVIDIA GTX 1050 Ti 4 GB VRAM (CUDA 11+) |
| **CPU** | Intel Core i5 Gen 8 / AMD Ryzen 5 3600 (4 core / 8 thread) |
| **RAM** | 8 GB |
| **Storage** | 50 GB SSD (5 GB app + model, sisanya video cache) |
| **OS** | Windows 10/11 64-bit atau Ubuntu 20.04+ |
| **Python** | 3.10 – 3.11 (conda/venv) |

### Kemampuan di Tier 1

| | |
|---|---|
| Kecepatan pipeline penuh | **~3–4 FPS** |
| Video 30 FPS → processing | **8–9× lebih lambat dari real-time** |
| Mode tanpa SlowFast | ~12–15 FPS |
| **Maksimal durasi video** | **3–5 menit** (agar selesai dalam ~30–45 menit) |
| **Ukuran file video** | Maks **500 MB** (720p H.264) |
| Live / real-time | ❌ Tidak bisa |

> **Contoh estimasi:** Video 5 menit (30 FPS) = 9.000 frame ÷ 3.5 FPS ≈ **43 menit processing**.

---

## Tier 2 — Recommended (Video Pertandingan 1 Kuarter)

> Proses lebih cepat, bisa handle 1 kuarter penuh (~10 menit).

| Komponen | Spesifikasi |
|---|---|
| **GPU** | NVIDIA GTX 1660 Ti 6 GB / RTX 2060 6 GB VRAM |
| **CPU** | Intel Core i7 Gen 10 / AMD Ryzen 7 5800X (8 core / 16 thread) |
| **RAM** | 16 GB |
| **Storage** | 100 GB SSD |
| **OS** | Windows 10/11 64-bit atau Ubuntu 20.04+ |

### Kemampuan di Tier 2

| | |
|---|---|
| Kecepatan pipeline penuh | **~10–15 FPS** |
| Rasio terhadap real-time | ~2–3× lebih lambat |
| **Maksimal durasi video** | **10–15 menit** (selesai dalam ~25–45 menit) |
| **Ukuran file video** | Maks **1.5 GB** (1080p H.264) |
| Live / real-time | ❌ Belum (bisa dengan mode ringan tanpa SlowFast) |

> **Contoh estimasi:** Video 10 menit (30 FPS) = 18.000 frame ÷ 12 FPS ≈ **25 menit processing**.

---

## Tier 3 — Optimal (Full Game + Real-Time Live)

> Bisa proses pertandingan penuh dan live feed kamera lapangan.

| Komponen | Spesifikasi |
|---|---|
| **GPU** | NVIDIA RTX 3080 10 GB / RTX 4070 12 GB VRAM atau lebih tinggi |
| **CPU** | Intel Core i9 Gen 12+ / AMD Ryzen 9 5900X (12+ core) |
| **RAM** | 32 GB |
| **Storage** | 200 GB NVMe SSD |
| **OS** | Ubuntu 22.04 LTS (server) atau Windows 11 |
| **Kamera (live)** | USB/IP kamera 1080p 30 FPS, latency < 100 ms |

### Kemampuan di Tier 3

| | |
|---|---|
| Kecepatan pipeline penuh | **25–35 FPS** |
| **Live / real-time** | ✅ Ya (30 FPS kamera) |
| **Maksimal durasi video** | Full game **40 menit** (selesai dalam ~50–60 menit) |
| **Ukuran file video** | Hingga **5 GB** (1080p 60 FPS) |
| Jumlah kamera live | 1 kamera (multi-kamera butuh beberapa GPU) |

---

## VRAM Breakdown (Pipeline Penuh)

| Model | VRAM |
|---|---|
| YOLOv8n (deteksi pemain + bola) | ~1.0 GB |
| SlowFast-R50 (action recognition) | ~3.5 GB |
| PaddleOCR (jersey number) | ~0.8 GB |
| Pose estimation | ~1.0 GB |
| Buffer & overhead | ~0.5 GB |
| **Total** | **~6.8 GB** |

> GTX 1050 Ti hanya 4 GB → model dijalankan bergantian (sequential), bukan paralel → lebih lambat.  
> RTX 3060 ke atas (8 GB+) → semua model bisa di-load sekaligus → signifikan lebih cepat.

---

## Ukuran File Video vs Durasi

| Resolusi | FPS | Format | Per menit | 5 menit | 10 menit | 40 menit |
|---|---|---|---|---|---|---|
| 720p | 30 | H.264 | ~60–100 MB | ~400 MB | ~800 MB | ~3.2 GB |
| 1080p | 30 | H.264 | ~100–200 MB | ~800 MB | ~1.5 GB | ~6 GB |
| 1080p | 60 | H.264 | ~200–400 MB | ~1.5 GB | ~3 GB | ~12 GB |

> **Rekomendasi upload:** 720p atau 1080p 30 FPS H.264. Hindari 4K — tidak menambah akurasi deteksi tapi jauh memperlambat pipeline.

---

## Kebutuhan Jaringan (Web App)

| Skenario | Bandwidth Upload | Bandwidth Download |
|---|---|---|
| Upload video rekaman | **10 Mbps** (untuk file < 1 GB) | 1 Mbps |
| Live streaming kamera ke server | **5–8 Mbps** per kamera | 2 Mbps |
| Akses dashboard (WebSocket) | < 1 Mbps | 2–5 Mbps |

---

## Browser Client

| | |
|---|---|
| **Browser** | Chrome 90+ / Edge 90+ / Firefox 88+ |
| **Fitur wajib** | WebSocket, ES2020, CSS Grid |
| **Resolusi layar** | Minimal 1280 × 720 (rekomendasi 1920 × 1080) |
| **RAM browser** | Minimal 4 GB tersedia |

---

## Ringkasan

| | Tier 1 (Minimum) | Tier 2 (Recommended) | Tier 3 (Optimal) |
|---|---|---|---|
| GPU | GTX 1050 Ti 4 GB | GTX 1660 Ti / RTX 2060 6 GB | RTX 3080 / RTX 4070 10+ GB |
| RAM | 8 GB | 16 GB | 32 GB |
| Max video | **5 menit / 500 MB** | **15 menit / 1.5 GB** | **40 menit / 5 GB** |
| Live real-time | ❌ | ❌ (bisa dengan lightweight mode) | ✅ |
| Use case | Demo singkat | 1 kuarter pertandingan | Full game + live |

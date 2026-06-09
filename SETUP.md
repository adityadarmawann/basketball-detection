# Setup — Device Baru

Panduan menjalankan Smart Vision Basketball Analytics dari awal di device baru.

---

## Prasyarat Sistem

| Kebutuhan | Versi Minimum | Catatan |
|---|---|---|
| OS | Ubuntu 20.04+ | atau Debian-based |
| Python | 3.10+ | disarankan 3.11/3.13 |
| Node.js | 20.x | wajib ≥18, disarankan 20 |
| RAM | 8 GB | 16 GB untuk GPU inference |
| GPU | Opsional | CUDA 11.8+ jika pakai GPU |

---

## 1. Clone Repository

```bash
git clone <repo-url>
cd smart-vision-cl
```

---

## 2. Node.js (wajib versi 20)

Cek versi dulu:
```bash
node --version
```

Kalau di bawah 18, upgrade via nvm:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20
nvm alias default 20
```

---

## 3. MongoDB

```bash
# Import GPG key
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
  sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# Tambah repo
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] \
  https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
  sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install
sudo apt update && sudo apt install -y mongodb-org

# Jalankan
sudo systemctl enable --now mongod
```

Verifikasi: `mongosh --eval "db.runCommand({ ping: 1 })"`

---

## 4. Redis

```bash
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
```

Verifikasi: `redis-cli ping` → harus balas `PONG`

---

## 5. Python Dependencies

Install dalam urutan berikut (urutan penting):

```bash
cd backend

# Step 1 — PyTorch GPU (WAJIB duluan, jangan lewati)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# Step 2 — Semua dependency lain
pip install -r requirements.txt

# Step 3 — BoxMOT (harus --no-deps agar tidak konflik numpy)
#   boxmot 10.0.43 mendeklarasikan numpy==1.23.1 di metadata-nya, padahal
#   runtime-compatible dengan numpy 2.x. --no-deps melewati cek tersebut.
pip install boxmot==10.0.43 lapx==0.9.4 --no-deps
```

> **CPU-only (tanpa GPU):** ganti step 1 dengan:
> ```bash
> pip install torch==2.2.2 torchvision==0.17.2
> ```
>
> **Warning "NumPy 1.x compiled"** saat import torch adalah normal — tidak menyebabkan crash.

---

## 6. Model Files

File `.pt` tidak ada di repository — harus dicopy manual ke folder `backend/models/`:

```
backend/models/
├── action_best_v1.pt       # SlowFast action recognition
├── best-object-basketball.pt  # YOLO deteksi bola
├── court_keypoints.pt      # homography lapangan
├── jersey_no.pt            # OCR nomor jersey (fine-tuned YOLOv8)
└── yolov8s-pose.pt         # pose estimation pemain
```

---

## 7. Environment Variables

Buat file `backend/.env`:
```env
MONGO_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
MODELS_PATH=./models
UPLOAD_PATH=./uploads
MAX_FPS_PROCESS=25
GPU_DEVICE=0
API_HOST=0.0.0.0
API_PORT=8000
```

Buat file `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_APP_NAME="Smart Vision Basketball Analytics"
```

> Kalau backend dan frontend beda mesin, ganti `localhost` dengan IP backend.

---

## 8. Inisialisasi Database

```bash
cd backend
python scripts/init_db.py
```

Membuat collections MongoDB beserta indexes yang dibutuhkan.

---

## 9. Frontend Dependencies

```bash
cd frontend
npm install
```

> Kalau muncul `npm audit` warning soal esbuild — abaikan, itu hanya berlaku di dev server dan tidak mempengaruhi fungsionalitas.
> Jangan jalankan `npm audit fix --force` karena akan upgrade Vite ke versi 8 yang tidak kompatibel.

---

## 10. Jalankan Aplikasi

Buka **dua terminal** terpisah:

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Buka browser: [http://localhost:5173](http://localhost:5173)

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'redis'`**
```bash
pip install redis
```

**`mongod` tidak bisa start**
```bash
sudo systemctl status mongod
sudo journalctl -u mongod --no-pager | tail -20
```

**Vite error `jsx invalid key` atau warning `esbuild deprecated`**
— Vite versi 8 terinstall. Downgrade ke Vite 5:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install vite@^5.4.0 --save-dev
npm install
```

**Pipeline tidak menghasilkan output analisis**
- Pastikan semua file `.pt` ada di `backend/models/`
- Cek Redis berjalan: `redis-cli ping`
- Cek log backend untuk error CUDA / model load

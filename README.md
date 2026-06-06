# Smart Vision Basketball Analytics Dashboard

Web application untuk analisis pertandingan basket berbasis Computer Vision (CV), terintegrasi dengan dashboard real-time menggunakan React dan FastAPI.

## 🏀 Tentang Proyek

**Smart Vision Campus League (CV-Based Sports Analytics)** adalah platform analitik basket yang menggabungkan:
- 5 model Computer Vision (Python/PyTorch) untuk deteksi pemain, pose, action recognition, jersey OCR, dan court detection
- Backend FastAPI dengan WebSocket untuk real-time data streaming
- Frontend React modern dengan D3.js dan Recharts untuk visualisasi
- MongoDB untuk penyimpanan data historis
- Redis untuk state management real-time

## 🛠 Tech Stack

### Frontend
- React 18 + TypeScript
- Vite for fast development
- Tailwind CSS for styling
- D3.js & Recharts for visualizations
- React Router v6
- Zustand for state management
- Axios for HTTP requests

### Backend
- FastAPI (Python 3.10+)
- WebSocket via FastAPI
- MongoDB (PyMongo)
- Redis for caching
- OpenCV + PyTorch + Ultralytics
- PaddleOCR for jersey number recognition

### DevOps
- Docker + Docker Compose
- MongoDB 7
- Redis 7

## 📁 Struktur Folder

```
smart-vision-cl/
├── backend/          # FastAPI application
├── frontend/         # React application
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose (recommended)
- Atau: Python 3.10+, Node.js 18+, MongoDB, Redis

### Development dengan Docker Compose

```bash
# Clone atau download project
cd smart-vision-cl

# Build dan run semua services
docker-compose up --build

# Services akan tersedia di:
# Frontend:  http://localhost:3000
# Backend:   http://localhost:8000
# MongoDB:   localhost:27017
# Redis:     localhost:6379
```

### Development Manual

#### Backend
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start MongoDB & Redis (in separate terminals or use services)
# mongod --dbpath ./data/db
# redis-server

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend
```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev

# Opens at http://localhost:5173
```

## 📖 API Documentation

Setelah backend berjalan, akses Swagger docs di: **http://localhost:8000/docs**

### Endpoint Utama
- `POST /api/match` — Buat match baru
- `POST /api/roster` — Input roster pemain
- `POST /api/upload-video` — Upload video pertandingan
- `GET /api/stats/live` — Real-time statistics
- `WS /ws/live` — WebSocket untuk frame updates
- `WS /ws/events` — WebSocket untuk game events

## 🎨 UI/UX

### Halaman Utama (/)
- Match setup form
- Video upload section
- Live video player dengan court overlay
- Scoreboard & MVP ranking
- Event feed

### Halaman Lineups (/match/lineups)
- Tabel statistik pemain per quarter
- Sortable columns
- Export to CSV

### Halaman MPI (/match/mpi)
- Physical metrics dashboard
- Distance, speed, acceleration metrics
- Jump height estimation
- Agility, endurance scores
- MPI composite index

## ⚙️ Konfigurasi

### Environment Variables

**Backend (.env)**
```
MONGO_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
MODELS_PATH=./models
UPLOAD_PATH=./uploads
MAX_FPS_PROCESS=25
GPU_DEVICE=0
API_HOST=0.0.0.0
API_PORT=8000
```

**Frontend (.env)**
```
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_APP_NAME="Smart Vision Basketball Analytics"
```

## 📊 Data Flow

```
Video Upload
    ↓
Video Processing Pipeline (Backend)
    ├── YOLO Detection (players, ball, hoop)
    ├── Pose Estimation (17 keypoints)
    ├── Court Detection + Homography
    ├── Action Recognition
    ├── Jersey Number OCR
    ├── Multi-Object Tracking (ByteTrack)
    ├── Event Engine (scoring, assists, etc)
    └── Stats Calculator (EFF, MPI, speed, distance)
    ↓
MongoDB (Historical Data) + Redis (Real-time State)
    ↓
WebSocket Broadcast
    ↓
Frontend Visualization (React + D3.js)
```

## 📝 Development Phases

1. ✅ **Phase 1: Scaffold** — Project structure & boilerplate
2. ⏳ **Phase 2: Frontend Pages** — Match, Lineups, MPI pages
3. ⏳ **Phase 3: Backend Routes** — API endpoints implementation
4. ⏳ **Phase 4: Database** — MongoDB & Redis integration
5. ⏳ **Phase 5: WebSocket** — Real-time data streaming
6. ⏳ **Phase 6: CV Pipeline** — Model integration
7. ⏳ **Phase 7: Export** — PDF & CSV export features
8. ⏳ **Phase 8: Deployment** — Docker finalization & production setup

## 🔗 Useful Links

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [D3.js](https://d3js.org/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)

## 📜 License

Copyright © 2024 Smart Vision Campus League. All rights reserved.

---

**Developed with ❤️ for Campus League Basketball Analytics**

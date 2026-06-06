# 🚀 Smart Vision Basketball Analytics — Quick Start Guide

## ✅ Completed: Phase 1 Scaffold

Struktur project sudah siap! Berikut status:

### Backend (FastAPI)
- ✅ Main FastAPI application setup
- ✅ API routes boilerplate (upload, stats, events, roster, match)
- ✅ Database layer (MongoDB, Redis)
- ✅ Pydantic schemas
- ✅ WebSocket infrastructure
- ✅ Docker setup
- ✅ requirements.txt

### Frontend (React + Vite)
- ✅ React 18 + TypeScript + Vite setup
- ✅ Tailwind CSS styling + custom color palette
- ✅ React Router navigation
- ✅ Zustand state management
- ✅ Custom hooks (WebSocket, video upload)
- ✅ Utility functions (court transform, MPI formula, export)
- ✅ Page components (Match, Lineups, MPI)
- ✅ Layout components (Navbar, Sidebar)
- ✅ Docker setup

### DevOps
- ✅ Docker Compose orchestration
- ✅ MongoDB + Redis services
- ✅ Network configuration

---

## 🏃 Menjalankan Project Secara Local

### Opsi 1: Dengan Docker Compose (Recommended)

```bash
cd smart-vision-cl

# Build semua services (first time, takes a few minutes)
docker-compose up --build

# Kemudian akses:
# Frontend:  http://localhost:3000
# Backend:   http://localhost:8000 (API docs: /docs)
# MongoDB:   localhost:27017
# Redis:     localhost:6379
```

Stop services:
```bash
docker-compose down
```

### Opsi 2: Development Manual Setup

#### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB running locally
- Redis running locally

#### Backend

```bash
cd backend

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access FastAPI docs: http://localhost:8000/docs

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Access app: http://localhost:5173

---

## 📂 Folder Structure

```
smart-vision-cl/
├── backend/
│   ├── api/
│   │   ├── routes/          # ✅ Upload, Stats, Events, Roster, Match
│   │   └── websocket.py     # 🟡 WebSocket manager (to implement)
│   ├── db/
│   │   ├── mongo.py         # ✅ MongoDB connection
│   │   └── redis_client.py  # ✅ Redis connection
│   ├── pipeline/            # 🟡 CV pipeline (to implement)
│   ├── schemas/             # ✅ Pydantic models
│   ├── models/              # 📦 Pre-trained .pt files go here
│   ├── uploads/             # 📁 Video uploads directory
│   ├── main.py              # ✅ FastAPI entry point
│   ├── requirements.txt      # ✅ Dependencies
│   └── Dockerfile           # ✅ Docker config
│
├── frontend/
│   ├── src/
│   │   ├── pages/           # ✅ Match, Lineups, MPI pages
│   │   ├── components/      # ✅ Layout, Match, Video, Roster, Dashboard, Lineup, MPI
│   │   ├── hooks/           # ✅ WebSocket, Video Upload
│   │   ├── store/           # ✅ Zustand state management
│   │   ├── utils/           # ✅ Court transform, MPI formula, Export
│   │   ├── types/           # ✅ TypeScript interfaces
│   │   ├── App.tsx          # ✅ Main app component
│   │   └── main.tsx         # ✅ Entry point
│   ├── index.html           # ✅ HTML template
│   ├── package.json         # ✅ Dependencies
│   ├── vite.config.ts       # ✅ Vite config
│   ├── tailwind.config.js   # ✅ Tailwind config
│   ├── tsconfig.json        # ✅ TypeScript config
│   └── Dockerfile           # ✅ Docker config
│
├── docker-compose.yml       # ✅ Service orchestration
├── README.md                # ✅ Full documentation
└── QUICKSTART.md            # 👈 This file
```

---

## 🔧 Development Workflow

### 1. Add Backend Routes
Endpoints dalam `/backend/api/routes/` sudah siap:
- [ ] Implement upload handler dengan video processing trigger
- [ ] Integrate MongoDB queries
- [ ] Implement WebSocket message broadcasting

### 2. Build Frontend Components
Component skeleton sudah ada di `/frontend/src/components/`:
- [ ] `match/` — Match header, form setup, stats bar, tabs, quarter filter
- [ ] `video/` — Upload, player, court overlay, court map
- [ ] `roster/` — Form input, player rows
- [ ] `dashboard/` — Scoreboard, MVP ranking, MPI, event feed
- [ ] `lineup/` — Stats table dengan sort & filter
- [ ] `mpi/` — Physical metrics, charts, composite score

### 3. Integrate WebSocket
- [ ] Implement WebSocket manager di backend
- [ ] Mock WebSocket data untuk development (simulasi 30fps frame updates)
- [ ] Update frontend hooks untuk receive data
- [ ] Test real-time updates di browser

### 4. Database Integration
- [ ] Create MongoDB collections (matches, players, events, stats, mpi_metrics)
- [ ] Implement CRUD operations
- [ ] Test data persistence

### 5. Video Processing Pipeline
- [ ] Setup model loading (YOLO, YOLOv8-Pose, etc)
- [ ] Implement frame processing loop
- [ ] Integrate CV inference with event engine
- [ ] Push processed data to Redis/MongoDB

---

## 📝 Environment Variables

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

---

## 🧪 Testing the Setup

### Test Backend Health
```bash
curl http://localhost:8000/health
# Response: {"status":"ok","service":"smart-vision-api"}
```

### Test FastAPI Docs
Open: http://localhost:8000/docs

### Test Frontend
Open: http://localhost:5173 (dev mode) or http://localhost:3000 (production)

You should see:
- Navbar dengan logo Campus League
- Hero banner
- Match setup form
- Navigation links

---

## 🎯 Next Steps (Priority)

1. **Enable & Test Backend Routes**
   - Uncomment imports dalam `main.py`
   - Test endpoints di Swagger

2. **Create Mock WebSocket Server**
   - Simulasi frame updates dengan player positions bergerak
   - Test frontend WebSocket hook

3. **Build Frontend Pages**
   - Complete Match page (form validation, roster management)
   - Complete Lineups page (stats table)
   - Complete MPI page (metrics cards & charts)

4. **Integrate Real Data**
   - Wire up API calls
   - Test database persistence
   - Connect WebSocket data stream

5. **Add CV Pipeline**
   - Download atau setup model files
   - Implement video processor
   - Test end-to-end inference

---

## 📚 Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **React Docs**: https://react.dev/
- **Tailwind CSS**: https://tailwindcss.com/
- **D3.js**: https://d3js.org/
- **Zustand**: https://github.com/pmndrs/zustand
- **Vite**: https://vitejs.dev/

---

## ❓ Troubleshooting

### Docker not building?
```bash
# Clear cache and rebuild
docker-compose down -v
docker-compose up --build
```

### Frontend not connecting to backend?
Check `.env` in frontend folder:
```
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### MongoDB connection error?
Ensure MongoDB is running and accessible at specified URL in `.env`

### Port already in use?
Change ports di `docker-compose.yml` atau stop conflicting services

---

**Happy coding! 🎉 Smart Vision Basketball Analytics** 🏀

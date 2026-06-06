# 🗺️ Smart Vision Basketball Analytics — Development Roadmap

## ✅ PHASE 1: PROJECT SCAFFOLD [COMPLETE]

### Backend Infrastructure
- [x] FastAPI app setup with Lifespan context manager
- [x] CORS middleware configuration
- [x] Environment variables (.env)
- [x] Python dependencies (requirements.txt) with 19 packages
- [x] Database connection modules (MongoDB, Redis)
- [x] WebSocket connection manager skeleton
- [x] Pydantic schemas for all data types
- [x] Dockerfile for backend
- [x] All 5 API route modules with boilerplate

### Frontend Infrastructure  
- [x] React 18 + Vite + TypeScript setup
- [x] Tailwind CSS with custom color palette
- [x] React Router with 3 main pages
- [x] Zustand store for match state
- [x] Custom hooks (WebSocket, video upload)
- [x] Utility functions (court transform, MPI formula, export)
- [x] Component folder structure (6 component groups)
- [x] Navbar and Sidebar components
- [x] Dockerfile for frontend (multi-stage build)

### DevOps & Documentation
- [x] Docker Compose orchestration (4 services)
- [x] MongoDB + Redis service configuration
- [x] .gitignore files
- [x] README.md (complete project documentation)
- [x] QUICKSTART.md (getting started guide)
- [x] API_REFERENCE.md (endpoint documentation)
- [x] DATABASE_SCHEMA.md (MongoDB schemas)
- [x] SCAFFOLD_COMPLETE.md (completion summary)
- [x] This roadmap document

---

## 🔄 PHASE 2: CORE COMPONENTS [Next - Est. 2-3 days]

### Match Page Components
- [ ] `MatchHeader.tsx` — Display match info (teams, score, quarter)
  - Team logos/badges
  - Live badge with animation
  - Score display with quarter & game clock
  - Shot clock counter
  
- [ ] `FormSetupMatch.tsx` — Match setup form
  - Team name inputs (A & B)
  - Category dropdown (Men's/Women's)
  - Round dropdown
  - Form validation
  - Submit handler

- [ ] `RosterManager.tsx` — Add/manage players
  - Jersey number input (0-99)
  - Player name input
  - Team assignment (A/B dropdown)
  - Delete player button
  - Real-time list update
  - Validation (no duplicate jerseys)

- [ ] `TeamStatsBar.tsx` — Team comparison bars
  - Display: PTS, REB, AST, FG%, 3P%, BLK, STL, TOV
  - Horizontal bar comparison (A vs B)
  - Color-coded (team-a = teal, team-b = dark)

- [ ] `QuarterFilter.tsx` — Q1/Q2/Q3/Q4 filter
  - Button group selector
  - "All" option
  - Connected to stats display

### Video Upload & Playback Components
- [ ] `VideoUpload.tsx` — Drag & drop upload
  - Drag & drop area
  - Browse button
  - Progress bar during upload
  - File info (name, size, duration)
  - Error handling

- [ ] `VideoPlayer.tsx` — Video playback with overlay
  - HTML5 video player
  - Play/pause controls
  - Progress bar
  - Volume control
  - Canvas overlay layer
  - FPS counter

- [ ] `CourtOverlay.tsx` — Canvas overlay for court visualization
  - Render player bounding boxes
  - Jersey numbers + names above boxes
  - Color by team
  - Speed indicator below boxes
  - Ball position + trajectory
  - Wasit/referee detection

- [ ] `CourtMap2D.tsx` — D3.js bird-eye court map
  - FIBA standard court drawing (28m × 15m)
  - Court elements (3PT line, paint, free throw, etc)
  - Player positions as colored dots
  - Ball position as yellow dot
  - Click player → show live stats panel
  - Legend

### Dashboard Components
- [ ] `Scoreboard.tsx` — Score display
  - Team A: score vs Team B
  - Quarter indicator
  - Game clock (MM:SS format)
  - Shot clock timer
  - Possession indicator

- [ ] `MvpRanking.tsx` — Leaderboard
  - Top 3 players by EFF + MPI
  - Gold badge for #1
  - Player name, jersey, stats
  - Update after each quarter

- [ ] `LiveEventFeed.tsx` — Event log
  - Scrollable list (auto-scroll on new event)
  - Timestamp, player name, event type, detail
  - Icons for each event type
  - Color-coded by team

---

## 📊 PHASE 3: LINEUPS PAGE [Est. 2-3 days]

- [ ] `PlayerStatsTable.tsx` — Detailed stats table
  - Sortable columns: #, NAME, MIN, PTS, 2P, 3P, FT, 2P%, OFF, DEF, TOT, AST, STL, TOV, C, R, M, R, +/-, EFF
  - Click column header to sort (ascending/descending toggle)
  - Highlight row with highest EFF (gold background)
  - +/- column: green for positive, red for negative
  - Quarter filter dropdown (All/Q1/Q2/Q3/Q4)
  - Export CSV button (top right)
  - Responsive design

---

## 📈 PHASE 4: MPI PAGE [Est. 2-3 days]

### Metrics Panels
- [ ] `PhysicalMetricsCard.tsx` — Metric cards
  - Distance covered (km)
  - Speed (Avg / Max in km/h)
  - Possession %
  - Jump height estimation
  - Acceleration / Deceleration
  - Agility, Endurance, Fatigue scores
  - Each with units and styling

- [ ] `RealtimeChart.tsx` — Recharts integration
  - Line chart: Speed over time
  - Line chart: Acceleration over time
  - Multi-player toggle
  - X-axis: frame / time
  - Y-axis: metric value
  - Update as data streams in
  - Max 200 data points for performance

- [ ] `MetricApproxPanel.tsx` — Approximation warning
  - ⚠️ Icon + yellow border
  - Text: "Data berbasis Computer Vision, bukan sensor IMU/GPS. Akurasi ±15-25%."
  - Toggle show/hide details

- [ ] `MpiCompositeCard.tsx` — MPI score display
  - Bar chart: MPI per pemain
  - Color gradient (red-yellow-green)
  - Display formula: 0.25P + 0.20A + 0.20E + 0.20Eff + 0.15C
  - Breakdown of each component

---

## 🔌 PHASE 5: BACKEND API INTEGRATION [Est. 3-4 days]

### Enable API Routes
- [ ] Uncomment imports in `main.py`
- [ ] Test all endpoints in Swagger docs
- [ ] Implement database queries
  - `GET /api/match/{match_id}`
  - `GET /api/roster/{match_id}`
  - `GET /api/stats/live`
  - `GET /api/stats/player/{id}`
  - `GET /api/stats/team/{id}`
  - `GET /api/stats/quarter/{q}`
  - `GET /api/events`

### Database Integration
- [ ] Create MongoDB collections with indexes
- [ ] Implement CRUD operations
- [ ] Connection pooling
- [ ] Query optimization
- [ ] Data validation

### WebSocket Implementation
- [ ] Complete WebSocket manager
- [ ] Handle client connections
- [ ] Broadcast frame updates to all clients
- [ ] Broadcast game events
- [ ] Reconnection logic
- [ ] Message queuing for high frequency updates

---

## 🌐 PHASE 6: REAL-TIME DATA FLOW [Est. 4-5 days]

### Mock WebSocket Server (Development)
- [ ] Create `mockWebSocket.ts` utility
- [ ] Simulate 30fps frame updates
- [ ] Generate fake player positions (moving around court)
- [ ] Generate fake events (FGM, REB, AST, etc) every 5-10 seconds
- [ ] Realistic game data
- [ ] Use for development before CV pipeline ready

### Frontend WebSocket Integration
- [ ] Connect to `/ws/live` endpoint
- [ ] Handle frame updates with `requestAnimationFrame`
- [ ] Update Zustand store
- [ ] Render player positions on canvas
- [ ] Update stats in real-time
- [ ] Handle disconnection/reconnection

### Backend WebSocket Broadcasting
- [ ] Implement connection manager
- [ ] Track active clients
- [ ] Broadcast to all connected clients
- [ ] Handle client disconnect
- [ ] Error handling

---

## 🎬 PHASE 7: VIDEO PROCESSING PIPELINE [Est. 5-7 days]

### Model Setup
- [ ] Download pre-trained model files (5 models)
- [ ] Place in `backend/models/`
- [ ] Implement lazy loading
- [ ] GPU device management
- [ ] Memory optimization

### Pipeline Implementation
- [ ] `detector.py` — YOLOv8 detection
- [ ] `pose.py` — YOLOv8-Pose 17 keypoints
- [ ] `court.py` — Court keypoints + homography
- [ ] `action.py` — Action classification
- [ ] `jersey_ocr.py` — Jersey number recognition
- [ ] `tracker.py` — ByteTrack multi-object tracking
- [ ] `event_engine.py` — Auto scoring logic
- [ ] `stats_calculator.py` — EFF, MPI, speed, distance

### Video Processing
- [ ] Implement main orchestrator
- [ ] Frame buffer management
- [ ] Background task processing
- [ ] Progress tracking
- [ ] Error handling & recovery

---

## 📤 PHASE 8: EXPORT FEATURES [Est. 1-2 days]

- [ ] PDF export with match summary
  - Header with match info
  - Stats table
  - Charts
  - Team logos
  
- [ ] CSV export
  - Tab-separated or comma-separated
  - All stat columns
  - One row per player
  - Download as file

---

## 🧪 PHASE 9: TESTING & OPTIMIZATION [Est. 2-3 days]

### Frontend Testing
- [ ] Unit tests for utility functions
- [ ] Component tests for isolated components
- [ ] Integration tests for page flows
- [ ] E2E tests for full user journey
- [ ] Performance profiling (Lighthouse)

### Backend Testing
- [ ] API endpoint tests (pytest)
- [ ] Database query tests
- [ ] WebSocket tests
- [ ] Load testing (concurrent connections)

### Performance Optimization
- [ ] Canvas rendering optimization
- [ ] WebSocket message debouncing
- [ ] Database query optimization
- [ ] Frontend bundle size analysis
- [ ] Lazy loading components

---

## 🚀 PHASE 10: DEPLOYMENT [Est. 1-2 days]

- [ ] Production Docker build
- [ ] Environment variable setup
- [ ] Database migration scripts
- [ ] SSL/HTTPS setup
- [ ] CI/CD pipeline
- [ ] Monitoring & logging
- [ ] Backup strategy

---

## 📋 Quick Checklist

### To Start Phase 2
- [ ] Run `docker-compose up --build`
- [ ] Test frontend loads at http://localhost:3000
- [ ] Test backend health at http://localhost:8000/health
- [ ] Check Swagger docs at http://localhost:8000/docs

### Before Phase 3
- [ ] All Phase 2 components built
- [ ] API endpoints returning mock data
- [ ] WebSocket mock running

### Before Phase 4
- [ ] Player stats table working
- [ ] Data flowing from backend

### Before Phase 5
- [ ] Frontend designs complete
- [ ] Database schema finalized
- [ ] API contracts defined

---

## 🎯 Success Criteria

**Phase 1 (Complete):** ✅
- Project structure in place
- All boilerplate code written
- Documentation complete
- Ready to start development

**Phase 2:** 🎯
- All component files exist
- Form submissions work
- Data displays in UI
- No hard errors

**Phase 3:** 🎯
- Stats table functional
- Sorting works
- Export buttons functional

**Phase 4:** 🎯
- All metric cards render
- Charts update in real-time
- Approximation warnings show

**Phase 5:** 🎯
- API endpoints tested
- Database persistence working
- Swagger docs accurate

**Phase 6:** 🎯
- Real-time updates flowing
- All UI elements updating
- No lag on 30fps stream

**Phase 7:** 🎯
- Video upload working
- Models loading successfully
- Inference completing
- Events auto-detected

**Phase 8:** 🎯
- PDF exports valid
- CSV downloads readable
- All stats included

**Phase 9:** 🎯
- All tests passing
- >90% code coverage
- Lighthouse score >80

**Phase 10:** 🎯
- Live server running
- Users can access app
- Data persists
- Scalable & stable

---

**Current Status: PHASE 1 ✅ COMPLETE**

**Ready to start PHASE 2 anytime!** 🚀

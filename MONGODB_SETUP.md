# MongoDB Setup — Smart Vision Basketball

Database name: `smart_vision_basketball`  
Default URL: `mongodb://localhost:27017`

---

## 1. Install MongoDB

```bash
# Import GPG key
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc \
  | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# Add repo
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] \
  https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" \
  | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install
sudo apt update && sudo apt install -y mongodb-org
```

---

## 2. Jalankan MongoDB

```bash
# Start & enable otomatis saat boot
sudo systemctl start mongod
sudo systemctl enable mongod

# Cek status
sudo systemctl status mongod

# Verifikasi koneksi
mongosh --eval "db.adminCommand('ping')"
# → { ok: 1 }  ✓
```

---

## 3. Init Database Sistem Ini

```bash
cd backend
python scripts/init_db.py
```

Output yang diharapkan:
```
✓ Created collection: matches
✓ Created collection: players
✓ Created collection: player_stats
✓ Created collection: events
✓ Created collection: mpi_metrics
✓ Created collection: frame_updates
✓ Created collection: videos
📊 Creating indexes...
✓ Index: matches.match_id
✓ Index: players (match_id, jersey_number)
✓ Index: player_stats (match_id, player_id) + (quarter)
✓ Index: events (match_id, timestamp)
✓ Index: mpi_metrics (match_id, player_id)
✅ Database initialization complete!
```

Dengan seed data dummy (opsional):
```bash
python scripts/init_db.py --seed
```

---

## 4. Konfigurasi .env

File `backend/.env` sudah ada dan sudah benar:
```env
MONGO_URL=mongodb://localhost:27017
```

Tidak perlu diubah untuk dev lokal. Kalau deploy ke server lain:
```env
MONGO_URL=mongodb://<user>:<password>@<host>:27017
```

---

## 5. Collections & Kegunaannya

| Collection | Diisi oleh | Isi |
|---|---|---|
| `matches` | `match.py` router + pipeline finalize | Data setup pertandingan |
| `players` | `roster.py` router | Roster pemain per match |
| `player_stats` | `stats.py` router | Statistik akhir per pemain |
| `events` | pipeline `video_processor.py` | Event game real-time (FGM, REB, dll) |
| `mpi_metrics` | pipeline | Movement Performance Index |
| `frame_updates` | (reserved) | — |
| `videos` | `upload.py` router | Metadata video yang diupload |

---

## 6. Cek Isi Database (opsional)

```bash
mongosh

use smart_vision_basketball

# Lihat semua matches
db.matches.find().pretty()

# Lihat events match tertentu
db.events.find({ match_id: "MATCH_ID_KAMU" }).pretty()

# Hitung dokumen per collection
db.matches.countDocuments()
db.events.countDocuments()
db.players.countDocuments()
```

---

## Troubleshooting

**MongoDB tidak mau start:**
```bash
sudo rm /tmp/mongodb-27017.sock 2>/dev/null
sudo chown -R mongodb:mongodb /var/lib/mongodb /var/log/mongodb
sudo systemctl restart mongod
```

**Port 27017 sudah terpakai:**
```bash
sudo lsof -i :27017   # cek proses mana
```

**Reset database (hapus semua data):**
```bash
mongosh --eval "use smart_vision_basketball; db.dropDatabase()"
# Lalu init ulang:
cd backend && python scripts/init_db.py
```

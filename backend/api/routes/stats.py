"""
Statistics endpoints:
- GET /stats/live              — Real-time stats snapshot
- GET /stats/player/{id}       — Per-player stats
- GET /stats/team/{team_id}    — Per-team stats
- GET /stats/quarter/{q}       — Per-quarter stats

MPI endpoints:
- GET /mpi/team                — All players MPI for a match (Redis)
- GET /mpi/{player_id}         — Single player MPI (Redis)
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Query
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client["smart_vision_basketball"]

from db.redis_client import get_redis
from db.mongo import get_db

# ── Default empty-stats structure returned when no live data is available ──────

def _empty_team_stats() -> dict:
    return {"pts": 0, "reb": 0, "ast": 0, "stl": 0,
            "blk": 0, "tov": 0, "fgm": 0, "fga": 0, "fg_pct": 0.0}

def _default_response() -> dict:
    return {
        "player_stats":  {},
        "team_stats":    {"A": _empty_team_stats(), "B": _empty_team_stats()},
        "mvp_ranking":   [],
        "possession_pct": {"A": 0.5, "B": 0.5},
        "score":         {"team_a": 0, "team_b": 0},
        "quarter":       1,
        "game_clock":    "10:00",
    }


@router.get("/stats/live")
async def get_live_stats(match_id: str = Query(None)):
    """
    Get current live stats snapshot.

    Priority:
      1. Redis key  stats:{match_id}  (written by VideoProcessor every N frames)
      2. Default empty structure      (never returns a plain message string)

    Query params
    ------------
    match_id : str, optional
        Match identifier. If omitted returns default structure.
    """
    if not match_id:
        return _default_response()

    # ── 1. Try Redis ──────────────────────────────────────────────────────────
    try:
        redis = await get_redis()
        if redis is not None:
            raw = await redis.get(f"stats:{match_id}")
            if raw:
                data = json.loads(raw)
                # Ensure all required top-level keys are present
                base = _default_response()
                base.update(data)
                return base
    except Exception as e:
        logger.warning("Redis read error for match %s: %s", match_id, e)

    # ── 2. Mongo: ringkasan GABUNGAN, permanen ────────────────────────────────
    # URUTAN INI PENTING dan pernah salah. Dulu Redis per kuarter didahulukan,
    # padahal isinya statistik MENTAH satu baris per track_id, sedangkan yang
    # dilihat pengguna adalah hasil penggabungan per nomor punggung dan tim.
    # Bedanya terukur: 488 pemain lawan 1292 baris. Kunci gabungan di Redis
    # hanya berlaku SATU hari, jadi tanpa urutan ini angka riwayat berubah
    # sendiri sehari setelah analisis, tanpa ada yang menyentuh apa pun.
    try:
        db = get_db()
        if db is not None:
            m = db.matches.find_one({"match_id": match_id},
                                    {"ringkasan.gabungan": 1})
            gab = ((m or {}).get("ringkasan") or {}).get("gabungan")
            if gab:
                base = _default_response()
                if gab.get("player_stats") is not None:
                    base["player_stats"] = gab["player_stats"]
                if gab.get("team_stats") is not None:
                    base["team_stats"] = gab["team_stats"]
                if gab.get("skor") is not None:
                    base["score"] = gab["skor"]
                base["sumber"] = "ringkasan_gabungan_mongo"
                return base
    except Exception as e:
        logger.warning("Ringkasan gabungan Mongo gagal untuk %s: %s", match_id, e)

    # ── 3. Redis per KUARTER, sebagai jalan terakhir yang masih berisi ────────
    # Isinya MENTAH, jadi hanya dipakai kalau ringkasan gabungan memang tidak
    # ada, misalnya analisis lama yang selesai sebelum ringkasan diperkenalkan.
    try:
        redis = await get_redis()
        if redis is not None:
            gabung = None
            for q in range(1, 5):
                raw = await redis.get(f"stats:{match_id}:q{q}")
                if raw:
                    gabung = json.loads(raw)      # kuarter terakhir yang ada menang
            if gabung:
                base = _default_response()
                base.update(gabung)
                base["sumber"] = "redis_kuarter_mentah"
                return base
    except Exception as e:
        logger.warning("Redis per-kuarter gagal untuk %s: %s", match_id, e)

    # ── 4. Mongo: ringkasan per kuarter, juga mentah ──────────────────────────
    try:
        db = get_db()
        if db is not None:
            m = db.matches.find_one({"match_id": match_id}, {"ringkasan": 1})
            ring = (m or {}).get("ringkasan") or {}
            terpilih = None
            for q in sorted(k for k in ring.keys() if k != "gabungan"):
                if ring[q]:
                    terpilih = ring[q]
            if terpilih:
                base = _default_response()
                if terpilih.get("player_stats") is not None:
                    base["player_stats"] = terpilih["player_stats"]
                if terpilih.get("team_stats") is not None:
                    base["team_stats"] = terpilih["team_stats"]
                if terpilih.get("skor") is not None:
                    base["score"] = terpilih["skor"]
                base["sumber"] = "ringkasan_kuarter_mongo"
                return base
    except Exception as e:
        logger.warning("Ringkasan Mongo gagal untuk %s: %s", match_id, e)

    # ── 5. Terakhir: bentuk kosong, JANGAN melempar galat ─────────────────────
    return _default_response()

# ── MPI helpers ───────────────────────────────────────────────────────────────

def _default_mpi_player() -> dict:
    return {
        "track_id":        0,
        "name":            "",
        "jersey_number":   0,
        "team":            "A",
        "mpi_composite":   0.0,
        "distance_m":      0.0,
        "avg_speed_kmh":   0.0,
        "max_speed_kmh":   0.0,
        "jump_height_cm":  0.0,
        "agility_score":   0.0,
        "endurance_score": 0.0,
        "fatigue_score":   0.0,
    }


async def _fetch_mpi_players(match_id: str) -> list:
    """
    Read MPI player list for a match.
    Tries dedicated  mpi:{match_id}  key first, then extracts the embedded
    mpi dict from each player entry in  stats:{match_id}  as fallback
    (VideoProcessor stores MPI inside player_stats, not in a separate key).
    """
    try:
        redis = await get_redis()
        if redis is None:
            return []

        # Primary: dedicated key
        raw = await redis.get(f"mpi:{match_id}")
        if raw:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                return data
            if isinstance(data, dict) and data:
                return list(data.values())

        # Fallback: extract from embedded player_stats
        raw = await redis.get(f"stats:{match_id}")
        ps_map = {}
        if raw:
            ps_map = (json.loads(raw) or {}).get("player_stats", {})
        if not ps_map:
            # Cadangan PERMANEN. Kunci stats:{id} hanya berlaku SATU hari,
            # jadi tab MPI adalah yang paling cepat kosong dari semuanya.
            # Ringkasan gabungan menyimpan player_stats yang sama beserta
            # bagian mpi di dalam tiap pemain, jadi angkanya tetap sama.
            try:
                db = get_db()
                if db is not None:
                    m = db.matches.find_one({"match_id": match_id},
                                            {"ringkasan.gabungan": 1})
                    gab = ((m or {}).get("ringkasan") or {}).get("gabungan") or {}
                    ps_map = gab.get("player_stats") or {}
            except Exception as e:
                logger.warning("Ringkasan Mongo untuk MPI gagal %s: %s",
                               match_id, e)
        if not ps_map:
            return []
        result    = []
        for tid_str, pdata in ps_map.items():
            mpi = pdata.get("mpi", {})
            if not mpi:
                continue
            tid = int(tid_str) if str(tid_str).isdigit() else 0
            result.append({
                "track_id":        tid,
                "name":            pdata.get("name", ""),
                "jersey_number":   pdata.get("jersey_number"),
                "team":            pdata.get("team", ""),
                **mpi,
            })
        return result
    except Exception as e:
        logger.warning("Redis MPI read error match=%s: %s", match_id, e)
        return []


# ── MPI endpoints — /mpi/team MUST be before /mpi/{player_id} ────────────────
# FastAPI matches routes top-to-bottom; the literal path "team" must be
# registered first so it isn't captured by the {player_id} path parameter.

@router.get("/mpi/team")
async def get_team_mpi(match_id: str = Query(...)):
    """
    Get MPI physical metrics for all players in a match.

    Reads from Redis key  mpi:{match_id}  (written by VideoProcessor after
    each stats_calculator update).  Returns an empty players list when the
    key is absent or Redis is unavailable — never raises 500.

    Query params
    ------------
    match_id : str  (required)
    """
    players = await _fetch_mpi_players(match_id)
    return {
        "match_id": match_id,
        "players":  players,
    }


@router.get("/mpi/{player_id}")
async def get_player_mpi(
    player_id: int,
    match_id:  str = Query(...),
):
    """
    Get MPI physical metrics for a single player.

    Reads from Redis key  mpi:{match_id}  and filters by
    track_id == player_id.

    Query params
    ------------
    match_id : str  (required)

    Raises 404 when the player is not found in the MPI data.
    """
    players = await _fetch_mpi_players(match_id)
    for player in players:
        if player.get("track_id") == player_id:
            return player
    raise HTTPException(
        status_code=404,
        detail=f"No MPI data for player_id={player_id} in match '{match_id}'",
    )


# ── Existing per-player / per-team / per-quarter endpoints ────────────────────

@router.get("/stats/player/{player_id}")
async def get_player_stats(player_id: int):
    """
    Get aggregated stats for a specific player.
    """
    try:
        player_stats = list(db.player_stats.find(
            {"player_id": player_id},
            {"_id": 0}
        ).sort("quarter", 1))

        if not player_stats:
            return {
                "player_id": player_id,
                "message": "No stats found"
            }

        return {
            "player_id": player_id,
            "count": len(player_stats),
            "stats": player_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/team/{team_id}")
async def get_team_stats(team_id: str):
    """
    Get aggregated stats for a team (A or B).
    """
    try:
        team_stats = list(db.player_stats.find(
            {"team": team_id},
            {"_id": 0}
        ).sort("quarter", 1))

        if not team_stats:
            return {"team_id": team_id, "message": "No stats found"}

        return {
            "team_id": team_id,
            "count": len(team_stats),
            "stats": team_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Bidang yang boleh DIJUMLAHKAN saat satu pemain muncul di beberapa blob.
_BISA_JUMLAH = (
    "pts", "fgm", "fga", "three_pm", "three_pa", "ftm", "fta",
    "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "min",
)


async def _blob_bersumber_kuarter(match_id: str) -> list:
    """Semua blob yang mungkin memuat quarter_stats, Redis dulu lalu Mongo."""
    hasil = []
    try:
        redis = await get_redis()
        if redis is not None:
            for q in range(1, 5):
                raw = await redis.get(f"stats:{match_id}:q{q}")
                if raw:
                    hasil.append((q, json.loads(raw)))
    except Exception as e:
        logger.warning("Baca blob kuarter dari Redis gagal %s: %s", match_id, e)
    if hasil:
        return hasil
    try:
        db = get_db()
        if db is not None:
            m = db.matches.find_one({"match_id": match_id}, {"ringkasan": 1})
            ring = (m or {}).get("ringkasan") or {}
            for k in sorted(k for k in ring.keys() if k != "gabungan"):
                if ring[k] and k.startswith("q") and k[1:].isdigit():
                    hasil.append((int(k[1:]), ring[k]))
    except Exception as e:
        logger.warning("Baca blob kuarter dari Mongo gagal %s: %s", match_id, e)
    return hasil


def _pemain_per_kuarter(blobs: list, quarter: int) -> dict:
    """Statistik satu kuarter, diturunkan dari quarter_stats tiap pemain.

    KENAPA BUKAN LANGSUNG DARI KUNCI stats:<id>:q<n>
    Kunci itu dinamai memakai kuarter AWAL unggahan, bukan kuarter yang sedang
    berjalan (`video_processor.py`: `q_key = f"stats:{id}:q{self._start_quarter}"`).
    Akibatnya satu video yang melintasi Q1 sampai Q4 menumpuk SELURUH
    statistiknya di `:q1`, dan tab Q2, Q3, Q4 tampil kosong padahal datanya ada.
    Itu gejala "cuma tab All yang terisi" yang dilaporkan pengguna.

    Angka per kuarter sebenarnya SUDAH dicatat stats_calculator di dalam tiap
    pemain sebagai `quarter_stats`, dengan bentuk bidang yang sama persis dengan
    `total_stats`. Fungsi ini membacanya dari sana, sehingga benar untuk dua
    duanya: unggahan per klip maupun satu video penuh.

    Pemain yang muncul di beberapa blob disatukan memakai aturan yang sama
    dengan `_merge_all_quarters`, yaitu per (nomor punggung, tim), dan memakai
    nama saat nomornya belum terbaca supaya tidak saling menelan.
    """
    qk = str(quarter)
    gabung: dict = {}
    for kuarter_kunci, blob in blobs:
        for pdata in (blob.get("player_stats") or {}).values():
            semua_qs = pdata.get("quarter_stats") or {}
            qs = semua_qs.get(qk)
            if not qs:
                # Pemain yang TIDAK punya rincian kuarter sama sekali tetap
                # dimasukkan ke kuarter milik blob ini. Mereka terdeteksi di
                # lapangan tapi tidak punya catatan statistik, dan di tab "All"
                # memang tampil sebagai baris nol. Tanpa langkah ini, baris
                # mereka ada di tab All tapi hilang di tab kuarter, dan itu
                # beda yang membingungkan. Terukur pada data nyata: 91 baris
                # seperti ini, NOL di antaranya punya angka bukan nol, jadi
                # tidak ada angka yang dikarang di sini.
                if semua_qs or kuarter_kunci != quarter:
                    continue
                qs = pdata.get("total_stats") or {}
            jn   = pdata.get("jersey_number")
            team = pdata.get("team", "")
            key  = (jn, team) if jn is not None else (pdata.get("name", ""), team)
            if key not in gabung:
                gabung[key] = {
                    "name":          pdata.get("name", ""),
                    "jersey_number": jn,
                    "team":          team,
                    "mpi":           pdata.get("mpi", {}),
                    "total_stats":   dict(qs),
                }
            else:
                tujuan = gabung[key]["total_stats"]
                for f in _BISA_JUMLAH:
                    v = qs.get(f, 0)
                    if isinstance(v, (int, float)):
                        tujuan[f] = tujuan.get(f, 0) + v

    for entri in gabung.values():
        ts  = entri["total_stats"]
        fgm = ts.get("fgm", 0); fga = ts.get("fga", 0)
        ts["fg_pct"] = round(fgm / fga, 3) if fga > 0 else 0.0
        ts["eff"] = (ts.get("pts", 0) + ts.get("reb", 0) + ts.get("ast", 0)
                     + ts.get("stl", 0) + ts.get("blk", 0)) \
                    - ((fga - fgm) + ts.get("tov", 0))

    return {str(i): v for i, v in enumerate(gabung.values())}


def _team_dari_pemain(player_stats: dict) -> dict:
    """team_stats diturunkan dari pemain, supaya cocok dengan tabelnya."""
    BIDANG = ("pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga")
    keluar = {"A": {f: 0 for f in BIDANG}, "B": {f: 0 for f in BIDANG}}
    for p in player_stats.values():
        t = p.get("team")
        if t not in ("A", "B"):
            continue
        ts = p.get("total_stats") or {}
        for f in BIDANG:
            keluar[t][f] += ts.get(f, 0) or 0
    for v in keluar.values():
        v["fg_pct"] = round(v["fgm"] / v["fga"], 3) if v["fga"] > 0 else 0.0
    return keluar


def _skor_dari_team(team_stats) -> dict:
    """Skor dari poin tim.

    Blob per kuarter yang ditulis pipeline TIDAK memuat bidang "score", jadi
    jalur Redis mengembalikan 0-0 padahal team_stats di blob yang sama menyebut
    8 dan 10. Ketidakcocokan itu nyata dan terukur. Menurunkannya dari poin tim
    membuat jalur Redis dan jalur Mongo mengembalikan angka yang sama, dan
    keduanya cocok dengan statistik tim yang ditampilkan di sebelahnya.
    """
    if not isinstance(team_stats, dict):
        return {"team_a": 0, "team_b": 0}
    return {
        "team_a": int((team_stats.get("A") or {}).get("pts", 0) or 0),
        "team_b": int((team_stats.get("B") or {}).get("pts", 0) or 0),
    }


@router.get("/stats/quarter/{quarter}")
async def get_quarter_stats(quarter: int, match_id: str = Query(...)):
    """
    Statistik SATU kuarter.

    Diturunkan dari `quarter_stats` milik tiap pemain, BUKAN dari isi kunci
    `stats:{match_id}:q{n}` apa adanya. Alasannya ada di `_pemain_per_kuarter`:
    kunci itu dinamai memakai kuarter AWAL unggahan, jadi satu video yang
    melintasi beberapa kuarter menumpuk semuanya di satu kunci dan tab kuarter
    lainnya tampil kosong.

    Bentuk keluarannya sama dengan /stats/live supaya frontend bisa memakai
    penormal yang sama.
    """
    if quarter not in [1, 2, 3, 4]:
        raise HTTPException(status_code=400, detail="Quarter must be 1-4")

    blobs = await _blob_bersumber_kuarter(match_id)
    pemain = _pemain_per_kuarter(blobs, quarter)
    if pemain:
        tim = _team_dari_pemain(pemain)
        base = _default_response()
        base["player_stats"] = pemain
        base["team_stats"]   = tim
        base["score"]        = _skor_dari_team(tim)
        base["quarter"]      = quarter
        base["sumber"]       = "quarter_stats"
        return base

    # Cadangan HANYA untuk data lama yang sama sekali tidak punya
    # quarter_stats. Penjagaan ini penting: tanpa dia, kuarter yang memang
    # TIDAK PERNAH diunggah akan menampilkan isi blob kuarter lain, sehingga
    # tab Q2 memperlihatkan angka Q1. Sudah terjadi saat diuji.
    ada_qs = any(
        (p.get("quarter_stats") or {})
        for _, blob in blobs
        for p in (blob.get("player_stats") or {}).values()
    )
    if ada_qs:
        # Datanya memang memuat quarter_stats, jadi kuarter ini benar benar
        # kosong. Mengembalikan bentuk kosong lebih jujur daripada menampilkan
        # angka kuarter lain.
        return _default_response()

    for _, blob in blobs:
        if not blob.get("player_stats"):
            continue
        base = _default_response()
        base.update({k: v for k, v in blob.items() if k != "skor"})
        if blob.get("skor") is not None:
            base["score"] = blob["skor"]
        base["quarter"] = quarter
        if not (base.get("score") or {}).get("team_a") and \
           not (base.get("score") or {}).get("team_b"):
            base["score"] = _skor_dari_team(base.get("team_stats"))
        base["sumber"] = "blob_kuarter_apa_adanya"
        return base

    return _default_response()

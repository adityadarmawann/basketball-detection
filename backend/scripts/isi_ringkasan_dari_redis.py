#!/usr/bin/env python
"""
Isi ringkasan kuarter di MongoDB dari statistik yang masih ada di Redis.

KENAPA PERLU
Ringkasan kuarter sempat GAGAL ditulis karena MongoDB menolak nama bidang
bertipe angka (track_id dan nomor kuarter), dengan pesan "documents must have
only string keys, key was 0". Akibatnya riwayat analisis tampil tanpa skor
walau analisisnya selesai dengan benar.

Pipeline-nya sudah diperbaiki, tapi analisis yang SUDAH selesai tidak akan
menulis ulang sendiri. Statistiknya masih utuh di Redis dengan masa berlaku
7 hari, jadi ringkasannya bisa diisi dari data ASLI itu. Tidak ada angka yang
dikarang di sini: kalau kuncinya sudah kedaluwarsa, barisnya dilewati dan
dilaporkan apa adanya.

Redis prod dan dev berada di basis data BERBEDA (prod db 0, dev db 1), dan satu
analisis hanya ada di salah satunya. Jadi keduanya diperiksa untuk tiap match.

PAKAI
    python scripts/isi_ringkasan_dari_redis.py           # lihat dulu, tanpa menulis
    python scripts/isi_ringkasan_dari_redis.py --tulis   # baru menulis
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis                                    # noqa: E402
from pymongo import MongoClient                 # noqa: E402

from pipeline.video_processor import (           # noqa: E402
    _kunci_teks, _skor_dari_team_stats,
)

MONGO_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGODB_DB", "smart_vision_basketball")
# Dasar URL Redis tanpa nomor basis data, supaya db 0 dan db 1 bisa disusun.
DASAR     = os.getenv("REDIS_URL", "redis://localhost:6379").rsplit("/", 1)[0] \
            if os.getenv("REDIS_URL", "").count("/") > 2 \
            else os.getenv("REDIS_URL", "redis://localhost:6379")


def _cari_gabungan(mid: str):
    """Blob GABUNGAN `stats:<mid>` kalau masih hidup, beserta nomor db-nya.

    Inilah yang DILIHAT pengguna di halaman hasil: barisnya sudah disatukan per
    nomor punggung dan tim oleh `_merge_all_quarters`. Sengaja diambil apa
    adanya, bukan dihitung ulang di sini, supaya tidak ada dua tempat yang
    menghitung hal yang sama lalu berbeda diam diam.

    Masa berlakunya cuma SATU hari, jadi untuk analisis yang lebih tua blob ini
    sudah tidak ada dan hanya ringkasan per kuarter yang bisa diisi.
    """
    for db in (0, 1):
        try:
            r = redis.from_url(f"{DASAR.rstrip('/')}/{db}", decode_responses=True)
            mentah = r.get(f"stats:{mid}")
            if mentah:
                return db, json.loads(mentah)
        except Exception:
            continue
    return None, None


def _cari_kuarter(mid: str) -> dict:
    """Semua kunci stats:<mid>:q<N> yang masih hidup, di db 0 maupun db 1."""
    temuan = {}
    for db in (0, 1):
        try:
            r = redis.from_url(f"{DASAR.rstrip('/')}/{db}", decode_responses=True)
            for k in r.scan_iter(f"stats:{mid}:q*"):
                q = k.rsplit(":q", 1)[-1]
                mentah = r.get(k)
                if mentah:
                    temuan[q] = (db, json.loads(mentah))
        except Exception as e:
            print(f"    db {db} tidak terbaca: {e}")
    return temuan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tulis", action="store_true",
                    help="benar benar menulis ke MongoDB")
    ap.add_argument("--match", default="",
                    help="batasi ke satu match_id saja")
    a = ap.parse_args()

    d = MongoClient(MONGO_URL)[MONGO_DB]
    saring = {"match_id": a.match} if a.match else {}
    cocok = list(d.matches.find(saring).sort("_id", -1))
    if not cocok:
        print("Tidak ada match yang cocok.")
        return 1

    print(f"{len(cocok)} match diperiksa. "
          f"{'MENULIS' if a.tulis else 'hanya melihat, tidak menulis'}.\n")
    terisi = dilewati = 0
    for m in cocok:
        mid = m.get("match_id")
        if not mid:
            continue
        sudah = sorted((m.get("ringkasan") or {}).keys())
        kuarter = _cari_kuarter(mid)
        print(f"  {mid}  status={m.get('status','?')}")
        print(f"    ringkasan sekarang : {sudah or 'kosong'}")
        if not kuarter:
            print("    di Redis           : tidak ada (kedaluwarsa atau dihapus)")
            print("    tindakan           : DILEWATI, tidak ada data asli\n")
            dilewati += 1
            continue

        # Gabungan lebih dulu: inilah yang dibaca halaman hasil.
        db_gab, gab = _cari_gabungan(mid)
        if gab is None:
            print("    gabungan           : tidak ada di Redis "
                  "(berlaku 1 hari, sudah lewat)")
        elif "gabungan" in sudah:
            print("    gabungan           : sudah ada, tidak ditimpa")
        else:
            gps = gab.get("player_stats") or {}
            print(f"    gabungan dari db{db_gab}: {len(gps)} pemain, "
                  f"skor {(gab.get('score') or {}).get('team_a','-')}-"
                  f"{(gab.get('score') or {}).get('team_b','-')}")
            if a.tulis:
                d.matches.update_one({"match_id": mid}, {"$set": {
                    "ringkasan.gabungan": _kunci_teks({
                        "skor":           gab.get("score"),
                        "player_stats":   gps,
                        "team_stats":     gab.get("team_stats"),
                        "mvp_ranking":    gab.get("mvp_ranking"),
                        "possession_pct": gab.get("possession_pct"),
                        "disimpan_pada":  time.time(),
                        "sumber":         f"redis db{db_gab}, diisi ulang",
                    })}})
                print("    gabungan           : DITULIS")
                terisi += 1

        for q, (db, s) in sorted(kuarter.items()):
            ts = s.get("team_stats")
            skor = _skor_dari_team_stats(ts)
            ps = s.get("player_stats") or {}
            print(f"    q{q} dari redis db{db}: {len(ps)} baris pemain, "
                  f"skor {(skor or {}).get('team_a','-')}-"
                  f"{(skor or {}).get('team_b','-')}")
            if f"q{q}" in sudah:
                print(f"    q{q}                : sudah ada, tidak ditimpa")
                continue
            if not a.tulis:
                continue
            muatan = _kunci_teks({
                "skor":           skor,
                "player_stats":   ps,
                "team_stats":     ts,
                "mvp_ranking":    s.get("mvp_ranking"),
                "possession_pct": s.get("possession_pct"),
                "disimpan_pada":  time.time(),
                # Ditandai supaya jelas angkanya diambil ulang dari Redis,
                # bukan ditulis pipeline saat analisis selesai.
                "sumber":         f"redis db{db}, diisi ulang",
            })
            d.matches.update_one({"match_id": mid},
                                 {"$set": {f"ringkasan.q{q}": muatan}})
            print(f"    q{q}                : DITULIS")
            terisi += 1
        print()

    print(f"Selesai. {terisi} kuarter ditulis, {dilewati} match dilewati.")
    if not a.tulis:
        print("Belum ada yang ditulis. Tambahkan --tulis untuk menulisnya.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

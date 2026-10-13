#!/usr/bin/env python
"""
Memeriksa bahwa halaman HASIL benar benar tampil lengkap, dengan merendernya
di Chrome tanpa tampilan dan membaca isi halamannya, SEKALIGUS memastikan
backend yang hidup memuat kode terbaru.

KENAPA ADA
Hasil pertandingan pernah tampil berbeda antara "baru selesai dianalisis" dan
"dibuka lagi dari riwayat": tanpa kotak sponsor, tanpa tab STATS dan LINEUPS,
tanpa riwayat event, dan tab LINEUP serta MPI kosong. Bedanya hanya ketahuan
satu per satu lewat laporan pengguna, dan itu melelahkan.

Skrip ini mengubah pemeriksaan itu jadi sekali jalan. Memeriksa SATU hal yang
tidak bisa dibuktikan dari kode saja: apa yang benar benar sampai ke layar.

PAKAI
    python scripts/periksa_tampilan.py <match_id> [alamat-web] [alamat-api]
    python scripts/periksa_tampilan.py 20261012_201407 http://127.0.0.1:5174 http://127.0.0.1:8001

Keluar dengan kode 1 kalau ada bagian wajib yang hilang, jadi bisa dipakai
sebagai penjaga sebelum menyatakan sesuatu sudah beres.
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time

import requests
import websockets

PORT   = int(os.getenv("PERIKSA_CDP_PORT", "9333"))
PROFIL = os.path.join(tempfile.gettempdir(), "periksa-tampilan-profil")
AUTH   = json.dumps({"state": {"user": {"name": "pemeriksa", "email": "uji@lokal"},
                               "isLoggedIn": True}, "version": 0})

# Bagian yang WAJIB ada di halaman hasil. Tiap baris pernah benar benar hilang.
WAJIB_HASIL = [
    ("video original",      "VIDEO ORIGINAL"),
    ("video analisis AI",   "VIDEO ANALISIS AI"),
    ("kotak sponsor",       "REWARD & APPRECIATE"),
    ("tab STATS",           "STATS"),
    ("tab LINEUPS",         "LINEUPS"),
    ("bar TEAM STATS",      "TEAM STATS"),
    ("riwayat event",       "RIWAYAT EVENT"),
    ("tombol Export CSV",   "EXPORT CSV"),
]
WAJIB_LINEUP = [("tabel pemain", "EFF"), ("pemilih kuarter", "Q1")]
WAJIB_MPI    = [("judul MPI", "PHYSICAL METRICS"), ("kartu penghargaan", "ENDURANCE")]


def nyalakan_chrome():
    os.makedirs(PROFIL, exist_ok=True)
    proc = subprocess.Popen(
        ["google-chrome", "--headless=new", f"--remote-debugging-port={PORT}",
         f"--user-data-dir={PROFIL}", "--no-sandbox", "--disable-gpu",
         "--window-size=1600,2400", "--hide-scrollbars", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            requests.get(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Chrome tidak kunjung siap")


class Sesi:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def kirim(self, metode, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": metode,
                                       "params": params}))
        while True:
            pesan = json.loads(await self.ws.recv())
            if pesan.get("id") == self.n:
                if "error" in pesan:
                    raise RuntimeError(pesan["error"])
                return pesan.get("result", {})


def tab_baru():
    for cara in (requests.put, requests.get):
        try:
            return cara(f"http://127.0.0.1:{PORT}/json/new?about:blank",
                        timeout=10).json()
        except Exception:
            continue
    daftar = requests.get(f"http://127.0.0.1:{PORT}/json/list", timeout=10).json()
    return next(t for t in daftar if t.get("type") == "page")


async def baca_halaman(url: str, tunggu: float, potret: str | None):
    tab = tab_baru()
    async with websockets.connect(tab["webSocketDebuggerUrl"],
                                  max_size=80_000_000) as ws:
        s = Sesi(ws)
        await s.kirim("Page.enable")
        await s.kirim("Runtime.enable")
        # Sesi login ditanam sebelum halaman dimuat, karena tab LINEUP dan MPI
        # dijaga ProtectedRoute dan tanpa ini hanya akan tampil halaman depan.
        await s.kirim("Page.addScriptToEvaluateOnNewDocument",
                      source=f"try{{localStorage.setItem('sv-auth',"
                             f"{json.dumps(AUTH)})}}catch(e){{}}")
        await s.kirim("Page.navigate", url=url)
        await asyncio.sleep(tunggu)
        hasil = await s.kirim("Runtime.evaluate",
                              expression="document.body.innerText || ''",
                              returnByValue=True)
        teks = hasil.get("result", {}).get("value") or ""
        if potret:
            try:
                tangkap = await s.kirim("Page.captureScreenshot", format="png",
                                        captureBeyondViewport=True)
            except Exception:
                # Halaman yang sangat panjang ditolak; potret yang terlihat saja.
                tangkap = await s.kirim("Page.captureScreenshot", format="png")
            with open(potret, "wb") as f:
                f.write(base64.b64decode(tangkap["data"]))
        return teks


def periksa_backend(api_url: str, match_id: str) -> list:
    """Memastikan backend yang HIDUP memuat kode terbaru, bukan cuma frontend.

    KENAPA PERLU DI SINI
    Pemeriksa ini awalnya hanya merender halaman, jadi ia tetap berkata
    "LENGKAP" walaupun backend-nya masih proses lama yang belum dinyalakan
    ulang. Tampilannya memang benar karena Vite memuat ulang frontend sendiri,
    tapi perilaku backend yang lama masih jalan, dan itu menyesatkan.

    Penandanya bidang `sumber` pada /api/stats/quarter/{q}. Kode lama tidak
    pernah mengirimkannya, kode baru selalu.
    """
    masalah = []
    try:
        r = requests.get(f"{api_url}/api/stats/quarter/1",
                         params={"match_id": match_id}, timeout=120)
        d = r.json()
    except Exception as e:
        print(f"  backend    TIDAK TERJANGKAU  ({api_url}: {e})")
        return ["backend tidak terjangkau"]

    sumber = d.get("sumber")
    pemain = len(d.get("player_stats") or {})
    if sumber:
        print(f"  backend    TERBARU   (sumber={sumber}, {pemain} pemain)")
    else:
        print(f"  backend    LAMA      (tanpa bidang sumber, {pemain} pemain)")
        print("      backend yang hidup belum dinyalakan ulang, jadi tab kuarter")
        print("      dan cadangan permanen masih memakai perilaku lama")
        masalah.append("backend belum dinyalakan ulang")
    return masalah


def nilai(nama_halaman, teks, wajib):
    besar = teks.upper()
    kurang = [label for label, kata in wajib if kata.upper() not in besar]
    tanda = "LENGKAP" if not kurang else "KURANG"
    print(f"  {nama_halaman:<10} {tanda:<8} ({len(teks)} huruf)")
    for label, kata in wajib:
        ada = kata.upper() in besar
        print(f"      {'ada   ' if ada else 'HILANG'}  {label}")
    return kurang


async def utama():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mid   = sys.argv[1]
    basis = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:5174"
    api   = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8001"
    keluar = os.path.join(tempfile.gettempdir(), "periksa-tampilan")
    os.makedirs(keluar, exist_ok=True)

    proc = nyalakan_chrome()
    gagal = []
    try:
        for jalur, nama, tunggu, wajib in (
            (f"/view/{mid}",     "riwayat", 25, WAJIB_HASIL),
            ("/match/lineups",   "lineup",  12, WAJIB_LINEUP),
            ("/match/mpi",       "mpi",     12, WAJIB_MPI),
        ):
            teks = await baca_halaman(basis + jalur, tunggu,
                                      os.path.join(keluar, f"{nama}.png"))
            gagal += nilai(nama, teks, wajib)
            print()
        gagal += periksa_backend(api, mid)
        print()
    finally:
        proc.terminate()

    print(f"  potret disimpan di {keluar}")
    if gagal:
        print(f"  GAGAL: {len(gagal)} bagian hilang -> {', '.join(gagal)}")
        return 1
    print("  Semua bagian wajib tampil.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(utama()))

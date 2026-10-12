"""
25_ambil_frame_sulit.py

Mengambil frame yang BENAR BENAR SULIT untuk deteksi bola, supaya bisa dilabeli
dan dipakai melatih ulang model. Bukan frame acak: pemilihannya memakai rekaman
keyakinan deteksi bola per frame dari skrip 19, jadi yang terpilih memang frame
tempat model sekarang gagal atau hampir gagal.

TIDAK memakai GPU. Hanya membaca rekaman JSON dan mengambil frame dengan cv2.

KENAPA FRAME INI YANG DIPILIH
Terukur pada klip ini: bola selebar 21 px di frame 1280 menyusut jadi 9,5 px
setelah frame diperkecil ke 576 untuk masuk detektor. Di situlah bola hilang
tepat saat menentukan. Jadi yang paling berharga untuk dilatih adalah:
  1. frame TANPA deteksi bola sama sekali, di dekat saat bola masuk
  2. frame dengan keyakinan LEMAH di dekat saat bola masuk
  3. frame dengan kotak bola SANGAT KECIL di mana pun
  4. frame tanpa deteksi di mana pun
Urutan itu dipakai sebagai prioritas.

Contoh:
    python eksperimen/25_ambil_frame_sulit.py
    JUMLAH=200 TUJUAN=/path/lain python eksperimen/25_ambil_frame_sulit.py
"""

import csv
import json
import os
import shutil
from pathlib import Path

import cv2

REKAM  = Path(os.getenv("REKAM", "/home/paul/rekam_bola/rfdetr.json"))
VIDEO  = os.getenv("VIDEO", "/home/paul/q1_4menit.mp4")
TUJUAN = Path(os.getenv("TUJUAN", "/home/paul/computer-vision/campus-league/"
                        "basketball/smart-vision-cl/test-video/object/object-2"))
JUMLAH = int(os.getenv("JUMLAH", "160"))

# Saat bola masuk, dalam DETIK pada potongan 240 detik. Sama dengan TRUTH di
# skrip 17. Potongan ini dipotong mulai detik 60 video penuh.
SAAT_MASUK = [19, 59, 90, 119, 146]
GESER_ASLI = 60          # detik awal potongan di video penuh
DEKAT_S    = 2.5         # dianggap "di dekat bola masuk" kalau beda <= ini

# Jarak minimum antar frame terpilih, dalam frame video. Di dekat bola masuk
# dibuat rapat supaya seluruh lintasan menembus ring terwakili; di luar itu
# dibuat jarang supaya tidak dapat 150 gambar yang nyaris sama.
JARAK_DEKAT = 4
JARAK_JAUH  = 30

KELAS = ["ball", "player", "hoop", "referee", "backboard"]


def main():
    d = json.load(open(REKAM))
    fps = float(d.get("fps") or 60.0)
    baris = d["baris"]

    def dekat_masuk(f):
        t = f / fps
        return min(abs(t - s) for s in SAAT_MASUK) <= DEKAT_S

    # Prioritas 1 sampai 4, seperti dijelaskan di atas.
    p1 = [b for b in baris if b["c"] == 0 and dekat_masuk(b["f"])]
    p2 = [b for b in baris if 0 < b["c"] < 0.25 and dekat_masuk(b["f"])]
    p3 = sorted([b for b in baris if 0 < b["w"] <= 16], key=lambda b: b["w"])
    p4 = [b for b in baris if b["c"] == 0 and not dekat_masuk(b["f"])]
    p5 = [b for b in baris if 0 < b["c"] < 0.25 and not dekat_masuk(b["f"])]

    alasan = {}
    for nama, grup in (("tanpa_deteksi_dekat_ring", p1),
                       ("keyakinan_lemah_dekat_ring", p2),
                       ("kotak_bola_sangat_kecil", p3),
                       ("tanpa_deteksi", p4),
                       ("keyakinan_lemah", p5)):
        for b in grup:
            alasan.setdefault(b["f"], (nama, b))

    # Ambil berurut prioritas, dengan jarak minimum supaya tidak nyaris kembar.
    pilih, dipakai = [], []

    def muat(f):
        jarak = JARAK_DEKAT if dekat_masuk(f) else JARAK_JAUH
        return all(abs(f - g) >= jarak for g in dipakai)

    for grup in (p1, p2, p3, p4, p5):
        for b in grup:
            if len(pilih) >= JUMLAH:
                break
            f = b["f"]
            if f in [x["f"] for x in pilih] or not muat(f):
                continue
            pilih.append(b)
            dipakai.append(f)
        if len(pilih) >= JUMLAH:
            break

    pilih.sort(key=lambda b: b["f"])
    print(f"  terpilih {len(pilih)} frame dari {len(baris)} yang direkam")
    for nama in ("tanpa_deteksi_dekat_ring", "keyakinan_lemah_dekat_ring",
                 "kotak_bola_sangat_kecil", "tanpa_deteksi", "keyakinan_lemah"):
        n = sum(1 for b in pilih if alasan[b["f"]][0] == nama)
        if n:
            print(f"    {nama:<28s} {n}")

    gambar = TUJUAN / "images"
    label  = TUJUAN / "labels"
    gambar.mkdir(parents=True, exist_ok=True)
    label.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"  video tidak bisa dibuka: {VIDEO}")
        return

    rows, ok_n = [], 0
    for b in pilih:
        f = b["f"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        t_klip = f / fps
        t_asli = t_klip + GESER_ASLI
        nama = (f"q1c_{f:06d}_{int(t_klip)//60:02d}m{int(t_klip)%60:02d}s.jpg")
        cv2.imwrite(str(gambar / nama), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # Berkas label KOSONG disiapkan supaya alat pelabelan langsung jalan.
        # Sengaja kosong: frame ini justru yang modelnya GAGAL, jadi pra-label
        # dari model yang sama akan salah atau kosong dan malah menyesatkan.
        (label / nama.replace(".jpg", ".txt")).touch()
        rows.append({
            "file": nama, "frame": f,
            "detik_potongan": f"{t_klip:.2f}",
            "detik_video_asli": f"{t_asli:.2f}",
            "alasan": alasan[f][0],
            "keyakinan_bola": f"{b['c']:.3f}",
            "lebar_kotak_px": f"{b['w']:.0f}",
        })
        ok_n += 1
    cap.release()

    with open(TUJUAN / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (TUJUAN / "classes.txt").write_text("\n".join(KELAS) + "\n")
    (TUJUAN / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\n\n"
        f"nc: {len(KELAS)}\nnames: {KELAS}\n")

    print(f"  tersimpan {ok_n} gambar di {gambar}")
    print(f"  manifest  {TUJUAN / 'manifest.csv'}")


if __name__ == "__main__":
    main()

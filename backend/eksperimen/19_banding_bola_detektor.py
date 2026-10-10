"""
19_banding_bola_detektor.py

Menjawab satu pertanyaan saja: KALAU SATU MODEL LEMAH DI KELAS BOLA, BISAKAH
MODEL LAIN MENUTUPINYA. Yang diukur cakupan deteksi BOLA per detektor pada
JENDELA FRAME YANG SAMA PERSIS, lalu disilangkan.

KENAPA DUA TAHAP, BUKAN SATU PROSES
Kartu grafis di mesin ini 8 GB. Memuat RF-DETR dan YOLO sekaligus sudah
terbukti membuat kehabisan memori pada percobaan SAM2. Jadi tiap detektor
direkam di PROSES SENDIRI lalu disilangkan dari berkas. Sebagai bonus, hasil
rekamannya bisa disilangkan ulang di ambang mana pun tanpa menjalankan GPU lagi.

KENAPA DIREKAM DI AMBANG RENDAH
Yang disimpan adalah keyakinan bola TERTINGGI per frame, direkam di ambang
0,10. Dari satu rekaman itu cakupan di ambang mana pun di atas 0,10 bisa
dihitung ulang, karena "terdeteksi di ambang t" sama dengan "keyakinan
tertinggi >= t". Jadi satu jalan GPU per detektor cukup untuk seluruh sapuan.

KENAPA grab() BUKAN cap.set()
Dua rekaman HARUS melihat frame yang sama persis, kalau tidak silangnya tidak
berarti apa apa. Seek sekali ke frame awal lalu grab() berurutan memberi indeks
frame yang pasti dan jauh lebih cepat daripada cap.set() tiap frame.

Cara pakai, BERURUTAN, jangan dua proses GPU sekaligus:

    # rekam RF-DETR, ambang bola 0,10 tanpa melonggarkan kelas lain
    MODE=rekam DETECTOR_METHOD=rfdetr RFDETR_CONF=0.35 RFDETR_CONF_BALL=0.10 \
      VIDEO=/path/klip.mp4 python eksperimen/19_banding_bola_detektor.py

    # rekam YOLO
    MODE=rekam DETECTOR_METHOD=yolo DETECTOR_CONF=0.10 \
      VIDEO=/path/klip.mp4 python eksperimen/19_banding_bola_detektor.py

    # silangkan, tanpa GPU
    MODE=gabung python eksperimen/19_banding_bola_detektor.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))
os.chdir(AKAR)

# WAJIB sebelum mengimpor pipeline: modul modul itu membaca os.getenv saat
# DIIMPOR. load_dotenv TIDAK menimpa env yang sudah ada, jadi env menang.
from dotenv import load_dotenv            # noqa: E402
load_dotenv(AKAR / ".env")

MODE    = os.getenv("MODE", "rekam").strip().lower()
MULAI   = float(os.getenv("MULAI", "72"))
DETIK   = float(os.getenv("DETIK", "143"))   # 72 s + 143 s = 215 s, memuat
                                             # kelima bola masuk kebenaran
LANGKAH = int(os.getenv("LANGKAH", "2"))     # 60 fps sumber jadi 30 fps efektif,
                                             # sama dengan yang dipakai pipeline
REKAM_DIR = Path(os.getenv("REKAM_DIR", "/tmp/rekam_bola"))

# Ambang yang disilangkan di MODE=gabung. Kiri RF-DETR, kanan YOLO.
# 0.35 dipilih untuk RF-DETR karena itu angka acuan yang sudah terukur, 0.25
# untuk YOLO karena itu bawaan DETECTOR_CONF.
AMBANG_RF   = float(os.getenv("AMBANG_RF", "0.35"))
AMBANG_YOLO = float(os.getenv("AMBANG_YOLO", "0.25"))


def jalan_rekam():
    import cv2
    from pipeline.detector import create_detector, DETECTOR_METHOD

    det = create_detector()
    print(f"detektor kelas {type(det).__name__} ambang {det.conf_threshold}"
          + (f" ambang bola {getattr(det, 'conf_ball', None)}"
             if hasattr(det, "conf_ball") else ""))

    cap = cv2.VideoCapture(os.environ["VIDEO"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    awal  = int(MULAI * fps)
    akhir = int((MULAI + DETIK) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, awal)

    baris = []
    t0 = time.perf_counter()
    i = awal
    while i < akhir:
        ok, fr = cap.read()
        if not ok:
            break
        d = det.detect(fr)
        b = d.get("ball")
        if b is None:
            baris.append({"f": i, "c": 0.0, "w": 0.0, "x": 0.0, "y": 0.0})
        else:
            x1, y1, x2, y2 = b["bbox"]
            baris.append({"f": i, "c": float(b["confidence"]),
                          "w": float(x2 - x1),
                          "x": float((x1 + x2) / 2), "y": float((y1 + y2) / 2)})
        # lewati frame di antaranya tanpa decode penuh
        for _ in range(LANGKAH - 1):
            if not cap.grab():
                break
        i += LANGKAH
    cap.release()
    lama = time.perf_counter() - t0

    REKAM_DIR.mkdir(parents=True, exist_ok=True)
    berkas = REKAM_DIR / f"{DETECTOR_METHOD}.json"
    berkas.write_text(json.dumps({
        "detektor": DETECTOR_METHOD,
        "ambang_rekam": float(det.conf_threshold),
        "ambang_bola_rekam": float(getattr(det, "conf_ball", det.conf_threshold)),
        "mulai": MULAI, "detik": DETIK, "langkah": LANGKAH, "fps": fps,
        "lama_detik": lama,
        "baris": baris,
    }))
    ada = sum(1 for r in baris if r["c"] > 0)
    print(f"{DETECTOR_METHOD}: {len(baris)} frame, bola ada {ada} "
          f"({ada/max(len(baris),1)*100:.1f}% di ambang rekam), "
          f"{lama/60:.1f} menit -> {berkas}")


def _muat(nama):
    berkas = REKAM_DIR / f"{nama}.json"
    if not berkas.exists():
        sys.exit(f"rekaman {berkas} belum ada. Jalankan MODE=rekam "
                 f"DETECTOR_METHOD={nama} dulu.")
    return json.loads(berkas.read_text())


def jalan_gabung():
    rf, yo = _muat("rfdetr"), _muat("yolo")

    # Jendela dan langkah HARUS sama, kalau tidak silangnya tak bermakna.
    for k in ("mulai", "detik", "langkah"):
        if rf[k] != yo[k]:
            sys.exit(f"jendela berbeda pada '{k}': rfdetr {rf[k]} lawan "
                     f"yolo {yo[k]}. Rekam ulang dengan jendela yang sama.")

    krf = {r["f"]: r for r in rf["baris"]}
    kyo = {r["f"]: r for r in yo["baris"]}
    frame = sorted(set(krf) & set(kyo))
    n = len(frame)
    print(f"\n=== cakupan BOLA per detektor | {n} frame bersama | "
          f"detik {rf['mulai']:.0f} sampai {rf['mulai']+rf['detik']:.0f} "
          f"| langkah {rf['langkah']} ===")
    print(f"  rekaman rfdetr di ambang bola {rf['ambang_bola_rekam']:.2f}, "
          f"yolo di ambang {yo['ambang_rekam']:.2f}")

    def silang(trf, tyo):
        dua = hrf = hyo = nol = 0
        for f in frame:
            a = krf[f]["c"] >= trf
            b = kyo[f]["c"] >= tyo
            if a and b:   dua += 1
            elif a:       hrf += 1
            elif b:       hyo += 1
            else:         nol += 1
        return dua, hrf, hyo, nol

    print("\n  --- SAPUAN cakupan satu detektor, dihitung ulang dari rekaman ---")
    print(f"  {'ambang':>7s}  {'RF-DETR':>9s}  {'YOLO':>9s}")
    for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        crf = sum(1 for f in frame if krf[f]["c"] >= t) / max(n, 1) * 100
        cyo = sum(1 for f in frame if kyo[f]["c"] >= t) / max(n, 1) * 100
        trf = "  (tak terekam)" if t < rf["ambang_bola_rekam"] - 1e-9 else ""
        tyo = "  (tak terekam)" if t < yo["ambang_rekam"] - 1e-9 else ""
        print(f"  {t:7.2f}  {crf:8.1f}%  {cyo:8.1f}%{trf}{tyo}")

    print(f"\n  --- SILANG pada RF-DETR {AMBANG_RF:.2f} lawan "
          f"YOLO {AMBANG_YOLO:.2f} ---")
    dua, hrf, hyo, nol = silang(AMBANG_RF, AMBANG_YOLO)
    pct = lambda v: f"{v:5d}  {v/max(n,1)*100:5.1f}%"
    print(f"    dua duanya menemukan bola   {pct(dua)}")
    print(f"    HANYA RF-DETR               {pct(hrf)}")
    print(f"    HANYA YOLO                  {pct(hyo)}")
    print(f"    tidak satu pun              {pct(nol)}")
    print(f"    cakupan RF-DETR saja        {pct(dua+hrf)}")
    print(f"    cakupan YOLO saja           {pct(dua+hyo)}")
    print(f"    cakupan GABUNGAN            {pct(dua+hrf+hyo)}")
    print(f"    tambahan gabungan atas RF-DETR  "
          f"+{hyo/max(n,1)*100:.1f} titik persen")

    # Kesepakatan posisi saat dua duanya menemukan bola. Kalau jaraknya besar,
    # salah satunya salah objek, dan "gabungan" berarti menggabung kesalahan.
    jarak = [float(np.hypot(krf[f]["x"] - kyo[f]["x"], krf[f]["y"] - kyo[f]["y"]))
             for f in frame
             if krf[f]["c"] >= AMBANG_RF and kyo[f]["c"] >= AMBANG_YOLO]
    if jarak:
        a = np.array(jarak)
        print(f"\n    jarak titik tengah bola saat dua duanya menemukan: "
              f"median {np.median(a):.1f} px  p90 {np.percentile(a,90):.1f} px  "
              f"maks {a.max():.1f} px  n={len(a)}")
        print(f"    sepakat (<25 px, kira kira satu lebar bola): "
              f"{int((a < 25).sum())}/{len(a)} = {(a<25).mean()*100:.1f}%")

    print("\n  --- LEBAR KOTAK BOLA, hanya frame yang lolos ambangnya ---")
    for nama, kunci, t in (("RF-DETR", krf, AMBANG_RF), ("YOLO", kyo, AMBANG_YOLO)):
        w = np.array([kunci[f]["w"] for f in frame if kunci[f]["c"] >= t]
                     or [0.0])
        c = np.array([kunci[f]["c"] for f in frame if kunci[f]["c"] >= t]
                     or [0.0])
        print(f"    {nama:<8s} lebar median {np.median(w):5.1f} px  "
              f"p10 {np.percentile(w,10):5.1f}  p90 {np.percentile(w,90):5.1f}  "
              f"| keyakinan median {np.median(c):.3f}  n={len(w)}")

    print(f"\n  waktu rekam: rfdetr {rf['lama_detik']/60:.1f} menit, "
          f"yolo {yo['lama_detik']/60:.1f} menit")


if __name__ == "__main__":
    if MODE == "gabung":
        jalan_gabung()
    else:
        jalan_rekam()

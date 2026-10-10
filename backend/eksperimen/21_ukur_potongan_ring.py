"""
21_ukur_potongan_ring.py

Mengukur jalur POTONGAN DI SEKITAR RING terhadap jalur frame penuh, pada frame
yang SAMA PERSIS, lalu menilai keduanya dengan ukuran yang benar.

SASARANNYA BUKAN CAKUPAN. Sudah terbukti 10 Oktober 2026 bahwa menaikkan
cakupan bola 69,4 jadi 83,6 persen justru MENURUNKAN skor 9 jadi 7, karena
sepertiga deteksi tambahan menaruh bola di posisi salah. Jadi yang dinilai di
sini KETEPATAN POSISI: berapa persen deteksi yang melompat dari lintasan
tetangganya yang kuat. Acuannya 3,0 persen pada deteksi kuat frame penuh.

CARA KERJANYA, dan kenapa begitu
Satu jalan GPU merekam DUA angka per frame: bola dari frame penuh, dan bola
dari potongan di sekitar rim. Potongan dijalankan lewat kode produksi yang
sebenarnya (RFDetrDetector._deteksi_ulang_ring) pada salinan hasil yang
bolanya dikosongkan, jadi yang terekam adalah calon potongan APA ADANYA tanpa
digabung. Dengan begitu aturan gabung bisa dinilai ulang di ambang mana pun
tanpa menjalankan GPU lagi, termasuk gerbang jaraknya.

MODE=rekam butuh GPU. MODE=nilai tidak.

    MODE=rekam DETECTOR_METHOD=rfdetr HOOP_ROI_NEAR_PX=0 HOOP_ROI_SIZE=448 \
      VIDEO=/home/paul/q1_4menit.mp4 python eksperimen/21_ukur_potongan_ring.py

    MODE=nilai REKAM=/home/paul/rekam_roi/roi_448.json \
      python eksperimen/21_ukur_potongan_ring.py
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

# WAJIB sebelum mengimpor pipeline: modul pipeline membaca os.getenv saat
# DIIMPOR. load_dotenv tidak menimpa env yang sudah ada, jadi env menang.
from dotenv import load_dotenv            # noqa: E402
load_dotenv(AKAR / ".env")

MODE    = os.getenv("MODE", "rekam").strip().lower()
# Jendela SAMA dengan skrip 19, supaya angkanya sebanding: memuat kelima bola
# masuk kebenaran.
MULAI   = float(os.getenv("MULAI", "72"))
DETIK   = float(os.getenv("DETIK", "143"))
LANGKAH = int(os.getenv("LANGKAH", "2"))     # 60 fps sumber jadi 30 fps efektif
REKAM   = Path(os.getenv("REKAM", "/home/paul/rekam_roi/roi.json"))

AMBANG_KUAT = float(os.getenv("AMBANG_KUAT", "0.25"))
JENDELA_F   = int(os.getenv("JENDELA_F", "3"))
LOMPAT_PX   = float(os.getenv("LOMPAT_PX", "60"))   # kira kira tiga lebar bola


# ═══════════════════════════════════════════════════════════════════
# REKAM
# ═══════════════════════════════════════════════════════════════════

def jalan_rekam():
    import cv2
    from pipeline.detector import create_detector
    from pipeline import detector_rfdetr as DR

    det = create_detector()
    if type(det).__name__ != "RFDetrDetector":
        sys.exit(f"detektor {type(det).__name__}, bukan RFDetrDetector. "
                 "Paket rfdetr hanya ada di env cl-gpu; penafsir lain MUNDUR "
                 "DIAM DIAM ke YOLO. Pakai "
                 "/home/paul/miniconda3/envs/cl-gpu/bin/python.")
    print(f"detektor {type(det).__name__} ambang {det.conf_threshold} "
          f"ambang bola {det.conf_ball}")
    print(f"potongan: SIZE={DR.HOOP_ROI_SIZE} NEAR_PX={DR.HOOP_ROI_NEAR_PX} "
          f"HIST_F={DR.HOOP_ROI_HIST_F} CONF={DR.HOOP_ROI_CONF} "
          f"EDGE_PX={DR.HOOP_ROI_EDGE_PX}")

    # Saklar produksi DIMATIKAN di dalam detect(), karena di sini potongan
    # dipanggil TERPISAH supaya calonnya terekam tanpa digabung.
    DR.HOOP_ROI_ENABLE = False

    cap = cv2.VideoCapture(os.environ["VIDEO"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    awal, akhir = int(MULAI * fps), int((MULAI + DETIK) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, awal)

    baris = []
    t_penuh = t_roi = 0.0
    n_roi = 0
    i = awal
    t0 = time.perf_counter()
    while i < akhir:
        ok, fr = cap.read()
        if not ok:
            break

        ta = time.perf_counter()
        d = det.detect(fr)
        t_penuh += time.perf_counter() - ta

        r = {"f": i, "c": 0.0, "w": 0.0, "x": 0.0, "y": 0.0,
             "rc": 0.0, "rw": 0.0, "rx": 0.0, "ry": 0.0,
             "hx": 0.0, "hy": 0.0, "gerbang": 0, "dbr": -1.0}

        b = d.get("ball")
        if b is not None:
            x1, y1, x2, y2 = b["bbox"]
            r.update(c=float(b["confidence"]), w=float(x2 - x1),
                     x=float((x1 + x2) / 2), y=float((y1 + y2) / 2))

        rim = det._pilih_rim(d)
        if rim is not None:
            r["hx"], r["hy"] = float(rim["center"][0]), float(rim["center"][1])
            # Jarak bola TERAKHIR DIKETAHUI ke rim, direkam supaya gerbang
            # HOOP_ROI_NEAR_PX bisa dinilai ulang di ambang mana pun.
            if (det._bola_terakhir is not None
                    and det._nf - det._bola_terakhir_nf <= DR.HOOP_ROI_HIST_F):
                r["dbr"] = float(np.hypot(
                    det._bola_terakhir[0] - r["hx"],
                    det._bola_terakhir[1] - r["hy"]))

            # Salinan yang bolanya DIKOSONGKAN: dengan begitu jalur gabung
            # selalu mengambil cabang "isi", jadi calon potongan terekam apa
            # adanya tanpa diubah aturan gabung.
            salinan = {"rims": d.get("rims"), "hoops": d.get("hoops"),
                       "ball": None}
            ta = time.perf_counter()
            det._deteksi_ulang_ring(fr, salinan)
            t_roi += time.perf_counter() - ta
            n_roi += 1
            r["gerbang"] = 1
            rb = salinan.get("ball")
            if rb is not None:
                x1, y1, x2, y2 = rb["bbox"]
                r.update(rc=float(rb["confidence"]), rw=float(x2 - x1),
                         rx=float((x1 + x2) / 2), ry=float((y1 + y2) / 2))

        baris.append(r)
        for _ in range(LANGKAH - 1):
            if not cap.grab():
                break
        i += LANGKAH
    cap.release()
    lama = time.perf_counter() - t0

    REKAM.parent.mkdir(parents=True, exist_ok=True)
    REKAM.write_text(json.dumps({
        "mulai": MULAI, "detik": DETIK, "langkah": LANGKAH, "fps": fps,
        "size": DR.HOOP_ROI_SIZE, "near_px": DR.HOOP_ROI_NEAR_PX,
        "hist_f": DR.HOOP_ROI_HIST_F, "roi_conf": DR.HOOP_ROI_CONF,
        "ambang": det.conf_threshold, "ambang_bola": det.conf_ball,
        "lama_detik": lama, "t_penuh": t_penuh, "t_roi": t_roi,
        "n_roi": n_roi, "roi_stat": det.roi_stat,
        "baris": baris,
    }))
    n = max(len(baris), 1)
    print(f"\n  {len(baris)} frame | {lama/60:.1f} menit total")
    print(f"  waktu detect frame penuh : {t_penuh/n*1000:6.1f} ms/frame")
    print(f"  waktu predict potongan   : {t_roi/max(n_roi,1)*1000:6.1f} ms "
          f"per potongan, dijalankan {n_roi}/{len(baris)} frame "
          f"({n_roi/n*100:.1f}%)")
    print(f"  roi_stat {det.roi_stat}")
    print(f"  -> {REKAM}")


# ═══════════════════════════════════════════════════════════════════
# NILAI, tanpa GPU
# ═══════════════════════════════════════════════════════════════════

def _lompat(titik, kuat):
    """Laju lompatan: persen deteksi yang jauh dari tetangga KUAT terdekat.

    Pembandingnya hanya deteksi KUAT, supaya ukuran lintasan tidak ikut
    tercemar deteksi lemah yang sedang dinilai. Pola ini sama dengan
    20_analisis_tambahan_bola.py, supaya angkanya sebanding dengan acuan 3,0
    persen di situ.
    """
    jarak = []
    for f, x, y in titik:
        tanpa_diri = {ff: p for ff, p in kuat.items() if ff != f}
        terdekat = None
        for d in range(1, JENDELA_F + 1):
            for ff in (f - d * 2, f + d * 2):      # langkah frame = 2
                k = tanpa_diri.get(ff)
                if k is not None:
                    j = float(np.hypot(x - k[0], y - k[1]))
                    if terdekat is None or j < terdekat:
                        terdekat = j
            if terdekat is not None:
                break
        if terdekat is not None:
            jarak.append(terdekat)
    if not jarak:
        return None
    a = np.array(jarak)
    return {"n": len(a), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "lompat": float((a > LOMPAT_PX).mean() * 100)}


def _lapor(nama, titik, kuat):
    s = _lompat(titik, kuat)
    if s is None:
        print(f"  {nama:<34s} tidak ada yang punya tetangga kuat")
        return
    print(f"  {nama:<34s} median {s['median']:6.1f} px  p90 {s['p90']:7.1f}  "
          f"MELOMPAT {s['lompat']:5.1f}%  n={s['n']}")


def jalan_nilai():
    d = json.loads(REKAM.read_text())
    baris = d["baris"]
    n = len(baris)
    print(f"\n=== potongan {d['size']} px | {n} frame | detik {d['mulai']:.0f} "
          f"sampai {d['mulai']+d['detik']:.0f} | ambang bola {d['ambang_bola']} ===")
    print(f"  gerbang saat rekam: NEAR_PX={d['near_px']} HIST_F={d['hist_f']}")
    print(f"  roi_stat {d['roi_stat']}")

    # ── 1. Cakupan, disilangkan ──
    pp = sum(1 for b in baris if b["c"] > 0)
    pr = sum(1 for b in baris if b["rc"] > 0)
    dua = sum(1 for b in baris if b["c"] > 0 and b["rc"] > 0)
    hp  = pp - dua
    hr  = pr - dua
    nol = n - dua - hp - hr
    ada_rim = sum(1 for b in baris if b["gerbang"])
    print(f"\n  --- CAKUPAN (bukan sasaran, hanya latar) ---")
    print(f"    frame yang punya rim        {ada_rim:5d}  {ada_rim/n*100:5.1f}%")
    print(f"    frame penuh menemukan bola  {pp:5d}  {pp/n*100:5.1f}%")
    print(f"    potongan menemukan bola     {pr:5d}  {pr/n*100:5.1f}%")
    print(f"    dua duanya                  {dua:5d}  {dua/n*100:5.1f}%")
    print(f"    HANYA frame penuh           {hp:5d}  {hp/n*100:5.1f}%")
    print(f"    HANYA potongan              {hr:5d}  {hr/n*100:5.1f}%")
    print(f"    tidak satu pun              {nol:5d}  {nol/n*100:5.1f}%")

    # ── 2. Kesepakatan posisi saat keduanya ada ──
    jar = np.array([float(np.hypot(b["x"] - b["rx"], b["y"] - b["ry"]))
                    for b in baris if b["c"] > 0 and b["rc"] > 0] or [0.0])
    print(f"\n  --- SAAT KEDUANYA ADA, n={len(jar)} ---")
    print(f"    jarak pusat: median {np.median(jar):.1f} px  "
          f"p90 {np.percentile(jar,90):.1f}  maks {jar.max():.1f}")
    for t in (10, 25, 60, 120):
        print(f"    sepakat dalam {t:3d} px : {(jar<t).mean()*100:5.1f}%")
    cp = np.array([b["c"] for b in baris if b["c"] > 0 and b["rc"] > 0] or [0.0])
    cr = np.array([b["rc"] for b in baris if b["c"] > 0 and b["rc"] > 0] or [0.0])
    print(f"    keyakinan median: frame penuh {np.median(cp):.3f}  "
          f"potongan {np.median(cr):.3f}")
    print(f"    potongan LEBIH yakin pada {(cr>cp).mean()*100:.1f}% frame")
    wp = np.array([b["w"] for b in baris if b["c"] > 0] or [0.0])
    wr = np.array([b["rw"] for b in baris if b["rc"] > 0] or [0.0])
    print(f"    lebar kotak median: frame penuh {np.median(wp):.1f} px  "
          f"potongan {np.median(wr):.1f} px")

    # ── 3. KETEPATAN POSISI, ini sasarannya ──
    # Lintasan acuan dibangun dari deteksi KUAT FRAME PENUH, karena itulah
    # acuan yang angkanya sudah ada (3,0 persen) dan tidak boleh ikut berubah
    # saat yang dinilai berganti.
    kuat = {b["f"]: (b["x"], b["y"]) for b in baris if b["c"] >= AMBANG_KUAT}
    print(f"\n  --- KETEPATAN POSISI, lintasan acuan dari deteksi KUAT "
          f"frame penuh (c >= {AMBANG_KUAT}), lompat = > {LOMPAT_PX:.0f} px ---")
    _lapor("frame penuh, kuat",
           [(b["f"], b["x"], b["y"]) for b in baris if b["c"] >= AMBANG_KUAT], kuat)
    _lapor("potongan, kuat",
           [(b["f"], b["rx"], b["ry"]) for b in baris if b["rc"] >= AMBANG_KUAT], kuat)
    _lapor("potongan, MENGISI celah penuh",
           [(b["f"], b["rx"], b["ry"]) for b in baris
            if b["rc"] >= AMBANG_KUAT and b["c"] < AMBANG_KUAT], kuat)

    # ── 3b. UJI CERMIN, supaya perbandingannya tidak berputar ──
    # Baris "frame penuh" di atas dinilai terhadap TETANGGANYA SENDIRI,
    # sedangkan baris "potongan" dinilai terhadap lintasan frame penuh. Itu
    # berat sebelah kalau kedua sumber punya geseran sistematis. Jadi di sini
    # lintasan acuannya DIBALIK: dibangun dari deteksi kuat POTONGAN, lalu
    # frame penuh yang dinilai. Kalau frame penuh juga melompat sekitar sama,
    # berarti keduanya setara dan selisih di atas cuma akibat cara ukur.
    kuat_r = {b["f"]: (b["rx"], b["ry"]) for b in baris if b["rc"] >= AMBANG_KUAT}
    print(f"\n  --- UJI CERMIN, lintasan acuan dari deteksi KUAT POTONGAN ---")
    _lapor("potongan, kuat (diri sendiri)",
           [(b["f"], b["rx"], b["ry"]) for b in baris if b["rc"] >= AMBANG_KUAT],
           kuat_r)
    _lapor("frame penuh, kuat",
           [(b["f"], b["x"], b["y"]) for b in baris if b["c"] >= AMBANG_KUAT],
           kuat_r)

    # ── 4. Sapuan aturan gabung ──
    # Hasil GABUNGAN dinilai dengan ukuran yang sama. OVERLAP menentukan kapan
    # potongan dipercaya mengganti frame penuh.
    print(f"\n  --- SAPUAN ATURAN GABUNG (isi celah + ganti kalau sepakat) ---")
    print(f"    {'overlap':>8s} {'cakupan':>8s} {'isi':>5s} {'ganti':>6s} "
          f"{'tolak':>6s} {'MELOMPAT':>9s} {'median':>8s}")
    for ov in (0, 25, 60, 120, 1e9):
        titik, isi, ganti, tolak = [], 0, 0, 0
        for b in baris:
            adap = b["c"] >= AMBANG_KUAT
            adar = b["rc"] >= AMBANG_KUAT
            if adar and not adap:
                titik.append((b["f"], b["rx"], b["ry"])); isi += 1
            elif adar and adap:
                j = np.hypot(b["x"] - b["rx"], b["y"] - b["ry"])
                if j <= ov:
                    titik.append((b["f"], b["rx"], b["ry"])); ganti += 1
                else:
                    titik.append((b["f"], b["x"], b["y"])); tolak += 1
            elif adap:
                titik.append((b["f"], b["x"], b["y"]))
        s = _lompat(titik, kuat)
        lbl = "tanpa batas" if ov > 1e8 else f"{ov:.0f} px"
        print(f"    {lbl:>8s} {len(titik)/n*100:7.1f}% {isi:5d} {ganti:6d} "
              f"{tolak:6d} " +
              (f"{s['lompat']:8.1f}% {s['median']:7.1f}" if s else "       -       -"))

    # ── 5. Gerbang jarak, biaya waktu ──
    print(f"\n  --- GERBANG JARAK: berapa frame yang perlu potongan ---")
    print(f"    {'NEAR_PX':>8s} {'jalan':>7s} {'% frame':>8s} "
          f"{'isi celah hilang':>18s}")
    for near in (0, 150, 200, 250, 300, 400, 600):
        jalan = sum(1 for b in baris
                    if b["gerbang"] and (near == 0 or
                                         (0 <= b["dbr"] <= near)))
        # Berapa pengisian celah yang HILANG kalau gerbangnya dipasang.
        hilang = sum(1 for b in baris
                     if b["rc"] >= AMBANG_KUAT and b["c"] < AMBANG_KUAT
                     and not (near == 0 or (0 <= b["dbr"] <= near)))
        lbl = "selalu" if near == 0 else f"{near}"
        print(f"    {lbl:>8s} {jalan:7d} {jalan/n*100:7.1f}% {hilang:18d}")

    tp, tr = d["t_penuh"], d["t_roi"]
    nr = max(d["n_roi"], 1)
    print(f"\n  --- WAKTU ---")
    print(f"    detect frame penuh  {tp/n*1000:6.1f} ms/frame")
    print(f"    predict potongan    {tr/nr*1000:6.1f} ms/potongan")
    print(f"    kalau potongan jalan di SEMUA frame: "
          f"+{tr/nr*1000/(tp/n*1000)*100:.0f}% waktu detektor")


if __name__ == "__main__":
    if MODE == "nilai":
        jalan_nilai()
    else:
        jalan_rekam()

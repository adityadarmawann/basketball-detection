"""
Setel ambang keyakinan dan RANSAC di court.py, dari data, bukan dari tebakan.

Latar belakangnya: semua ambang di court.py disetel KETIKA tabel koordinat masih
salah 12 dari 33 titik. Dengan tabel yang sudah benar, titik yang dulu dibuang
sebagai pencilan bisa jadi sekarang sah, dan sebaliknya. Jadi ambangnya perlu
diukur ulang.

Tiga mutu yang diukur, dan ketiganya harus dilihat bersama karena saling tarik:
    CAKUPAN   berapa persen frame berhasil terkalibrasi. Naik itu bagus, tapi
              kalau naiknya dari homografi yang asal asalan justru berbahaya.
    KETEPATAN galat proyeksi ulang inlier, dalam meter. Turun itu bagus.
    KESTABILAN seberapa goyang posisi lapangan sebuah titik piksel tetap antara
              dua frame berurutan saat kamera hampir diam. Turun itu bagus.
              Ini yang paling penting untuk jarak tempuh dan kecepatan, karena
              homografi yang bergetar menghasilkan gerakan palsu.

Pemakaian:
    python3 -u eksperimen/08_setel_ambang_court.py <kp.npz> [kp2.npz ...]
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pipeline.court as C

# titik piksel tetap yang dipakai mengukur kestabilan, tersebar di bingkai
UJI_PIKSEL = np.array([[w, h] for w in (0.3, 0.5, 0.7) for h in (0.4, 0.6, 0.8)],
                      dtype=np.float64)


def kalibrasi(k, amb_umum, amb_rendah, amb_tinggi, ransac, lewati):
    """Tiru compute_homography dengan ambang tertentu. Kembalikan (H, galat, n)."""
    px, ct = [], []
    for i, lab in enumerate(C.KP_INDEX_TO_LABEL):
        if lab in lewati:
            continue
        a = (amb_tinggi if lab in C.HIGH_CONF_LABELS
             else amb_rendah if lab in C.LOW_CONF_LABELS else amb_umum)
        if k[i, 2] < a:
            continue
        px.append(k[i, :2]); ct.append(C.LABEL_TO_COURT[lab])
    if len(px) < C.MIN_KEYPOINTS:
        return None, None, len(px)
    P = np.array(px, np.float32); T = np.array(ct, np.float32)
    try:
        H_ct2px, mask = cv2.findHomography(T, P, cv2.RANSAC,
                                           ransacReprojThreshold=ransac)
    except cv2.error:
        return None, None, len(px)
    if H_ct2px is None or mask is None or int(mask.sum()) < C.MIN_KEYPOINTS:
        return None, None, len(px)
    inl = mask.ravel().astype(bool)
    # sebaran: inlier harus mencakup minimal 2 dari 4 separuh lapangan
    ci = T[inl]
    sisi = sum([bool((ci[:, 0] < 14).any()), bool((ci[:, 0] > 14).any()),
                bool((ci[:, 1] > 7.5).any()), bool((ci[:, 1] < 7.5).any())])
    if sisi < 2:
        return None, None, len(px)
    try:
        H = np.linalg.inv(H_ct2px.astype(np.float64))
    except np.linalg.LinAlgError:
        return None, None, len(px)
    # PERINGATAN: galat proyeksi ulang pada INLIER SAJA TIDAK BOLEH dipakai
    # menilai mutu, karena bisa diakali. Model yang membuang lebih banyak titik
    # menyisakan inlier yang pas pasan cocok, jadi galatnya terlihat kecil.
    # Terbukti di data: tabel koordinat yang SALAH memberi galat inlier 0,053 m
    # sementara yang BENAR 0,068 m, padahal yang salah meleset 2,7 meter.
    #
    # Dua ukuran di bawah ini tidak bisa diakali begitu:
    #   rasio inlier       berapa persen titik terlihat yang SETUJU dengan H
    #   galat tinggal satu titik diuji TANPA ikut membentuk H
    proy = cv2.perspectiveTransform(P[inl].reshape(-1, 1, 2).astype(np.float64),
                                    H).reshape(-1, 2)
    galat = float(np.median(np.linalg.norm(proy - T[inl], axis=1)))
    rasio = float(inl.sum()) / len(P)
    loo = []
    for j in range(len(P)):
        sel = [i for i in range(len(P)) if i != j]
        if len(sel) < 4:
            continue
        Hj, _ = cv2.findHomography(T[sel], P[sel], cv2.RANSAC,
                                   ransacReprojThreshold=ransac)
        if Hj is None:
            continue
        try:
            Hji = np.linalg.inv(Hj.astype(np.float64))
        except np.linalg.LinAlgError:
            continue
        q = cv2.perspectiveTransform(P[j].reshape(1, 1, 2).astype(np.float64),
                                     Hji).reshape(2)
        e = float(np.linalg.norm(q - T[j]))
        if np.isfinite(e) and e < 50:
            loo.append(e)
    return H, (galat, rasio, float(np.median(loo)) if loo else float("nan")), int(inl.sum())


GESER_DIAM_PX = 10.0   # kamera dianggap diam kalau geser keypoint di bawah ini


def geser_piksel(a, b):
    """Geser rata rata keypoint yang sama sama terlihat di dua sampel."""
    m = (a[:, 2] > 0.35) & (b[:, 2] > 0.35)
    if m.sum() < 2:
        return None
    return float(np.mean(np.linalg.norm(b[m, :2] - a[m, :2], axis=1)))


def nilai(data, amb_umum, amb_rendah, amb_tinggi, ransac, lewati, bingkai):
    """KESTABILAN hanya dihitung pada pasangan sampel yang kameranya DIAM.

    Tanpa saringan ini, goyangan homografi tercampur dengan gerakan kamera yang
    asli, dan angkanya jadi tidak berarti. Pada data sampel per detik, sebagian
    besar pasangan memang kameranya bergerak, jadi yang tersaring sedikit tapi
    itulah yang benar benar mengukur getaran H.
    """
    cakup, galat, goyang, rasio, loo = 0, [], [], [], []
    Hsebelum, ksebelum = None, None
    for k in data:
        H, g, _ = kalibrasi(k, amb_umum, amb_rendah, amb_tinggi, ransac, lewati)
        if H is None:
            Hsebelum, ksebelum = None, None
            continue
        cakup += 1
        galat.append(g[0]); rasio.append(g[1])
        if np.isfinite(g[2]):
            loo.append(g[2])
        diam = (ksebelum is not None
                and (geser_piksel(ksebelum, k) or 1e9) < GESER_DIAM_PX)
        if Hsebelum is not None and diam:
            pts = (UJI_PIKSEL * bingkai).reshape(-1, 1, 2)
            a = cv2.perspectiveTransform(pts, Hsebelum).reshape(-1, 2)
            b = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
            d = np.linalg.norm(a - b, axis=1)
            d = d[np.isfinite(d)]
            if d.size:
                goyang.append(float(np.median(d)))
        Hsebelum, ksebelum = H, k
    return {
        "cakupan": cakup / max(len(data), 1) * 100,
        "galat": float(np.median(galat)) if galat else float("nan"),
        "goyang": float(np.median(goyang)) if goyang else float("nan"),
        "n_diam": len(goyang),
        "rasio": float(np.mean(rasio))*100 if rasio else float("nan"),
        "loo": float(np.median(loo)) if loo else float("nan"),
    }


def main() -> int:
    kumpulan = []
    for f in sys.argv[1:]:
        d = np.load(f)
        kumpulan.append((os.path.basename(f), d["kps"], None))
    print(f"berkas: {[n for n,_,_ in kumpulan]}\n")

    # bingkai gambar untuk titik uji kestabilan; diambil dari nama berkas
    bingkai = {"kp25.npz": (1920, 1080), "kp60.npz": (1280, 720)}

    dasar = (C.CONF_THRESHOLD, C.LOW_CONF_THRESHOLD, C.HIGH_CONF_THRESHOLD, 20.0,
             C.H_SKIP_LABELS)
    print(f"{'setelan':40s} {'cakupan':>9s} {'inlier':>9s} {'galat LOO':>9s} {'goyang':>9s}")
    print("-" * 80)

    def baris(nama, au, ar, at, rs, lw):
        hasil = []
        for nm, kps, _ in kumpulan:
            b = bingkai.get(nm, (1920, 1080))
            hasil.append(nilai(kps, au, ar, at, rs, lw, np.array(b, float)))
        c = np.mean([h["cakupan"] for h in hasil])
        g = np.nanmean([h["galat"] for h in hasil])
        y = np.nanmean([h["goyang"] for h in hasil])
        r = np.nanmean([h["rasio"] for h in hasil])
        l = np.nanmean([h["loo"] for h in hasil])
        print(f"{nama:40s} {c:8.1f}% {r:8.1f}% {l:8.3f}m {y:8.3f}m")
        return c, r, l, y

    baris("SEKARANG 0.35 / 0.20 / 0.55, ransac 20, skip 40", *dasar)
    print()
    for au in (0.25, 0.30, 0.35, 0.40, 0.50):
        baris(f"ambang umum {au}", au, dasar[1], dasar[2], dasar[3], dasar[4])
    print()
    for at in (0.40, 0.55, 0.70, 0.85):
        baris(f"ambang tinggi {at}", dasar[0], dasar[1], at, dasar[3], dasar[4])
    print()
    for rs in (8.0, 12.0, 20.0, 30.0, 45.0):
        baris(f"ransac {rs} px", dasar[0], dasar[1], dasar[2], rs, dasar[4])
    print()
    baris("tanpa H_SKIP (titik 40 ikut dipakai)", *dasar[:4], set())
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
12_uji_tahap_classify.py

Mengukur tahap kedua CLASSIFY di jersey_ocr: apakah menimpa tebakan angka dari
model detect dengan model classify memperbaiki bacaan nomor, atau merusaknya.

Diuji lewat kelas JerseyOCR yang sebenarnya, bukan model telanjang, supaya
penyambungan di _batch_digit_ocr dan _parse_digit_result ikut teruji.

Masukannya dibuat menyerupai yang dilihat sistem: kotak gabungan seluruh digit
di satu gambar dianggap kotak NOMOR, lalu diperbesar NUMBER_CROP_PAD seperti
yang dilakukan pipeline, baru diberikan ke pembaca digit.

Saklarnya dibaca saat impor, jadi skrip ini dijalankan DUA KALI:
    DIGIT_CLS_ENABLE=0 python eksperimen/12_uji_tahap_classify.py
    DIGIT_CLS_ENABLE=1 python eksperimen/12_uji_tahap_classify.py
"""

import os
import sys
import time
import glob
import random
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.jersey_ocr import (            # noqa: E402
    JerseyOCR, MODELS_DIR, MODEL_FILENAME,
    NUMBER_CROP_PAD, DIGIT_CLS_ENABLE, DIGIT_CLS_PAD, DIGIT_CLS_MIN_CONF,
)

DATASET = Path("/home/paul/computer-vision/campus-league/basketball/"
               "dataset/number/dataset_raw/test")
N_GAMBAR = int(os.getenv("N_GAMBAR", "400"))
TURUNKAN = os.getenv("TURUNKAN", "0") == "1"   # tiru hilangnya resolusi di sistem


def muat_contoh():
    """Bangun (potongan_nomor, nomor_asli) dari gambar uji.

    Hanya gambar dengan 1 atau 2 kotak yang dipakai. Gambar dengan 3 kotak atau
    lebih memuat BEBERAPA pemain, jadi digitnya tidak boleh dirangkai jadi satu
    nomor; merangkainya akan membuat label yang salah.
    """
    contoh = []
    berkas = sorted(glob.glob(str(DATASET / "images" / "*.jpg")))
    random.Random(0).shuffle(berkas)
    for p in berkas:
        lbl = p.replace("/images/", "/labels/").replace(".jpg", ".txt")
        if not os.path.exists(lbl):
            continue
        baris = [l.split() for l in open(lbl).read().splitlines() if len(l.split()) >= 5]
        if not (1 <= len(baris) <= 2):
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        kotak = []
        for q in baris:
            c = int(float(q[0])); cx, cy, bw, bh = map(float, q[1:5])
            kotak.append((c, (cx - bw / 2) * W, (cy - bh / 2) * H,
                          (cx + bw / 2) * W, (cy + bh / 2) * H))
        kotak.sort(key=lambda k: k[1])                     # kiri ke kanan
        asli = int("".join(str(k[0]) for k in kotak))
        if not (0 <= asli <= 99):
            continue

        # kotak NOMOR = gabungan semua kotak digit, lalu diperbesar seperti pipeline
        nx1 = min(k[1] for k in kotak); ny1 = min(k[2] for k in kotak)
        nx2 = max(k[3] for k in kotak); ny2 = max(k[4] for k in kotak)
        bw, bh = nx2 - nx1, ny2 - ny1
        px, py = int(bw * NUMBER_CROP_PAD), int(bh * NUMBER_CROP_PAD)
        x1 = max(0, int(nx1) - px); y1 = max(0, int(ny1) - py)
        x2 = min(W, int(nx2) + px); y2 = min(H, int(ny2) + py)
        if x2 <= x1 or y2 <= y1:
            continue
        pot = im[y1:y2, x1:x2]
        if pot.size == 0:
            continue

        if TURUNKAN:
            # Sistem melihat digit dengan sekitar 0,24 detail nyata dibanding
            # dataset latih. Turunkan lalu naikkan lagi untuk menirunya.
            h0, w0 = pot.shape[:2]
            kecil = cv2.resize(pot, (max(4, int(w0 * 0.24)), max(4, int(h0 * 0.24))),
                               interpolation=cv2.INTER_AREA)
            pot = cv2.resize(kecil, (w0, h0), interpolation=cv2.INTER_LINEAR)

        contoh.append((pot, asli))
        if len(contoh) >= N_GAMBAR:
            break
    return contoh


def main():
    print("=" * 70)
    print(f"  tahap classify : {'HIDUP' if DIGIT_CLS_ENABLE else 'MATI'}"
          + (f"   pad={DIGIT_CLS_PAD}  ambang={DIGIT_CLS_MIN_CONF}"
             if DIGIT_CLS_ENABLE else ""))
    print(f"  resolusi diturunkan ke 0,24 : {'YA' if TURUNKAN else 'tidak'}")
    print("=" * 70)

    contoh = muat_contoh()
    print(f"  {len(contoh)} potongan nomor dibangun dari {DATASET.name}\n")

    ocr = JerseyOCR(model_path=os.path.join(MODELS_DIR, MODEL_FILENAME), device="cpu")
    ocr.load_digit_model()
    ocr.load_digit_cls_model()
    if ocr._digit_model is None:
        print("  model detect tidak termuat, berhenti.")
        return
    if DIGIT_CLS_ENABLE and ocr._digit_cls_model is None:
        print("  PERINGATAN: saklar hidup tapi model classify tidak termuat.\n")

    potongan = [c for c, _ in contoh]
    asli     = [a for _, a in contoh]

    t0 = time.perf_counter()
    hasil = ocr._batch_digit_ocr(potongan, centred=True)
    lama = time.perf_counter() - t0

    benar = kosong = salah = 0
    contoh_salah = []
    for h, a in zip(hasil, asli):
        if h is None:
            kosong += 1
        elif h[0] == a:
            benar += 1
        else:
            salah += 1
            if len(contoh_salah) < 10:
                contoh_salah.append((a, h[0]))

    n = len(contoh)
    print(f"  benar        : {benar:4d}  ({benar/n*100:5.2f} %)")
    print(f"  salah baca   : {salah:4d}  ({salah/n*100:5.2f} %)")
    print(f"  tidak terbaca: {kosong:4d}  ({kosong/n*100:5.2f} %)")
    print(f"\n  waktu total  : {lama:.2f} s   ({lama/n*1000:.1f} ms per potongan, CPU)")
    if DIGIT_CLS_ENABLE:
        print(f"  classify menimpa tebakan detect : {ocr._cls_dipakai} kali")
        print(f"  classify ditolak / gagal        : {ocr._cls_dilewat} kali")
    if contoh_salah:
        print(f"\n  contoh salah (asli -> terbaca): {contoh_salah}")


if __name__ == "__main__":
    main()

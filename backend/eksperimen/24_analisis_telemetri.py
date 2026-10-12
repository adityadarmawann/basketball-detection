"""
24_analisis_telemetri.py

Membaca rekaman 23_telemetri_gerbang.py lalu menjawab pertanyaan yang
diperlukan untuk memilih ambang saklar B dan saklar C DARI DATA:

  1. Setiap rentetan dwell yang pernah MENCAPAI ambang: panjangnya, berapa
     piksel bola turun sepanjang rentetan, jarak terdekat ke pusat kotak rim,
     dan apakah berujung kejadian.
  2. Di sekitar tiap bola masuk menurut papan skor: frame NYATA yang paling
     dalam di cincin, beserta posisi tegaknya relatif pusat dan tepi bawah
     kotak rim. Ini yang menentukan apakah saklar C bisa meloloskannya.
  3. Berapa banyak frame di SELURUH klip yang akan memenuhi syarat inti cincin
     saklar C untuk beberapa pilihan pecahan. Ini ukuran paparan palsunya.

Tidak memakai GPU. Tidak memutuskan apa pun; yang memutuskan skrip 17.

Contoh:
    python eksperimen/24_analisis_telemetri.py
"""

import json
import math
import os
import sys

BERKAS = os.getenv("TELEMETRI_OUT", "/home/paul/telemetri_gerbang.json")
TRUTH = [19, 59, 90, 119, 146]


def jam(t):
    return f"{int(t)//60}:{t%60:04.1f}"


def pusat_rim(r):
    return ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)


def main():
    with open(BERKAS) as f:
        data = json.load(f)
    meta, fr, kej = data["meta"], data["frame"], data["kejadian"]
    ambang = meta["dwell_ambang"]
    print(f"  meta: {meta}")
    print(f"  {len(fr)} frame gerbang, {len(kej)} kejadian skor\n")

    # ── 1. Rentetan dwell ───────────────────────────────────────────────
    print(f"  ═══ RENTETAN DWELL yang mencapai ambang {ambang} ═══")
    print("     waktu    panjang  turun_px  d_rim_min  nyata_di_inti  kejadian")
    rentet, sekarang = [], []
    for r in fr:
        if r["d1"] > r["d0"] or (r["d1"] > 0 and sekarang):
            sekarang.append(r)
            if r["d1"] == 0:
                rentet.append(sekarang)
                sekarang = []
        elif sekarang:
            rentet.append(sekarang)
            sekarang = [] if r["d1"] == 0 else [r]
    if sekarang:
        rentet.append(sekarang)

    panjang_kej, panjang_tanpa = [], []
    for s in rentet:
        maks = max(x["d1"] for x in s)
        if maks < ambang:
            continue
        ys = [x["b"][1] for x in s if x["b"]]
        turun = (max(ys) - ys[0]) if ys else 0.0
        drims = []
        inti_min = None
        for x in s:
            if x["b"] and x["rim"]:
                cx, cy = pusat_rim(x["rim"])
                d = math.dist(x["b"], (cx, cy))
                drims.append(d)
                if x["nyata"] and x["b"][1] >= cy:
                    dx = abs(x["b"][0] - cx)
                    half = (x["rim"][2] - x["rim"][0]) / 2.0
                    f = dx / half if half else 9.9
                    inti_min = f if inti_min is None else min(inti_min, f)
        ada_kej = sum(x["kej"] for x in s)
        (panjang_kej if ada_kej else panjang_tanpa).append(maks)
        print(f"     {jam(s[0]['t']):>7s}  {maks:7d}  {turun:8.1f}  "
              f"{(min(drims) if drims else -1):9.1f}  "
              f"{('%.2f' % inti_min) if inti_min is not None else '    -':>13s}  "
              f"{'YA ' + str(ada_kej) if ada_kej else 'tidak':>8s}")

    print(f"\n     panjang dwell saat BERUJUNG kejadian : {sorted(panjang_kej)}")
    print(f"     panjang dwell TANPA kejadian         : {sorted(panjang_tanpa)}")

    # ── 2. Sekitar tiap bola masuk menurut papan skor ───────────────────
    print(f"\n  ═══ FRAME NYATA PALING DALAM di sekitar tiap bola masuk ═══")
    print("     Kolom: dx/halfw = jarak mendatar dibagi separuh lebar kotak rim,")
    print("            dy_pusat = y bola dikurangi y pusat rim (positif = di bawah),")
    print("            dy_bawah = y bola dikurangi tepi bawah kotak rim.")
    for tt in TRUTH:
        jend = [r for r in fr if abs(r["t"] - tt) <= 8 and r["b"] and r["rim"]]
        nyata = [r for r in jend if r["nyata"]]
        if not nyata:
            print(f"     {tt//60}:{tt%60:02d}  tidak ada frame nyata dengan kotak rim")
            continue
        def kedalaman(r):
            cx, cy = pusat_rim(r["rim"])
            return math.dist(r["b"], (cx, cy))
        nyata.sort(key=kedalaman)
        print(f"     ── bola masuk {tt//60}:{tt%60:02d} "
              f"({len(jend)} frame, {len(nyata)} nyata) ──")
        for r in nyata[:6]:
            cx, cy = pusat_rim(r["rim"])
            half = (r["rim"][2] - r["rim"][0]) / 2.0
            print(f"        {jam(r['t']):>7s} d_rim={kedalaman(r):5.1f} "
                  f"dx/halfw={abs(r['b'][0]-cx)/half if half else 9.9:5.2f} "
                  f"dy_pusat={r['b'][1]-cy:6.1f} dy_bawah={r['b'][1]-r['rim'][3]:6.1f} "
                  f"dwell={r['d0']}->{r['d1']}")

    # ── 3. Paparan syarat inti cincin di SELURUH klip ───────────────────
    print(f"\n  ═══ PAPARAN syarat INTI CINCIN di seluruh klip ═══")
    print("     pecahan  frame_nyata_lolos  dari_total_nyata_dgn_rim")
    tot = [r for r in fr if r["nyata"] and r["b"] and r["rim"]]
    for frac in (0.15, 0.25, 0.35, 0.5, 0.75, 1.0):
        n = 0
        for r in tot:
            cx, cy = pusat_rim(r["rim"])
            half = (r["rim"][2] - r["rim"][0]) / 2.0
            if half and abs(r["b"][0] - cx) <= frac * half and r["b"][1] >= cy:
                n += 1
        print(f"     {frac:7.2f}  {n:17d}  {len(tot):24d}")

    # ── 4. Ukuran kotak rim ─────────────────────────────────────────────
    lebar = sorted((r["rim"][2] - r["rim"][0]) for r in fr if r["rim"])
    tinggi = sorted((r["rim"][3] - r["rim"][1]) for r in fr if r["rim"])
    if lebar:
        print(f"\n     kotak rim: lebar median {lebar[len(lebar)//2]:.1f} px, "
              f"tinggi median {tinggi[len(tinggi)//2]:.1f} px, n={len(lebar)}")
    print(f"     kejadian skor: " +
          ", ".join(f"{jam(k['t'])}{'/3' if k['tiga'] else '/2'}" for k in kej))


if __name__ == "__main__":
    main()

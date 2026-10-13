"""
31_simulasi_saring_lapangan.py

Mengukur dampak penyaring FILTER_LUAR_LAPANGAN TANPA menjalankan pipeline,
dengan mensimulasikannya pada data frame yang sudah ada.

Kenapa bisa: data frame yang disimpan sistem sudah memuat `cp`, yaitu posisi
lapangan tiap pemain, dan `cp` bernilai kosong persis ketika `pixel_to_court`
menolak titik itu karena di luar batas 2 meter dari garis lapangan. Jadi
keputusan yang akan diambil penyaring sudah terekam di data.

Yang diukur:
  1. Berapa deteksi yang dibuang.
  2. Berapa track_id yang HILANG SELURUHNYA, yaitu yang tidak pernah sekali pun
     berada di dalam lapangan. Inilah yang paling berharga, karena track seperti
     itu murni penonton atau panitia dan selama ini ikut jadi entri statistik.
  3. Berapa track yang justru TERPOTONG, yaitu pemain sungguhan yang kadang
     keluar batas. Ini risikonya, dan harus kecil.

Contoh:
    python eksperimen/31_simulasi_saring_lapangan.py /tmp/frames_prod.json
"""

import json
import sys
from collections import defaultdict


def main(path):
    d = json.load(open(path))
    total = dibuang = 0
    per_track_dalam = defaultdict(int)
    per_track_luar = defaultdict(int)
    frame_dilewati = 0

    for e in d:
        pl = e.get("p") or []
        if not pl:
            continue
        di_dalam = [p for p in pl if p.get("cp")]
        total += len(pl)
        # Pengaman: kalau tidak satu pun terpetakan, penyaring TIDAK bekerja.
        if not di_dalam:
            frame_dilewati += 1
            for p in pl:
                per_track_luar[p.get("i")] += 0   # tidak dihitung sebagai luar
            continue
        dibuang += len(pl) - len(di_dalam)
        for p in pl:
            (per_track_dalam if p.get("cp") else per_track_luar)[p.get("i")] += 1

    semua = set(per_track_dalam) | set(per_track_luar)
    hantu = [t for t in semua if per_track_dalam.get(t, 0) == 0]
    terpotong = [t for t in semua
                 if per_track_dalam.get(t, 0) > 0 and per_track_luar.get(t, 0) > 0]

    print(f"\n  === {len(d)} frame, {total} deteksi pemain ===")
    print(f"  dibuang            : {dibuang} ({dibuang/max(total,1)*100:.1f}%)")
    print(f"  frame dilewati     : {frame_dilewati} "
          f"(tidak terkalibrasi, penyaring sengaja tidak bekerja)")
    print(f"\n  track_id seluruhnya: {len(semua)}")
    print(f"    HILANG total     : {len(hantu)} "
          f"({len(hantu)/max(len(semua),1)*100:.0f}%)  <- murni bukan pemain")
    print(f"    terpotong sebagian: {len(terpotong)} "
          f"({len(terpotong)/max(len(semua),1)*100:.0f}%)  <- pemain yang kadang keluar")
    print(f"    utuh              : {len(semua)-len(hantu)-len(terpotong)}")

    if terpotong:
        rasio = sorted(per_track_luar[t] / (per_track_dalam[t] + per_track_luar[t])
                       for t in terpotong)
        tengah = rasio[len(rasio) // 2]
        print(f"\n  pada track yang terpotong, bagian yang dibuang: "
              f"median {tengah*100:.0f}% dari hidupnya")
        print(f"  (kalau angka ini kecil, pemain aslinya hampir tidak terganggu)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/frames_prod.json")

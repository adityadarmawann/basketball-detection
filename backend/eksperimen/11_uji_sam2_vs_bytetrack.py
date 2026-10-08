"""
Uji nyata: SAM2 lawan ByteTrack di frame yang sama persis.

Detektor dijalankan SEKALI per frame, keluarannya dipakai kedua pelacak, jadi
perbandingannya adil. Sudut kamera yang berpindah ditangani dengan mendaftarkan
ulang objek ke SAM2, memakai sinyal camera_cut dari court.py.

Yang dibandingkan:
    jumlah track_id unik        makin sedikit makin baik (lebih sedikit pecah)
    umur track                  makin panjang makin baik
    langkah mustahil            lompatan di atas 36 km/jam, tanda identitas tertukar

Pemakaian:
    python3 -u eksperimen/11_uji_sam2_vs_bytetrack.py <video> <n_frame> [mulai]
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

STRIDE = int(os.getenv("PROCESS_STRIDE", "2"))
COURT_EVERY = int(os.getenv("COURT_EVERY_N", "3"))
MAX_OBJ = int(os.getenv("SAM2_MAX_OBJ", "16"))


def buat_sam2():
    from ultralytics.models.sam import SAM2DynamicInteractivePredictor
    from ultralytics.utils.checks import check_imgsz
    p = SAM2DynamicInteractivePredictor(
        overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024,
                       model=os.getenv("SAM2_MODEL", "sam2_t.pt"),
                       save=False, verbose=False),
        max_obj_num=MAX_OBJ)
    p.setup_model(model=None, verbose=False)
    p.imgsz = check_imgsz(p.args.imgsz, stride=32, min_dim=2)
    return p


def main() -> int:
    video = sys.argv[1]
    n_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    mulai = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    from pipeline.detector import BasketballDetector
    from pipeline.tracker import PlayerTracker
    from pipeline.court import CourtMapper

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if mulai:
        cap.set(cv2.CAP_PROP_POS_FRAMES, mulai)
    print(f"{os.path.basename(video)}  fps {fps:.0f}  "
          f"{n_frame} frame dari {mulai}  ({n_frame/fps:.0f} detik video)")

    det = BasketballDetector(); det.load_model()
    trk = PlayerTracker(frame_rate=int(round(fps / STRIDE)))
    crt = CourtMapper(); crt.load_model()
    sam = buat_sam2()
    print("model siap\n")

    hasil_bt, hasil_s2 = [], []
    n_cut, n_daftar, sam_siap, sid = 0, 0, False, 0
    tunda, kurang, n_pegang = 0, 0, 0
    TUNDA_DAFTAR = int(os.getenv("SAM2_TUNDA_DAFTAR", "3"))
    t_bt, t_s2 = [], []
    fid, court = mulai, None
    t_awal = time.time()

    def siap(fr):
        sam.batch = [None, [fr]]
        return sam.preprocess([fr])

    with torch.autocast("cuda", dtype=torch.bfloat16):
        while fid < mulai + n_frame:
            ok, fr = cap.read()
            if not ok:
                break
            if (fid - mulai) % STRIDE != 0:
                fid += 1
                continue
            d = det.detect(fr)
            kotak = [p["bbox"] for p in d["players"]][:MAX_OBJ]

            # ── ByteTrack ──
            t0 = time.time()
            tr = trk.update(fr, d)
            t_bt.append(time.time() - t0)
            for p in tr.get("tracked_players", []):
                c = p.get("center")
                if c:
                    hasil_bt.append((fid, int(p["track_id"]), c[0], c[1]))

            # ── sinyal ganti sudut ──
            if fid % COURT_EVERY == 0:
                court = crt.process_frame(fr)
            cut = bool(court and court.get("camera_cut"))
            if cut:
                n_cut += 1

            # ── SAM2, daftar ulang saat ganti sudut atau belum siap ──
            # PENTING: pendaftaran JANGAN di frame ganti sudut itu sendiri.
            # Di frame itu gambar paling kabur dan kamera sedang bergerak, jadi
            # detektor cuma menemukan sedikit pemain, dan SAM2 akan memegang
            # sedikit objek saja sampai ganti sudut berikutnya. Versi pertama
            # uji ini melakukan itu dan hasilnya cuma 2,4 pemain per frame,
            # padahal detektor menemukan 8,8. Jadi pendaftaran DITUNDA beberapa
            # frame sampai tampilan tenang.
            t0 = time.time()
            if cut:
                tunda = TUNDA_DAFTAR
                sam_siap = False
            if tunda > 0:
                tunda -= 1
                perlu = (tunda == 0)
            else:
                perlu = not sam_siap
            # Tambal juga kalau objek yang terpegang jauh lebih sedikit daripada
            # yang terdeteksi, tanda banyak pemain baru masuk layar.
            if sam_siap and kotak and n_pegang < len(kotak) * 0.6:
                kurang += 1
                if kurang >= 5:
                    perlu, kurang = True, 0
            else:
                kurang = 0
            if perlu and kotak:
                sam = buat_sam2()          # memori bersih untuk objek baru
                ids = list(range(len(kotak)))
                sam.inference(siap(fr), bboxes=kotak, obj_ids=ids,
                              update_memory=True)
                peta = {i: sid + i for i in ids}     # id global unik
                sid += len(ids)
                sam_siap = True
                n_daftar += 1
            elif sam_siap:
                try:
                    m, _ = sam.inference(siap(fr))
                    mask = (m > 0.0).cpu().numpy()
                    n_pegang = 0
                    for i in range(mask.shape[0]):
                        ys, xs = np.nonzero(mask[i])
                        if len(xs) < 50:
                            continue
                        n_pegang += 1
                        H, W = mask.shape[-2:]
                        sx, sy = fr.shape[1] / W, fr.shape[0] / H
                        hasil_s2.append((fid, peta.get(i, -1),
                                         float(xs.mean() * sx),
                                         float(ys.mean() * sy)))
                except Exception as e:
                    sam_siap = False
            t_s2.append(time.time() - t0)
            fid += 1
            if (fid - mulai) % 200 == 0:
                print(f"  frame {fid-mulai}/{n_frame}  "
                      f"{(time.time()-t_awal):.0f}s  ganti sudut {n_cut}", flush=True)
    cap.release()

    print(f"\nselesai {(time.time()-t_awal)/60:.1f} menit, "
          f"ganti sudut {n_cut}, pendaftaran SAM2 {n_daftar}")
    print(f"waktu per frame: ByteTrack {np.median(t_bt)*1000:.0f} ms, "
          f"SAM2 {np.median(t_s2)*1000:.0f} ms\n")

    dt = STRIDE / fps
    for nama, h in (("ByteTrack", hasil_bt), ("SAM2", hasil_s2)):
        if not h:
            print(f"{nama}: tidak ada hasil"); continue
        a = np.array(h, float)
        f_, t_, x_, y_ = a[:, 0], a[:, 1].astype(int), a[:, 2], a[:, 3]
        u, c = np.unique(t_, return_counts=True)
        o = np.lexsort((f_, t_))
        tt, ff, xx, yy = t_[o], f_[o], x_[o], y_[o]
        ok = (tt[1:] == tt[:-1]) & ((ff[1:] - ff[:-1]) <= STRIDE * 2)
        lang = np.hypot(xx[1:] - xx[:-1], yy[1:] - yy[:-1])[ok]      # piksel
        print(f"{nama:10s} track_id unik {len(u):5d}   "
              f"umur median {np.median(c)*dt:5.2f} detik   "
              f"umur p90 {np.percentile(c,90)*dt:5.2f} detik   "
              f"sampel {len(a)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

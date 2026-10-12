"""
Verifikasi pipeline PENUH: pastikan perubahan saklar tidak mematikan deteksi.

Menjalankan VideoProcessor._process_frame yang asli, jadi seluruh rantai ikut
diuji: detektor, pelacak, homografi, pose, OCR nomor, klasifikasi tim,
klasifikasi aksi, mesin kejadian, dan statistik. Tanpa Mongo, Redis, atau
WebSocket.

Yang dihitung dan dibandingkan:
    pemain terdeteksi, bola, wasit, ring
    frame terkalibrasi (homografi hidup atau mati)
    pose terdeteksi
    nomor punggung TERKONFIRMASI  <- paling rawan mati oleh NUMBER_VALIDATOR
    tim A/B terpasang             <- paling rawan mati oleh TEAM_METHOD
    aksi terklasifikasi
    kejadian terbit
    jarak dan percepatan statistik

Pemakaian:
    python3 -u eksperimen/04_verifikasi_penuh.py <video> <n_frame> <label>
Setelan saklarnya diambil dari env, jadi pemanggil yang mengatur.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

# VERIF_LOG=1 menyalakan log INFO, supaya kejadian seperti ganti sudut kamera
# dan pelepasan kuncian nomor benar benar terlihat, bukan cuma diasumsikan.
JENIS_KEJADIAN: dict = {}

if os.getenv("VERIF_LOG") == "1":
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")


def main() -> int:
    video = sys.argv[1]
    n_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    label = sys.argv[3] if len(sys.argv) > 3 else "?"

    from pipeline.video_processor import VideoProcessor, PROCESS_STRIDE
    import pipeline.stats_calculator as S, pipeline.court as C
    import pipeline.jersey_ocr as J, pipeline.detector as D
    import pipeline.action as A, pipeline.event_engine as E

    saklar = {
        "PATH_CLEAN_METHOD": S.PATH_CLEAN_METHOD,
        "KEYPOINT_SMOOTH_METHOD": C.KEYPOINT_SMOOTH_METHOD,
        "NUMBER_VALIDATOR": J.NUMBER_VALIDATOR,
        "TEAM_METHOD": J.TEAM_METHOD,
        "DETECTOR_METHOD": D.DETECTOR_METHOD,
        "ACTION_SOURCE": A.ACTION_SOURCE,
        "SHOT_METHOD": E.SHOT_METHOD,
    }
    print(f"=== {label} ===")
    for k, v in saklar.items():
        print(f"  {k:24s} = {v}")

    vp = VideoProcessor()
    vp._init_pipeline(vp._models_path)
    komp = {n: getattr(vp, f"_{n}") is not None for n in
            ("detector", "tracker", "court", "pose", "jersey", "action")}
    komp["events"] = vp._events_engine is not None
    komp["stats"] = vp._stats is not None
    print("  komponen termuat        = " +
          ", ".join(f"{k}:{'ya' if v else 'TIDAK'}" for k, v in komp.items()))
    mati = [k for k, v in komp.items() if not v]
    if mati:
        print(f"  PERINGATAN komponen mati: {mati}")

    # Roster penting: jersey_ocr menyaring nomor yang TIDAK ada di daftar
    # pemain. Menguji tanpa roster membuat nomor palsu ikut lolos, jadi
    # perbandingan presisinya tidak mencerminkan keadaan produksi.
    roster_file = os.getenv("VERIF_ROSTER")
    if roster_file and os.path.exists(roster_file):
        with open(roster_file) as f:
            vp._roster = json.load(f)
        print(f"  roster dimuat           = {len(vp._roster)} entri")
    else:
        vp._roster = {}
        print("  roster                  = KOSONG (tanpa penyaringan)")
    vp._current_quarter = 1
    vp._lock_quarter = True

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ── Samakan DASAR WAKTU dengan produksi, ini WAJIB ───────────────────────
    # Skrip ini memanggil _process_frame LANGSUNG, jadi koreksi fps yang ada di
    # process_video TIDAK PERNAH jalan. Pelangkahan frame memang sudah benar
    # (PROCESS_STRIDE), TAPI StatsCalculator tetap memakai TARGET_FPS=25 bawaan.
    # Akibatnya jeda SEBENARNYA 33,3 ms (2 frame dari 60 fps) sementara rumus
    # memakai 1/fps = 40 ms, meleset 1,2 kali. Batas langkah jalur bersih juga
    # ikut longgar, 0,400 m lawan 0,333 m yang benar.
    # Produksi menyetel fps efektif 30 sehingga 1/fps = 33,3 ms dan COCOK.
    try:
        from pipeline.video_processor import PROCESS_STRIDE as _STRIDE
    except Exception:
        _STRIDE = 2
    _eff_fps = max(1, round(fps / _STRIDE))
    vp._source_fps = fps
    vp._target_fps = _eff_fps
    if getattr(vp, "_stats", None) is not None:
        vp._stats.fps = _eff_fps
    if getattr(vp, "_events_engine", None) is not None and hasattr(vp._events_engine, "set_fps"):
        vp._events_engine.set_fps(_eff_fps)
    if getattr(vp, "_tracker", None) is not None and hasattr(vp._tracker, "frame_rate"):
        vp._tracker.frame_rate = _eff_fps
    print(f"  dasar waktu             = langkah {_STRIDE}, fps efektif {_eff_fps}")
    # VERIF_START memungkinkan menguji SEGMEN tertentu, misalnya bagian yang
    # memuat time out, tanpa memproses seluruh video dari awal.
    mulai = int(os.getenv("VERIF_START", "0"))
    if mulai:
        cap.set(cv2.CAP_PROP_POS_FRAMES, mulai)
        print(f"  mulai dari frame        = {mulai} ({mulai/fps:.0f} detik)")

    c = dict(pemain=0, bola=0, wasit=0, ring=0, kalib=0, pose=0,
             nomor=0, tim=0, aksi=0, kejadian=0, frame=0, galat=0)
    nomor_unik, tim_unik, tid_unik = set(), set(), set()
    t0 = time.time()
    fid = mulai
    while fid < mulai + n_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if fid % PROCESS_STRIDE != 0:
            fid += 1
            continue
        try:
            fd = vp._process_frame(frame, fid, int(fid * 1000 / fps))
        except Exception as e:
            c["galat"] += 1
            if c["galat"] <= 3:
                print(f"  GALAT frame {fid}: {type(e).__name__}: {e}")
            fid += 1
            continue
        c["frame"] += 1
        d = fd.get("detections", {}) or {}
        tr = fd.get("tracking", {}) or {}
        pl = tr.get("tracked_players", []) or []
        c["pemain"] += len(pl)
        c["bola"] += 1 if tr.get("tracked_ball") else 0
        c["wasit"] += len(tr.get("tracked_referees", []) or [])
        c["ring"] += len(d.get("hoops", []) or [])
        if (fd.get("court", {}) or {}).get("is_calibrated"):
            c["kalib"] += 1
        c["pose"] += len((fd.get("pose", {}) or {}).get("poses", []) or [])
        for jr in (fd.get("jersey", {}) or {}).get("jersey_results", []) or []:
            if jr.get("jersey_number") is not None:
                c["nomor"] += 1
                nomor_unik.add(jr["jersey_number"])
            if jr.get("team") in ("A", "B"):
                c["tim"] += 1
                tim_unik.add(jr["team"])
        c["aksi"] += len((fd.get("actions", {}) or {}).get("actions", []) or [])
        _ev = fd.get("events", []) or []
        c["kejadian"] += len(_ev)
        # Rincian JENIS kejadian. Cacah total saja menyembunyikan sebab: saat
        # DIGIT_CLS dinyalakan cacahnya turun 35 jadi 27, dan tanpa rincian
        # tidak bisa dibedakan apakah yang hilang kejadian palsu yang terkoreksi
        # atau kejadian sah yang ikut tertahan.
        for _e in _ev:
            _j = str(_e.get("type") or _e.get("jenis") or "TANPA_JENIS").upper()
            JENIS_KEJADIAN[_j] = JENIS_KEJADIAN.get(_j, 0) + 1
        for p in pl:
            tid_unik.add(p["track_id"])
        fid += 1
    cap.release()
    lama = time.time() - t0

    print(f"\n  {'frame diproses':26s} {c['frame']}")
    print(f"  {'galat frame':26s} {c['galat']}")
    print(f"  {'waktu':26s} {lama:.1f}s  ({c['frame']/max(lama,1e-9):.2f} frame/detik)")
    print(f"  {'pemain terdeteksi':26s} {c['pemain']}  ({c['pemain']/max(c['frame'],1):.1f} per frame)")
    print(f"  {'bola terdeteksi':26s} {c['bola']}/{c['frame']} frame")
    print(f"  {'wasit terdeteksi':26s} {c['wasit']}")
    print(f"  {'ring terdeteksi':26s} {c['ring']}")
    print(f"  {'frame TERKALIBRASI':26s} {c['kalib']}/{c['frame']}  "
          f"({c['kalib']/max(c['frame'],1)*100:.0f}%)")
    print(f"  {'pose terdeteksi':26s} {c['pose']}")
    print(f"  {'nomor TERKONFIRMASI':26s} {c['nomor']} bacaan, nomor unik {sorted(nomor_unik)}")
    print(f"  {'tim terpasang':26s} {c['tim']} bacaan, tim unik {sorted(tim_unik)}")
    print(f"  {'aksi terklasifikasi':26s} {c['aksi']}")
    print(f"  {'kejadian terbit':26s} {c['kejadian']}")
    if JENIS_KEJADIAN:
        for _j, _n in sorted(JENIS_KEJADIAN.items(), key=lambda x: -x[1]):
            print(f"    {_j:24s} {_n}")
    print(f"  {'track_id unik':26s} {len(tid_unik)}")

    if vp._stats:
        tids = [t for t in vp._stats._all_tids
                if vp._stats._playing_frames.get(t, 0) >= 20]
        if tids:
            jar = [vp._stats._calc_mpi_raw(t) for t in tids]
            d_tot = sum(x["distance_m"] for x in jar)
            acc = [x["acceleration_ms2"] for x in jar]
            print(f"  {'statistik: track >=20f':26s} {len(tids)}")
            print(f"  {'statistik: total jarak':26s} {d_tot:.1f} m")
            print(f"  {'statistik: acc median':26s} {np.median(acc):.1f} m/s2")
            print(f"  {'statistik: acc mustahil':26s} "
                  f"{sum(1 for a in acc if a > 10)}/{len(acc)}")
        else:
            print(f"  statistik: belum ada track yang cukup panjang")

    hasil = {"label": label, "saklar": saklar, "cacah": c,
             "nomor_unik": sorted(nomor_unik), "tim_unik": sorted(tim_unik),
             "track_unik": len(tid_unik), "detik": round(lama, 1)}
    # Rincian jenis kejadian ikut disimpan supaya bisa disilangkan antar
    # percobaan tanpa menjalankan ulang pipeline.
    hasil["jenis_kejadian"] = dict(JENIS_KEJADIAN)

    out = os.getenv("VERIF_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(hasil, f, indent=2)
        print(f"\n  tersimpan: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

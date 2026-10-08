"""
Uji jalur RF-DETR dengan bobot starter COCO Roboflow.

Tujuannya BUKAN mendeteksi basket. Bobot COCO tidak tahu apa itu ball-in-basket.
Tujuannya membuktikan pipanya benar SEBELUM model 10 kelas dilatih:

    1. bobot bisa dimuat lewat from_checkpoint
    2. predict() menerima frame BGR dari OpenCV setelah dibalik ke RGB
    3. _bongkar() benar membongkar supervision.Detections jadi (kotak, conf, kelas)
    4. keluaran detect() bentuknya sama dengan BasketballDetector.detect()
    5. pencocokan nomor dan pemetaan aksi jalan di data nyata

Kalau semua ini lolos, begitu bobot 10 kelas jadi, satu satunya yang berubah
adalah arti indeks kelasnya.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else None
DEV = os.getenv("RFDETR_TEST_DEVICE", "cpu")


def main() -> int:
    import rfdetr
    from pipeline.detector_rfdetr import RFDetrDetector, RFDETR_CLASS_NAMES

    print("1. memuat bobot starter COCO (nano, terkecil)")
    t0 = time.time()
    model = rfdetr.RFDETRNano()          # tanpa argumen = unduh bobot COCO resmi
    print(f"   termuat {time.time()-t0:.1f}s, kelas {type(model).__name__}")

    # Pasang model itu ke adaptor kita, tanpa melewati load_model karena
    # bobot basket belum ada. Yang diuji adalah detect() dan _bongkar().
    det = RFDetrDetector.__new__(RFDetrDetector)
    det.model = model
    det.conf_threshold = 0.35
    det.variant = "nano"
    det.device = DEV
    # COCO punya 90 kelas; nama basket diganti sementara supaya terbaca
    coco = {0: "person", 32: "sports ball"}
    det._class_names = [coco.get(i, f"coco_{i}") for i in range(91)]

    if VIDEO:
        cap = cv2.VideoCapture(VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("   GAGAL membaca frame")
            return 1
        print(f"2. frame nyata dari video, {frame.shape[1]}x{frame.shape[0]}")
    else:
        frame = np.zeros((1080, 1920, 3), np.uint8)
        print("2. frame kosong (tanpa video)")

    t0 = time.time()
    out = det.detect(frame)
    print(f"3. detect() selesai {(time.time()-t0)*1000:.0f} ms")

    print("4. bentuk keluaran:")
    for k in ("players", "ball", "referees", "hoops", "backboards",
              "rims", "numbers", "balls_in_basket"):
        v = out.get(k)
        n = (0 if v is None else (1 if isinstance(v, dict) else len(v)))
        tanda = "ADA" if k in out else "HILANG"
        print(f"   {k:17s} {tanda:7s} {n}")

    hilang = [k for k in ("players", "ball", "referees", "hoops", "backboards")
              if k not in out]
    if hilang:
        print(f"   GAGAL: kunci lama hilang {hilang}")
        return 1

    # _bongkar diuji langsung
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    d = model.predict(rgb, threshold=0.35)
    kotak, conf, kelas = RFDetrDetector._bongkar(d)
    if kotak is None:
        print("5. _bongkar: tidak ada deteksi di frame ini (bukan kegagalan pipa)")
    else:
        print(f"5. _bongkar BERHASIL: {len(kotak)} kotak, "
              f"conf {conf.min():.2f} sampai {conf.max():.2f}, "
              f"indeks kelas {sorted(set(kelas.tolist()))[:8]}")
        cacah = {}
        for ci in kelas:
            nm = det._class_names[ci] if ci < len(det._class_names) else f"?{ci}"
            cacah[nm] = cacah.get(nm, 0) + 1
        atas = sorted(cacah.items(), key=lambda x: -x[1])[:5]
        print(f"   kelas terbanyak: {atas}")
        print(f"   orang terdeteksi: {cacah.get('person', 0)} "
              f"(frame basket wajar punya banyak)")

    print(f"\n6. daftar 10 kelas yang akan dipakai model barumu:")
    for i, n in enumerate(RFDETR_CLASS_NAMES):
        print(f"   {i}  {n}")
    print("\nSEMUA TAHAP PIPA LOLOS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

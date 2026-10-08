"""
Klasifikasi tim A/B dengan sematan SigLIP, cara Roboflow.

Dipakai sebagai ALTERNATIF metode sumbu LAB yang sudah ada di jersey_ocr.py,
bukan pengganti. Saklarnya TEAM_METHOD:

    TEAM_METHOD=lab      (bawaan) sumbu LAB, sudah terukur di campus league
    TEAM_METHOD=siglip   sematan SigLIP lalu pengelompokan 2 kelompok

Cara kerja (sama seperti roboflow/sports TeamClassifier):
    1. potongan gambar pemain dilewatkan ke menara penglihatan SigLIP,
       keluarannya vektor 768 dimensi per potongan
    2. dimensi diturunkan ke 3 dengan PCA, supaya pengelompokan tidak terseret
       dimensi yang isinya derau
    3. KMeans dua kelompok memisahkan dua seragam

Bedanya dengan Roboflow:
    - Roboflow memakai UMAP; di sini PCA dipakai lebih dulu karena umap-learn
      belum terpasang. Pasang umap-learn lalu setel TEAM_SIGLIP_REDUCER=umap
      kalau mau sama persis.
    - Roboflow melatih sekali di awal dari kumpulan potongan satu video.
      Di sini perilakunya sama: fit() dipanggil sekali saat kalibrasi, lalu
      predict() untuk frame berikutnya.

Catatan penting soal label: KMeans tidak tahu mana A mana B. Penamaan diambil
dari potongan kalibrasi yang sudah berlabel (dari pemilihan tim yang ada), jadi
A/B tidak pernah tertukar di tengah pertandingan.

Perangkat bawaan CPU. Perangkat ini sering dipakai melatih model lain, jadi
GPU hanya diambil kalau TEAM_SIGLIP_DEVICE=cuda disetel dengan sengaja.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("TEAM_SIGLIP_MODEL", "google/siglip-base-patch16-224")
DEVICE   = os.getenv("TEAM_SIGLIP_DEVICE", "cpu").strip().lower()
BATCH    = int(os.getenv("TEAM_SIGLIP_BATCH", "16"))
REDUCER  = os.getenv("TEAM_SIGLIP_REDUCER", "pca").strip().lower()
N_COMPS  = int(os.getenv("TEAM_SIGLIP_COMPONENTS", "3"))
MIN_FIT  = int(os.getenv("TEAM_SIGLIP_MIN_FIT", "24"))


class TeamClassifierSigLIP:
    """Pengelompokan dua tim dari sematan SigLIP.

    Pemakaian:
        tc = TeamClassifierSigLIP()
        tc.fit(crops, labels)        # labels opsional: list "A"/"B"/None
        tc.predict_one(crop)         # -> "A" | "B" | None
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._reducer = None
        self._kmeans = None
        self._cluster_to_team: dict[int, str] = {}
        self._siap = False

    # ── pemuatan model ────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Muat menara penglihatan SigLIP. False kalau gagal, pemanggil wajib
        jatuh kembali ke metode LAB."""
        if self._model is not None:
            return True
        try:
            import torch
            # Hanya pengolah GAMBAR yang dipakai. AutoProcessor ikut menarik
            # tokenizer teks SigLIP yang butuh sentencepiece, padahal menara
            # teksnya tidak dipakai di sini.
            from transformers import AutoImageProcessor, SiglipVisionModel
        except Exception as e:
            logger.warning("SigLIP butuh transformers dan torch (%s)", e)
            return False
        try:
            self._torch = torch
            self._processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            model = SiglipVisionModel.from_pretrained(MODEL_ID)
            model.eval()
            dev = DEVICE
            if dev.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("TEAM_SIGLIP_DEVICE=cuda tapi CUDA tidak ada; pakai CPU")
                dev = "cpu"
            self._device = dev
            self._model = model.to(dev)
            logger.info("SigLIP %s dimuat di %s", MODEL_ID, dev)
            return True
        except Exception as e:
            logger.warning("Gagal memuat SigLIP %s (%s)", MODEL_ID, e)
            self._model = None
            return False

    # ── sematan ───────────────────────────────────────────────────────────────

    def embed(self, crops: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        """(N, 768) sematan dari potongan BGR. None kalau model tidak bisa dimuat."""
        if not self.load():
            return None
        crops = [c for c in crops if c is not None and c.size > 0]
        if not crops:
            return None
        torch = self._torch
        keluar = []
        for i in range(0, len(crops), BATCH):
            potong = crops[i : i + BATCH]
            # OpenCV memberi BGR, SigLIP menunggu RGB. .copy() wajib: irisan
            # langkah negatif tidak bisa dijadikan tensor torch.
            rgb = [np.ascontiguousarray(c[:, :, ::-1]) for c in potong]
            try:
                inp = self._processor(images=rgb, return_tensors="pt")
            except Exception as e:
                logger.warning("Prapemrosesan SigLIP gagal (%s)", e)
                return None
            inp = {k: v.to(self._device) for k, v in inp.items()}
            with torch.no_grad():
                out = self._model(**inp)
            # pooler_output kalau ada, kalau tidak rata rata token
            vek = getattr(out, "pooler_output", None)
            if vek is None:
                vek = out.last_hidden_state.mean(dim=1)
            keluar.append(vek.float().cpu().numpy())
        return np.concatenate(keluar, axis=0)

    # ── pelatihan sekali di awal ──────────────────────────────────────────────

    def fit(
        self,
        crops: Sequence[np.ndarray],
        labels: Optional[Sequence[Optional[str]]] = None,
    ) -> bool:
        """Latih peredam dimensi dan KMeans dari potongan kalibrasi.

        labels sejajar dengan crops, isinya "A", "B", atau None. Dipakai HANYA
        untuk menamai kelompok, bukan untuk melatih. Tanpa label, kelompok 0
        dinamai A dan kelompok 1 dinamai B, jadi A/B bisa tertukar.
        """
        if len(crops) < MIN_FIT:
            logger.info("SigLIP fit ditunda: %d potongan, minimal %d",
                        len(crops), MIN_FIT)
            return False
        X = self.embed(crops)
        if X is None or len(X) < MIN_FIT:
            return False
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except Exception as e:
            logger.warning("SigLIP butuh scikit-learn (%s)", e)
            return False

        if REDUCER == "umap":
            try:
                import umap
                self._reducer = umap.UMAP(n_components=N_COMPS, random_state=0)
            except Exception as e:
                logger.info("umap-learn tidak ada (%s); pakai PCA", e)
                self._reducer = PCA(n_components=min(N_COMPS, X.shape[1]))
        else:
            self._reducer = PCA(n_components=min(N_COMPS, X.shape[1]))

        try:
            Z = self._reducer.fit_transform(X)
            self._kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(Z)
        except Exception as e:
            logger.warning("Pengelompokan SigLIP gagal (%s)", e)
            return False

        self._cluster_to_team = self._namai(self._kmeans.labels_, labels)
        self._siap = True
        logger.info("SigLIP siap: kelompok %s", self._cluster_to_team)
        return True

    @staticmethod
    def _namai(kelompok: np.ndarray, labels) -> dict[int, str]:
        """Pasangkan nomor kelompok ke "A"/"B" lewat suara terbanyak label asli."""
        if labels is None:
            return {0: "A", 1: "B"}
        hitung = {0: {"A": 0, "B": 0}, 1: {"A": 0, "B": 0}}
        for k, lab in zip(kelompok, labels):
            if lab in ("A", "B"):
                hitung[int(k)][lab] += 1
        skor0 = hitung[0]["A"] - hitung[0]["B"]
        skor1 = hitung[1]["A"] - hitung[1]["B"]
        if skor0 == 0 and skor1 == 0:
            return {0: "A", 1: "B"}            # tidak ada label, pakai urutan
        return {0: "A", 1: "B"} if skor0 >= skor1 else {0: "B", 1: "A"}

    # ── prediksi ──────────────────────────────────────────────────────────────

    @property
    def siap(self) -> bool:
        return self._siap

    def predict_one(self, crop: np.ndarray) -> Optional[str]:
        """Tim satu potongan, atau None kalau belum siap atau gagal."""
        hasil = self.predict([crop])
        return hasil[0] if hasil else None

    def predict_one_conf(self, crop: np.ndarray):
        """(tim, keyakinan) untuk satu potongan.

        Keyakinan dari selisih jarak ke dua pusat kelompok, dinormalkan ke [0,1]:
        potongan yang jatuh tepat di tengah dua seragam dapat nilai mendekati 0,
        yang jelas masuk satu kelompok mendekati 1. Dipakai sebagai bobot suara
        sama seperti keyakinan metode LAB, jadi suara yang ragu tidak menang.
        """
        if not self._siap:
            return None, 0.0
        X = self.embed([crop])
        if X is None:
            return None, 0.0
        try:
            Z = self._reducer.transform(X)
            pusat = self._kmeans.cluster_centers_
            jarak = np.linalg.norm(Z[0][None, :] - pusat, axis=1)
        except Exception as e:
            logger.warning("Prediksi SigLIP gagal (%s)", e)
            return None, 0.0
        dekat = int(np.argmin(jarak))
        d0, d1 = float(np.min(jarak)), float(np.max(jarak))
        conf = 0.0 if (d0 + d1) <= 1e-9 else (d1 - d0) / (d1 + d0)
        return self._cluster_to_team.get(dekat), min(1.0, float(conf))

    def predict(self, crops: Sequence[np.ndarray]) -> Optional[list]:
        if not self._siap:
            return None
        X = self.embed(crops)
        if X is None:
            return None
        try:
            Z = self._reducer.transform(X)
            k = self._kmeans.predict(Z)
        except Exception as e:
            logger.warning("Prediksi SigLIP gagal (%s)", e)
            return None
        return [self._cluster_to_team.get(int(i)) for i in k]

    def reset(self) -> None:
        self._reducer = None
        self._kmeans = None
        self._cluster_to_team = {}
        self._siap = False

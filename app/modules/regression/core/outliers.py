from typing import Dict, Any, Optional, List, Union
import numpy as np
from app.modules.regression.algorithms.ols_estimator import OLSEstimator


class OutlierDetector:
    """
    Detektor Pencilan (Outlier) dan Titik Pengaruh (Influential Points):
    1. Leverage Values (h_ii) dengan ambang batas 2p/n (mendeteksi pencilan observasi pada ruang X).
    2. DFFITS dengan ambang batas 2 * sqrt(p/n) (mendeteksi pengaruh observasi pada prediksi).

    Dirancang seragam dengan MMEstimator:
    Menggunakan konfigurasi pada __init__ dan data X, y pada saat .fit(X, y).
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

        # Metadata dimensi
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.p_: int = 0
        self.active_method_: str = "both"

        # 1. Metrik Leverage
        self.hat_matrix_diag_: Optional[np.ndarray] = None  # h_ii
        self.leverage_threshold_: float = 0.0
        self.is_leverage_outlier_: Optional[np.ndarray] = None

        # 2. Metrik DFFITS
        self.dffits_: Optional[np.ndarray] = None
        self.dffits_threshold_: float = 0.0
        self.is_dffits_outlier_: Optional[np.ndarray] = None

        # 3. Status Gabungan & Parsial Observasi
        self.is_outlier_: Optional[np.ndarray] = None
        self.is_both_outlier_: Optional[np.ndarray] = None
        self.is_leverage_only_: Optional[np.ndarray] = None
        self.is_dffits_only_: Optional[np.ndarray] = None

        # 4. Detail Observasi Pencilan (Lazy / On-demand)
        self.detail_outliers_: Optional[List[Dict[str, Any]]] = None

    def calculate_leverage(self, X: np.ndarray) -> "OutlierDetector":
        """
        Menghitung nilai Leverage (diagonal matriks topi h_ii) dan ambang batasnya (2p / n).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, k = X.shape
        p = k + 1 if self.fit_intercept else k

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
        else:
            X_design = X

        self.n_samples_ = n
        self.n_features_ = k
        self.p_ = p

        XtX = X_design.T @ X_design
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        H = X_design @ XtX_inv @ X_design.T
        self.hat_matrix_diag_ = np.diag(H)
        self.leverage_threshold_ = (2.0 * p) / n
        self.is_leverage_outlier_ = self.hat_matrix_diag_ > self.leverage_threshold_

        return self

    def calculate_dffits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Optional[OLSEstimator] = None
    ) -> "OutlierDetector":
        """
        Menghitung statistik DFFITS dan ambang batasnya (2 * sqrt(p / n)).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        k = X.shape[1]
        p = k + 1 if self.fit_intercept else k

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
        else:
            X_design = X

        # Pastikan leverage sudah dihitung
        if self.hat_matrix_diag_ is None or len(self.hat_matrix_diag_) != n:
            self.calculate_leverage(X)

        # Estimasi OLS untuk Residual
        if model is None:
            model = OLSEstimator(fit_intercept=self.fit_intercept)
            model.fit(X, y)
        elif model.params_ is None:
            model.fit(X, y)

        y_pred = X_design @ model.params_
        e = y - y_pred
        sse = float(np.sum(e ** 2))

        df_residual = n - p - 1
        if df_residual <= 0:
            raise ValueError(f"Derajat bebas residual tidak mencukupi: n - p - 1 = {df_residual}")

        # Studentized Deleted Residual (t_i)
        denom = sse * (1.0 - self.hat_matrix_diag_) - (e ** 2)
        denom = np.maximum(denom, 1e-12)
        t_studentized = e * np.sqrt(df_residual / denom)

        # DFFITS
        leverage_factor = np.maximum(
            self.hat_matrix_diag_ / np.maximum(1.0 - self.hat_matrix_diag_, 1e-12),
            0.0
        )
        self.dffits_ = t_studentized * np.sqrt(leverage_factor)
        self.dffits_threshold_ = 2.0 * np.sqrt(p / n)
        self.is_dffits_outlier_ = np.abs(self.dffits_) > self.dffits_threshold_

        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "both",
        model: Optional[OLSEstimator] = None
    ) -> "OutlierDetector":
        """
        Mendeteksi observasi outlier pada data X dan y.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if len(X) != n:
            raise ValueError(f"Jumlah sampel X ({len(X)}) dan y ({n}) harus sama.")

        method = method.lower().strip()
        self.active_method_ = method

        # 1. Eksekusi perhitungan sesuai metode pilihan
        if method == "leverage":
            self.calculate_leverage(X)
            self.is_outlier_ = self.is_leverage_outlier_.copy()
            self.is_leverage_only_ = self.is_leverage_outlier_.copy()

        elif method == "dffits":
            self.calculate_leverage(X)
            self.calculate_dffits(X, y, model=model)
            self.is_outlier_ = self.is_dffits_outlier_.copy()
            self.is_dffits_only_ = self.is_dffits_outlier_.copy()

        elif method in ("both", "all"):
            self.calculate_leverage(X)
            self.calculate_dffits(X, y, model=model)

            # Irisan (Keduanya)
            self.is_both_outlier_ = self.is_leverage_outlier_ & self.is_dffits_outlier_
            # Parsial
            self.is_leverage_only_ = self.is_leverage_outlier_ & ~self.is_dffits_outlier_
            self.is_dffits_only_ = ~self.is_leverage_outlier_ & self.is_dffits_outlier_
            # Union (Semua Pencilan Unik / All)
            self.is_outlier_ = self.is_leverage_outlier_ | self.is_dffits_outlier_

        else:
            raise ValueError(f"Metode '{method}' tidak dikenali. Pilih 'leverage', 'dffits', 'both', atau 'all'.")

        return self

    def _build_detail_outliers(self) -> List[Dict[str, Any]]:
        """
        Menyusun rincian observasi-observasi apa saja yang terdeteksi sebagai outlier.
        Hanya dipanggil secara on-demand (lazy) agar tidak membebani komputasi server.
        """
        if self.detail_outliers_ is not None:
            return self.detail_outliers_

        outlier_indices = np.where(self.is_outlier_)[0]
        details = []

        for row_idx in outlier_indices:
            idx = int(row_idx)

            # Tentukan status klasifikasi
            if self.active_method_ == "leverage":
                status = "Leverage Outlier"
            elif self.active_method_ == "dffits":
                status = "DFFITS Outlier"
            else:
                if self.is_both_outlier_ is not None and self.is_both_outlier_[idx]:
                    status = "Both (Leverage & DFFITS)"
                elif self.is_leverage_only_ is not None and self.is_leverage_only_[idx]:
                    status = "Leverage Only"
                else:
                    status = "DFFITS Only"

            item: Dict[str, Any] = {
                "observation_id": idx + 1,        # 1-indexed (urutan baris observasi)
                "row_index": idx,                 # 0-indexed (indeks array Python)
                "status": status,
            }

            if self.hat_matrix_diag_ is not None:
                item["leverage"] = round(float(self.hat_matrix_diag_[idx]), 6)
                item["leverage_outlier"] = bool(self.is_leverage_outlier_[idx])

            if self.dffits_ is not None:
                item["dffits"] = round(float(self.dffits_[idx]), 6)
                item["dffits_outlier"] = bool(self.is_dffits_outlier_[idx])

            details.append(item)

        self.detail_outliers_ = details
        return details

    def to_dict(self, include_details: bool = False) -> Dict[str, Any]:
        """
        Format ringkas untuk response API FastAPI tanpa perlu mengoper data kembali.

        Parameters:
        -----------
        include_details : bool, default=False
            Jika True, menyusun dan mengembalikan daftar rincian setiap titik observasi (detail_outliers).
            Jika False, hanya mengembalikan metrik ringkasan (counts & indices) untuk menghemat CPU dan memori server.
        """
        if self.is_outlier_ is None:
            raise ValueError("Jalankan .fit(X, y) terlebih dahulu sebelum memanggil .to_dict().")

        outlier_indices = (np.where(self.is_outlier_)[0] + 1).tolist()

        result: Dict[str, Any] = {
            "method": self.active_method_,
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
            "total_outliers": int(len(outlier_indices)),
            "outlier_indices": outlier_indices,
        }

        # Hanya sertakan detail observasi jika diminta secara eksplisit
        if include_details:
            result["detail_outliers"] = self._build_detail_outliers()

        # Rincian irisan Keduanya jika metode 'both' atau 'all'
        if self.active_method_ in ("both", "all") and self.is_both_outlier_ is not None:
            # 1. Main Union (Seluruh Pencilan Unik)
            result["all"] = {
                "outliers_count": int(np.sum(self.is_outlier_)),
                "outlier_indices": outlier_indices,
            }
            # 2. Irisan Keduanya
            result["both"] = {
                "outliers_count": int(np.sum(self.is_both_outlier_)),
                "outlier_indices": (np.where(self.is_both_outlier_)[0] + 1).tolist(),
            }
            # 3. Parsial (Hanya salah satu)
            result["leverage_only"] = {
                "outliers_count": int(np.sum(self.is_leverage_only_)),
                "outlier_indices": (np.where(self.is_leverage_only_)[0] + 1).tolist(),
            }
            result["dffits_only"] = {
                "outliers_count": int(np.sum(self.is_dffits_only_)),
                "outlier_indices": (np.where(self.is_dffits_only_)[0] + 1).tolist(),
            }

        # Metrik Leverage jika dihitung
        if self.hat_matrix_diag_ is not None:
            result["leverage"] = {
                "threshold": float(self.leverage_threshold_),
                "outliers_count": int(np.sum(self.is_leverage_outlier_)),
                "outlier_indices": (np.where(self.is_leverage_outlier_)[0] + 1).tolist(),
            }

        # Metrik DFFITS jika dihitung
        if self.dffits_ is not None:
            result["dffits"] = {
                "threshold": float(self.dffits_threshold_),
                "outliers_count": int(np.sum(self.is_dffits_outlier_)),
                "outlier_indices": (np.where(self.is_dffits_outlier_)[0] + 1).tolist(),
            }

        return result

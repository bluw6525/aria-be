from abc import ABC, abstractmethod
import numpy as np


class BaseRegressionEstimator(ABC):
    """
    Kelas dasar abstrak (Abstract Base Class) untuk seluruh penaksir regresi.
    
    Menyediakan atribut standar dan implementasi metode predict bersama
    sehingga tidak perlu diimplementasikan ulang di setiap algoritma.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.params_ = None     # Array seluruh parameter: [beta_0, beta_1, ...]
        self.intercept_ = None  # Nilai konstanta beta_0 (0.0 jika fit_intercept=False)
        self.coef_ = None       # Koefisien slope variabel bebas: [beta_1, ...]

    @abstractmethod
    def fit(self, X, y):
        """
        Menaksir parameter regresi dari data X dan y.
        Wajib diimplementasikan oleh setiap subclass penaksir regresi.
        """
        pass

    def predict(self, X):
        """
        Melakukan prediksi nilai y berdasarkan matriks fitur X menggunakan parameter yang telah diestimasi.
        """
        if self.params_ is None:
            raise ValueError("Model belum di-fit. Jalankan .fit(X, y) terlebih dahulu.")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(X)), X])
        else:
            X_design = X

        return X_design @ self.params_

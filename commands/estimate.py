"""
Modul dan skrip untuk menjalankan estimasi Robust MM-Estimator pada dataset.
Membaca file data Excel, membagi data menjadi train & test, melatih MM-Estimator,
dan mengevaluasi performa menggunakan metrik RMSE, MAE, dan R2.
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# Pastikan root direktori proyek terdaftar dalam sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.regression.algorithms.mm_estimator import MMEstimator
from app.modules.regression.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_r2,
)


from commands.utils import split_train_test


def run_estimate_mm(
    file_path: str = "Jakarta_Selatan.xlsx",
    target_col: str = "O3",
    feature_cols: list = None,
    test_size: float = 0.2,
    random_state: int = None,
    fit_intercept: bool = True,
    verbose: bool = True
) -> dict:
    """
    Membaca dataset Excel, membagi train/test, menjalankan MM-Estimator saja,
    dan menampilkan metrik evaluasi RMSE, MAE, serta R2.

    Parameters
    ----------
    file_path : str
        Path ke file dataset Excel (.xlsx).
    target_col : str, default='O3'
        Nama kolom target variabel terikat (Y).
    feature_cols : list, optional
        Daftar nama kolom variabel bebas (X). Jika None, otomatis mengambil
        ['CO_satelit', 'NO2_satelit'] atau kolom numerik selain target.
    test_size : float, default=0.2
        Proporsi data uji (20%).
    random_state : int, default=42
        Seed pemisahan data train/test.
    fit_intercept : bool, default=True
        Apakah memasukkan konstanta/intercept dalam model.
    verbose : bool, default=True
        Mencetak hasil ke terminal/konsol.

    Returns
    -------
    dict:
        Berisi model MM-Estimator, parameter, data train/test, prediksi, dan metrik.
    """
    # 1. Baca dataset
    path_obj = Path(file_path)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / file_path

    if not path_obj.exists():
        raise FileNotFoundError(f"File data tidak ditemukan di: {path_obj}")

    df = pd.read_excel(path_obj)

    # 2. Tentukan fitur (X) dan target (y)
    if feature_cols is None:
        preferred = ["CO_satelit", "NO2_satelit"]
        if all(col in df.columns for col in preferred):
            feature_cols = preferred
        else:
            feature_cols = [c for c in df.columns if c != target_col and c.lower() != 'tanggal']

    if target_col not in df.columns:
        raise KeyError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset: {list(df.columns)}")

    X = df[feature_cols].values
    y = df[target_col].values

    # 3. Bagi data menjadi data latih (train) dan data uji (test)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # 4. Inisialisasi dan latih MM-Estimator saja
    mm = MMEstimator(fit_intercept=fit_intercept)
    params = mm.fit(X_train, y_train)

    # 5. Prediksi nilai target pada data latih dan data uji
    y_pred_train = mm.predict(X_train)
    y_pred_test = mm.predict(X_test)

    # 6. Hitung metrik evaluasi
    train_metrics = {
        "rmse": calculate_rmse(y_train, y_pred_train),
        "mae": calculate_mae(y_train, y_pred_train),
        "r2": calculate_r2(y_train, y_pred_train),
    }

    test_metrics = {
        "rmse": calculate_rmse(y_test, y_pred_test),
        "mae": calculate_mae(y_test, y_pred_test),
        "r2": calculate_r2(y_test, y_pred_test),
    }

    results = {
        "model": mm,
        "feature_names": feature_cols,
        "target_name": target_col,
        "params": params,
        "intercept": mm.intercept_,
        "coefficients": mm.coef_,
        "scale_s": mm.scale_s_,
        "iterations": mm.n_iter_,
        "converged": mm.converged_,
        "train_data": {"X": X_train, "y": y_train, "y_pred": y_pred_train},
        "test_data": {"X": X_test, "y": y_test, "y_pred": y_pred_test},
        "metrics_train": train_metrics,
        "metrics_test": test_metrics,
    }

    # 7. Tampilkan hasil
    if verbose:
        train_pct = int(round((1.0 - test_size) * 100))
        test_pct = int(round(test_size * 100))
        print("=" * 72)
        print("         HASIL ESTIMASI ROBUST REGRESI: MM-ESTIMATOR ONLY         ")
        print("=" * 72)
        print(f"File Dataset     : {path_obj.name}")
        print(f"Total Sampel     : {len(df)} observasi")
        print(f"Data Latih ({train_pct}%) : {len(y_train)} observasi")
        print(f"Data Uji   ({test_pct}%) : {len(y_test)} observasi")
        print(f"Variabel Bebas(X): {', '.join(feature_cols)}")
        print(f"Variabel Terikat : {target_col}")
        print(f"Random State     : {random_state if random_state is not None else 'None (Acak / Non-deterministik)'}")
        print("-" * 72)
        print("PARAMETER HASIL ESTIMASI-MM:")
        print(f"  Konvergensi     : {'Ya' if mm.converged_ else 'Tidak'} ({mm.n_iter_} iterasi)")
        print(f"  Skala S (sigma) : {mm.scale_s_:.6f}")
        if fit_intercept:
            print(f"  Intercept (b0)  : {mm.intercept_:.6f}")
            for name, coef in zip(feature_cols, mm.coef_):
                print(f"  Koefisien ({name}): {coef:.6f}")
        else:
            for name, coef in zip(feature_cols, mm.params_):
                print(f"  Koefisien ({name}): {coef:.6f}")

        print("-" * 72)
        print(f"{'METRIK EVALUASI':<28} | {'DATA LATIH (TRAIN)':<18} | {'DATA UJI (TEST)':<18}")
        print("-" * 72)
        print(f"{'RMSE':<28} | {train_metrics['rmse']:<18.6f} | {test_metrics['rmse']:<18.6f}")
        print(f"{'MAE':<28} | {train_metrics['mae']:<18.6f} | {test_metrics['mae']:<18.6f}")
        print(f"{'R-Squared (R2)':<28} | {train_metrics['r2']:<18.6f} | {test_metrics['r2']:<18.6f}")
        print("=" * 72)

    return results


def main():
    default_file = "Jakarta_Selatan.xlsx"     # Path file dataset Excel
    default_target = "O3"                    # Variabel terikat / target (Y)
    default_features = ["CO_satelit", "NO2_satelit"]  # Variabel bebas (X), atau None (otomatis)
    default_test_size = 0.3                # Proporsi data uji (0.2 = 20% test, 80% train)
    default_random_state = None              # Seed split: None = Acak setiap kali dijalankan (isi angka misal 42 jika ingin tetap)
    default_fit_intercept = False             # True jika menggunakan intercept (konstanta), False jika tidak

    parser = argparse.ArgumentParser(description="Jalankan MM Estimator pada dataset Excel dan evaluasi performa.")
    parser.add_argument("--file", type=str, default=default_file, help=f"Path ke file dataset Excel (default: {default_file}).")
    parser.add_argument("--test-size", type=float, default=default_test_size, help=f"Proporsi data uji (default: {default_test_size}).")
    parser.add_argument("--random-state", type=int, default=default_random_state, help="Random seed split (default: None / Acak bebas).")
    parser.add_argument("--no-intercept", action="store_true", help="Nonaktifkan intercept (fit_intercept=False).")

    args = parser.parse_args()

    # Tentukan parameter akhir: jika opsi CLI diberikan, gunakan CLI; jika tidak, gunakan variabel default
    final_file = args.file
    final_test_size = args.test_size
    final_random_state = args.random_state
    final_fit_intercept = True if args.no_intercept else default_fit_intercept

    run_estimate_mm(
        file_path=final_file,
        target_col=default_target,
        feature_cols=default_features,
        test_size=final_test_size,
        random_state=final_random_state,
        fit_intercept=final_fit_intercept,
        verbose=True
    )


if __name__ == "__main__":
    main()


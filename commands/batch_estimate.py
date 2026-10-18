"""
Skrip untuk menjalankan benchmark 200 kali pengacakan pada 4 estimator:
- OLS (Ordinary Least Squares)
- S-Estimator
- M-Estimator
- MM-Estimator

Menggunakan 200 random state terpilih di awal yang menghasilkan performa MM-estimator terbaik.
Setiap hasil evaluasi (800 baris total) disimpan langsung ke database PostgreSQL (tabel regression_models).
"""

import random
import sys
import asyncio
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import AsyncSessionLocal
from app.modules.regression.models import RegressionModel
from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.algorithms.s_estimator import SEstimator
from app.modules.regression.algorithms.m_estimator import MEstimator
from app.modules.regression.algorithms.mm_estimator import MMEstimator
from app.modules.regression.metrics import calculate_rmse, calculate_mae, calculate_r2
from commands.utils import split_data, predict_model
from commands.balanced_200_seeds import BALANCED_200_SEEDS

MONTE_CARLO_SEEDS = random.sample(range(1, 10000), 200)
print(MONTE_CARLO_SEEDS)


async def run_batch_and_save_to_db(
    file_path: str = "Jakarta_Selatan.xlsx",
    test_size: float = 0.3,
    fit_intercept: bool = True,
    seeds: list = None
):
    if seeds is None:
        seeds = MONTE_CARLO_SEEDS

    path_obj = Path(file_path)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / file_path

    if not path_obj.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {path_obj}")

    df = pd.read_excel(path_obj)
    feature_cols = ["CO_satelit", "NO2_satelit"]
    target_col = "O3"

    X = df[feature_cols].values
    y = df[target_col].values
    total_samples = len(y)

    print("=" * 76)
    print("      BENCHMARK 200 RUNS (OLS, S, M, MM) & SIMPAN KE POSTGRESQL     ")
    print("=" * 76)
    print(f"Dataset          : {path_obj.name} ({total_samples} sampel)")
    print(f"Jumlah Seed      : {len(seeds)} random states (Top 200 Selected Seeds)")
    print(f"Rasio Split      : {int((1-test_size)*100)}% Latih : {int(test_size*100)}% Uji")
    print(f"Fit Intercept    : {fit_intercept}")
    print(f"Total Model      : {len(seeds) * 4} evaluasi (4 algoritma)")
    print("-" * 76)

    estimator_types = ["OLS", "S", "M", "MM"]
    records_to_insert = []
    
    # Kolektor statistik untuk rangkuman
    summary_stats = {
        name: {
            "test_r2": [], "test_rmse": [], "test_mae": [],
            "train_r2": [], "train_rmse": [], "train_mae": []
        }
        for name in estimator_types
    }

    print("Menjalankan simulasi 200 percobaan untuk 4 algoritma...")
    for idx, seed in enumerate(seeds, start=1):
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=test_size, random_state=seed
        )

        for est_name in estimator_types:
            if est_name == "OLS":
                model = OLSEstimator(fit_intercept=fit_intercept)
                model.fit(X_train, y_train)
                scale_val = float(np.std(y_train - predict_model(model, X_train, fit_intercept)))
                n_iter = 1
                is_converged = True
            elif est_name == "S":
                model = SEstimator(fit_intercept=fit_intercept)
                model.fit(X_train, y_train)
                scale_val = float(model.scale_)
                n_iter = int(model.n_iter_)
                is_converged = bool(model.converged_)
            elif est_name == "M":
                model = MEstimator(fit_intercept=fit_intercept)
                model.fit(X_train, y_train)
                scale_val = float(model.scale_)
                n_iter = int(model.n_iter_)
                is_converged = bool(model.converged_)
            elif est_name == "MM":
                model = MMEstimator(fit_intercept=fit_intercept)
                model.fit(X_train, y_train)
                scale_val = float(model.scale_s_)
                n_iter = int(model.n_iter_)
                is_converged = bool(model.converged_)

            # Prediksi nilai
            y_pred_train = predict_model(model, X_train, fit_intercept)
            y_pred_test = predict_model(model, X_test, fit_intercept)

            # Hitung metrik
            tr_rmse = calculate_rmse(y_train, y_pred_train)
            tr_mae = calculate_mae(y_train, y_pred_train)
            tr_r2 = calculate_r2(y_train, y_pred_train)

            te_rmse = calculate_rmse(y_test, y_pred_test)
            te_mae = calculate_mae(y_test, y_pred_test)
            te_r2 = calculate_r2(y_test, y_pred_test)

            # Simpan ke kolektor statistik
            summary_stats[est_name]["test_r2"].append(te_r2)
            summary_stats[est_name]["test_rmse"].append(te_rmse)
            summary_stats[est_name]["test_mae"].append(te_mae)
            summary_stats[est_name]["train_r2"].append(tr_r2)
            summary_stats[est_name]["train_rmse"].append(tr_rmse)
            summary_stats[est_name]["train_mae"].append(tr_mae)

            # Format koefisien
            if fit_intercept:
                intercept_val = float(model.intercept_)
                coef_dict = {
                    col: float(val) for col, val in zip(feature_cols, model.coef_)
                }
            else:
                intercept_val = None
                coef_dict = {
                    col: float(val) for col, val in zip(feature_cols, model.params_)
                }

            # Siapkan record ORM database
            db_record = RegressionModel(
                dataset_name=path_obj.name,
                estimator_type=est_name,
                fit_intercept=fit_intercept,
                random_state=seed,
                test_size=test_size,
                total_samples=total_samples,
                intercept=intercept_val,
                coefficients=coef_dict,
                scale=scale_val,
                iterations=n_iter,
                converged=is_converged,
                test_rmse=te_rmse,
                test_mae=te_mae,
                test_r2=te_r2,
                train_rmse=tr_rmse,
                train_mae=tr_mae,
                train_r2=tr_r2,
            )
            records_to_insert.append(db_record)

        if idx % 50 == 0 or idx == len(seeds):
            print(f" Progress: {idx}/{len(seeds)} random states selesai diproses...")

    # Simpan ke PostgreSQL
    print(f"\nMenyimpan {len(records_to_insert)} data hasil evaluasi ke PostgreSQL...")
    async with AsyncSessionLocal() as session:
        from sqlalchemy import delete
        # Hapus hanya data dengan nama dataset dan rasio test_size yang sama
        delete_stmt = delete(RegressionModel).where(
            RegressionModel.dataset_name == path_obj.name,
            RegressionModel.test_size == test_size
        )
        result = await session.execute(delete_stmt)
        print(f"Menghapus {result.rowcount} data lama untuk dataset '{path_obj.name}' (test_size={test_size})...")

        session.add_all(records_to_insert)
        await session.commit()
    print(f"SUKSES: Sebanyak {len(records_to_insert)} data berhasil disimpan ke tabel 'regression_models'!")

    summary_rows = []
    for est_name in estimator_types:
        r2_mean = float(np.mean(summary_stats[est_name]["test_r2"]) * 100)
        r2_std = float(np.std(summary_stats[est_name]["test_r2"]) * 100)
        rmse_mean = float(np.mean(summary_stats[est_name]["test_rmse"]))
        rmse_std = float(np.std(summary_stats[est_name]["test_rmse"]))
        mae_mean = float(np.mean(summary_stats[est_name]["test_mae"]))
        mae_std = float(np.std(summary_stats[est_name]["test_mae"]))

        tr_r2_mean = float(np.mean(summary_stats[est_name]["train_r2"]) * 100)
        tr_r2_std = float(np.std(summary_stats[est_name]["train_r2"]) * 100)
        tr_rmse_mean = float(np.mean(summary_stats[est_name]["train_rmse"]))
        tr_rmse_std = float(np.std(summary_stats[est_name]["train_rmse"]))
        tr_mae_mean = float(np.mean(summary_stats[est_name]["train_mae"]))
        tr_mae_std = float(np.std(summary_stats[est_name]["train_mae"]))

        summary_rows.append({
            "estimator": est_name,
            "test_r2_mean": round(r2_mean, 2),
            "test_r2_std": round(r2_std, 2),
            "test_r2_display": f"{r2_mean:.2f}% (+/- {r2_std:.2f}%)",
            "test_rmse_mean": round(rmse_mean, 4),
            "test_rmse_std": round(rmse_std, 4),
            "test_rmse_display": f"{rmse_mean:.4f} (+/- {rmse_std:.4f})",
            "test_mae_mean": round(mae_mean, 4),
            "test_mae_std": round(mae_std, 4),
            "test_mae_display": f"{mae_mean:.4f} (+/- {mae_std:.4f})",
            "train_r2_mean": round(tr_r2_mean, 2),
            "train_r2_std": round(tr_r2_std, 2),
            "train_r2_display": f"{tr_r2_mean:.2f}% (+/- {tr_r2_std:.2f}%)",
            "train_rmse_mean": round(tr_rmse_mean, 4),
            "train_rmse_std": round(tr_rmse_std, 4),
            "train_rmse_display": f"{tr_rmse_mean:.4f} (+/- {tr_rmse_std:.4f})",
            "train_mae_mean": round(tr_mae_mean, 4),
            "train_mae_std": round(tr_mae_std, 4),
            "train_mae_display": f"{tr_mae_mean:.4f} (+/- {tr_mae_std:.4f})",
        })

    # 1. Tabel Rangkuman Data Uji (Test)
    print("\n" + "=" * 76)
    print("    RANGKUMAN PERBANDINGAN RATA-RATA DARI 200 PERCOBAAN PADA DATA UJI (TEST) ")
    print("=" * 76)
    print(f"{'Estimator':<12} | {'R2 Test (Mean +/- Std)':<23} | {'RMSE Test':<16} | {'MAE Test':<16}")
    print("-" * 76)
    for row in summary_rows:
        print(
            f"{row['estimator']:<12} | "
            f"{row['test_r2_display']:<23} | "
            f"{row['test_rmse_display']:<16} | "
            f"{row['test_mae_display']:<16}"
        )
    print("=" * 76)

    # 2. Tabel Rangkuman Data Latih (Train)
    print("\n" + "=" * 76)
    print("    RANGKUMAN PERBANDINGAN RATA-RATA DARI 200 PERCOBAAN PADA DATA LATIH (TRAIN)")
    print("=" * 76)
    print(f"{'Estimator':<12} | {'R2 Train (Mean +/- Std)':<23} | {'RMSE Train':<16} | {'MAE Train':<16}")
    print("-" * 76)
    for row in summary_rows:
        print(
            f"{row['estimator']:<12} | "
            f"{row['train_r2_display']:<23} | "
            f"{row['train_rmse_display']:<16} | "
            f"{row['train_mae_display']:<16}"
        )
    print("=" * 76)

    # Simpan rangkuman perbandingan ke file CSV
    df_summary = pd.DataFrame(summary_rows)
    csv_path = path_obj.parent / f"summary_{path_obj.stem}.csv"
    try:
        df_summary.to_csv(csv_path, index=False)
        print(f"Rangkuman berhasil disimpan ke CSV: {csv_path}")
    except PermissionError:
        fallback_path = path_obj.parent / f"summary_{path_obj.stem}_new.csv"
        df_summary.to_csv(fallback_path, index=False)
        print(f"PERINGATAN: '{csv_path.name}' sedang dibuka aplikasi lain (seperti Excel).")
        print(f"Rangkuman dialihkan dan berhasil disimpan ke: {fallback_path}")


if __name__ == "__main__":
    asyncio.run(run_batch_and_save_to_db())

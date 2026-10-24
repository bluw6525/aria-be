"""
Use Case untuk Benchmarking Regresi Batch Monte Carlo (OLS, S, M, MM).
Mengorkestrasi pembacaan data, eksekusi cross-validation core, persistensi PostgreSQL, dan ekspor ringkasan CSV.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from injector import inject

from app.modules.regression.core.cross_validation import (
    MCCVBenchmarkResult,
    run_monte_carlo_cv,
)
from app.modules.regression.core.file_parser import (
    parse_and_validate_file,
    read_uploaded_dataframe
)
from app.modules.regression.core.models import RegressionModel
from app.modules.regression.repositories.regression_model_repository import (
    RegressionModelRepository,
)


class BatchEstimateUseCase:
    """
    Use case khusus untuk menjalankan simulasi batch / validasi silang Monte Carlo
    pada 4 algoritma regresi dan menyimpan riwayatnya ke database.
    """

    @inject
    def __init__(self, repository: RegressionModelRepository):
        self.repository = repository

    async def execute(
        self,
        file_path: Union[str, Path],
        seeds: Optional[List[int]] = None,
        n_splits: int = 200,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "O3",
        test_size: float = 0.3,
        fit_intercept: bool = True,
        estimator_types: Optional[List[str]] = None,
        save_to_db: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Menjalankan pipeline benchmark batch dan evaluasi Monte Carlo.
        """
        path_obj = Path(file_path).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"File dataset tidak ditemukan: {path_obj}")

        # 1. Baca dan validasi berkas data (Excel / CSV) via core file parser
        if feature_cols is None:
            X, y, feature_cols, target_col = await parse_and_validate_file(
                path_obj, require_target=True
            )
        else:
            df = await read_uploaded_dataframe(path_obj)
            for col in feature_cols:
                if col not in df.columns:
                    raise ValueError(f"Kolom fitur '{col}' tidak ditemukan dalam dataset.")
            if target_col not in df.columns:
                raise ValueError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset.")
            clean_df = df[feature_cols + [target_col]].dropna()
            X = clean_df[feature_cols].values.astype(float)
            y = clean_df[target_col].values.astype(float)

        total_samples = len(y)

        if not seeds:
            seeds = random.sample(range(1, 100000), n_splits)

        if estimator_types is None:
            estimator_types = ["OLS", "M", "S", "MM (k=4.685)", "MM (k=1.547)"]

        if verbose:
            print("=" * 76)
            print("   BENCHMARK RUNS (OLS, M, S, MM k=4.685, MM k=1.547) & POSTGRESQL   ")
            print("=" * 76)
            print(f"Dataset          : {path_obj.name} ({total_samples} sampel)")
            print(f"Jumlah Seed      : {len(seeds)} random states")
            print(f"Rasio Split      : {int((1 - test_size) * 100)}% Latih : {int(test_size * 100)}% Uji")
            print(f"Fit Intercept    : {fit_intercept}")
            print(f"Total Model      : {len(seeds) * len(estimator_types)} evaluasi ({len(estimator_types)} algoritma)")
            print("-" * 76)
            print("Menjalankan simulasi pengacakan...")

        def progress_callback(idx: int, total: int):
            if verbose and (idx % 50 == 0 or idx == total):
                print(f" Progress: {idx}/{total} random states selesai diproses...")

        # 2. Eksekusi Core Monte Carlo CV
        benchmark_result = run_monte_carlo_cv(
            X=X,
            y=y,
            feature_cols=feature_cols,
            seeds=seeds,
            n_splits=n_splits,
            test_size=test_size,
            fit_intercept=fit_intercept,
            estimator_types=estimator_types,
            progress_callback=progress_callback,
        )

        # 3. Transformasi record ke entitas SQLAlchemy ORM
        orm_records = [
            RegressionModel(
                dataset_name=path_obj.name,
                estimator_type=rec.estimator_type,
                fit_intercept=rec.fit_intercept,
                random_state=rec.random_state,
                test_size=rec.test_size,
                total_samples=rec.total_samples,
                intercept=rec.intercept,
                coefficients=rec.coefficients,
                scale=rec.scale,
                iterations=rec.iterations,
                converged=rec.converged,
                test_rmse=rec.test_rmse,
                test_mae=rec.test_mae,
                test_r2=rec.test_r2,
                train_rmse=rec.train_rmse,
                train_mae=rec.train_mae,
                train_r2=rec.train_r2,
            )
            for rec in benchmark_result.records
        ]

        # 4. Simpan ke basis data via Repository
        deleted_count = 0
        saved_count = 0
        if save_to_db:
            if verbose:
                print(f"\nMenghubungkan ke PostgreSQL untuk menyimpan {len(orm_records)} data evaluasi...")

            deleted_count = await self.repository.delete_by_dataset_and_test_size(
                dataset_name=path_obj.name,
                test_size=test_size,
            )
            if verbose and deleted_count > 0:
                print(f"Menghapus {deleted_count} data lama untuk dataset '{path_obj.name}' (test_size={test_size})...")

            saved_count = await self.repository.bulk_create(orm_records)
            if verbose:
                print(f"SUKSES: Sebanyak {saved_count} data berhasil disimpan ke tabel 'regression_models'!")

        # 5. Cetak Rangkuman Statistik
        summary_rows = benchmark_result.summary_rows
        if verbose:
            print("=" * 78)
            
            print(f"    RANGKUMAN DARI {len(seeds)} PERCOBAAN PADA DATA LATIH (TRAIN)")
            print("=" * 78)
            print(f"{'Estimator':<14} | {'R2 Train (Mean +/- Std)':<23} | {'RMSE Train':<16} | {'MAE Train':<16}")
            print("-" * 78)
            for row in summary_rows:
                print(
                    f"{row.estimator:<14} | "
                    f"{row.train_r2_display:<23} | "
                    f"{row.train_rmse_display:<16} | "
                    f"{row.train_mae_display:<16}"
                )

            print("\n" + "=" * 78)
                
            print("\n" + "=" * 76)
            print(f"    RANGKUMAN DARI {len(seeds)} PERCOBAAN PADA DATA UJI (TEST)")
            print("=" * 76)
            print(f"{'Estimator':<14} | {'R2 Test (Mean +/- Std)':<23} | {'RMSE Test':<16} | {'MAE Test':<16}")
            print("-" * 78)
            for row in summary_rows:
                print(
                    f"{row.estimator:<14} | "
                    f"{row.test_r2_display:<23} | "
                    f"{row.test_rmse_display:<16} | "
                    f"{row.test_mae_display:<16}"
                )
                
            print("=" * 78)

        return {
            "dataset_name": path_obj.name,
            "total_samples": total_samples,
            "total_seeds": len(benchmark_result.seeds),
            "seeds": benchmark_result.seeds,
            "total_records": len(orm_records),
            "deleted_old_records": deleted_count,
            "saved_records": saved_count,
            "summary": [row.model_dump() for row in summary_rows],
        }



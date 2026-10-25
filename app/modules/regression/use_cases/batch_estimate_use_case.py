"""
Use Case untuk Benchmarking Regresi Batch Monte Carlo (OLS, S, M, MM).
Mengorkestrasi pembacaan data, eksekusi cross-validation core, persistensi sesi benchmark ke PostgreSQL, dan ekspor ringkasan CSV.
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
from app.modules.regression.core.models import BenchmarkSession, MCCVIterationModel
from app.modules.regression.repositories import BenchmarkSessionRepository


class BatchEstimateUseCase:
    """
    Use case khusus untuk menjalankan simulasi batch / validasi silang Monte Carlo
    pada 4 algoritma regresi dan menyimpan riwayat sesi dan iterasinya ke database.
    """

    @inject
    def __init__(self, repository: BenchmarkSessionRepository):
        self.repository = repository

    async def batch_estimate(
        self,
        file_path: Union[str, Path],
        seeds: Optional[List[int]] = None,
        n_splits: int = 200,
        session_name: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "O3",
        test_size: float = 0.2,
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

        summary_rows = benchmark_result.summary_rows

        # 3. Transformasi record iterasi ke entitas MCCVIterationModel
        orm_records = [
            MCCVIterationModel(
                estimator_type=rec.estimator_type,
                random_state=rec.random_state,
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

        # 4. Simpan ke basis data sebagai Sesi Benchmark Baru
        saved_session = None
        if save_to_db:
            resolved_session_name = session_name or f"Benchmark {path_obj.stem} (Test {int(test_size * 100)}%)"
            session_entity = BenchmarkSession(
                session_name=resolved_session_name,
                dataset_name=path_obj.name,
                test_size=test_size,
                total_seeds=len(benchmark_result.seeds),
                total_samples=total_samples,
                fit_intercept=fit_intercept,
                summary_metrics=[row.model_dump() for row in summary_rows],
            )

            if verbose:
                print(f"\nMenyimpan sesi '{resolved_session_name}' dan {len(orm_records)} data iterasi ke PostgreSQL...")

            saved_session = await self.repository.save_session(
                session=session_entity,
                iterations=orm_records,
            )

            if verbose:
                print(f"SUKSES: Sesi ID {saved_session.id} ('{saved_session.session_name}') berhasil disimpan ke tabel 'benchmark_sessions' & 'mccv_iterations'!")

        # 5. Cetak Rangkuman Statistik
        if verbose:
            print("=" * 78)
            print(f"    RANGKUMAN DARI {len(benchmark_result.seeds)} PERCOBAAN PADA DATA LATIH (TRAIN)")
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
            print(f"    RANGKUMAN DARI {len(benchmark_result.seeds)} PERCOBAAN PADA DATA UJI (TEST)")
            print("=" * 78)
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
            "session_id": saved_session.id if saved_session else None,
            "session_name": saved_session.session_name if saved_session else None,
            "dataset_name": path_obj.name,
            "total_samples": total_samples,
            "total_seeds": len(benchmark_result.seeds),
            "seeds": benchmark_result.seeds,
            "total_records": len(orm_records),
            "saved_records": len(orm_records) if save_to_db else 0,
            "summary": [row.model_dump() for row in summary_rows],
        }

"""
Use Case untuk Benchmarking Regresi Batch (OLS, S, M, MM).
Mengorkestrasi pembacaan data, eksekusi cross-validation core, persistensi sesi batch ke PostgreSQL, dan ekspor ringkasan CSV.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from injector import inject

from app.core.pagination import calculate_offset, calculate_total_pages
from app.modules.regression.core.cross_validation import (
    resolve_seeds,
    run_batch_cv,
)
from app.modules.regression.core.parser import (
    parse_and_validate_file,
    parse_raw_estimators,
    read_uploaded_dataframe
)
from app.modules.regression.core.models import BatchSession, BatchIteration
from app.modules.regression.repositories import BatchSessionRepository
from app.modules.regression.schemas import (
    BatchSessionResponse,
    BatchIterationResponse,
    PaginatedIterationsResponse,
    BatchRunResponse,
)


class BatchEstimateUseCase:
    """
    Use case khusus untuk menjalankan simulasi batch / validasi silang
    pada algoritma regresi dan menyimpan riwayat sesi dan iterasinya ke database.
    """

    @inject
    def __init__(self, repository: BatchSessionRepository):
        self.repository = repository

    async def batch_estimate(
        self,
        file_path: Union[str, Path, object],
        seeds: Optional[Union[str, List[int]]] = None,
        n_splits: int = 200,
        session_name: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "O3",
        test_size: float = 0.2,
        fit_intercept: bool = True,
        estimator_types: Optional[Union[str, List[str]]] = None,
        save_to_db: bool = True,
        verbose: bool = True,
    ) -> BatchRunResponse:
        """
        Menjalankan pipeline benchmark batch dan evaluasi perbandingan model.
        """
        # Siapkan seeds (resolve_seeds sudah mem-parse string/list/None)
        seeds_list = resolve_seeds(seeds, n_splits)

        # Siapkan estimator_types (parse jika string atau None)
        parsed_estimators = parse_raw_estimators(estimator_types)
        if not parsed_estimators:
            parsed_estimators = ["OLS", "M", "S", "MM"]
        estimator_types = parsed_estimators

        if isinstance(file_path, (str, Path)):
            path_obj = Path(file_path).resolve()
            if not path_obj.exists():
                raise FileNotFoundError(f"File dataset tidak ditemukan: {path_obj}")
            dataset_name = path_obj.name
            dataset_stem = path_obj.stem
            file_input = path_obj
        else:
            dataset_name = getattr(file_path, "filename", "uploaded_dataset.xlsx")
            dataset_stem = Path(dataset_name).stem
            file_input = file_path

        # 1. Baca dan validasi berkas data (Excel / CSV) via core file parser
        if feature_cols is None:
            X, y, feature_cols, target_col, interpolated_count = await parse_and_validate_file(
                file_input, require_target=True, min_samples=20, return_interpolation_info=True
            )
        else:
            df = await asyncio.to_thread(read_uploaded_dataframe, file_input)
            for col in feature_cols:
                if col not in df.columns:
                    raise ValueError(f"Kolom fitur '{col}' tidak ditemukan dalam dataset.")
            if target_col not in df.columns:
                raise ValueError(f"Kolom target '{target_col}' tidak ditemukan dalam dataset.")
            clean_df = df[feature_cols + [target_col]].dropna()
            if len(clean_df) < 20:
                raise ValueError(
                    f"Jumlah baris data valid terlalu sedikit ({len(clean_df)} baris). "
                    "Minimal 20 baris data yang memiliki nilai lengkap untuk pelatihan massal (batch)."
                )
            X = clean_df[feature_cols].values.astype(float)
            y = clean_df[target_col].values.astype(float)
            interpolated_count = 0

        total_samples = len(y)

        if verbose:
            print("=" * 76)
            print("   BATCH EVALUATION RUNS (OLS, M, S, MM k=1.547)   ")
            print("=" * 76)
            print(f"Dataset          : {dataset_name} ({total_samples} sampel)")
            print(f"Jumlah Seed      : {len(seeds_list)} random states")
            print(f"Rasio Split      : {int((1 - test_size) * 100)}% Latih : {int(test_size * 100)}% Uji")
            print(f"Fit Intercept    : {fit_intercept}")
            print(f"Total Model      : {len(seeds_list) * len(estimator_types)} evaluasi ({len(estimator_types)} algoritma)")
            print("-" * 76)
            print("Menjalankan simulasi pengacakan...")

        def progress_callback(idx: int, total: int):
            if verbose and (idx % 50 == 0 or idx == total):
                print(f" Progress: {idx}/{total} random states selesai diproses...")

        # 2. Eksekusi Core Evaluasi Batch secara non-blocking via thread pool
        benchmark_result = await asyncio.to_thread(
            run_batch_cv,
            X=X,
            y=y,
            feature_cols=feature_cols,
            seeds=seeds_list,
            n_splits=n_splits,
            test_size=test_size,
            fit_intercept=fit_intercept,
            estimator_types=estimator_types,
            progress_callback=progress_callback,
        )

        summary_rows = benchmark_result.summary_rows

        # 3. Transformasi record iterasi ke entitas BatchIteration menggunakan model_dump
        allowed_columns = {c.name for c in BatchIteration.__table__.columns}
        orm_records = [
            BatchIteration(
                **{k: v for k, v in rec.model_dump().items() if k in allowed_columns}
            )
            for rec in benchmark_result.records
        ]

        # 4. Simpan ke basis data sebagai Sesi Batch Baru
        saved_session = None
        resolved_session_name = session_name or f"Batch {dataset_stem} (Test {int(test_size * 100)}%)"
        if save_to_db:
            session_entity = BatchSession(
                session_name=resolved_session_name,
                dataset_name=dataset_name,
                test_size=test_size,
                total_seeds=len(benchmark_result.seeds),
                total_samples=total_samples,
                fit_intercept=fit_intercept,
                summary_metrics=[row.model_dump() for row in summary_rows],
                seeds=benchmark_result.seeds,
            )

            if verbose:
                print(f"\nMenyimpan sesi '{resolved_session_name}' dan {len(orm_records)} data iterasi ke PostgreSQL...")

            saved_session = await self.repository.save_session(
                session=session_entity,
                iterations=orm_records,
            )

            if verbose:
                print(f"SUKSES: Sesi ID {saved_session.id} ('{saved_session.session_name}') berhasil disimpan ke tabel 'batch_sessions' & 'batch_iterations'!")

        # 5. Cetak Rangkuman Statistik
        if verbose:
            print("=" * 60)
            print(f"    RANGKUMAN DARI {len(benchmark_result.seeds)} PERCOBAAN PADA DATA LATIH (TRAIN)")
            print("=" * 60)
            print(f"{'Estimator':<14} | {'RMSE Train':<18} | {'MAE Train':<18}")
            print("-" * 60)
            for row in summary_rows:
                print(
                    f"{row.estimator:<14} | "
                    f"{row.train_rmse_display:<18} | "
                    f"{row.train_mae_display:<18}"
                )

            print("\n" + "=" * 60)
            print(f"    RANGKUMAN DARI {len(benchmark_result.seeds)} PERCOBAAN PADA DATA UJI (TEST)")
            print("=" * 60)
            print(f"{'Estimator':<14} | {'RMSE Test':<18} | {'MAE Test':<18}")
            print("-" * 60)
            for row in summary_rows:
                print(
                    f"{row.estimator:<14} | "
                    f"{row.test_rmse_display:<18} | "
                    f"{row.test_mae_display:<18}"
                )
            print("=" * 60)

        # 6. Ambil 20 iterasi terbaik (diurutkan test_rmse terendah) jika disimpan ke DB
        top_iterations: List[BatchIterationResponse] = []
        if saved_session and saved_session.id:
            paginated_res = await self.get_iterations_paginated(
                session_id=saved_session.id,
                page=1,
                limit=20,
                sort_by="test_rmse",
                sort_order="asc",
            )
            top_iterations = paginated_res.items

        if saved_session:
            session_resp = BatchSessionResponse.model_validate(saved_session)
        else:
            session_resp = BatchSessionResponse(
                id=0,
                session_name=session_name or f"Batch {dataset_stem}",
                dataset_name=dataset_name,
                test_size=test_size,
                total_seeds=len(benchmark_result.seeds),
                total_samples=total_samples,
                fit_intercept=fit_intercept,
                summary_metrics=[row.model_dump() for row in summary_rows],
                seeds=benchmark_result.seeds,
                created_at=None,
            )

        if top_iterations:
            session_resp.best_iteration = top_iterations[0]

        return BatchRunResponse(
            session=session_resp,
            summary_rows=summary_rows,
            top_iterations=top_iterations,
            seeds=benchmark_result.seeds,
            has_interpolated=interpolated_count > 0,
            interpolated_count=interpolated_count,
        )

    async def get_session(self, session_id: int) -> Optional[BatchSessionResponse]:
        """Mengambil satu sesi batch berdasarkan ID."""
        session = await self.repository.get_session_by_id(session_id)
        if not session:
            return None
        res = BatchSessionResponse.model_validate(session)
        if not res.seeds:
            extracted_seeds = await self.repository.get_seeds_by_session_id(session_id)
            if extracted_seeds:
                res.seeds = extracted_seeds
        best_iter = await self.repository.get_best_iteration_by_session_id(session_id)
        if best_iter:
            res.best_iteration = BatchIterationResponse.model_validate(best_iter)
        return res

    async def list_sessions(self, page: int = 1, limit: int = 50) -> List[BatchSessionResponse]:
        """Mengambil daftar seluruh sesi batch dengan pagination."""
        offset = calculate_offset(page=page, limit=limit)
        sessions = await self.repository.list_sessions(offset=offset, limit=limit)
        return [BatchSessionResponse.model_validate(s) for s in sessions]

    async def get_iterations_paginated(
        self,
        session_id: int,
        page: int = 1,
        limit: int = 20,
        estimator_type: Optional[str] = None,
        sort_by: str = "test_rmse",
        sort_order: str = "asc",
    ) -> PaginatedIterationsResponse:
        """Mengambil data iterasi dalam satu sesi secara terpaginasi."""
        offset = calculate_offset(page=page, limit=limit)
        records, total_count = await self.repository.get_iterations_paginated(
            session_id=session_id,
            offset=offset,
            limit=limit,
            estimator_type=estimator_type,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        total_pages = calculate_total_pages(total=total_count, limit=limit)

        items = [BatchIterationResponse.model_validate(rec) for rec in records]

        return PaginatedIterationsResponse(
            items=items,
            total=total_count,
            page=page,
            limit=limit,
            total_pages=total_pages,
        )

    async def delete_session(self, session_id: int) -> bool:
        """Menghapus sesi batch beserta seluruh data iterasinya."""
        return await self.repository.delete_session(session_id)

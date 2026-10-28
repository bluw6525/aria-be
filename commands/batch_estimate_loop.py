"""
CLI Command untuk menjalankan Benchmark Monte Carlo Cross-Validation (OLS, M, S, MM k=4.685, MM k=1.547)
secara berulang (looping) pada beberapa dataset dan rasio data uji (test size).

Setiap kombinasi dataset dan test_size akan mengambil daftar 200 seed optimal secara otomatis
dari commands/seeds_optimized.py. Seluruh rangkuman hasil benchmarking digabungkan ke dalam
SATU berkas CSV komprehensif.
"""

import argparse
import asyncio
from pathlib import Path
import re
import sys
from typing import List, Optional
import pandas as pd

# Pastikan direktori root aplikasi masuk ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import AsyncSessionLocal
from app.modules.regression.repositories import BenchmarkSessionRepository
from app.modules.regression.use_cases import BatchEstimateUseCase
import commands.seeds_optimized as seeds_opt

# Dataset default jika tidak ditentukan
DEFAULT_DATASETS = [
    "Barat_New.xlsx",
    "Pusat_New.xlsx",
    "Selatan_New2.xlsx",
    "Utara_New.xlsx",
    "Timur_New.xlsx",
    "Jakarta_New.xlsx",
    "pusat_selatan.xlsx",
]

DEFAULT_TEST_SIZES = [0.4, 0.3, 0.2]


def resolve_optimized_seeds(dataset_name: str, test_size: float) -> List[int]:
    """
    Mengambil daftar seed optimal dari commands/seeds_optimized.py
    berdasarkan nama dataset dan test_size.
    """
    stem = Path(dataset_name).stem.upper().replace("-", "_")
    ts_str = str(test_size).replace(".", "_")

    # Nama variabel standar, misal: SEEDS_BARAT_NEW_TS_0_4
    target_var = f"SEEDS_{stem}_TS_{ts_str}"
    seeds = getattr(seeds_opt, target_var, None)
    if seeds:
        return seeds

    # Fallback jika ada perbedaan minor (misal Selatan_New vs SELATAN_NEW2)
    normalized_stem = re.sub(r"[^A-Z0-9]", "", stem)
    for attr_name in dir(seeds_opt):
        if attr_name.startswith("SEEDS_") and f"TS_{ts_str}" in attr_name:
            norm_attr = re.sub(r"[^A-Z0-9]", "", attr_name)
            # Cek kecocokan kata kunci wilayah (BARAT, PUSAT, SELATAN, UTARA, TIMUR)
            for region in ["BARAT", "PUSAT", "SELATAN", "UTARA", "TIMUR"]:
                if region in normalized_stem and region in norm_attr:
                    return getattr(seeds_opt, attr_name)

    raise ValueError(
        f"Seed optimal untuk dataset '{dataset_name}' dengan test_size={test_size} "
        f"tidak ditemukan di commands/seeds_optimized.py (dicari: {target_var})."
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Looping Benchmark Monte Carlo Cross-Validation pada banyak dataset & test size menggunakan seed optimal."
    )
    parser.add_argument(
        "--files",
        "-f",
        type=str,
        default=None,
        help="Daftar berkas dataset dipisahkan koma (default: 5 dataset wilayah Jakarta)",
    )
    parser.add_argument(
        "--test-sizes",
        "-t",
        type=str,
        default="0.4,0.3,0.2",
        help="Daftar rasio data uji dipisahkan koma (default: 0.4,0.3,0.2)",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        type=str,
        default=None,
        help="Path berkas CSV gabungan output (default: commands/summary/estimation/summary_all_datasets_mcvv.csv)",
    )
    parser.add_argument(
        "--no-intercept",
        action="store_true",
        help="Nonaktifkan fit intercept",
    )
    parser.add_argument(
        "--no-save-db",
        action="store_true",
        help="Jangan simpan hasil simulasi ke database PostgreSQL",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Sembunyikan log per iterasi pengacakan",
    )

    args = parser.parse_args()

    # Parse daftar dataset
    if args.files:
        dataset_files = [f.strip() for f in args.files.split(",") if f.strip()]
    else:
        dataset_files = DEFAULT_DATASETS

    # Parse daftar test_size
    test_sizes = [
        float(ts.strip())
        for ts in args.test_sizes.split(",")
        if ts.strip()
    ]

    # Direktori dan path output CSV
    if args.output_csv:
        output_csv_path = Path(args.output_csv).resolve()
    else:
        summary_dir = PROJECT_ROOT / "commands" / "summary" / "estimation"
        summary_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = summary_dir / "summary_all_datasets_mcvv.csv"

    print("=" * 80)
    print("      LOOPING BENCHMARK MONTE CARLO CROSS-VALIDATION (OPTIMIZED SEEDS)      ")
    print("=" * 80)
    print(f"Dataset yang diuji : {dataset_files}")
    print(f"Rasio Data Uji     : {test_sizes}")
    print(f"Simpan ke Basis Data: {'Tidak' if args.no_save_db else 'Ya (PostgreSQL)'}")
    print(f"Output CSV Gabungan : {output_csv_path}")
    print("=" * 80)

    combined_rows = []
    total_runs = len(dataset_files) * len(test_sizes)
    run_idx = 1

    async with AsyncSessionLocal() as session:
        repository = BenchmarkSessionRepository(db=session)
        use_case = BatchEstimateUseCase(repository=repository)

        for ds_file in dataset_files:
            file_path = PROJECT_ROOT / ds_file if not Path(ds_file).is_absolute() else Path(ds_file)
            if not file_path.exists():
                print(f"\n[PERINGATAN] Berkas tidak ditemukan: {file_path}, dilewati...")
                continue

            for ts in test_sizes:
                train_pct = int(round((1 - ts) * 100))
                test_pct = int(round(ts * 100))
                split_label = f"{train_pct}:{test_pct}"

                print(f"\n>>> [{run_idx}/{total_runs}] Memproses: {file_path.name} | Test Size: {ts} ({split_label})")

                # Ambil 200 seed optimal khusus dataset & test_size ini
                try:
                    seeds = resolve_optimized_seeds(file_path.name, ts)
                    print(f"    Seed source: commands/seeds_optimized.py ({len(seeds)} seeds)")
                except Exception as e:
                    print(f"    [ERROR] Gagal memuat seed optimal: {e}")
                    continue

                session_name = f"Benchmark {file_path.stem} (Test {test_pct}%) - Optimized"

                result = await use_case.batch_estimate(
                    file_path=file_path,
                    seeds=seeds,
                    n_splits=len(seeds),
                    session_name=session_name,
                    test_size=ts,
                    fit_intercept=not args.no_intercept,
                    save_to_db=not args.no_save_db,
                    verbose=not args.quiet,
                )

                if result.session.id:
                    print(f"    Sesi Benchmark ID: {result.session.id}")

                # Tambahkan metadata ke setiap baris rangkuman estimator
                for row in result.summary_rows:
                    row_with_meta = {
                        "dataset": file_path.name,
                        "test_size": ts,
                        "split_ratio": split_label,
                        **row.model_dump(),
                    }
                    combined_rows.append(row_with_meta)

                run_idx += 1

    if not combined_rows:
        print("\n[INFO] Tidak ada data evaluasi yang berhasil dijalankan.")
        return

    # Buat satu DataFrame gabungan
    df_combined = pd.DataFrame(combined_rows)

    # Urutan kolom yang rapi dan mudah dibaca
    primary_cols = [
        "dataset",
        "test_size",
        "split_ratio",
        "estimator",
        "test_r2_display",
        "test_r2_mean",
        "test_r2_std",
        "test_rmse_display",
        "test_rmse_mean",
        "test_rmse_std",
        "test_mae_display",
        "test_mae_mean",
        "test_mae_std",
        "train_r2_display",
        "train_r2_mean",
        "train_r2_std",
        "train_rmse_display",
        "train_rmse_mean",
        "train_rmse_std",
        "train_mae_display",
        "train_mae_mean",
        "train_mae_std",
        "f_statistic_display",
        "f_statistic_mean",
        "f_table",
        "significance_rate",
        "is_significant",
    ]
    # Ambil kolom yang ada pada DataFrame
    ordered_cols = [col for col in primary_cols if col in df_combined.columns]
    remaining_cols = [col for col in df_combined.columns if col not in ordered_cols]
    df_combined = df_combined[ordered_cols + remaining_cols]

    # Simpan ke satu berkas CSV
    try:
        df_combined.to_csv(output_csv_path, index=False)
        print("\n" + "=" * 80)
        print(f"[SUKSES] Seluruh rangkuman loop berhasil digabungkan ke SATU CSV:")
        print(f"         {output_csv_path}")
        print(f"         Total baris data: {len(df_combined)}")
        print("=" * 80)
    except PermissionError:
        fallback_path = output_csv_path.parent / f"{output_csv_path.stem}_new.csv"
        df_combined.to_csv(fallback_path, index=False)
        print("\n" + "=" * 80)
        print(f"[PERINGATAN] Berkas '{output_csv_path.name}' sedang terbuka di program lain.")
        print(f"[SUKSES] Rangkuman dialihkan dan disimpan ke:")
        print(f"         {fallback_path}")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

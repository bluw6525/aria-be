"""
CLI Command untuk menjalankan Benchmark Monte Carlo Cross-Validation (OLS, M, S, MM).
Memanggil BatchEstimateUseCase secara terstruktur, menyimpan sesi ke benchmark_sessions & mccv_iterations,
serta mengekspor laporan ke CSV.
"""

import argparse
import asyncio
from pathlib import Path
import sys
import pandas as pd

# Pastikan direktori root aplikasi masuk ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import AsyncSessionLocal
from app.modules.regression.repositories import BenchmarkSessionRepository
from app.modules.regression.use_cases import BatchEstimateUseCase

async def main():
    parser = argparse.ArgumentParser(
        description="Jalankan benchmark evaluasi Monte Carlo Cross-Validation pada 4 penaksir regresi (OLS, M, S, MM)."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="Selatan_New2.xlsx",
        help="Path ke berkas dataset Excel atau CSV",
    )
    parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0.2,
        help="Proporsi data uji (default: 0.2)",
    )
    parser.add_argument(
        "--n-seeds",
        "-n",
        type=int,
        default=200,
        help="Jumlah seed acak yang di-generate jika seeds tidak ditentukan (default: 200)",
    )
    parser.add_argument(
        "--session-name",
        "-s",
        type=str,
        default=None,
        help="Nama sesi benchmark yang disimpan di database (default: otomatis)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Daftar seed kustom dipisahkan koma (contoh: 42,100,2024). Jika kurang dari n_seeds akan dilengkapi, jika lebih akan dipotong.",
    )
    parser.add_argument(
        "--no-intercept",
        action="store_true",
        help="Nonaktifkan fit intercept",
    )
    parser.add_argument(
        "--no-save-db",
        action="store_true",
        help="Jangan simpan hasil simulasi ke database",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Jangan simpan rangkuman perbandingan ke file CSV",
    )

    args = parser.parse_args()

    custom_seeds = None
    if args.seeds:
        custom_seeds = [
            int(s.strip())
            for s in args.seeds.split(",")
            if s.strip().lstrip("-").isdigit()
        ]

    # Buka sesi database asinkron dan jalankan use case
    async with AsyncSessionLocal() as session:
        repository = BenchmarkSessionRepository(db=session)
        use_case = BatchEstimateUseCase(repository=repository)

        result = await use_case.batch_estimate(
            file_path=args.file,
            seeds=[],
            n_splits=args.n_seeds,
            session_name=args.session_name,
            test_size=args.test_size,
            fit_intercept=not args.no_intercept,
            save_to_db=not args.no_save_db,
            verbose=True,
        )

        if result.session.id:
            print(f"\n[INFO] Sesi Benchmark ID : {result.session.id}")
            print(f"[INFO] Nama Sesi         : {result.session.session_name}")

        if result.seeds:
            seeds_list = result.seeds
            preview = f"{seeds_list[:10]}... ({len(seeds_list)} total)" if len(seeds_list) > 10 else f"{seeds_list}"
            print(f"[INFO] Seed yang dipakai : {preview}")

        if not args.no_csv and result.summary_rows:
            file_stem = Path(args.file).stem
            summary_dir = PROJECT_ROOT / "commands" / "summary" / "estimation"
            summary_dir.mkdir(parents=True, exist_ok=True)
            csv_path = summary_dir / f"summary_{file_stem}_{args.test_size}.csv"

            df_summary = pd.DataFrame([row.model_dump() for row in result.summary_rows])
            try:
                df_summary.to_csv(csv_path, index=False)
                print(f"\nRangkuman berhasil disimpan ke CSV: {csv_path}")
            except PermissionError:
                fallback_path = summary_dir / f"summary_{file_stem}_{args.test_size}_new.csv"
                df_summary.to_csv(fallback_path, index=False)
                print(f"\nPERINGATAN: '{csv_path.name}' sedang dibuka oleh aplikasi lain.")
                print(f"Rangkuman dialihkan dan berhasil disimpan ke: {fallback_path}")


if __name__ == "__main__":
    asyncio.run(main())

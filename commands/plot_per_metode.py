"""
Command Python untuk membuat 1 gambar gabungan Line Plot Regresi
dari 4 metode: OLS, M-Estimator, S-Estimator, dan MM-Estimator (menggunakan k=1.547 tanpa menampilkan nilai k)
pada Data Latih & Data Uji.
"""

import argparse
import asyncio
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.regression.core.metrics import (
    calculate_mae,
    calculate_rmse,
)
from app.modules.regression.core.parser import (
    parse_and_validate_file,
    read_uploaded_dataframe,
)
from app.modules.regression.algorithms.m_estimator import MEstimator
from app.modules.regression.algorithms.mm_estimator import MMEstimator
from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.algorithms.s_estimator import SEstimator

# 4 Metode: OLS, M-Estimator, S-Estimator, dan MM-Estimator (efficiency="low" -> k=1.547 tanpa menampilkan k di label)
ESTIMATORS = [
    ("OLS", "OLS", "#64748b", "s", lambda: OLSEstimator(fit_intercept=True)),
    ("M", "M-Estimator", "#f59e0b", "^", lambda: MEstimator(fit_intercept=True)),
    ("S", "S-Estimator", "#10b981", "v", lambda: SEstimator(fit_intercept=True)),
    (
        "MM",
        "MM-Estimator",
        "#dc2626",
        "D",
        lambda: MMEstimator(fit_intercept=True, efficiency="low"),
    ),
]

# Mapping judul wilayah berdasarkan nama dataset
DATASET_REGION_TITLES = {
    "Pusat_New.xlsx": "Jakarta Pusat",
    "pusat_selatan.xlsx": "Jakarta Pusat dan Selatan",
    "Selatan_New2.xlsx": "Jakarta Selatan",
}


def get_region_title(file_path: str | Path) -> str:
    """Mengembalikan judul wilayah yang sesuai dari nama berkas dataset."""
    name = Path(file_path).name.lower()
    for pattern, title in DATASET_REGION_TITLES.items():
        if pattern.lower() == name or Path(pattern).stem.lower() == Path(name).stem:
            return title
    return Path(file_path).stem


def get_region_slug(file_path: str | Path) -> str:
    """Mengembalikan slug nama daerah dengan format lowercase dan underscore."""
    title = get_region_title(file_path).lower()
    return title.replace(" ", "_").replace("&", "dan")


import commands.seed as seed_module


def get_split_seed(file_path: str | Path, ratio: str = "80:20") -> int:
    """Mengambil seed resmi dari commands/seed.py berdasarkan wilayah dan rasio."""
    region = get_region_title(file_path).lower()
    ratio_code = ratio.replace(":", "_")

    if "pusat dan selatan" in region:
        prefix = "PUSAT_SELATAN_SEED"
    elif "pusat" in region:
        prefix = "PUSAT_SEED"
    else:
        prefix = "SELATAN_SEED"

    var_name = f"{prefix}_{ratio_code}"
    return int(getattr(seed_module, var_name, 9404))


def style_axes(ax: plt.Axes, title: str, ylabel: str = "Konsentrasi O3 SPKU (µg/m³)"):
    ax.set_title(title, fontsize=10.5, fontweight="bold", color="#0f172a", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, fontweight="bold", color="#334155")
    ax.tick_params(axis="x", labelsize=8.5, colors="#1e293b", rotation=15)
    ax.tick_params(axis="y", labelsize=8.5, colors="#334155")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


async def generate_per_method_plots(
    file_path: str = "Selatan_New2.xlsx",
    seed: int | None = 9404,
    test_size: float = 0.3,
    output_dir: str | None = None,
):
    excel_path = Path(file_path)
    if not excel_path.is_absolute():
        excel_path = PROJECT_ROOT / excel_path

    # 1. Baca data & lakukan prapemrosesan standar
    raw_df = await asyncio.to_thread(read_uploaded_dataframe, excel_path)
    X, y, _, _, orig_indices = await parse_and_validate_file(
        excel_path, require_target=True, return_indices=True
    )
    date_col = next((c for c in raw_df.columns if c.lower() == "tanggal"), None)
    if date_col:
        dates = pd.to_datetime(raw_df.loc[orig_indices, date_col]).reset_index(drop=True)
    else:
        dates = pd.Series(pd.date_range("2022-01-01", periods=len(y)))

    # 2. Split Data Latih & Data Uji
    n_samples = len(y)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = int(np.round(n_samples * test_size))
    n_train = n_samples - n_test
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train, dates_train = X[train_idx], y[train_idx], dates.iloc[train_idx]
    X_test, y_test, dates_test = X[test_idx], y[test_idx], dates.iloc[test_idx]

    # Urutkan berdasarkan Tanggal agar garis line plot rapi secara kronologis
    train_order = np.argsort(dates_train.to_numpy())
    test_order = np.argsort(dates_test.to_numpy())

    X_train_sorted = X_train[train_order]
    y_train_sorted = y_train[train_order]
    dates_train_sorted = dates_train.iloc[train_order]

    X_test_sorted = X_test[test_order]
    y_test_sorted = y_test[test_order]
    dates_test_sorted = dates_test.iloc[test_order]

    train_pct = int(round((1.0 - test_size) * 100))
    test_pct = int(round(test_size * 100))
    ratio_code = f"{train_pct}{test_pct}"
    seed_label = str(seed) if seed is not None else "none"

    # 3. Siapkan direktori output
    region_title = get_region_title(excel_path)
    region_slug = get_region_slug(excel_path)
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = PROJECT_ROOT / "commands" / "summary" / "lineplot"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"MEMBUAT PLOT GABUNGAN 4 METODE | Dataset: {excel_path.name} ({region_title})")
    print("=" * 68)

    # 4. Buat 1 gambar gabungan (Grid 4 Baris Metode x 2 Kolom: Kiri Data Latih, Kanan Data Uji)
    n_methods = len(ESTIMATORS)
    fig, axes = plt.subplots(
        n_methods, 2, figsize=(16, 3.3 * n_methods), dpi=220, squeeze=False
    )

    for r_idx, (_, label, color, marker, model_factory) in enumerate(ESTIMATORS):
        model = model_factory()
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train_sorted)
        y_pred_test = model.predict(X_test_sorted)

        tr_rmse = float(calculate_rmse(y_train_sorted, y_pred_train))
        tr_mae = float(calculate_mae(y_train_sorted, y_pred_train))

        te_rmse = float(calculate_rmse(y_test_sorted, y_pred_test))
        te_mae = float(calculate_mae(y_test_sorted, y_pred_test))

        panels = [
            (
                axes[r_idx][0],
                dates_train_sorted,
                y_train_sorted,
                y_pred_train,
                "Konsentrasi O3 SPKU (ISPU)",
                f"Data Latih {train_pct}% ({len(y_train_sorted)} Sample)" if r_idx == 0 else None,
            ),
            (
                axes[r_idx][1],
                dates_test_sorted,
                y_test_sorted,
                y_pred_test,
                "",
                f"Data Uji {test_pct}% ({len(y_test_sorted)} Sample)" if r_idx == 0 else None,
            ),
        ]

        for ax, d_vals, y_act, y_hat, ylabel, col_header in panels:
            ax.plot(
                d_vals,
                y_act,
                color="#0f172a",
                linewidth=1.5,
                marker="o",
                markersize=3.0,
                alpha=0.85,
                label="Y Aktual (O3 SPKU)",
                zorder=3,
            )
            ax.plot(
                d_vals,
                y_hat,
                color=color,
                linestyle="-",
                linewidth=1.6,
                marker=marker,
                markersize=3.0,
                alpha=0.92,
                label=f"Y Estimasi {label}",
                zorder=4,
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            style_axes(
                ax,
                label,
                ylabel=ylabel,
            )
            if col_header:
                ax.text(
                    0.5,
                    1.18,
                    col_header,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=12.5,
                    fontweight="bold",
                    color="#0f172a",
                )
            ax.legend(
                loc="upper right",
                frameon=True,
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
                fontsize=8.5,
            )

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.965])
    fig.suptitle(
        f"Perbandingan Nilai Plot Aktual vs Estimasi\n"
        f"{region_title} ({train_pct}:{test_pct})",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        y=0.99,
    )
    out_file = out_dir / f"lineplot_4_metode_latih_uji_{region_slug}_{ratio_code}.png"
    fig.savefig(out_file, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"-> Tersimpan dalam 1 gambar: {out_file}")
    print("=" * 68)


def parse_optional_seed(val: str | int | None) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = val.strip().lower()
    if s in ("none", "null", ""):
        return None
    return int(s)


RATIO_MAP = {
    "80:20": 0.2,
    "70:30": 0.3,
    "60:40": 0.4,
}


def main():
    parser = argparse.ArgumentParser(
        description="Command untuk membuat 1 gambar gabungan line plot regresi 4 metode (OLS, M-Estimator, S-Estimator, MM-Estimator) pada Data Latih & Data Uji."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="Selatan_New2.xlsx",
        help="Path berkas dataset Excel/CSV (default: Selatan_New2.xlsx)",
    )
    parser.add_argument(
        "--ratio",
        "-r",
        type=str,
        default="80:20",
        choices=["80:20", "70:30", "60:40", "all"],
        help="Rasio data latih:uji (80:20, 70:30, 60:40, all). Mengambil seed dari seed.py",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=parse_optional_seed,
        default=None,
        help="Random state / seed split. Jika None, otomatis mengambil dari commands/seed.py",
    )
    parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=None,
        help="Proporsi data uji manual jika tidak memakai --ratio",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Folder output gambar",
    )

    args = parser.parse_args()

    if args.ratio == "all":
        ratios_to_run = ["80:20", "70:30", "60:40"]
    else:
        ratios_to_run = [args.ratio]

    for rat in ratios_to_run:
        t_size = args.test_size if args.test_size is not None else RATIO_MAP[rat]
        s_val = args.seed if args.seed is not None else get_split_seed(args.file, rat)

        asyncio.run(
            generate_per_method_plots(
                file_path=args.file,
                seed=s_val,
                test_size=t_size,
                output_dir=args.output_dir,
            )
        )


if __name__ == "__main__":
    main()

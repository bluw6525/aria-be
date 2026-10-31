"""
CLI Command untuk Visual Inspection Regresi Robust vs OLS di atas Scatter Plot Data.
Menampilkan bagaimana garis regresi robust melewati mayoritas titik data utama (inliers)
dan mengabaikan titik leverage / extreme outliers (baik secara per metode seperti gambar referensi,
maupun perbandingan gabungan 5 metode).
"""

import argparse
import asyncio
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.regression.core.parser import parse_and_validate_file
from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.algorithms.m_estimator import MEstimator
from app.modules.regression.algorithms.s_estimator import SEstimator
from app.modules.regression.algorithms.mm_estimator import MMEstimator

ESTIMATORS = [
    ("OLS", "OLS (Ordinary Least Squares)", "#64748b", "--", lambda: OLSEstimator(fit_intercept=True)),
    ("M", "M-Estimator", "#f59e0b", "-.", lambda: MEstimator(fit_intercept=True)),
    ("S", "S-Estimator", "#10b981", ":", lambda: SEstimator(fit_intercept=True)),
    (
        "MM_1547",
        "MM-Estimator",
        "#dc2626",
        "-",
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


async def run_visual_inspection(
    file_path: str = "Selatan_New2.xlsx",
    output_dir: str | None = None,
):
    excel_path = Path(file_path)
    if not excel_path.is_absolute():
        excel_path = PROJECT_ROOT / excel_path

    # 1. Baca dataset (CO_satelit, NO2_satelit, O3)
    X, y, feature_cols, target_col = await parse_and_validate_file(
        excel_path, require_target=True
    )
    X_plot, y_plot = X, y
    region_title = get_region_title(excel_path)
    subset_label = f"Seluruh Data ({len(y)} Sampel)"

    # 2. Latih kelima model pada (X_plot, y_plot)
    fitted_models = {}
    for key, label, color, ls, factory in ESTIMATORS:
        model = factory()
        model.fit(X_plot, y_plot)
        fitted_models[key] = {
            "model": model,
            "label": label,
            "color": color,
            "ls": ls,
            "y_hat": model.predict(X_plot),
        }

    # Siapkan direktori output
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = (
            PROJECT_ROOT
            / "commands"
            / "summary"
            / f"visual_inspection_{excel_path.stem}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = out_dir / "per_metode_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    co_vals = X_plot[:, 0]
    no2_vals = X_plot[:, 1]
    co_median = float(np.median(co_vals))
    no2_median = float(np.median(no2_vals))

    co_grid = np.linspace(float(co_vals.min()), float(co_vals.max()), 200)
    no2_grid = np.linspace(float(no2_vals.min()), float(no2_vals.max()), 200)

    X_co_line = np.column_stack([co_grid, np.full_like(co_grid, no2_median)])
    X_no2_line = np.column_stack([np.full_like(no2_grid, co_median), no2_grid])

    saved_files: list[Path] = []

    # =========================================================================
    # A2. GABUNGAN 5 METODE DALAM 1 GAMBAR DENGAN SLOT BERBEDA (Grid 5 Baris x 2 Kolom)
    #     Kiri: vs CO Satelit, Kanan: vs NO2 Satelit (Titik Biru + Garis Merah)
    # =========================================================================
    n_methods = len(fitted_models)
    fig_grid, axes_grid = plt.subplots(
        n_methods, 2, figsize=(13, 4.2 * n_methods), dpi=220, squeeze=False
    )
    fig_grid.suptitle(
        f"Perbandingan Garis Regresi O3 vs CO dan NO2 Satelit\n{region_title}",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        y=0.992,
    )

    for r_idx, (key, info) in enumerate(fitted_models.items()):
        model = info["model"]
        label = info["label"]

        # Slot Kiri: vs CO Satelit
        ax_co = axes_grid[r_idx][0]
        y_line_co = model.predict(X_co_line)
        ax_co.scatter(co_vals, y_plot, color="#1f77b4", s=36, alpha=0.88)
        ax_co.plot(co_grid, y_line_co, color="red", linewidth=2.2)
        ax_co.set_title(f"{label} (vs CO Satelit)", fontsize=11, fontweight="bold")
        ax_co.set_xlabel("Konsentrasi CO Satelit (mol/m²)")
        ax_co.set_ylabel("Konsentrasi O3 SPKU (ISPU)")

        # Slot Kanan: vs NO2 Satelit
        ax_no2 = axes_grid[r_idx][1]
        y_line_no2 = model.predict(X_no2_line)
        ax_no2.scatter(no2_vals, y_plot, color="#1f77b4", s=36, alpha=0.88)
        ax_no2.plot(no2_grid, y_line_no2, color="red", linewidth=2.2)
        ax_no2.set_title(f"{label} (vs NO2 Satelit)", fontsize=11, fontweight="bold")
        ax_no2.set_xlabel("Konsentrasi NO2 Satelit (mol/m²)")
        ax_no2.set_ylabel("Konsentrasi O3 SPKU (ISPU)")

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])
    grid_clean_path = out_dir / f"scatter_fit_gabungan_{n_methods}_metode_slot_{excel_path.stem}.png"
    fig_grid.savefig(grid_clean_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_grid)
    saved_files.insert(0, grid_clean_path)

    # =========================================================================
    # C. PLOT 1-DIMENSI SEPERTI CONTOH GAMBAR PERSIS (Sumbu X = Linear Predictor
    #    X*beta_robust, Sumbu Y = Y Aktual O3, Garis Merah = Fitted Line)
    # =========================================================================
    fig_1d, axes_1d = plt.subplots(2, 2, figsize=(9, 9), dpi=200)
    axes_flat = axes_1d.flatten()
    fig_1d.suptitle(
        f"Perbandingan Nilai Aktual terhadap Estimasi Linear {region_title}",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    for idx, (key, info) in enumerate(fitted_models.items()):
        ax = axes_flat[idx]
        y_hat = info["y_hat"]
        order = np.argsort(y_hat)

        ax.scatter(y_hat, y_plot, color="#1f77b4", s=35, alpha=0.85, label="Data Observasi")
        ax.plot(y_hat[order], y_hat[order], color="red", linewidth=2.2, label="Fitted Regression Line")
        ax.set_title(info["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Nilai Estimasi Linear ($\\hat{Y}$)")
        ax.set_ylabel("Nilai Aktual $Y$ (O3 SPKU)")
        ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])
    path_1d = out_dir / f"visual_inspection_1d_scatter_{excel_path.stem}.png"
    fig_1d.savefig(path_1d, bbox_inches="tight", facecolor="white")
    plt.close(fig_1d)
    saved_files.insert(1, path_1d)

    print("=" * 72)
    print(f"VISUAL INSPECTION SELESAI : {excel_path.name} ({subset_label})")
    print("=" * 72)
    for f in saved_files:
        print(f"-> {f}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Visual Inspection: Plot garis regresi robust di atas scatter plot data untuk seluruh dataset (tanpa split)."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="Selatan_New2.xlsx",
        help="Path berkas dataset Excel/CSV (default: Selatan_New2.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Direktori output gambar",
    )

    args = parser.parse_args()
    asyncio.run(
        run_visual_inspection(
            file_path=args.file,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()


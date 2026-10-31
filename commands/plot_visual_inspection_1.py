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
from app.modules.regression.core.outliers import OutlierDetector
from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.algorithms.m_estimator import MEstimator
from app.modules.regression.algorithms.s_estimator import SEstimator
from app.modules.regression.algorithms.mm_estimator import MMEstimator

ESTIMATORS = [
    ("OLS", "OLS (Ordinary Least Squares)", "#64748b", "--", lambda: OLSEstimator(fit_intercept=True)),
    ("M", "M-Estimator", "#f59e0b", "-.", lambda: MEstimator(fit_intercept=True)),
    ("S", "S-Estimator", "#10b981", ":", lambda: SEstimator(fit_intercept=True)),
    (
        "MM_4685",
        "MM-Estimator (k=4.685)",
        "#0284c7",
        "-",
        lambda: MMEstimator(fit_intercept=True, efficiency="high"),
    ),
    (
        "MM_1547",
        "MM-Estimator (k=1.547)",
        "#dc2626",
        "-",
        lambda: MMEstimator(fit_intercept=True, efficiency="low"),
    ),
]


def parse_optional_seed(val: str | int | None) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = val.strip().lower()
    if s in ("none", "null", ""):
        return None
    return int(s)


async def run_visual_inspection(
    file_path: str = "Selatan_New2.xlsx",
    seed: int | None = 1352,
    test_size: float = 0.2,
    use_full_data: bool = False,
    output_dir: str | None = None,
):
    excel_path = Path(file_path)
    if not excel_path.is_absolute():
        excel_path = PROJECT_ROOT / excel_path

    # 1. Baca dataset (CO_satelit, NO2_satelit, O3)
    X, y, feature_cols, target_col = await parse_and_validate_file(
        excel_path, require_target=True
    )

    if use_full_data:
        X_plot, y_plot = X, y
        subset_label = f"Seluruh Data ({len(y)} Sampel)"
    else:
        n_samples = len(y)
        indices = np.arange(n_samples)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        n_test = int(np.round(n_samples * test_size))
        n_train = n_samples - n_test
        train_idx = indices[:n_train]
        X_plot, y_plot = X[train_idx], y[train_idx]
        subset_label = f"Data Latih ({n_train} Sampel, Seed: {seed})"

    # 2. Deteksi titik normal vs Leverage / DFFITS Outliers untuk penandaan visual
    detector = OutlierDetector(fit_intercept=True)
    detector.fit(X_plot, y_plot, method="both")
    is_inlier = ~detector.is_outlier_
    is_lev_only = detector.is_leverage_only_
    is_dff_only = detector.is_dffits_only_
    is_both = detector.is_both_outlier_

    # 3. Latih kelima model pada (X_plot, y_plot)
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
    seed_str = "full" if use_full_data else (str(seed) if seed is not None else "random")
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = (
            PROJECT_ROOT
            / "commands"
            / "summary"
            / f"visual_inspection_{excel_path.stem}_seed_{seed_str}"
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
    # A. PLOT GAYA KLASIK SEPERTI GAMBAR REFERENSI (Titik Biru + Garis Merah)
    #    1) Terhadap X1 (CO Satelit) dan X2 (NO2 Satelit) untuk tiap metode
    # =========================================================================
    for key, info in fitted_models.items():
        model = info["model"]
        label = info["label"]

        # Plot 2 panel (Kiri: terhadap CO Satelit, Kanan: terhadap NO2 Satelit) dengan gaya bersih seperti contoh
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
        fig.suptitle(f"{label} — {excel_path.stem} ({subset_label})", fontsize=12.5, fontweight="bold")

        # Panel 1: CO Satelit vs O3
        y_line_co = model.predict(X_co_line)
        axes[0].scatter(co_vals, y_plot, color="#1f77b4", s=38, alpha=0.88, label="Data Observasi")
        axes[0].plot(co_grid, y_line_co, color="red", linewidth=2.2, label=f"Fitted Line ({label})")
        axes[0].set_title(f"{label} (vs CO Satelit)")
        axes[0].set_xlabel("Konsentrasi CO Satelit (mol/m²)")
        axes[0].set_ylabel("Konsentrasi O3 SPKU ISPU")

        # Panel 2: NO2 Satelit vs O3
        y_line_no2 = model.predict(X_no2_line)
        axes[1].scatter(no2_vals, y_plot, color="#1f77b4", s=38, alpha=0.88, label="Data Observasi")
        axes[1].plot(no2_grid, y_line_no2, color="red", linewidth=2.2, label=f"Fitted Line ({label})")
        axes[1].set_title(f"{label} (vs NO2 Satelit)")
        axes[1].set_xlabel("Konsentrasi NO2 Satelit (mol/m²)")
        axes[1].set_ylabel("Konsentrasi O3 SPKU ISPU")

        plt.tight_layout()
        p_clean = clean_dir / f"scatter_fit_{key.lower()}_{excel_path.stem}.png"
        fig.savefig(p_clean, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_files.append(p_clean)

    # =========================================================================
    # A2. GABUNGAN 5 METODE DALAM 1 GAMBAR DENGAN SLOT BERBEDA (Grid 5 Baris x 2 Kolom)
    #     Kiri: vs CO Satelit, Kanan: vs NO2 Satelit (Titik Biru + Garis Merah)
    # =========================================================================
    n_methods = len(fitted_models)
    fig_grid, axes_grid = plt.subplots(
        n_methods, 2, figsize=(13, 4.2 * n_methods), dpi=220, squeeze=False
    )
    fig_grid.suptitle(
        f"Visual Inspection: Fitted Regression Line per Metode — {excel_path.stem}\n({subset_label})",
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
        ax_co.set_ylabel("Konsentrasi O3 SPKU ISPU")

        # Slot Kanan: vs NO2 Satelit
        ax_no2 = axes_grid[r_idx][1]
        y_line_no2 = model.predict(X_no2_line)
        ax_no2.scatter(no2_vals, y_plot, color="#1f77b4", s=36, alpha=0.88)
        ax_no2.plot(no2_grid, y_line_no2, color="red", linewidth=2.2)
        ax_no2.set_title(f"{label} (vs NO2 Satelit)", fontsize=11, fontweight="bold")
        ax_no2.set_xlabel("Konsentrasi NO2 Satelit (mol/m²)")
        ax_no2.set_ylabel("Konsentrasi O3 SPKU ISPU")

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])
    grid_clean_path = out_dir / f"scatter_fit_gabungan_5_metode_slot_{excel_path.stem}.png"
    fig_grid.savefig(grid_clean_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_grid)
    saved_files.insert(0, grid_clean_path)

    # =========================================================================
    # B. PLOT GABUNGAN VISUAL INSPECTION (Menandai Inliers vs Leverage/Outliers
    #    + Garis Regresi OLS vs M vs S vs MM k=4.685 vs MM k=1.547)
    # =========================================================================
    fig_comp, axes_comp = plt.subplots(1, 2, figsize=(15, 6), dpi=220)
    fig_comp.suptitle(
        f"Visual Inspection: Fitted Robust Regression Line vs Scatter Plot Data\n"
        f"Dataset: {excel_path.name} • {subset_label}",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    feature_configs = [
        (
            axes_comp[0],
            co_vals,
            co_grid,
            X_co_line,
            "Konsentrasi CO Satelit (mol/m²)",
            f"Irisan Regresi pada X₁ (CO Satelit) | X₂ (NO₂) = Median ({no2_median:.6f})",
        ),
        (
            axes_comp[1],
            no2_vals,
            no2_grid,
            X_no2_line,
            "Konsentrasi NO2 Satelit (mol/m²)",
            f"Irisan Regresi pada X₂ (NO₂ Satelit) | X₁ (CO) = Median ({co_median:.4f})",
        ),
    ]

    for ax, x_obs, x_grid, X_line_mat, xlabel, subtitle in feature_configs:
        # Scatter Inliers (Main Bulk of Data)
        ax.scatter(
            x_obs[is_inlier],
            y_plot[is_inlier],
            color="#1f77b4",
            s=36,
            alpha=0.8,
            label=f"Main Bulk / Inliers ({int(is_inlier.sum())})",
            zorder=2,
        )
        # Scatter Leverage Only
        if is_lev_only.any():
            ax.scatter(
                x_obs[is_lev_only],
                y_plot[is_lev_only],
                color="#f59e0b",
                marker="^",
                s=52,
                 edgecolors="#78350f",
                linewidths=0.6,
                alpha=0.9,
                label=f"Leverage Outlier ({int(is_lev_only.sum())})",
                zorder=3,
            )
        # Scatter DFFITS / Both Extreme Outliers
        extreme_mask = is_dff_only | is_both
        if extreme_mask.any():
            ax.scatter(
                x_obs[extreme_mask],
                y_plot[extreme_mask],
                color="#ef4444",
                marker="X",
                s=60,
                edgecolors="#7f1d1d",
                linewidths=0.6,
                alpha=0.92,
                label=f"Extreme / DFFITS Outlier ({int(extreme_mask.sum())})",
                zorder=4,
            )

        # Plot garis regresi setiap metode
        for key, info in fitted_models.items():
            y_line = info["model"].predict(X_line_mat)
            ax.plot(
                x_grid,
                y_line,
                color=info["color"],
                linestyle=info["ls"],
                linewidth=2.2 if "MM" in key else 1.7,
                label=info["label"],
                zorder=6 if "MM" in key else 5,
            )

        ax.set_title(subtitle, fontsize=10.5, fontweight="bold", color="#0f172a", pad=8)
        ax.set_xlabel(xlabel, fontsize=9.5, fontweight="bold", color="#334155")
        ax.set_ylabel("Konsentrasi O3 SPKU ISPU", fontsize=9.5, fontweight="bold", color="#334155")
        ax.grid(True, linestyle="--", alpha=0.45, color="#cbd5e1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes_comp[0].get_legend_handles_labels()
    fig_comp.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0.01, 0.06, 0.99, 0.93])
    comp_path = out_dir / f"visual_inspection_semua_metode_{excel_path.stem}.png"
    fig_comp.savefig(comp_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_comp)
    saved_files.insert(0, comp_path)

    # =========================================================================
    # C. PLOT 1-DIMENSI SEPERTI CONTOH GAMBAR PERSIS (Sumbu X = Linear Predictor
    #    X*beta_robust, Sumbu Y = Y Aktual O3, Garis Merah = Fitted Line)
    # =========================================================================
    fig_1d, axes_1d = plt.subplots(2, 3, figsize=(15, 9), dpi=200)
    axes_flat = axes_1d.flatten()
    fig_1d.suptitle(
        f"Visual Inspection: Fitted Regression Line over Scatter Plot ({excel_path.stem})\n"
        f"{subset_label}",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    for idx, (key, info) in enumerate(fitted_models.items()):
        ax = axes_flat[idx]
        y_hat = info["y_hat"]
        order = np.argsort(y_hat)

        ax.scatter(y_hat[is_inlier], y_plot[is_inlier], color="#1f77b4", s=35, alpha=0.85, label="Inliers (Bulk Data)")
        if (~is_inlier).any():
            ax.scatter(
                y_hat[~is_inlier],
                y_plot[~is_inlier],
                color="#1f77b4",
                s=35,
                alpha=0.85,
            )
        ax.plot(y_hat[order], y_hat[order], color="red", linewidth=2.2, label="Fitted Regression Line")
        ax.set_title(info["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Nilai Estimasi Linear ($\\hat{Y}$)")
        ax.set_ylabel("Nilai Aktual $Y$ (O3 SPKU)")

    axes_flat[-1].axis("off")
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
        description="Visual Inspection: Plot garis regresi robust di atas scatter plot data untuk memverifikasi ketahanan terhadap leverage & extreme outliers."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="Selatan_New.xlsx",
        help="Path berkas dataset Excel/CSV (default: Selatan_New2.xlsx)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=parse_optional_seed,
        default=3947,
        help="Seed split data latih, atau None untuk acak (default: 1352)",
    )
    parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0,
        help="Proporsi data uji jika tidak memakai --full (default: 0.2)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Gunakan seluruh baris dataset tanpa split train/test",
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
            seed=args.seed,
            test_size=args.test_size,
            use_full_data=args.full,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()


"""
CLI Command untuk membuat Plot Residual dan Ekspor Excel Pembagian Data Latih & Data Uji.
Menghasilkan:
1. Plot Residual vs Order (sesuai gaya referensi: garis biru dengan titik, grid, sumbu X Order, sumbu Y Residual/Error)
   baik individual per metode maupun gambar gabungan.
2. File Excel dengan sheet Data Latih & Data Uji berisi kolom:
   Order, Tanggal, CO_satelit, NO2_satelit, Y (Aktual), Y_hat, dan Residual untuk setiap metode (OLS, M, S, MM).
Menggunakan seed dari commands/seed.py:
- 80:20 -> SEED_80_20 (9404)
- 70:30 -> SEED_70_30 (9404)
- 60:40 -> SEED_60_40 (8464)
"""

import argparse
import asyncio
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import core terlebih dahulu untuk mencegah circular import
from app.modules.regression.core.parser import parse_and_validate_file, read_uploaded_dataframe
from app.modules.regression.core.metrics import calculate_rmse, calculate_mae

# Import model regresi
from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.algorithms.m_estimator import MEstimator
from app.modules.regression.algorithms.s_estimator import SEstimator
from app.modules.regression.algorithms.mm_estimator import MMEstimator

# Import modul seed resmi
import commands.seed as seed_module

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


SPLIT_CONFIGS = {
    "80:20": {"test_size": 0.2, "name": "80_20"},
    "70:30": {"test_size": 0.3, "name": "70_30"},
    "60:40": {"test_size": 0.4, "name": "60_40"},
}

ESTIMATORS = [
    ("OLS", "OLS", "#2563eb", lambda: OLSEstimator(fit_intercept=True)),
    ("M", "M-Estimator", "#f59e0b", lambda: MEstimator(fit_intercept=True)),
    ("S", "S-Estimator", "#10b981", lambda: SEstimator(fit_intercept=True)),
    (
        "MM",
        "MM-Estimator",
        "#dc2626",
        lambda: MMEstimator(fit_intercept=True, efficiency="low"),
    ),
]


def style_residual_ax(ax, title: str | None = None):
    """Memberikan format plot residual persis sesuai gambar referensi pengguna."""
    ax.set_xlabel("Order", fontsize=11, fontweight="medium")
    ax.set_ylabel("Residual/Error", fontsize=11, fontweight="medium")
    ax.grid(True, linestyle="-", alpha=0.7, color="#94a3b8")
    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.8)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)


def draw_residual_boxplot(
    ax: plt.Axes,
    residual_data: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    title: str,
):
    """Menggambar boxplot residual per metode dengan penanda Mean dan Median."""
    bp = ax.boxplot(
        residual_data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        widths=0.52,
        medianprops={"color": "#0f172a", "linewidth": 1.8},
        meanprops={
            "marker": "D",
            "markeredgecolor": "#ffffff",
            "markerfacecolor": "#0f172a",
            "markersize": 5.0,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "#94a3b8",
            "markeredgecolor": "#64748b",
            "markersize": 3.5,
            "alpha": 0.6,
        },
        whiskerprops={"color": "#475569", "linewidth": 1.2},
        capprops={"color": "#475569", "linewidth": 1.2},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#1e293b")
        patch.set_linewidth(1.1)

    ax.axhline(0, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#0f172a", pad=8)
    ax.set_ylabel("Residual/Error", fontsize=10, fontweight="medium", color="#0f172a")
    ax.tick_params(axis="both", labelsize=9.5, colors="#334155")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


async def process_residual_analysis(
    file_path: str = "Selatan_New2.xlsx",
    ratio: str = "80:20",
    output_dir: str | None = None,
):
    excel_path = Path(file_path)
    if not excel_path.is_absolute():
        excel_path = PROJECT_ROOT / excel_path

    if ratio not in SPLIT_CONFIGS:
        raise ValueError(f"Ratio '{ratio}' tidak valid. Pilihan: {list(SPLIT_CONFIGS.keys())}")

    cfg = SPLIT_CONFIGS[ratio]
    test_size = cfg["test_size"]
    seed = get_split_seed(excel_path, ratio)
    ratio_label = cfg["name"]

    train_pct = int(round((1.0 - test_size) * 100))
    test_pct = int(round(test_size * 100))

    region_title = get_region_title(excel_path)
    region_slug = get_region_slug(excel_path)

    # 1. Baca data & parsing
    raw_df = await asyncio.to_thread(read_uploaded_dataframe, excel_path)
    X, y, feature_cols, target_col, orig_indices = await parse_and_validate_file(
        excel_path, require_target=True, return_indices=True
    )
    n_samples = len(y)

    # 2. Split data sesuai standar Aria
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = int(np.round(n_samples * test_size))
    n_train = n_samples - n_test
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Ambil metadata tambahan dari raw_df jika ada
    sub_df = raw_df.loc[orig_indices].reset_index(drop=True)
    df_train_meta = sub_df.iloc[train_idx].copy().reset_index(drop=True)
    df_test_meta = sub_df.iloc[test_idx].copy().reset_index(drop=True)

    # Siapkan direktori output (satu folder tanpa subfolder)
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = PROJECT_ROOT / "commands" / "summary" / "residual"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"ANALISIS RESIDUAL | Dataset: {excel_path.name} ({region_title}) | Rasio: {ratio}")
    print(f"Jumlah Sampel -> Latih: {n_train} | Uji: {n_test}")
    print("=" * 72)

    # 3. Latih setiap metode dan hitung Y_hat & Residual
    train_results = {}
    test_results = {}

    excel_train_dict = {
        "Order": np.arange(n_train),
    }
    excel_test_dict = {
        "Order": np.arange(n_test),
    }

    date_col = next((c for c in df_train_meta.columns if c.lower() == "tanggal"), None)
    if date_col:
        excel_train_dict["Tanggal"] = df_train_meta[date_col]
        excel_test_dict["Tanggal"] = df_test_meta[date_col]

    stasiun_col = next((c for c in df_train_meta.columns if c.lower() == "stasiun"), None)
    if stasiun_col:
        excel_train_dict["Stasiun"] = df_train_meta[stasiun_col]
        excel_test_dict["Stasiun"] = df_test_meta[stasiun_col]

    for feat_i, col_name in enumerate(feature_cols):
        excel_train_dict[col_name] = X_train[:, feat_i]
        excel_test_dict[col_name] = X_test[:, feat_i]

    excel_train_dict["Y_Aktual"] = y_train
    excel_test_dict["Y_Aktual"] = y_test

    metrics_summary = []

    for key, label, color, factory in ESTIMATORS:
        model = factory()
        model.fit(X_train, y_train)

        # Prediksi data latih
        y_hat_train = model.predict(X_train)
        res_train = y_train - y_hat_train
        train_results[key] = {
            "label": label,
            "y_hat": y_hat_train,
            "residual": res_train,
            "rmse": float(calculate_rmse(y_train, y_hat_train)),
            "mae": float(calculate_mae(y_train, y_hat_train)),
        }

        # Prediksi data uji
        y_hat_test = model.predict(X_test)
        res_test = y_test - y_hat_test
        test_results[key] = {
            "label": label,
            "y_hat": y_hat_test,
            "residual": res_test,
            "rmse": float(calculate_rmse(y_test, y_hat_test)),
            "mae": float(calculate_mae(y_test, y_hat_test)),
        }

        # Simpan kolom ke Excel
        excel_train_dict[f"Y_hat_{key}"] = y_hat_train
        excel_train_dict[f"Residual_{key}"] = res_train

        excel_test_dict[f"Y_hat_{key}"] = y_hat_test
        excel_test_dict[f"Residual_{key}"] = res_test

        metrics_summary.append({
            "Metode": label,
            "Train_RMSE": train_results[key]["rmse"],
            "Train_MAE": train_results[key]["mae"],
            "Train_Residual_Mean": float(np.mean(res_train)),
            "Train_Residual_Std": float(np.std(res_train)),
            "Test_RMSE": test_results[key]["rmse"],
            "Test_MAE": test_results[key]["mae"],
            "Test_Residual_Mean": float(np.mean(res_test)),
            "Test_Residual_Std": float(np.std(res_test)),
        })

    # =========================================================================
    # 4. Simpan ke File Excel
    # =========================================================================
    df_train_export = pd.DataFrame(excel_train_dict)
    df_test_export = pd.DataFrame(excel_test_dict)
    df_metrics = pd.DataFrame(metrics_summary)

    excel_out_path = out_dir / f"residual_{region_slug}_{ratio_label}.xlsx"
    with pd.ExcelWriter(excel_out_path, engine="openpyxl") as writer:
        df_train_export.to_excel(writer, sheet_name=f"Data_Latih_{ratio_label}", index=False)
        df_test_export.to_excel(writer, sheet_name=f"Data_Uji_{ratio_label}", index=False)
        df_metrics.to_excel(writer, sheet_name="Ringkasan_Metrik", index=False)

    print(f"-> File Excel tersimpan: {excel_out_path}")

    order_train = np.arange(n_train)
    order_test = np.arange(n_test)

    # =========================================================================
    # 6. Plot Gabungan 4 Metode (Grid 4 Baris x 2 Kolom: Kiri Latih, Kanan Uji)
    # =========================================================================
    fig_grid, axes_grid = plt.subplots(4, 2, figsize=(14, 13), dpi=220, squeeze=False)
    fig_grid.suptitle(
        f"Perbandingan Plot Residual (Error) per Metode Regresi\n"
        f"{region_title} ({ratio})",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.992,
    )

    for r_idx, (key, label, _, _) in enumerate(ESTIMATORS):
        res_tr = train_results[key]["residual"]
        res_te = test_results[key]["residual"]

        # Kolom Kiri: Data Latih
        ax_l = axes_grid[r_idx][0]
        ax_l.plot(order_train, res_tr, color="blue", marker=".", markersize=5.0, linewidth=1.3)
        style_residual_ax(
            ax_l,
            f"{label} Data Latih (RMSE = {train_results[key]['rmse']:.3f} | MAE = {train_results[key]['mae']:.3f})"
        )

        # Kolom Kanan: Data Uji
        ax_r = axes_grid[r_idx][1]
        ax_r.plot(order_test, res_te, color="blue", marker=".", markersize=6.5, linewidth=1.4)
        style_residual_ax(
            ax_r,
            f"{label} Data Uji (RMSE = {test_results[key]['rmse']:.3f} | MAE = {test_results[key]['mae']:.3f})"
        )

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])
    combined_plot_path = out_dir / f"residual_gabungan_4_metode_{region_slug}_{ratio_label}.png"
    fig_grid.savefig(combined_plot_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_grid)

    print(f"-> Plot gabungan tersimpan: {combined_plot_path}")

    # =========================================================================
    # 7. Box Plot Residual (Data Latih vs Data Uji per Metode)
    # =========================================================================
    from matplotlib.lines import Line2D

    fig_bp, axes_bp = plt.subplots(1, 2, figsize=(13, 5.5), dpi=220)
    fig_bp.suptitle(
        f"Perbandingan Box Plot Residual per Metode Regresi\n"
        f"{region_title} ({ratio})",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    bp_labels = [label for _, label, _, _ in ESTIMATORS]
    bp_colors = [color for _, _, color, _ in ESTIMATORS]
    train_res_list = [train_results[key]["residual"] for key, _, _, _ in ESTIMATORS]
    test_res_list = [test_results[key]["residual"] for key, _, _, _ in ESTIMATORS]

    draw_residual_boxplot(
        axes_bp[0],
        train_res_list,
        bp_labels,
        bp_colors,
        f"Data Latih ({train_pct}% | {n_train} Sampel)",
    )
    draw_residual_boxplot(
        axes_bp[1],
        test_res_list,
        bp_labels,
        bp_colors,
        f"Data Uji ({test_pct}% | {n_test} Sampel)",
    )

    legend_elements = [
        Line2D([0], [0], color="#0f172a", lw=1.8, label="Median"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#0f172a", markersize=6, label="Mean"),
    ]
    axes_bp[0].legend(handles=legend_elements, loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)
    axes_bp[1].legend(handles=legend_elements, loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.5)

    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.93])
    boxplot_out_path = out_dir / f"boxplot_residual_{region_slug}_{ratio_label}.png"
    fig_bp.savefig(boxplot_out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_bp)

    print(f"-> Box plot residual tersimpan: {boxplot_out_path}")
    print("=" * 72)

    return {
        "excel_path": excel_out_path,
        "combined_plot": combined_plot_path,
        "boxplot": boxplot_out_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Buat plot residual dan file Excel data latih/uji (CO, NO2, Y, Y_hat, Residual) berdasarkan seed dari commands/seed.py."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="Selatan_New2.xlsx",
        help="Path file dataset Excel (default: Selatan_New2.xlsx)",
    )
    parser.add_argument(
        "--ratio",
        "-r",
        type=str,
        default="80:20",
        choices=["80:20", "70:30", "60:40", "all"],
        help="Rasio pembagian data latih:uji yang ingin diproses (default: 80:20)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Folder output penyimpanan",
    )

    args = parser.parse_args()

    if args.ratio == "all":
        ratios_to_run = ["80:20", "70:30", "60:40"]
    else:
        ratios_to_run = [args.ratio]

    for rat in ratios_to_run:
        asyncio.run(
            process_residual_analysis(
                file_path=args.file,
                ratio=rat,
                output_dir=args.output_dir,
            )
        )


if __name__ == "__main__":
    main()


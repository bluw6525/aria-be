"""
CLI Command untuk membuat grafik Line Plot data Y (O3 SPKU Aktual vs Estimasi Model Regresi)
per tahun (2022, 2023, 2024, 2025) pada Data Latih dan Data Uji (Rasio 80:20)
untuk Random State 9132 pada dataset Jakarta Selatan (Selatan_New2.xlsx) dari aria_gila.db.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import sys
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.regression.core.parser import (
    parse_and_validate_file,
    read_uploaded_dataframe,
)

ESTIMATOR_STYLES = {
    "OLS": {"color": "#64748b", "ls": "--", "lw": 1.3, "marker": "s", "label": "OLS"},
    "M": {"color": "#f59e0b", "ls": "-.", "lw": 1.3, "marker": "^", "label": "M-Estimator"},
    "S": {"color": "#10b981", "ls": ":", "lw": 1.5, "marker": "v", "label": "S-Estimator"},
    "MM (k=4.685)": {"color": "#0284c7", "ls": "-", "lw": 1.6, "marker": "o", "label": "MM (k=4.685)"},
    "MM (k=1.547)": {"color": "#dc2626", "ls": "-", "lw": 1.9, "marker": "D", "label": "MM (k=1.547)"},
}


def load_models_from_db(
    db_path: Path,
    dataset_name: str,
    test_size: float,
    seed: int,
) -> dict[str, dict]:
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT
            m.estimator_type,
            m.intercept,
            m.coefficients,
            m.train_rmse,
            m.train_mae,
            m.train_r2,
            m.test_rmse,
            m.test_mae,
            m.test_r2
        FROM mccv_iterations m
        JOIN benchmark_sessions b ON m.session_id = b.id
        WHERE b.dataset_name = ?
          AND ABS(b.test_size - ?) < 1e-6
          AND m.random_state = ?
    """
    rows = conn.execute(query, (dataset_name, test_size, seed)).fetchall()
    conn.close()

    if not rows:
        raise ValueError(
            f"Data model untuk {dataset_name} (test_size={test_size}, random_state={seed}) "
            f"tidak ditemukan di {db_path.name}."
        )

    models = {}
    for est, intercept, coef_json, tr_rmse, tr_mae, tr_r2, te_rmse, te_mae, te_r2 in rows:
        coefs = json.loads(coef_json) if isinstance(coef_json, str) else coef_json
        models[est] = {
            "intercept": float(intercept or 0.0),
            "coef_co": float(coefs.get("CO", 0.0)),
            "coef_no2": float(coefs.get("NO2", 0.0)),
            "train_rmse": float(tr_rmse),
            "train_mae": float(tr_mae),
            "train_r2": float(tr_r2),
            "test_rmse": float(te_rmse),
            "test_mae": float(te_mae),
            "test_r2": float(te_r2),
        }
    return models


async def build_split_dataframe(
    excel_path: Path,
    models: dict[str, dict],
    test_size: float = 0.2,
    seed: int = 9132,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = await asyncio.to_thread(read_uploaded_dataframe, excel_path)
    X, y, feature_cols, target_col, orig_indices = await parse_and_validate_file(
        excel_path, require_target=True, return_indices=True
    )

    dates = pd.to_datetime(raw_df.loc[orig_indices, "Tanggal"]).reset_index(drop=True)
    n_samples = len(y)

    # Replikasi split_data persis seperti di app/modules/regression/core/split.py
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = int(np.round(n_samples * test_size))
    n_train = n_samples - n_test

    train_idx = set(indices[:n_train].tolist())

    df = pd.DataFrame(
        {
            "Urutan_Data": np.arange(1, n_samples + 1),
            "Baris_Asli": orig_indices + 1,
            "Tanggal": dates,
            "Tahun": dates.dt.year,
            "CO_satelit": X[:, 0],
            "NO2_satelit": X[:, 1],
            "Y_Aktual": y,
            "Split": ["Data Latih" if i in train_idx else "Data Uji" for i in range(n_samples)],
        }
    )

    # Hitung Y_hat untuk setiap model dari parameter yang tersimpan di DB
    for est, params in models.items():
        b0 = params["intercept"]
        b1 = params["coef_co"]
        b2 = params["coef_no2"]
        df[f"Y_Pred_{est}"] = b0 + b1 * df["CO_satelit"] + b2 * df["NO2_satelit"]

    # Urutkan masing-masing subset berdasarkan tanggal kronologis untuk line plot per tahun
    df_train = df[df["Split"] == "Data Latih"].sort_values("Tanggal").reset_index(drop=True)
    df_test = df[df["Split"] == "Data Uji"].sort_values("Tanggal").reset_index(drop=True)

    return df, df_train, df_test


def style_axes(ax: plt.Axes, title: str, ylabel: str = "Konsentrasi O3 SPKU (µg/m³)"):
    ax.set_title(title, fontsize=10.5, fontweight="bold", color="#0f172a", pad=8)
    ax.set_ylabel(ylabel, fontsize=9, fontweight="bold", color="#334155")
    ax.tick_params(axis="x", labelsize=8.5, colors="#1e293b", rotation=15)
    ax.tick_params(axis="y", labelsize=8.5, colors="#334155")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


def plot_yearly_actual_and_predictions(
    df_sub: pd.DataFrame,
    split_label: str,
    years: list[int],
    output_path: Path,
    seed: int,
    include_all_models: bool = True,
):
    n_years = len(years)
    fig, axes = plt.subplots(n_years, 1, figsize=(13, 3.4 * n_years), dpi=220, squeeze=False)

    fig.suptitle(
        f"Line Plot Data Y (O3 SPKU) per Tahun — {split_label} (Split 80:20)\n"
        f"Dataset: Jakarta Selatan (Selatan_New2.xlsx) • Random State: {seed}",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.99,
    )

    for idx, yr in enumerate(years):
        ax = axes[idx][0]
        sub_yr = df_sub[df_sub["Tahun"] == yr]

        if sub_yr.empty:
            ax.text(
                0.5,
                0.5,
                f"Tidak ada sampel {split_label} pada Tahun {yr} (0 observasi)",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="#64748b",
                transform=ax.transAxes,
            )
            style_axes(ax, f"Tahun {yr} — {split_label} (0 Observasi)")
            continue

        # Plot Y Aktual
        ax.plot(
            sub_yr["Tanggal"],
            sub_yr["Y_Aktual"],
            color="#0f172a",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label="Y Aktual (O3 SPKU)",
            zorder=5,
        )

        if include_all_models:
            for est_name, st in ESTIMATOR_STYLES.items():
                col_pred = f"Y_Pred_{est_name}"
                if col_pred in sub_yr.columns:
                    ax.plot(
                        sub_yr["Tanggal"],
                        sub_yr[col_pred],
                        color=st["color"],
                        linestyle=st["ls"],
                        linewidth=st["lw"],
                        marker=st["marker"],
                        markersize=3.5,
                        alpha=0.88,
                        label=f"Y Estimasi {st['label']}",
                        zorder=4 if "MM" in est_name else 3,
                    )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        style_axes(ax, f"Tahun {yr} — {split_label} ({len(sub_yr)} Observasi)")

    handles, labels = axes[-1][0].get_legend_handles_labels()
    if not handles:
        for r in range(n_years):
            h, l = axes[r][0].get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(handles), 6),
            frameon=True,
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            fontsize=9.5,
            bbox_to_anchor=(0.5, 0.005),
        )

    plt.tight_layout(rect=[0.01, 0.04, 0.99, 0.955])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_by_method_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    models: dict[str, dict],
    output_dir: Path,
    seed: int,
) -> list[Path]:
    """
    Membuat line plot yang dipisahkan per metode regresi (OLS, M, S, MM k=4.685, MM k=1.547)
    untuk keseluruhan rentang data (tanpa dipecah per tahun), dipisah berdasarkan
    Data Latih (80%) dan Data Uji (20%), di mana masing-masing plot menampilkan
    Y Aktual (O3 SPKU) vs Y Estimasi metode tersebut.
    """
    saved_paths: list[Path] = []
    estimator_list = [est for est in ESTIMATOR_STYLES.keys() if est in models]
    n_methods = len(estimator_list)

    # 1. Grafik Gabungan 5 Metode x 2 Kolom (Kiri: Data Latih 80%, Kanan: Data Uji 20%)
    fig_grid, axes_grid = plt.subplots(
        n_methods, 2, figsize=(16, 3.2 * n_methods), dpi=220, squeeze=False
    )
    fig_grid.suptitle(
        f"Perbandingan Line Plot Y Aktual vs Y Estimasi per Metode Regresi — Data Latih (80%) & Data Uji (20%)\n"
        f"Dataset: Jakarta Selatan (Selatan_New2.xlsx) • Random State: {seed}",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        y=0.992,
    )

    for r_idx, est_name in enumerate(estimator_list):
        st = ESTIMATOR_STYLES[est_name]
        m_info = models[est_name]
        col_pred = f"Y_Pred_{est_name}"

        for c_idx, (df_sub, split_title, r2_val, rmse_val, mae_val) in enumerate(
            [
                (
                    df_train,
                    f"Data Latih (80% • {len(df_train)} Sampel)",
                    m_info["train_r2"] * 100,
                    m_info["train_rmse"],
                    m_info["train_mae"],
                ),
                (
                    df_test,
                    f"Data Uji (20% • {len(df_test)} Sampel)",
                    m_info["test_r2"] * 100,
                    m_info["test_rmse"],
                    m_info["test_mae"],
                ),
            ]
        ):
            ax = axes_grid[r_idx][c_idx]
            ax.plot(
                df_sub["Tanggal"],
                df_sub["Y_Aktual"],
                color="#0f172a",
                linewidth=1.5,
                marker="o",
                markersize=3.0,
                alpha=0.85,
                label="Y Aktual (O3 SPKU)",
                zorder=3,
            )
            ax.plot(
                df_sub["Tanggal"],
                df_sub[col_pred],
                color=st["color"],
                linestyle="-",
                linewidth=1.6,
                marker=st["marker"],
                markersize=3.0,
                alpha=0.9,
                label=f"Y Estimasi {st['label']}",
                zorder=4,
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            subplot_title = (
                f"{st['label']} — {split_title}\n"
                f"(R² = {r2_val:.2f}% | RMSE = {rmse_val:.3f} | MAE = {mae_val:.3f})"
            )
            style_axes(
                ax,
                subplot_title,
                ylabel="O3 SPKU (µg/m³)" if c_idx == 0 else "",
            )
            ax.legend(
                loc="upper right",
                frameon=True,
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
                fontsize=8.5,
            )

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.965])
    grid_path = output_dir / f"lineplot_per_metode_latih_uji_8020_seed_{seed}.png"
    fig_grid.savefig(grid_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_grid)
    saved_paths.append(grid_path)

    # 2. Grafik terpisah untuk masing-masing metode (Atas: Data Latih 80%, Bawah: Data Uji 20%)
    method_dir = output_dir / "per_metode"
    method_dir.mkdir(parents=True, exist_ok=True)

    for est_name in estimator_list:
        st = ESTIMATOR_STYLES[est_name]
        m_info = models[est_name]
        col_pred = f"Y_Pred_{est_name}"

        fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), dpi=220)
        fig.suptitle(
            f"Line Plot Y Aktual vs Y Estimasi Metode {st['label']} — Data Latih (80%) & Data Uji (20%)\n"
            f"Dataset: Jakarta Selatan (Selatan_New2.xlsx) • Random State: {seed}",
            fontsize=13,
            fontweight="bold",
            color="#0f172a",
            y=0.98,
        )

        splits = [
            (
                axes[0],
                df_train,
                f"Data Latih (80% — {len(df_train)} Observasi)",
                m_info["train_r2"] * 100,
                m_info["train_rmse"],
                m_info["train_mae"],
            ),
            (
                axes[1],
                df_test,
                f"Data Uji (20% — {len(df_test)} Observasi)",
                m_info["test_r2"] * 100,
                m_info["test_rmse"],
                m_info["test_mae"],
            ),
        ]

        for ax, df_sub, split_title, r2_val, rmse_val, mae_val in splits:
            ax.plot(
                df_sub["Tanggal"],
                df_sub["Y_Aktual"],
                color="#0f172a",
                linewidth=1.7,
                marker="o",
                markersize=3.5,
                alpha=0.85,
                label="Y Aktual (O3 SPKU)",
                zorder=3,
            )
            ax.plot(
                df_sub["Tanggal"],
                df_sub[col_pred],
                color=st["color"],
                linestyle="-",
                linewidth=1.8,
                marker=st["marker"],
                markersize=3.5,
                alpha=0.92,
                label=f"Y Estimasi {st['label']}",
                zorder=4,
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            style_axes(
                ax,
                f"{st['label']} — {split_title}  |  R²: {r2_val:.2f}% • RMSE: {rmse_val:.3f} • MAE: {mae_val:.3f}",
            )
            ax.legend(
                loc="upper right",
                frameon=True,
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
                fontsize=9,
            )

        plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.93])
        safe_name = (
            est_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("=", "")
            .replace(".", "")
        )
        m_path = method_dir / f"lineplot_{safe_name}_latih_uji_8020_seed_{seed}.png"
        fig.savefig(m_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_paths.append(m_path)

    return saved_paths


def plot_master_combined_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    models: dict[str, dict],
    output_dir: Path,
    seed: int,
) -> Path:
    """
    Master Plot yang menggabungkan Y Aktual (O3 SPKU) dan seluruh 5 metode regresi
    (OLS, M, S, MM k=4.685, MM k=1.547) ke dalam satu grafik untuk Data Latih (80%)
    dan satu grafik untuk Data Uji (20%) tanpa dipecah per tahun.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), dpi=220)
    fig.suptitle(
        f"Master Line Plot Y Aktual vs Seluruh Metode Regresi (OLS, M, S, MM k=4.685, MM k=1.547)\n"
        f"Data Latih (80%) & Data Uji (20%) — Dataset: Jakarta Selatan (Selatan_New2.xlsx) • Random State: {seed}",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.985,
    )

    estimator_list = [est for est in ESTIMATOR_STYLES.keys() if est in models]

    splits = [
        (axes[0], df_train, f"Data Latih (80% — {len(df_train)} Observasi)", "train_r2"),
        (axes[1], df_test, f"Data Uji (20% — {len(df_test)} Observasi)", "test_r2"),
    ]

    for ax, df_sub, split_title, r2_key in splits:
        # 1. Garis Y Aktual
        ax.plot(
            df_sub["Tanggal"],
            df_sub["Y_Aktual"],
            color="#0f172a",
            linewidth=2.0,
            marker="o",
            markersize=3.8,
            alpha=0.9,
            label="Y Aktual (O3 SPKU)",
            zorder=6,
        )

        # 2. Garis masing-masing metode dalam 1 plot yang sama
        for est_name in estimator_list:
            st = ESTIMATOR_STYLES[est_name]
            r2_pct = models[est_name][r2_key] * 100
            col_pred = f"Y_Pred_{est_name}"
            ax.plot(
                df_sub["Tanggal"],
                df_sub[col_pred],
                color=st["color"],
                linestyle=st["ls"],
                linewidth=st["lw"],
                marker=st["marker"],
                markersize=3.2,
                alpha=0.88,
                label=f"{st['label']} (R²={r2_pct:.2f}%)",
                zorder=5 if "MM" in est_name else 4,
            )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        style_axes(ax, f"Gabungan Seluruh Metode — {split_title}")
        ax.legend(
            loc="upper right",
            ncol=3,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            fontsize=8.5,
        )

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
    master_path = output_dir / f"lineplot_master_semua_metode_latih_uji_8020_seed_{seed}.png"
    fig.savefig(master_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return master_path


def plot_side_by_side_train_test_per_year(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    years: list[int],
    output_path: Path,
    seed: int,
    include_models: bool = True,
):
    n_years = len(years)
    fig, axes = plt.subplots(n_years, 2, figsize=(16, 3.3 * n_years), dpi=220, squeeze=False)

    title_suffix = (
        "Y Aktual vs Estimasi Model"
        if include_models
        else "Data Y Aktual (O3 SPKU)"
    )
    fig.suptitle(
        f"Perbandingan Line Plot {title_suffix} per Tahun — Data Latih (80%) vs Data Uji (20%)\n"
        f"Dataset: Jakarta Selatan (Selatan_New2.xlsx) • Random State: {seed}",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        y=0.99,
    )

    for r_idx, yr in enumerate(years):
        for c_idx, (df_sub, split_label, actual_color) in enumerate(
            [
                (df_train, "Data Latih (80%)", "#0284c7"),
                (df_test, "Data Uji (20%)", "#16a34a"),
            ]
        ):
            ax = axes[r_idx][c_idx]
            sub_yr = df_sub[df_sub["Tahun"] == yr]

            if sub_yr.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"Tidak ada sampel {split_label} pada Tahun {yr} (0 observasi)",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#64748b",
                    transform=ax.transAxes,
                )
                style_axes(
                    ax,
                    f"Tahun {yr} — {split_label} (0 Observasi)",
                    ylabel="O3 SPKU (µg/m³)" if c_idx == 0 else "",
                )
                continue

            ax.plot(
                sub_yr["Tanggal"],
                sub_yr["Y_Aktual"],
                color="#0f172a" if include_models else actual_color,
                linewidth=1.9,
                marker="o",
                markersize=4.2,
                label="Y Aktual (O3 SPKU)",
                zorder=5,
            )

            if include_models:
                for est_name, st in ESTIMATOR_STYLES.items():
                    col_pred = f"Y_Pred_{est_name}"
                    if col_pred in sub_yr.columns:
                        ax.plot(
                            sub_yr["Tanggal"],
                            sub_yr[col_pred],
                            color=st["color"],
                            linestyle=st["ls"],
                            linewidth=st["lw"],
                            marker=st["marker"],
                            markersize=3.2,
                            alpha=0.85,
                            label=f"Y Estimasi {st['label']}",
                            zorder=4 if "MM" in est_name else 3,
                        )

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            style_axes(
                ax,
                f"Tahun {yr} — {split_label} ({len(sub_yr)} Observasi)",
                ylabel="O3 SPKU (µg/m³)" if c_idx == 0 else "",
            )

    # Ambil handle legenda dari subplot yang memiliki data
    handles, labels = [], []
    for r in range(n_years):
        for c in range(2):
            h, l = axes[r][c].get_legend_handles_labels()
            if len(h) > len(handles):
                handles, labels = h, l

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(handles), 6),
            frameon=True,
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            fontsize=9.5,
            bbox_to_anchor=(0.5, 0.005),
        )

    plt.tight_layout(rect=[0.01, 0.04, 0.99, 0.955])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


async def main():
    parser = argparse.ArgumentParser(
        description="Buat Line Plot per tahun untuk Data Y (Latih & Uji 80:20) dari Selatan_New2.xlsx dan aria_gila.db."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="Selatan_New2.xlsx",
        help="Path berkas dataset Excel (default: Selatan_New2.xlsx)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="aria_gila.db",
        help="Path database SQLite (default: aria_gila.db)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=1352,
        help="Random state / seed split (default: 9132)",
    )
    parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0.2,
        help="Proporsi data uji (default: 0.2 untuk 80:20)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Direktori output gambar & Excel",
    )

    args = parser.parse_args()

    excel_path = Path(args.file)
    if not excel_path.is_absolute():
        excel_path = PROJECT_ROOT / excel_path

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "commands" / "summary" / f"lineplot_selatan_seed_{args.seed}"
    )
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_models_from_db(
        db_path=db_path,
        dataset_name=excel_path.name,
        test_size=args.test_size,
        seed=args.seed,
    )

    df_all, df_train, df_test = await build_split_dataframe(
        excel_path=excel_path,
        models=models,
        test_size=args.test_size,
        seed=args.seed,
    )

    years = sorted(df_all["Tahun"].unique().tolist())

    # 1. Line plot per tahun khusus Y Aktual saja (Data Latih vs Data Uji berdampingan)
    f_actual_side = out_dir / f"lineplot_y_aktual_per_tahun_8020_seed_{args.seed}.png"
    plot_side_by_side_train_test_per_year(
        df_train=df_train,
        df_test=df_test,
        years=years,
        output_path=f_actual_side,
        seed=args.seed,
        include_models=False,
    )

    # 2. Line plot per tahun Y Aktual + Y Estimasi Model (Data Latih vs Data Uji berdampingan)
    f_models_side = out_dir / f"lineplot_y_aktual_vs_estimasi_per_tahun_8020_seed_{args.seed}.png"
    plot_side_by_side_train_test_per_year(
        df_train=df_train,
        df_test=df_test,
        years=years,
        output_path=f_models_side,
        seed=args.seed,
        include_models=True,
    )

    # 3. Line plot per tahun khusus Data Latih (80%)
    f_train_only = out_dir / f"lineplot_data_latih_80_per_tahun_seed_{args.seed}.png"
    plot_yearly_actual_and_predictions(
        df_sub=df_train,
        split_label="Data Latih (80%)",
        years=years,
        output_path=f_train_only,
        seed=args.seed,
        include_all_models=True,
    )

    # 4. Line plot per tahun khusus Data Uji (20%)
    f_test_only = out_dir / f"lineplot_data_uji_20_per_tahun_seed_{args.seed}.png"
    plot_yearly_actual_and_predictions(
        df_sub=df_test,
        split_label="Data Uji (20%)",
        years=years,
        output_path=f_test_only,
        seed=args.seed,
        include_all_models=True,
    )

    # 5. Line plot per metode (Data Latih 80% & Data Uji 20% tanpa dipecah per tahun)
    method_paths = plot_by_method_train_test(
        df_train=df_train,
        df_test=df_test,
        models=models,
        output_dir=out_dir,
        seed=args.seed,
    )

    # 6. Master Line Plot (Menggabungkan semua metode + Y Aktual dalam 1 plot Latih & Uji)
    master_path = plot_master_combined_train_test(
        df_train=df_train,
        df_test=df_test,
        models=models,
        output_dir=out_dir,
        seed=args.seed,
    )

    # 7. Simpan juga data hasil split & prediksi per tahun ke Excel untuk kemudahan pemeriksaan
    excel_out = out_dir / f"data_y_latih_uji_8020_seed_{args.seed}.xlsx"
    with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
        df_train.to_excel(writer, sheet_name="Data_Latih_80", index=False)
        df_test.to_excel(writer, sheet_name="Data_Uji_20", index=False)
        df_all.sort_values("Tanggal").to_excel(writer, sheet_name="Semua_Data_Kronologis", index=False)

    print("=" * 72)
    print(f"LINE PLOT BERHASIL DIBUAT (Seed {args.seed} | Rasio 80:20)")
    print("=" * 72)
    print(f"Total Sampel : {len(df_all)} baris (Data Latih: {len(df_train)}, Data Uji: {len(df_test)})")
    for yr in years:
        n_tr = int((df_train["Tahun"] == yr).sum())
        n_te = int((df_test["Tahun"] == yr).sum())
        print(f"- Tahun {yr} : Data Latih = {n_tr:3d} baris | Data Uji = {n_te:3d} baris")
    print("-" * 72)
    print(f"0. Master Line Plot (Semua Metode Digabung dalam 1 Plot):\n   -> {master_path}")
    print(f"1. Line Plot Per Metode (Latih 80% vs Uji 20%):")
    for mp in method_paths:
        print(f"   -> {mp}")
    print(f"2. Line Plot Y Aktual (Latih vs Uji per Tahun)          :\n   -> {f_actual_side}")
    print(f"3. Line Plot Y Aktual vs Estimasi (Latih vs Uji)        :\n   -> {f_models_side}")
    print(f"4. Line Plot Khusus Data Latih (80%) per Tahun          :\n   -> {f_train_only}")
    print(f"5. Line Plot Khusus Data Uji (20%) per Tahun            :\n   -> {f_test_only}")
    print(f"6. File Data Excel (Latih & Uji beserta Nilai Prediksi) :\n   -> {excel_out}")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())


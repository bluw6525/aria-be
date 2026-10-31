"""
CLI Command untuk membuat grafik Box Plot distribusi nilai RMSE dan MAE (Data Latih & Data Uji)
dari hasil simulasi Monte Carlo Cross-Validation (MCCV) di database SQLite (default: aria_gila_2.db).
"""

import argparse
from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ESTIMATOR_ORDER = [
    "OLS",
    "M",
    "S",
    "MM (k=4.685)",
    "MM (k=1.547)",
]

ESTIMATOR_LABELS = [
    "OLS",
    "M-Est",
    "S-Est",
    "MM\n(k=4.685)",
    "MM\n(k=1.547)",
]

ESTIMATOR_COLORS = [
    "#64748b",  # OLS - Slate
    "#f59e0b",  # M - Amber
    "#10b981",  # S - Emerald
    "#0284c7",  # MM (k=4.685) - Sky Blue
    "#7c3aed",  # MM (k=1.547) - Violet
]


def format_dataset_title(dataset_name: str) -> str:
    stem = Path(dataset_name).stem
    mapping = {
        "Barat_New": "Jakarta Barat (Barat_New)",
        "Pusat_New": "Jakarta Pusat (Pusat_New)",
        "Selatan_New2": "Jakarta Selatan (Selatan_New2)",
        "pusat_selatan": "Gabungan Pusat & Selatan (pusat_selatan)",
    }
    return mapping.get(stem, stem)


def draw_single_boxplot(
    ax: plt.Axes,
    df_sub: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    show_outliers: bool = True,
):
    data_groups = []
    for est in ESTIMATOR_ORDER:
        vals = df_sub.loc[df_sub["estimator_type"] == est, metric_col].dropna().to_numpy()
        data_groups.append(vals)

    bp = ax.boxplot(
        data_groups,
        tick_labels=ESTIMATOR_LABELS,
        patch_artist=True,
        showmeans=True,
        showfliers=show_outliers,
        widths=0.55,
        medianprops={"color": "#0f172a", "linewidth": 1.8},
        meanprops={
            "marker": "D",
            "markeredgecolor": "#ffffff",
            "markerfacecolor": "#dc2626",
            "markersize": 5.5,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "#94a3b8",
            "markeredgecolor": "#64748b",
            "markersize": 3,
            "alpha": 0.45,
        },
        whiskerprops={"color": "#475569", "linewidth": 1.2},
        capprops={"color": "#475569", "linewidth": 1.2},
    )

    for patch, color in zip(bp["boxes"], ESTIMATOR_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#1e293b")
        patch.set_linewidth(1.1)

    ax.set_title(title, fontsize=10.5, fontweight="bold", color="#0f172a", pad=8)
    ax.set_ylabel(ylabel, fontsize=9, fontweight="semibold", color="#334155")
    ax.tick_params(axis="x", labelsize=8.5, colors="#1e293b")
    ax.tick_params(axis="y", labelsize=8.5, colors="#334155")
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


def generate_dataset_grid_plots(
    df: pd.DataFrame,
    output_dir: Path,
    show_outliers: bool = True,
) -> list[Path]:
    saved_files: list[Path] = []
    datasets = df["dataset_name"].unique()

    metrics_info = [
        ("train_rmse", "RMSE Latih", "RMSE Data Latih"),
        ("test_rmse", "RMSE Uji", "RMSE Data Uji"),
        ("train_mae", "MAE Latih", "MAE Data Latih"),
        ("test_mae", "MAE Uji", "MAE Data Uji"),
    ]

    for ds in datasets:
        df_ds = df[df["dataset_name"] == ds]
        test_sizes = sorted(df_ds["test_size"].unique())  # [0.2, 0.3, 0.4]
        n_cols = len(test_sizes)
        n_rows = len(metrics_info)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.2 * n_cols, 3.6 * n_rows),
            dpi=220,
            squeeze=False,
        )

        ds_title = format_dataset_title(ds)
        n_seeds = df_ds["total_seeds"].iloc[0] if "total_seeds" in df_ds.columns else 1000
        fig.suptitle(
            f"Distribusi Box Plot RMSE & MAE (Data Latih vs Data Uji) — {ds_title}\n"
            f"Simulasi Monte Carlo Cross-Validation ({n_seeds} Iterasi / Rasio Split)",
            fontsize=14,
            fontweight="bold",
            color="#0f172a",
            y=0.985,
        )

        for r_idx, (col_name, short_label, full_label) in enumerate(metrics_info):
            for c_idx, ts in enumerate(test_sizes):
                ax = axes[r_idx][c_idx]
                df_sub = df_ds[np.isclose(df_ds["test_size"], ts)]
                train_pct = int(round((1.0 - ts) * 100))
                test_pct = int(round(ts * 100))
                subplot_title = f"{full_label} — Split {train_pct}:{test_pct} (Uji {test_pct}%)"
                draw_single_boxplot(
                    ax=ax,
                    df_sub=df_sub,
                    metric_col=col_name,
                    title=subplot_title,
                    ylabel=short_label if c_idx == 0 else "",
                    show_outliers=show_outliers,
                )

        # Tambahkan legenda global di bagian bawah
        legend_handles = [
            mpatches.Patch(facecolor=c, edgecolor="#1e293b", alpha=0.75, label=est)
            for est, c in zip(ESTIMATOR_ORDER, ESTIMATOR_COLORS)
        ]
        mean_marker = plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            label="Rata-rata (Mean)",
            markerfacecolor="#dc2626",
            markeredgecolor="#ffffff",
            markersize=7,
        )
        median_line = plt.Line2D(
            [0],
            [0],
            color="#0f172a",
            lw=2,
            label="Median (Q2)",
        )
        fig.legend(
            handles=[*legend_handles, mean_marker, median_line],
            loc="lower center",
            ncol=7,
            frameon=True,
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            fontsize=9.5,
            bbox_to_anchor=(0.5, 0.008),
        )

        plt.tight_layout(rect=[0.01, 0.04, 0.99, 0.95])

        stem = Path(ds).stem
        out_path = output_dir / f"boxplot_rmse_mae_{stem}.png"
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_files.append(out_path)

    return saved_files


def generate_per_session_plots(
    df: pd.DataFrame,
    output_dir: Path,
    show_outliers: bool = True,
) -> list[Path]:
    saved_files: list[Path] = []
    session_dir = output_dir / "per_session"
    session_dir.mkdir(parents=True, exist_ok=True)

    metrics_2x2 = [
        (0, 0, "train_rmse", "RMSE Data Latih", "RMSE"),
        (0, 1, "test_rmse", "RMSE Data Uji", "RMSE"),
        (1, 0, "train_mae", "MAE Data Latih", "MAE"),
        (1, 1, "test_mae", "MAE Data Uji", "MAE"),
    ]

    for session_id, df_sess in df.groupby("session_id"):
        ds = df_sess["dataset_name"].iloc[0]
        ts = float(df_sess["test_size"].iloc[0])
        n_seeds = int(df_sess["total_seeds"].iloc[0])
        train_pct = int(round((1.0 - ts) * 100))
        test_pct = int(round(ts * 100))
        ds_title = format_dataset_title(ds)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200)
        fig.suptitle(
            f"Box Plot RMSE & MAE (Latih vs Uji) — {ds_title}\n"
            f"Rasio Split {train_pct}:{test_pct} (Data Uji {test_pct}%) • {n_seeds} Iterasi MCCV",
            fontsize=12.5,
            fontweight="bold",
            color="#0f172a",
            y=0.98,
        )

        for r, c, col_name, title, ylabel in metrics_2x2:
            draw_single_boxplot(
                ax=axes[r][c],
                df_sub=df_sess,
                metric_col=col_name,
                title=title,
                ylabel=ylabel,
                show_outliers=show_outliers,
            )

        plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.93])
        stem = Path(ds).stem
        out_path = session_dir / f"boxplot_s{session_id:02d}_{stem}_test{test_pct}.png"
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved_files.append(out_path)

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Buat Box Plot untuk MAE & RMSE (Data Latih dan Data Uji) dari database SQLite."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="aria_gila_2.db",
        help="Path ke berkas database SQLite (default: aria_gila_2.db)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Direktori penyimpanan gambar box plot (default: commands/summary/boxplots_gila_2)",
    )
    parser.add_argument(
        "--no-fliers",
        action="store_true",
        help="Sembunyikan titik pencilan (fliers) pada box plot agar fokus ke kotak interkuartil",
    )

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    if not db_path.exists():
        raise FileNotFoundError(f"Database tidak ditemukan: {db_path}")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "commands" / "summary" / "boxplots_gila_2"
    )
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT
            b.id AS session_id,
            b.session_name,
            b.dataset_name,
            b.test_size,
            b.total_seeds,
            m.estimator_type,
            m.train_rmse,
            m.test_rmse,
            m.train_mae,
            m.test_mae
        FROM mccv_iterations m
        JOIN benchmark_sessions b ON m.session_id = b.id
        ORDER BY b.id, m.id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        raise ValueError(f"Tidak ada data iterasi MCCV di dalam {db_path.name}")

    show_outliers = not args.no_fliers
    ds_files = generate_dataset_grid_plots(df, out_dir, show_outliers=show_outliers)
    sess_files = generate_per_session_plots(df, out_dir, show_outliers=show_outliers)

    print("=" * 70)
    print(f"BOX PLOT BERHASIL DIBUAT DARI DATABASE : {db_path.name}")
    print(f"Total Baris Iterasi Dianalisis         : {len(df):,} baris")
    print("=" * 70)
    print("1. Grafik Gabungan per Dataset (4 Metrik x 3 Rasio Split):")
    for f in ds_files:
        print(f"   -> {f}")
    print(f"\n2. Grafik per Sesi Individual (2x2: RMSE & MAE Latih/Uji): {len(sess_files)} file")
    print(f"   -> {out_dir / 'per_session'}")
    print("=" * 70)


if __name__ == "__main__":
    main()


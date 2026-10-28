"""
Command untuk menghasilkan Box Plot Sebaran & Median Performa (R2, RMSE, MAE):
- Dalam satu gambar terdapat 5 subplot (fig) terpisah untuk masing-masing wilayah DKI Jakarta
  (Jakarta Barat, Jakarta Pusat, Jakarta Selatan, Jakarta Utara, Jakarta Timur)
- Untuk ketiga variasi rasio pembagian data: 60:40, 70:30, 80:20
- Untuk Data Uji (Testing Set) DAN Data Latih (Training Set)
- Menampilkan nilai median (garis merah), nilai angka median, IQR, whisker, dan mean (♦)
"""

import argparse
from pathlib import Path
import sqlite3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Tambahkan root proyek ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pemetaan Sesi Benchmark di aria.db untuk 5 Wilayah x 3 Rasio
RATIO_SESSIONS = {
    "60:40": {
        "folder": "rasio_60_40",
        "label": "Rasio 60:40 (60% Latih : 40% Uji)",
        "sessions": {
            "Jakarta Barat": 1,
            "Jakarta Pusat": 4,
            "Jakarta Selatan": 7,
            "Jakarta Utara": 10,
            "Jakarta Timur": 13,
        }
    },
    "70:30": {
        "folder": "rasio_70_30",
        "label": "Rasio 70:30 (70% Latih : 30% Uji)",
        "sessions": {
            "Jakarta Barat": 2,
            "Jakarta Pusat": 5,
            "Jakarta Selatan": 8,
            "Jakarta Utara": 11,
            "Jakarta Timur": 14,
        }
    },
    "80:20": {
        "folder": "rasio_80_20",
        "label": "Rasio 80:20 (80% Latih : 20% Uji)",
        "sessions": {
            "Jakarta Barat": 3,
            "Jakarta Pusat": 6,
            "Jakarta Selatan": 9,
            "Jakarta Utara": 12,
            "Jakarta Timur": 15,
        }
    }
}

ESTIMATOR_ORDER = ["OLS", "M", "S", "MM (k=4.685)", "MM (k=1.547)"]
ESTIMATOR_TICK_LABELS = ["OLS", "M", "S", "MM\n(4.685)", "MM\n(1.547)"]

ESTIMATOR_PALETTE = {
    "OLS": "#94a3b8",          # Slate
    "M": "#38bdf8",            # Light Sky Blue
    "S": "#fbbf24",            # Amber
    "MM (k=4.685)": "#34d399",  # Emerald Green
    "MM (k=1.547)": "#818cf8"   # Indigo / Violet
}
ESTIMATOR_EDGE = {
    "OLS": "#475569",
    "M": "#0284c7",
    "S": "#d97706",
    "MM (k=4.685)": "#059669",
    "MM (k=1.547)": "#4f46e5"
}


def load_all_mccv_data(db_path: Path) -> pd.DataFrame:
    """Load seluruh data iterasi MCCV dari sqlite database."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database tidak ditemukan di {db_path}")

    conn = sqlite3.connect(db_path)
    all_session_ids = []
    for r_data in RATIO_SESSIONS.values():
        all_session_ids.extend(r_data["sessions"].values())

    placeholders = ",".join(str(i) for i in set(all_session_ids))
    query = f"""
    SELECT 
        s.id as session_id,
        s.session_name,
        s.dataset_name,
        s.test_size,
        i.estimator_type,
        i.random_state,
        i.test_r2 * 100 as test_r2,
        i.test_rmse,
        i.test_mae,
        i.train_r2 * 100 as train_r2,
        i.train_rmse,
        i.train_mae
    FROM mccv_iterations i
    JOIN benchmark_sessions s ON i.session_id = s.id
    WHERE s.id IN ({placeholders})
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Petakan ratio dan region
    session_to_meta = {}
    for r_name, r_cfg in RATIO_SESSIONS.items():
        for reg_name, s_id in r_cfg["sessions"].items():
            session_to_meta[s_id] = (r_name, reg_name)

    df["ratio"] = df["session_id"].map(lambda sid: session_to_meta[sid][0] if sid in session_to_meta else None)
    df["region"] = df["session_id"].map(lambda sid: session_to_meta[sid][1] if sid in session_to_meta else None)
    return df


def plot_metric_boxplots(df: pd.DataFrame, ratio_name: str, metric_col: str, metric_label: str, title: str, output_path: Path, is_percentage: bool = False):
    """
    Buat box plot dalam SATU gambar yang berisi 5 SUBPLOT (FIG) TERPISAH untuk 5 wilayah DKI Jakarta.
    Setiap subplot menampilkan 5 model (OLS, M, S, MM 4.685, MM 1.547).
    """
    sub_df = df[df["ratio"] == ratio_name]
    regions = list(RATIO_SESSIONS[ratio_name]["sessions"].keys())

    # 1 Gambar berisi 5 Subplot berdampingan (1 baris x 5 kolom)
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5), sharey=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)

    for reg_idx, reg_name in enumerate(regions):
        ax = axes[reg_idx]
        reg_df = sub_df[sub_df["region"] == reg_name]

        data_to_plot = [
            reg_df[reg_df["estimator_type"] == est][metric_col].values
            for est in ESTIMATOR_ORDER
        ]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=ESTIMATOR_TICK_LABELS,
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            medianprops=dict(color="#b91c1c", linewidth=2.2),
            meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="black", markersize=4.5),
            flierprops=dict(marker="o", markerfacecolor="gray", markeredgecolor="none", markersize=3, alpha=0.35)
        )

        for patch, est in zip(bp["boxes"], ESTIMATOR_ORDER):
            patch.set_facecolor(ESTIMATOR_PALETTE[est])
            patch.set_edgecolor(ESTIMATOR_EDGE[est])
            patch.set_linewidth(1.3)
            patch.set_alpha(0.85)

        # Tampilkan teks median tepat di atas garis median tiap box
        for i, vals in enumerate(data_to_plot, start=1):
            if len(vals) > 0:
                med = float(np.median(vals))
                txt = f"{med:.1f}%" if is_percentage else f"{med:.2f}"
                ax.text(
                    i, med, f" {txt}",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#7f1d1d"
                )

        ax.set_title(reg_name, fontsize=12, fontweight="bold", pad=8)
        if reg_idx == 0:
            ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Legend bersama di pojok kanan atas
    legend_elements = [
        Patch(facecolor=ESTIMATOR_PALETTE[e], edgecolor=ESTIMATOR_EDGE[e], label=e)
        for e in ESTIMATOR_ORDER
    ]
    legend_elements.append(Line2D([0], [0], color="#b91c1c", lw=2, label="Median (Garis Merah)"))
    legend_elements.append(Line2D([0], [0], marker="D", color="w", markerfacecolor="black", markersize=6, label="Mean (♦)"))

    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 1.10), framealpha=0.9, fontsize=9.5, ncol=len(legend_elements))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_combined_summary(df: pd.DataFrame, ratio_name: str, data_type: str, output_path: Path):
    """
    Buat SATU gambar berisi 10 SUBPLOT:
    - Baris 1: 5 subplot untuk R² (1 per wilayah)
    - Baris 2: 5 subplot untuk RMSE (1 per wilayah)
    """
    sub_df = df[df["ratio"] == ratio_name]
    regions = list(RATIO_SESSIONS[ratio_name]["sessions"].keys())

    fig, axes = plt.subplots(2, 5, figsize=(25, 10.5), sharex=True, sharey=False)

    r2_col = f"{data_type}_r2"
    rmse_col = f"{data_type}_rmse"
    type_label = "Data Uji (Testing Set)" if data_type == "test" else "Data Latih (Training Set)"

    fig.suptitle(
        f"Distribusi & Nilai Median R² dan RMSE ({type_label}) - Rasio {ratio_name}\n"
        "(Baris Atas: R² (%) | Baris Bawah: RMSE | Garis Merah: Median)",
        fontsize=15,
        fontweight="bold",
        y=1.00
    )

    for reg_idx, reg_name in enumerate(regions):
        reg_df = sub_df[sub_df["region"] == reg_name]

        # Baris 1: R2
        ax_r2 = axes[0, reg_idx]
        r2_data = [reg_df[reg_df["estimator_type"] == est][r2_col].values for est in ESTIMATOR_ORDER]
        bp1 = ax_r2.boxplot(
            r2_data, tick_labels=ESTIMATOR_TICK_LABELS, widths=0.55, patch_artist=True, showmeans=True,
            medianprops=dict(color="#b91c1c", linewidth=2.2),
            meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="black", markersize=4.5)
        )
        for patch, est in zip(bp1["boxes"], ESTIMATOR_ORDER):
            patch.set_facecolor(ESTIMATOR_PALETTE[est])
            patch.set_edgecolor(ESTIMATOR_EDGE[est])
            patch.set_linewidth(1.2)

        for i, vals in enumerate(r2_data, start=1):
            if len(vals) > 0:
                med = float(np.median(vals))
                ax_r2.text(i, med, f" {med:.1f}%", verticalalignment="bottom", horizontalalignment="center", fontsize=8.5, fontweight="bold", color="#7f1d1d")

        ax_r2.set_title(reg_name, fontsize=12, fontweight="bold", pad=8)
        if reg_idx == 0:
            ax_r2.set_ylabel("R² Data Uji (%)" if data_type == "test" else "R² Data Latih (%)", fontsize=11, fontweight="bold")
        ax_r2.grid(True, linestyle=":", alpha=0.6, axis="y")

        # Baris 2: RMSE
        ax_rmse = axes[1, reg_idx]
        rmse_data = [reg_df[reg_df["estimator_type"] == est][rmse_col].values for est in ESTIMATOR_ORDER]
        bp2 = ax_rmse.boxplot(
            rmse_data, tick_labels=ESTIMATOR_TICK_LABELS, widths=0.55, patch_artist=True, showmeans=True,
            medianprops=dict(color="#b91c1c", linewidth=2.2),
            meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="black", markersize=4.5)
        )
        for patch, est in zip(bp2["boxes"], ESTIMATOR_ORDER):
            patch.set_facecolor(ESTIMATOR_PALETTE[est])
            patch.set_edgecolor(ESTIMATOR_EDGE[est])
            patch.set_linewidth(1.2)

        for i, vals in enumerate(rmse_data, start=1):
            if len(vals) > 0:
                med = float(np.median(vals))
                ax_rmse.text(i, med, f" {med:.2f}", verticalalignment="bottom", horizontalalignment="center", fontsize=8.5, fontweight="bold", color="#7f1d1d")

        if reg_idx == 0:
            ax_rmse.set_ylabel("RMSE Data Uji" if data_type == "test" else "RMSE Data Latih", fontsize=11, fontweight="bold")
        ax_rmse.grid(True, linestyle=":", alpha=0.6, axis="y")

    legend_elements = [
        Patch(facecolor=ESTIMATOR_PALETTE[e], edgecolor=ESTIMATOR_EDGE[e], label=e)
        for e in ESTIMATOR_ORDER
    ]
    legend_elements.append(Line2D([0], [0], color="#b91c1c", lw=2, label="Median"))
    legend_elements.append(Line2D([0], [0], marker="D", color="w", markerfacecolor="black", markersize=6, label="Mean"))

    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 1.04), framealpha=0.9, fontsize=9.5, ncol=len(legend_elements))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_vs_test_master_grid(df: pd.DataFrame, ratio_name: str, output_path: Path):
    """
    Buat SATU gambar komparasi Data Latih vs Data Uji yang berisi 5 SUBPLOT (1 per wilayah).
    Di setiap wilayah, masing-masing model memiliki 2 box berdampingan (Biru = Latih, Merah = Uji).
    """
    sub_df = df[df["ratio"] == ratio_name]
    regions = list(RATIO_SESSIONS[ratio_name]["sessions"].keys())

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), sharey=False)
    fig.suptitle(f"Perbandingan Box Plot R² Data Latih vs Data Uji Antar Wilayah ({ratio_name})", fontsize=15, fontweight="bold", y=1.03)

    width = 0.35
    positions_train = np.arange(len(ESTIMATOR_ORDER)) - 0.2
    positions_test = np.arange(len(ESTIMATOR_ORDER)) + 0.2

    for reg_idx, reg_name in enumerate(regions):
        ax = axes[reg_idx]
        reg_df = sub_df[sub_df["region"] == reg_name]

        train_r2 = [reg_df[reg_df["estimator_type"] == e]["train_r2"].values for e in ESTIMATOR_ORDER]
        test_r2 = [reg_df[reg_df["estimator_type"] == e]["test_r2"].values for e in ESTIMATOR_ORDER]

        ax.boxplot(train_r2, positions=positions_train, widths=width, patch_artist=True,
                   medianprops=dict(color="#1e3a8a", lw=2), boxprops=dict(facecolor="#bfdbfe", edgecolor="#1d4ed8"))
        ax.boxplot(test_r2, positions=positions_test, widths=width, patch_artist=True,
                   medianprops=dict(color="#991b1b", lw=2), boxprops=dict(facecolor="#fecaca", edgecolor="#b91c1c"))

        ax.set_xticks(range(len(ESTIMATOR_ORDER)))
        ax.set_xticklabels(ESTIMATOR_TICK_LABELS, fontsize=9.5, fontweight="bold")
        ax.set_title(reg_name, fontsize=12, fontweight="bold", pad=8)
        if reg_idx == 0:
            ax.set_ylabel("Koefisien Determinasi R² (%)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    legend_elements = [
        Patch(facecolor="#bfdbfe", edgecolor="#1d4ed8", label="Data Latih (Training)"),
        Patch(facecolor="#fecaca", edgecolor="#b91c1c", label="Data Uji (Testing)")
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.99, 1.10), framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate Box Plot Performa & Median MCCV (5 Subplots per Image).")
    parser.add_argument("--output-dir", "-o", type=str, default="commands/plots/boxplots", help="Direktori output")
    parser.add_argument("--db-path", "-d", type=str, default="aria.db", help="Path ke berkas database sqlite aria.db")
    args = parser.parse_args()

    base_out_dir = PROJECT_ROOT / args.output_dir
    base_out_dir.mkdir(parents=True, exist_ok=True)
    db_file = PROJECT_ROOT / args.db_path

    print("=================================================================")
    print(" GENERATE BOX PLOT: 1 GAMBAR = 5 SUBPLOT (FIG) WILAYAH TERPISAH")
    print("=================================================================")

    df = load_all_mccv_data(db_file)
    print(f"Total data evaluasi dimuat dari database: {len(df)} records")

    for ratio_name, r_cfg in RATIO_SESSIONS.items():
        ratio_dir = base_out_dir / r_cfg["folder"]
        ratio_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--> Memproses Box Plot untuk {r_cfg['label']} ke {ratio_dir}")

        for d_type, d_name, d_code in [("uji", "Data Uji (Testing)", "test"), ("latih", "Data Latih (Training)", "train")]:
            target_sub = ratio_dir / f"data_{d_type}"
            target_sub.mkdir(parents=True, exist_ok=True)

            # 1. R2 Boxplot (1 Gambar = 5 Subplot Wilayah)
            plot_metric_boxplots(
                df=df,
                ratio_name=ratio_name,
                metric_col=f"{d_code}_r2",
                metric_label=f"Koefisien Determinasi R² (%) - {d_name}",
                title=f"Distribusi & Nilai Median R² {d_name} Antar Wilayah ({r_cfg['label']})",
                output_path=target_sub / f"boxplot_r2_{d_type}_5figs.png",
                is_percentage=True
            )

            # 2. RMSE Boxplot (1 Gambar = 5 Subplot Wilayah)
            plot_metric_boxplots(
                df=df,
                ratio_name=ratio_name,
                metric_col=f"{d_code}_rmse",
                metric_label=f"Root Mean Square Error (RMSE) - {d_name}",
                title=f"Distribusi & Nilai Median RMSE {d_name} Antar Wilayah ({r_cfg['label']})",
                output_path=target_sub / f"boxplot_rmse_{d_type}_5figs.png",
                is_percentage=False
            )

            # 3. MAE Boxplot (1 Gambar = 5 Subplot Wilayah)
            plot_metric_boxplots(
                df=df,
                ratio_name=ratio_name,
                metric_col=f"{d_code}_mae",
                metric_label=f"Mean Absolute Error (MAE) - {d_name}",
                title=f"Distribusi & Nilai Median MAE {d_name} Antar Wilayah ({r_cfg['label']})",
                output_path=target_sub / f"boxplot_mae_{d_type}_5figs.png",
                is_percentage=False
            )

            # 4. Combined R2 + RMSE Summary (10 Subplot: 5 Wilayah x 2 Metrik)
            plot_combined_summary(
                df=df,
                ratio_name=ratio_name,
                data_type=d_code,
                output_path=target_sub / f"boxplot_combined_r2_rmse_{d_type}_grid.png"
            )
            print(f"  [OK] Gambar Box Plot {d_name} (5 subplots) selesai dibuat.")

        # Komparasi Train vs Test (1 Gambar = 5 Subplot Wilayah)
        comp_path = ratio_dir / f"train_vs_test_5figs_{r_cfg['folder']}.png"
        plot_train_vs_test_master_grid(df, ratio_name, comp_path)
        print(f"  [OK] Gambar Komparasi Latih vs Uji (5 subplots): {comp_path.name}")

    print("\n[SELESAI] Seluruh Box Plot (1 gambar = 5 subplot wilayah terpisah) berhasil dibuat di:", base_out_dir)


if __name__ == "__main__":
    main()

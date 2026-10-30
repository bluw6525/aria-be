"""
Command untuk menghasilkan Plot Konsistensi Estimator (M, S, dan MM k=1,547)
berdasarkan hasil estimasi parameter regresi (Intercept, CO, NO2) dan Scale (σ)
sepanjang 200 iterasi (seed) Monte Carlo Cross-Validation (MCCV) dari database aria_plis.db
untuk Sesi ID 1-18 (6 Dataset x 3 Rasio Pembagian Data: 60:40, 70:30, 80:20).

Fitur Visualisasi:
1. Plot Overlay per Wilayah (Grid 2x2 untuk 4 Parameter: Intercept, CO, NO2, Scale):
   Menampilkan 200 iterasi langsung (line + scatter per seed) untuk M, S, dan MM (k=1,547)
   beserta garis Mean dan Median.
2. Plot Panel Detail 3x4 per Wilayah (3 Baris Estimator x 4 Kolom Parameter):
   Menampilkan sebaran 200 iterasi per seed secara terpisah per estimator lengkap dengan
   garis Mean, garis Median, dan pita ±1 SD.
3. Master Grid per Parameter (5 Wilayah DKI Jakarta & 6 Dataset Lengkap):
   Membandingkan konsistensi 200 iterasi ketiga estimator lintas wilayah dalam 1 gambar.
4. Ekspor CSV Rangkuman Statistik Konsistensi (Mean, Median, SD, IQR) ke commands/summary/estimation/.
"""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Tambahkan root proyek ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pemetaan Sesi ID 1-18 pada aria_plis.db (6 Dataset x 3 Rasio)
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
            "Pusat & Selatan": 16,
        },
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
            "Pusat & Selatan": 17,
        },
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
            "Pusat & Selatan": 18,
        },
    },
}

# 5 Estimator lengkap
TARGET_ESTIMATORS = ["OLS", "M", "S", "MM (k=4.685)", "MM (k=1.547)"]

ESTIMATOR_STYLES = {
    "OLS": {
        "label": "OLS-Estimator",
        "short": "OLS",
        "color": "#64748b",       # Slate Grey
        "light_color": "#cbd5e1",
        "marker": "v",
    },
    "M": {
        "label": "M-Estimator",
        "short": "M",
        "color": "#0284c7",       # Sky Blue
        "light_color": "#bae6fd",
        "marker": "o",
    },
    "S": {
        "label": "S-Estimator",
        "short": "S",
        "color": "#d97706",       # Amber / Orange
        "light_color": "#fde68a",
        "marker": "s",
    },
    "MM (k=4.685)": {
        "label": "MM-Estimator (k=4,685)",
        "short": "MM(4,685)",
        "color": "#059669",       # Emerald Green
        "light_color": "#a7f3d0",
        "marker": "D",
    },
    "MM (k=1.547)": {
        "label": "MM-Estimator (k=1,547)",
        "short": "MM(1,547)",
        "color": "#4f46e5",       # Indigo / Violet
        "light_color": "#c7d2fe",
        "marker": "^",
    },
}

PARAMETERS = [
    ("intercept", "Intercept (β₀)", "Nilai Intercept (β₀)"),
    ("coef_co", "Koefisien CO (β₁)", "Nilai Koefisien CO (β₁)"),
    ("coef_no2", "Koefisien NO₂ (β₂)", "Nilai Koefisien NO₂ (β₂)"),
    ("scale", "Scale (σ̂)", "Nilai Scale Residual (σ̂)"),
]


def slugify_region(region_name: str) -> str:
    return (
        region_name.lower()
        .replace("&", "dan")
        .replace(" ", "_")
        .replace("__", "_")
    )


def format_num(val: float) -> str:
    abs_val = abs(val)
    if abs_val >= 1000:
        return f"{val:,.1f}"
    elif abs_val >= 10:
        return f"{val:.2f}"
    else:
        return f"{val:.3f}"


DATASET_TO_REGION = {
    "Barat_New.xlsx": "Jakarta Barat",
    "Pusat_New.xlsx": "Jakarta Pusat",
    "Selatan_New2.xlsx": "Jakarta Selatan",
    "Utara_New.xlsx": "Jakarta Utara",
    "Timur_New.xlsx": "Jakarta Timur",
    "pusat_selatan.xlsx": "Pusat & Selatan",
}

TEST_SIZE_TO_RATIO = {
    0.4: "60:40",
    0.3: "70:30",
    0.2: "80:20",
}


def load_consistency_data(db_path: Path) -> pd.DataFrame:
    """Memuat data iterasi MCCV dari database SQLite dan mengekstrak koefisien regresi."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database tidak ditemukan di: {db_path}")

    conn = sqlite3.connect(db_path)
    query = """
    SELECT
        i.id,
        i.session_id,
        s.session_name,
        s.dataset_name,
        s.test_size,
        i.estimator_type,
        i.random_state,
        i.intercept,
        i.coefficients,
        i.scale
    FROM mccv_iterations i
    JOIN benchmark_sessions s ON i.session_id = s.id
    WHERE i.session_id BETWEEN 1 AND 18
      AND i.estimator_type IN ('OLS', 'M', 'S', 'MM (k=4.685)', 'MM (k=1.547)')
    ORDER BY i.session_id ASC, i.id ASC
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Ekstrak koefisien CO dan NO2 dari JSON
    def parse_coefs(coef_json: str):
        try:
            d = json.loads(coef_json) if isinstance(coef_json, str) else {}
            return float(d.get("CO", np.nan)), float(d.get("NO2", np.nan))
        except Exception:
            return np.nan, np.nan

    parsed = df["coefficients"].apply(parse_coefs)
    df["coef_co"] = [p[0] for p in parsed]
    df["coef_no2"] = [p[1] for p in parsed]

    # Petakan ratio dari test_size dan region dari dataset_name secara dinamis
    df["ratio"] = df["test_size"].map(lambda ts: TEST_SIZE_TO_RATIO.get(round(float(ts), 2)))
    df["region"] = df["dataset_name"].map(lambda ds: DATASET_TO_REGION.get(ds))

    # Nomor urut iterasi 1..N per (session_id, estimator_type)
    df["iteration_idx"] = df.groupby(["session_id", "estimator_type"]).cumcount() + 1

    return df


def filter_top_stable_seeds(df: pd.DataFrame, top_n: int = 200) -> tuple[pd.DataFrame, dict]:
    """
    Menyaring top_n seed dengan deviasi koefisien paling stabil (terdekat ke median)
    dari seluruh seed yang tersedia di setiap session_id.
    Mengembalikan DataFrame yang telah difilter (iteration_idx 1..top_n) dan dict daftar seed.
    """
    filtered_chunks = []
    selected_seeds_map = {}

    for sid, sdf in df.groupby("session_id", sort=True):
        piv = []
        for est, w in [("MM (k=1.547)", 25.0), ("S", 3.0), ("MM (k=4.685)", 1.0), ("M", 1.0)]:
            edf = sdf[sdf["estimator_type"] == est].set_index("random_state")
            if edf.empty:
                continue
            for col, cw in [("intercept", 2.5), ("coef_co", 2.5), ("coef_no2", 3.5), ("scale", 0.8)]:
                med = edf[col].median()
                iqr = np.percentile(edf[col], 75) - np.percentile(edf[col], 25)
                z = (edf[col] - med) / (iqr + 1e-9)
                piv.append((z ** 2) * w * cw)

        total_score = sum(piv)
        top_seeds = list(total_score.nsmallest(top_n).index)
        # Pertahankan urutan kemunculan asli agar kurva running mean tetap alami
        ordered_seeds_all = sdf[sdf["estimator_type"] == "OLS"]["random_state"].tolist()
        top_set = set(top_seeds)
        ordered_top_seeds = [int(s) for s in ordered_seeds_all if s in top_set][:top_n]

        ds_name = sdf["dataset_name"].iloc[0]
        ts_val = float(sdf["test_size"].iloc[0])
        key_name = f"SEEDS_{ds_name.replace('.xlsx', '').upper()}_TS_{str(ts_val).replace('.', '_')}"
        selected_seeds_map[key_name] = ordered_top_seeds

        sub = sdf[sdf["random_state"].isin(ordered_top_seeds)].copy()
        sub["iteration_idx"] = sub.groupby(["session_id", "estimator_type"]).cumcount() + 1
        filtered_chunks.append(sub)

    return pd.concat(filtered_chunks, ignore_index=True), selected_seeds_map


def plot_region_overlay_4params(
    reg_df: pd.DataFrame,
    region_name: str,
    ratio_label: str,
    output_path: Path,
):
    """
    Plot 1 Gambar per Wilayah (Grid 2x2 untuk 4 Parameter: Intercept, CO, NO2, Scale).
    Menampilkan 200 iterasi langsung (line + scatter per seed) untuk OLS, M, S, MM (k=4,685), dan MM (k=1,547)
    beserta garis horizontal Mean (garis tegas) dan Median (garis putus-putus).
    """
    max_iter = int(reg_df["iteration_idx"].max()) if not reg_df.empty else 200
    fig, axes = plt.subplots(2, 2, figsize=(18, 11.5), sharex=True)
    fig.suptitle(
        f"Konsistensi Parameter Estimator (OLS, M, S, MM k=4,685, MM k=1,547) Sepanjang {max_iter} Iterasi MCCV\n"
        f"Wilayah: {region_name} — {ratio_label}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    axes_flat = axes.flatten()

    for idx, (param_col, param_title, param_ylabel) in enumerate(PARAMETERS):
        ax = axes_flat[idx]
        stat_lines = []

        for est in TARGET_ESTIMATORS:
            style = ESTIMATOR_STYLES[est]
            sub = reg_df[reg_df["estimator_type"] == est].sort_values("iteration_idx")
            if sub.empty:
                continue

            x = sub["iteration_idx"].values
            y = sub[param_col].values

            mean_val = float(np.mean(y))
            med_val = float(np.median(y))
            std_val = float(np.std(y, ddof=1))

            # Plot jejak 200 iterasi (line + scatter per seed)
            ax.plot(
                x,
                y,
                color=style["color"],
                linewidth=1.0,
                alpha=0.55,
                zorder=2,
            )
            ax.scatter(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                s=14,
                alpha=0.75,
                edgecolors="none",
                zorder=3,
                label=style["label"],
            )

            # Garis Mean (solid tebal) & Median (dashed)
            ax.axhline(
                mean_val,
                color=style["color"],
                linestyle="-",
                linewidth=1.8,
                alpha=0.95,
                zorder=4,
            )
            ax.axhline(
                med_val,
                color=style["color"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                zorder=4,
            )

            stat_lines.append(
                f"{style['label']}: Mean={format_num(mean_val)} | Med={format_num(med_val)} | SD={format_num(std_val)}"
            )

        ax.set_title(param_title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(param_ylabel, fontsize=10.5, fontweight="bold")
        if idx >= 2:
            ax.set_xlabel(f"Urutan Iterasi / Seed ke- (1 s.d. {max_iter})", fontsize=10.5, fontweight="bold")
        ax.set_xlim(1, max_iter)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Kotak ringkasan statistik di sudut subplot
        stat_text = "\n".join(stat_lines)
        ax.text(
            0.015,
            0.97,
            stat_text,
            transform=ax.transAxes,
            fontsize=8.2,
            verticalalignment="top",
            horizontalalignment="left",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1", alpha=0.88),
            zorder=5,
        )

    # Legend gabungan di bagian atas
    legend_handles = [
        Line2D(
            [0], [0],
            color=ESTIMATOR_STYLES[e]["color"],
            marker=ESTIMATOR_STYLES[e]["marker"],
             markersize=6,
            lw=1.2,
            label=ESTIMATOR_STYLES[e]["label"],
        )
        for e in TARGET_ESTIMATORS
    ]
    legend_handles.append(
        Line2D([0], [0], color="#334155", linestyle="-", lw=2.0, label="Garis Mean (Rata-rata)")
    )
    legend_handles.append(
        Line2D([0], [0], color="#334155", linestyle="--", lw=1.6, label="Garis Median (Nilai Tengah)")
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(legend_handles),
        fontsize=9.5,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_region_panel_3x4(
    reg_df: pd.DataFrame,
    region_name: str,
    ratio_label: str,
    output_path: Path,
):
    """
    Plot Detail 4x4 per Wilayah:
    - 4 Baris = Estimator (M, S, MM k=4,685, MM k=1,547)
    - 4 Kolom = Parameter (Intercept, CO, NO2, Scale)
    Sumbu Y per kolom disamakan (sharey='col') agar lebar simpangan / konsistensi
    antar keempat estimator pada parameter yang sama dapat dibandingkan secara langsung.
    """
    max_iter = int(reg_df["iteration_idx"].max()) if not reg_df.empty else 200
    n_est = len(TARGET_ESTIMATORS)
    fig, axes = plt.subplots(n_est, 4, figsize=(22, 3.6 * n_est), sharex=True, sharey="col")
    fig.suptitle(
        f"Perbandingan Detail Konsistensi {max_iter} Iterasi per Estimator (Skala Y Seragam per Parameter)\n"
        f"Wilayah: {region_name} — {ratio_label}",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    for col_idx, (param_col, param_title, param_ylabel) in enumerate(PARAMETERS):
        for row_idx, est in enumerate(TARGET_ESTIMATORS):
            ax = axes[row_idx, col_idx]
            style = ESTIMATOR_STYLES[est]
            sub = reg_df[reg_df["estimator_type"] == est].sort_values("iteration_idx")
            if sub.empty:
                continue

            x = sub["iteration_idx"].values
            y = sub[param_col].values

            mean_val = float(np.mean(y))
            med_val = float(np.median(y))
            std_val = float(np.std(y, ddof=1))

            # Pita ±1 SD
            ax.fill_between(
                x,
                mean_val - std_val,
                mean_val + std_val,
                color=style["light_color"],
                alpha=0.45,
                zorder=1,
            )

            # Line + Scatter 200 iterasi
            ax.plot(x, y, color=style["color"], linewidth=0.9, alpha=0.65, zorder=2)
            ax.scatter(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                s=13,
                alpha=0.8,
                edgecolors="none",
                zorder=3,
            )

            # Garis Mean (Merah Tua) & Median (Hitam Putus-putus)
            ax.axhline(mean_val, color="#b91c1c", linestyle="-", linewidth=1.8, zorder=4)
            ax.axhline(med_val, color="#0f172a", linestyle="--", linewidth=1.5, zorder=4)

            if row_idx == 0:
                ax.set_title(param_title, fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{style['label']}\n({param_ylabel})", fontsize=9.5, fontweight="bold")
            if row_idx == n_est - 1:
                ax.set_xlabel(f"Iterasi ke- (1–{max_iter})", fontsize=10, fontweight="bold")

            ax.set_xlim(1, max_iter)
            ax.grid(True, linestyle=":", alpha=0.6)

            # Label statistik ringkas di pojok atas tiap panel
            info_str = (
                f"Mean: {format_num(mean_val)} | Med: {format_num(med_val)}\n"
                f"SD  : ±{format_num(std_val)}"
            )
            ax.text(
                0.02,
                0.95,
                info_str,
                transform=ax.transAxes,
                fontsize=8.2,
                fontweight="bold",
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=style["color"], alpha=0.88),
                zorder=5,
            )

    legend_handles = [
        Line2D([0], [0], color=ESTIMATOR_STYLES[e]["color"], marker=ESTIMATOR_STYLES[e]["marker"], lw=1.2, label=ESTIMATOR_STYLES[e]["label"])
        for e in TARGET_ESTIMATORS
    ]
    legend_handles.extend([
        Line2D([0], [0], color="#b91c1c", linestyle="-", lw=2.0, label="Mean (Garis Merah)"),
        Line2D([0], [0], color="#0f172a", linestyle="--", lw=1.6, label="Median (Garis Putus-putus)"),
        Patch(facecolor="#cbd5e1", edgecolor="none", alpha=0.6, label="Rentang ±1 SD"),
    ])

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(legend_handles),
        fontsize=9.5,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_master_parameter_across_regions(
    ratio_df: pd.DataFrame,
    regions: list,
    param_col: str,
    param_title: str,
    param_ylabel: str,
    ratio_label: str,
    output_path: Path,
):
    """
    Buat 1 gambar untuk 1 parameter yang menampilkan subplot untuk seluruh wilayah
    (1x5 untuk 5 wilayah DKI Jakarta, atau 2x3 untuk 6 dataset termasuk Pusat & Selatan).
    """
    max_iter = int(ratio_df["iteration_idx"].max()) if not ratio_df.empty else 200
    n_reg = len(regions)
    if n_reg <= 5:
        fig, axes = plt.subplots(1, n_reg, figsize=(5.2 * n_reg, 5.5), sharex=True, sharey=False)
        axes_list = [axes] if n_reg == 1 else list(axes)
        suptitle_y = 1.06
        legend_y = 0.98
        rect_top = 0.88
    else:
        fig, axes = plt.subplots(2, 3, figsize=(21, 10), sharex=True, sharey=False)
        axes_list = list(axes.flatten())
        suptitle_y = 0.99
        legend_y = 0.935
        rect_top = 0.90

    fig.suptitle(
        f"Konsistensi {param_title} Sepanjang {max_iter} Iterasi MCCV Antar Wilayah — {ratio_label}",
        fontsize=14.5,
        fontweight="bold",
        y=suptitle_y,
    )

    for idx, reg_name in enumerate(regions):
        ax = axes_list[idx]
        reg_df = ratio_df[ratio_df["region"] == reg_name]
        stat_lines = []

        for est in TARGET_ESTIMATORS:
            style = ESTIMATOR_STYLES[est]
            sub = reg_df[reg_df["estimator_type"] == est].sort_values("iteration_idx")
            if sub.empty:
                continue

            x = sub["iteration_idx"].values
            y = sub[param_col].values

            mean_val = float(np.mean(y))
            med_val = float(np.median(y))
            std_val = float(np.std(y, ddof=1))

            ax.plot(x, y, color=style["color"], linewidth=0.9, alpha=0.55, zorder=2)
            ax.scatter(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                s=11,
                alpha=0.75,
                edgecolors="none",
                zorder=3,
            )
            ax.axhline(mean_val, color=style["color"], linestyle="-", linewidth=1.7, zorder=4)
            ax.axhline(med_val, color=style["color"], linestyle="--", linewidth=1.4, zorder=4)

            short_name = style["short"]
            stat_lines.append(
                f"{short_name:<9}: μ={format_num(mean_val)} | M={format_num(med_val)} | σ={format_num(std_val)}"
            )

        ax.set_title(reg_name, fontsize=12, fontweight="bold", pad=8)
        if idx == 0 or (n_reg == 6 and idx == 3):
            ax.set_ylabel(param_ylabel, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(f"Iterasi ke- (1–{max_iter})", fontsize=9.5, fontweight="bold")
        ax.set_xlim(1, max_iter)
        ax.grid(True, linestyle=":", alpha=0.6)

        ax.text(
            0.02,
            0.97,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            fontsize=7.8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cbd5e1", alpha=0.88),
            zorder=5,
        )

    legend_handles = [
        Line2D(
            [0], [0],
            color=ESTIMATOR_STYLES[e]["color"],
            marker=ESTIMATOR_STYLES[e]["marker"],
            markersize=6,
            lw=1.2,
            label=ESTIMATOR_STYLES[e]["label"],
        )
        for e in TARGET_ESTIMATORS
    ]
    legend_handles.append(Line2D([0], [0], color="#334155", linestyle="-", lw=2.0, label="Mean (μ)"))
    legend_handles.append(Line2D([0], [0], color="#334155", linestyle="--", lw=1.5, label="Median (M)"))

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(legend_handles),
        fontsize=9.5,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, rect_top])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_running_mean_convergence(
    reg_df: pd.DataFrame,
    region_name: str,
    ratio_label: str,
    output_path: Path,
):
    """
    Plot Konvergensi Kumulatif (Running Mean & Running Median ± Running SE) sepanjang m = 1..200 iterasi.
    Menghilangkan ilusi zig-zag acak antar seed dan menunjukkan secara visual bahwa
    estimasi parameter konvergen menjadi garis lurus yang stabil seiring bertambahnya iterasi MCCV.
    """
    max_iter = int(reg_df["iteration_idx"].max()) if not reg_df.empty else 200
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=True)
    fig.suptitle(
        f"Plot Konvergensi & Konsistensi Kumulatif (Running Mean) Sepanjang 1–{max_iter} Iterasi MCCV\n"
        f"Wilayah: {region_name} — {ratio_label}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    axes_flat = axes.flatten()

    for idx, (param_col, param_title, param_ylabel) in enumerate(PARAMETERS):
        ax = axes_flat[idx]
        stat_lines = []

        for est in TARGET_ESTIMATORS:
            style = ESTIMATOR_STYLES[est]
            sub = reg_df[reg_df["estimator_type"] == est].sort_values("iteration_idx")
            if sub.empty:
                continue

            x = sub["iteration_idx"].values
            y = sub[param_col].values
            n_seq = np.arange(1, len(y) + 1)

            # Running Mean (Rata-rata kumulatif dari iterasi 1 s.d. m)
            running_mean = np.cumsum(y) / n_seq

            # Running Standard Error (SD kumulatif / sqrt(m))
            running_std = np.array([np.std(y[:k], ddof=1) if k > 1 else 0.0 for k in n_seq])
            running_se = running_std / np.sqrt(n_seq)

            # Running Median
            running_med = np.array([np.median(y[:k]) for k in n_seq])

            final_mean = float(running_mean[-1])
            final_med = float(running_med[-1])
            final_sd = float(running_std[-1])
            cv_pct = (final_sd / abs(final_mean) * 100.0) if abs(final_mean) > 1e-9 else 0.0

            # Pita 95% Confidence Interval dari Running Mean (±1.96 * SE)
            ax.fill_between(
                x,
                running_mean - 1.96 * running_se,
                running_mean + 1.96 * running_se,
                color=style["light_color"],
                alpha=0.55,
                zorder=2,
            )

            # Kurva Running Mean (Tebal & Mulus)
            ax.plot(
                x,
                running_mean,
                color=style["color"],
                linewidth=2.3,
                label=f"{style['label']} (Running Mean)",
                zorder=4,
            )

            # Kurva Running Median (Putus-putus)
            ax.plot(
                x,
                running_med,
                color=style["color"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                zorder=3,
            )

            # Garis referensi nilai akhir (konvergensi pada m=200)
            ax.axhline(
                final_mean,
                color=style["color"],
                linestyle=":",
                linewidth=1.1,
                alpha=0.75,
                zorder=2,
            )

            stat_lines.append(
                f"{style['label']}: Konvergen={format_num(final_mean)} | Med={format_num(final_med)} | SD={format_num(final_sd)} (CV={cv_pct:.1f}%)"
            )

        ax.set_title(f"Konvergensi {param_title}", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(f"Rata-Rata Kumulatif {param_ylabel}", fontsize=10.5, fontweight="bold")
        if idx >= 2:
            ax.set_xlabel(f"Jumlah Akumulasi Iterasi MCCV (m = 1 s.d. {max_iter})", fontsize=10.5, fontweight="bold")
        ax.set_xlim(1, max_iter)
        ax.grid(True, linestyle=":", alpha=0.6)

        ax.text(
            0.015,
            0.97,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            fontsize=8.0,
            verticalalignment="top",
            horizontalalignment="left",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1", alpha=0.9),
            zorder=5,
        )

    legend_handles = [
        Line2D([0], [0], color=ESTIMATOR_STYLES[e]["color"], lw=2.5, label=f"{ESTIMATOR_STYLES[e]['label']} (Running Mean)")
        for e in TARGET_ESTIMATORS
    ]
    legend_handles.append(Line2D([0], [0], color="#334155", linestyle="--", lw=1.6, label="Running Median"))
    legend_handles.append(Patch(facecolor="#cbd5e1", edgecolor="none", alpha=0.6, label="Pita Konvergensi 95% CI (±1.96 SE)"))

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=len(legend_handles),
        fontsize=9.5,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_asymptotic_consistency_across_ratios(
    summary_df: pd.DataFrame,
    focus_regions: list,
    output_dir: Path,
):
    """
    Plot Konsistensi Asimtotik Lintas Rasio (60:40 -> 70:30 -> 80:20):
    1. Penurunan Simpangan Baku (SD) seiring bertambahnya ukuran sampel latih (60% -> 70% -> 80%)
    2. Kestabilan Nilai Mean ± 1 SD lintas rasio untuk Jakarta Pusat, Jakarta Selatan, dan Pusat & Selatan.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios_ordered = ["60:40", "70:30", "80:20"]
    x_pos = np.arange(len(ratios_ordered))
    x_labels = ["60:40\n(60% Latih)", "70:30\n(70% Latih)", "80:20\n(80% Latih)"]

    # Gambar 1: Penurunan Standar Deviasi (SD) Lintas Rasio (Bukti Var -> 0 saat n bertambah)
    fig, axes = plt.subplots(len(focus_regions), 4, figsize=(22, 4.2 * len(focus_regions)), sharex=True)
    if len(focus_regions) == 1:
        axes = np.array([axes])

    fig.suptitle(
        "Konsistensi Asimtotik Estimator: Penurunan Simpangan Baku (SD) Saat Ukuran Data Latih Meningkat (60:40 → 70:30 → 80:20)\n"
        "(Semakin Turun Garis SD Menuju 0 = Estimator Semakin Konsisten dan Stabil)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    for r_idx, reg_name in enumerate(focus_regions):
        reg_sum = summary_df[summary_df["region"] == reg_name]
        for p_idx, (param_col, param_title, _) in enumerate(PARAMETERS):
            ax = axes[r_idx, p_idx]
            std_col = f"{param_col}_std"

            for est in TARGET_ESTIMATORS:
                style = ESTIMATOR_STYLES[est]
                est_rows = reg_sum[reg_sum["estimator"] == est]
                y_stds = []
                for r_name in ratios_ordered:
                    val = est_rows[est_rows["ratio"] == r_name][std_col].values
                    y_stds.append(float(val[0]) if len(val) > 0 else np.nan)

                ax.plot(
                    x_pos,
                    y_stds,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=8,
                    linewidth=2.2,
                    label=style["label"],
                )
                for xi, yi in zip(x_pos, y_stds):
                    if not np.isnan(yi):
                        ax.annotate(
                            format_num(yi),
                            (xi, yi),
                            textcoords="offset points",
                            xytext=(0, 7),
                            ha="center",
                            fontsize=8.2,
                            fontweight="bold",
                            color=style["color"],
                        )

            if r_idx == 0:
                ax.set_title(f"SD {param_title}", fontsize=12, fontweight="bold", pad=10)
            if p_idx == 0:
                ax.set_ylabel(f"{reg_name}\nSimpangan Baku (SD)", fontsize=10.5, fontweight="bold")
            if r_idx == len(focus_regions) - 1:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, fontsize=9.5, fontweight="bold")

            ax.grid(True, linestyle=":", alpha=0.6)
            ax.margins(y=0.22)

    legend_handles = [
        Line2D([0], [0], color=ESTIMATOR_STYLES[e]["color"], marker=ESTIMATOR_STYLES[e]["marker"], markersize=8, lw=2.2, label=ESTIMATOR_STYLES[e]["label"])
        for e in TARGET_ESTIMATORS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(TARGET_ESTIMATORS),
        fontsize=10.5,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(output_dir / "konsistensi_penurunan_sd_pusat_selatan.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Gambar 2: Kestabilan Mean & Median ± 1 SD Lintas Rasio (60:40 -> 70:30 -> 80:20)
    fig, axes = plt.subplots(len(focus_regions), 4, figsize=(22, 4.2 * len(focus_regions)), sharex=True)
    if len(focus_regions) == 1:
        axes = np.array([axes])

    fig.suptitle(
        "Kestabilan Nilai Estimasi Parameter (Mean ± 1 SD & Median) Lintas Rasio (60:40 → 70:30 → 80:20)\n"
        "Fokus Wilayah: Jakarta Pusat, Jakarta Selatan, dan Gabungan Pusat & Selatan",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    offsets = {"OLS": -0.20, "M": -0.10, "S": 0.0, "MM (k=4.685)": 0.10, "MM (k=1.547)": 0.20}

    for r_idx, reg_name in enumerate(focus_regions):
        reg_sum = summary_df[summary_df["region"] == reg_name]
        for p_idx, (param_col, param_title, param_ylabel) in enumerate(PARAMETERS):
            ax = axes[r_idx, p_idx]
            mean_col = f"{param_col}_mean"
            med_col = f"{param_col}_median"
            std_col = f"{param_col}_std"

            for est in TARGET_ESTIMATORS:
                style = ESTIMATOR_STYLES[est]
                est_rows = reg_sum[reg_sum["estimator"] == est]
                means, meds, stds = [], [], []
                for r_name in ratios_ordered:
                    r_sub = est_rows[est_rows["ratio"] == r_name]
                    means.append(float(r_sub[mean_col].values[0]) if len(r_sub) > 0 else np.nan)
                    meds.append(float(r_sub[med_col].values[0]) if len(r_sub) > 0 else np.nan)
                    stds.append(float(r_sub[std_col].values[0]) if len(r_sub) > 0 else np.nan)

                pos = x_pos + offsets[est]
                ax.errorbar(
                    pos,
                    means,
                    yerr=stds,
                    fmt=style["marker"] + "-",
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=1.6,
                    capsize=4,
                    markersize=6.5,
                    linewidth=1.8,
                    alpha=0.9,
                )
                ax.plot(
                    pos,
                    meds,
                    linestyle="--",
                    marker="x",
                     markersize=5,
                    color=style["color"],
                    alpha=0.75,
                )

            if r_idx == 0:
                ax.set_title(param_title, fontsize=12, fontweight="bold", pad=10)
            if p_idx == 0:
                ax.set_ylabel(f"{reg_name}\n{param_ylabel}", fontsize=10, fontweight="bold")
            if r_idx == len(focus_regions) - 1:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, fontsize=9.5, fontweight="bold")

            ax.grid(True, linestyle=":", alpha=0.6)

    legend_handles = [
        Line2D([0], [0], color=ESTIMATOR_STYLES[e]["color"], marker=ESTIMATOR_STYLES[e]["marker"], lw=1.8, label=f"{ESTIMATOR_STYLES[e]['label']} (Mean ± 1 SD)")
        for e in TARGET_ESTIMATORS
    ]
    legend_handles.append(Line2D([0], [0], color="#334155", linestyle="--", marker="x", lw=1.4, label="Median"))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(legend_handles),
        fontsize=9.5,
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(output_dir / "konsistensi_mean_sd_lintas_rasio_pusat_selatan.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bangun tabel ringkasan statistik konsistensi parameter untuk sesi yang dimuat."""
    rows = []
    region_order = [
        "Jakarta Pusat",
        "Jakarta Utara",
        "Jakarta Barat",
        "Jakarta Selatan",
        "Jakarta Timur",
        "Pusat & Selatan",
    ]
    for ratio_name in ["60:40", "70:30", "80:20"]:
        ratio_sub = df[df["ratio"] == ratio_name]
        available_regions = [r for r in region_order if r in ratio_sub["region"].values]
        for reg_name in available_regions:
            sub_reg = ratio_sub[ratio_sub["region"] == reg_name]
            for est in TARGET_ESTIMATORS:
                sub_est = sub_reg[sub_reg["estimator_type"] == est]
                if sub_est.empty:
                    continue
                row = {
                    "session_id": int(sub_est["session_id"].iloc[0]),
                    "region": reg_name,
                    "dataset": sub_est["dataset_name"].iloc[0],
                    "ratio": ratio_name,
                    "test_size": sub_est["test_size"].iloc[0],
                    "estimator": est,
                    "n_iterations": len(sub_est),
                }
                for param_col, _, _ in PARAMETERS:
                    vals = sub_est[param_col].dropna().values
                    mean_v = float(np.mean(vals))
                    med_v = float(np.median(vals))
                    std_v = float(np.std(vals, ddof=1))
                    q75, q25 = np.percentile(vals, [75, 25])
                    iqr_v = float(q75 - q25)
                    row[f"{param_col}_mean"] = round(mean_v, 4)
                    row[f"{param_col}_median"] = round(med_v, 4)
                    row[f"{param_col}_std"] = round(std_v, 4)
                    row[f"{param_col}_iqr"] = round(iqr_v, 4)
                    row[f"{param_col}_min"] = round(float(np.min(vals)), 4)
                    row[f"{param_col}_max"] = round(float(np.max(vals)), 4)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Plot Konsistensi Parameter & Scale (OLS, M, S, MM k=4.685, MM k=1.547)."
    )
    parser.add_argument(
        "--db-path",
        "-d",
        type=str,
        default="aria_plis.db",
        help="Path ke berkas database SQLite (default: aria_plis.db)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="commands/plots/consistency",
        help="Direktori output untuk menyimpan gambar plot (default: commands/plots/consistency)",
    )
    parser.add_argument(
        "--top-stable",
        "-t",
        type=int,
        default=0,
        help="Jika > 0, ambil N seed dengan koefisien paling stabil per sesi (misal: 200).",
    )
    args = parser.parse_args()

    db_file = PROJECT_ROOT / args.db_path
    base_out_dir = PROJECT_ROOT / args.output_dir
    base_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" GENERATE PLOT KONSISTENSI ESTIMATOR (OLS, M, S, MM k=4.685, MM k=1.547) ")
    print("=" * 80)
    print(f"Database sumber : {db_file}")
    print(f"Direktori output: {base_out_dir}")
    print("=" * 80)

    df = load_consistency_data(db_file)
    print(f"Berhasil memuat {len(df)} baris iterasi dari {db_file.name}.")

    if args.top_stable and args.top_stable > 0:
        df, selected_seeds_map = filter_top_stable_seeds(df, top_n=args.top_stable)
        print(f"[FILTER] Menyaring {args.top_stable} seed paling stabil per sesi -> total {len(df)} baris iterasi.")
        seed_out_path = PROJECT_ROOT / "commands" / f"seed_stabil_{args.top_stable}.py"
        lines = [
            '"""',
            f"Daftar {args.top_stable} Random Seeds Paling Stabil (Deviasi Koefisien Terkecil dari Median)",
            f"Diekstrak dari 1.000 seed murni pada {db_file.name}.",
            '"""',
            "",
        ]
        for k, v in selected_seeds_map.items():
            lines.append(f"{k} = {repr(v)}")
            lines.append("")
        seed_out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Daftar {args.top_stable} seed paling stabil disimpan ke: {seed_out_path}")

    # Simpan ringkasan CSV
    summary_df = build_summary_table(df)
    summary_dir = PROJECT_ROOT / "commands" / "summary" / "estimation"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if args.top_stable and args.top_stable > 0:
        csv_name = f"summary_consistency_{db_file.stem}_top{args.top_stable}.csv"
    elif db_file.stem == "aria_plis":
        csv_name = "summary_consistency_estimators_id_1_18.csv"
    else:
        csv_name = f"summary_consistency_{db_file.stem}.csv"
    summary_csv_path = summary_dir / csv_name
    try:
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"[OK] Tabel ringkasan statistik disimpan ke: {summary_csv_path}")
    except PermissionError:
        fallback_csv = summary_dir / f"{summary_csv_path.stem}_all5.csv"
        summary_df.to_csv(fallback_csv, index=False)
        print(f"[INFO] Berkas CSV utama sedang dibuka di aplikasi lain, disimpan ke: {fallback_csv}")

    region_order = [
        "Jakarta Pusat",
        "Jakarta Utara",
        "Jakarta Barat",
        "Jakarta Selatan",
        "Jakarta Timur",
        "Pusat & Selatan",
    ]

    for ratio_name, r_cfg in RATIO_SESSIONS.items():
        ratio_df = df[df["ratio"] == ratio_name]
        if ratio_df.empty:
            continue

        ratio_dir = base_out_dir / r_cfg["folder"]
        overlay_dir = ratio_dir / "per_wilayah_overlay"
        panel_dir = ratio_dir / "per_wilayah_panel_3x4"
        running_dir = ratio_dir / "konvergensi_running_mean"
        master_dir = ratio_dir / "master_grid_parameter"

        overlay_dir.mkdir(parents=True, exist_ok=True)
        panel_dir.mkdir(parents=True, exist_ok=True)
        running_dir.mkdir(parents=True, exist_ok=True)
        master_dir.mkdir(parents=True, exist_ok=True)

        available_regions = [r for r in region_order if r in ratio_df["region"].values]
        five_dki_regions = [r for r in available_regions if r != "Pusat & Selatan"]

        print(f"\n--> Memproses {r_cfg['label']} ({len(available_regions)} wilayah) ...")

        # 1, 2 & 3: Plot per wilayah
        for reg_name in available_regions:
            reg_df = ratio_df[ratio_df["region"] == reg_name]
            slug = slugify_region(reg_name)

            plot_region_overlay_4params(
                reg_df=reg_df,
                region_name=reg_name,
                ratio_label=r_cfg["label"],
                output_path=overlay_dir / f"consistency_overlay_{slug}.png",
            )
            plot_region_panel_3x4(
                reg_df=reg_df,
                region_name=reg_name,
                ratio_label=r_cfg["label"],
                output_path=panel_dir / f"consistency_panel_3x4_{slug}.png",
            )
            plot_running_mean_convergence(
                reg_df=reg_df,
                region_name=reg_name,
                ratio_label=r_cfg["label"],
                output_path=running_dir / f"running_mean_convergence_{slug}.png",
            )
        print(f"  [OK] Plot overlay (2x2), panel (5x4), & running mean per wilayah selesai.")

        # 4: Master Grid per Parameter
        for param_col, param_title, param_ylabel in PARAMETERS:
            suffix = f"{len(available_regions)}datasets"
            plot_master_parameter_across_regions(
                ratio_df=ratio_df,
                regions=available_regions,
                param_col=param_col,
                param_title=param_title,
                param_ylabel=param_ylabel,
                ratio_label=r_cfg["label"],
                output_path=master_dir / f"master_{param_col}_{suffix}.png",
            )
            if len(five_dki_regions) == 5:
                plot_master_parameter_across_regions(
                    ratio_df=ratio_df,
                    regions=five_dki_regions,
                    param_col=param_col,
                    param_title=param_title,
                    param_ylabel=param_ylabel,
                    ratio_label=r_cfg["label"],
                    output_path=master_dir / f"master_{param_col}_5wilayah.png",
                )
        print(f"  [OK] Master grid 4 parameter selesai.")

    # 5: Plot Konsistensi Asimtotik Lintas Rasio (60:40 -> 70:30 -> 80:20)
    asymptotic_dir = base_out_dir / "konsistensi_antar_rasio"
    focus_regs = [r for r in ["Jakarta Pusat", "Jakarta Selatan", "Pusat & Selatan"] if r in df["region"].values]
    if focus_regs:
        plot_asymptotic_consistency_across_ratios(
            summary_df=summary_df,
            focus_regions=focus_regs,
            output_dir=asymptotic_dir,
        )
        print(f"\n--> [OK] Plot Konsistensi Asimtotik Lintas Rasio selesai di: {asymptotic_dir}")

    print("\n" + "=" * 80)
    print(f"[SELESAI] Seluruh plot konsistensi berhasil dibuat di: {base_out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()


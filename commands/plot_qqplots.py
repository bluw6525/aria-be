"""
Command untuk menghasilkan Q-Q Plot Residual menggunakan statsmodels.api.qqplot (line='s'):
- Menggunakan Y aktual (kolom 'O3') dan X prediktor ('CO_satelit', 'NO2_satelit') dari file Excel resmi per wilayah:
  * Jakarta Barat  : Barat_New.xlsx
  * Jakarta Pusat  : Pusat_New.xlsx
  * Jakarta Selatan: Selatan_New2.xlsx
  * Jakarta Utara  : Utara_New.xlsx
  * Jakarta Timur  : Timur_New.xlsx
- Menghasilkan 5 model (OLS, M, S, MM k=4.685, MM k=1.547)
- Mencakup 3 Rasio Pembagian Data (60:40, 70:30, 80:20)
- Mencakup Data Uji (Testing Set) DAN Data Latih (Training Set)
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Tambahkan root proyek ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.algorithms.m_estimator import MEstimator
from app.modules.regression.algorithms.s_estimator import SEstimator
from app.modules.regression.algorithms.mm_estimator import MMEstimator
from app.modules.regression.core.split import split_data
from app.modules.regression.core.metrics import calculate_r2, calculate_rmse

# 1. Pemetaan File Excel Resmi per Wilayah & Kolom Fitur / Target
REGION_CONFIG = {
    "Jakarta Barat": {
        "file": "Barat_New.xlsx",
        "target_col": "O3",
        "feature_cols": ["CO_satelit", "NO2_satelit"]
    },
    "Jakarta Pusat": {
        "file": "Pusat_New.xlsx",
        "target_col": "O3",
        "feature_cols": ["CO_satelit", "NO2_satelit"]
    },
    "Jakarta Selatan": {
        "file": "Selatan_New2.xlsx",
        "target_col": "O3",
        "feature_cols": ["CO_satelit", "NO2_satelit"]
    },
    "Jakarta Utara": {
        "file": "Utara_New.xlsx",
        "target_col": "O3",
        "feature_cols": ["CO_satelit", "NO2_satelit"]
    },
    "Jakarta Timur": {
        "file": "Timur_New.xlsx",
        "target_col": "O3",
        "feature_cols": ["CO_satelit", "NO2_satelit"]
    }
}

# 2. Optimal Seeds per Rasio Data hasil evaluasi MCCV
RATIO_CONFIG = {
    "60:40": {
        "test_size": 0.4,
        "folder": "rasio_60_40",
        "label": "Rasio 60:40 (60% Latih : 40% Uji)",
        "seeds": {
            "Jakarta Barat": 884,
            "Jakarta Pusat": 482,
            "Jakarta Selatan": 368,
            "Jakarta Utara": 444,
            "Jakarta Timur": 360,
        }
    },
    "70:30": {
        "test_size": 0.3,
        "folder": "rasio_70_30",
        "label": "Rasio 70:30 (70% Latih : 30% Uji)",
        "seeds": {
            "Jakarta Barat": 26,
            "Jakarta Pusat": 859,
            "Jakarta Selatan": 727,
            "Jakarta Utara": 562,
            "Jakarta Timur": 763,
        }
    },
    "80:20": {
        "test_size": 0.2,
        "folder": "rasio_80_20",
        "label": "Rasio 80:20 (80% Latih : 20% Uji)",
        "seeds": {
            "Jakarta Barat": 561,
            "Jakarta Pusat": 98,
            "Jakarta Selatan": 385,
            "Jakarta Utara": 839,
            "Jakarta Timur": 735,
        }
    }
}

# 3. Konfigurasi 5 Model Estimator
ESTIMATORS = {
    "OLS": {"name": "OLS", "color": "#64748b"},
    "M": {"name": "M-Estimator (Huber)", "color": "#0284c7"},
    "S": {"name": "S-Estimator", "color": "#d97706"},
    "MM (k=4.685)": {"name": "MM (k=4,685)", "color": "#059669"},
    "MM (k=1.547)": {"name": "MM (k=1,547)", "color": "#4f46e5"}
}


def fit_model_and_get_residuals(estimator_key: str, X_train, y_train, X_eval, y_eval):
    """
    Melatih model pada data latih dan menghitung residual pada data evaluasi (uji/latih):
    Residual = Y_aktual - Y_prediksi
    """
    if estimator_key == "OLS":
        model = OLSEstimator(fit_intercept=True)
    elif estimator_key == "M":
        model = MEstimator(fit_intercept=True)
    elif estimator_key == "S":
        model = SEstimator(fit_intercept=True)
    elif estimator_key == "MM (k=4.685)":
        model = MMEstimator(tuning_constant=4.685, fit_intercept=True)
    elif estimator_key == "MM (k=1.547)":
        model = MMEstimator(tuning_constant=1.547, fit_intercept=True)
    else:
        raise ValueError(f"Model tidak dikenal: {estimator_key}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    residuals = y_eval - y_pred  # Residual = Y_aktual - Y_prediksi
    r2 = calculate_r2(y_eval, y_pred)
    rmse = calculate_rmse(y_eval, y_pred)
    return residuals, r2, rmse


def plot_qq_single_region(region_name: str, ratio_name: str, data_type: str, output_dir: Path):
    """
    Menghasilkan 1 file gambar berisi 5 SUBPLOT Q-Q Plot (sm.qqplot line='s')
    untuk 5 model regresi pada wilayah tertentu.
    """
    cfg_region = REGION_CONFIG[region_name]
    cfg_ratio = RATIO_CONFIG[ratio_name]
    excel_path = PROJECT_ROOT / cfg_region["file"]

    if not excel_path.exists():
        print(f"[WARN] File Excel {excel_path} tidak ditemukan!")
        return None

    # 1. Baca data dari Excel resmi
    df = pd.read_excel(excel_path)
    X = df[cfg_region["feature_cols"]].values
    y = df[cfg_region["target_col"]].values  # Y aktual (O3)

    # 2. Split data latih dan uji berdasarkan seed optimal
    seed = cfg_ratio["seeds"][region_name]
    test_size = cfg_ratio["test_size"]
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=seed)

    # Tentukan subset evaluasi (uji vs latih)
    if data_type == "uji":
        X_eval, y_eval = X_test, y_test
        data_label = "Data Uji (Testing Set)"
    else:
        X_eval, y_eval = X_train, y_train
        data_label = "Data Latih (Training Set)"

    # 3. Siapkan Figure dengan 5 Subplot (1 baris x 5 kolom)
    fig, axes = plt.subplots(1, 5, figsize=(23, 5), sharey=False)
    fig.suptitle(
        f"Normal Q-Q Plot Residual {data_label} - {region_name}\n"
        f"[{cfg_ratio['label']} | Seed: {seed} | Sumber: {cfg_region['file']}]",
        fontsize=13,
        fontweight="bold",
        y=1.03
    )

    for idx, (est_key, est_meta) in enumerate(ESTIMATORS.items()):
        ax = axes[idx]
        residuals, r2, rmse = fit_model_and_get_residuals(est_key, X_train, y_train, X_eval, y_eval)

        # 4. MEMBUAT Q-Q PLOT DENGAN statsmodels sm.qqplot(residuals, line='s')
        sm.qqplot(residuals, line="s", ax=ax)

        # Kustomisasi estetika garis & titik dari sm.qqplot
        lines = ax.get_lines()
        if len(lines) >= 2:
            # lines[0] adalah titik-titik residual sampel
            lines[0].set_markerfacecolor(est_meta["color"])
            lines[0].set_markeredgecolor("none")
            lines[0].set_markersize(5.0)
            lines[0].set_alpha(0.75)
            # lines[1] adalah garis referensi 's' (standardized line)
            lines[1].set_color("#dc2626")  # Merah tegas
            lines[1].set_linewidth(1.8)
            lines[1].set_linestyle("--")

        ax.set_title(
            f"{est_meta['name']}\nR²: {r2:.1f}% | RMSE: {rmse:.2f}",
            fontsize=11,
            fontweight="bold",
            pad=6
        )
        ax.set_xlabel("Theoretical Quantiles", fontsize=10)
        if idx == 0:
            ax.set_ylabel(f"Sample Residuals ({data_type.upper()})", fontsize=10, fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    slug = region_name.lower().replace(" ", "_")
    out_file = output_dir / f"qqplot_{slug}_{data_type}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


def plot_qq_master_grid(ratio_name: str, data_type: str, output_dir: Path):
    """
    Menghasilkan 1 file gambar MASTER GRID (5 baris wilayah x 5 kolom model)
    menggunakan sm.qqplot(residuals, line='s').
    """
    cfg_ratio = RATIO_CONFIG[ratio_name]
    regions = list(REGION_CONFIG.keys())
    test_size = cfg_ratio["test_size"]
    data_label = "Data Uji (Testing Set)" if data_type == "uji" else "Data Latih (Training Set)"

    fig, axes = plt.subplots(5, 5, figsize=(23, 20), sharex=True, sharey=False)
    fig.suptitle(
        f"Master Grid Normal Q-Q Plot Residual {data_label} Antar Wilayah DKI Jakarta\n"
        f"[{cfg_ratio['label']} | Menggunakan sm.qqplot(line='s')]",
        fontsize=15,
        fontweight="bold",
        y=0.995
    )

    for r_idx, reg_name in enumerate(regions):
        cfg_reg = REGION_CONFIG[reg_name]
        excel_path = PROJECT_ROOT / cfg_reg["file"]
        if not excel_path.exists():
            continue

        df = pd.read_excel(excel_path)
        X = df[cfg_reg["feature_cols"]].values
        y = df[cfg_reg["target_col"]].values
        seed = cfg_ratio["seeds"][reg_name]

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=seed)
        X_eval, y_eval = (X_test, y_test) if data_type == "uji" else (X_train, y_train)

        for c_idx, (est_key, est_meta) in enumerate(ESTIMATORS.items()):
            ax = axes[r_idx, c_idx]
            residuals, r2, rmse = fit_model_and_get_residuals(est_key, X_train, y_train, X_eval, y_eval)

            sm.qqplot(residuals, line="s", ax=ax)
            lines = ax.get_lines()
            if len(lines) >= 2:
                lines[0].set_markerfacecolor(est_meta["color"])
                lines[0].set_markeredgecolor("none")
                lines[0].set_markersize(4.0)
                lines[0].set_alpha(0.70)
                lines[1].set_color("#dc2626")
                lines[1].set_linewidth(1.5)
                lines[1].set_linestyle("--")

            if r_idx == 0:
                ax.set_title(f"{est_meta['name']}", fontsize=11, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"{reg_name}\nResiduals", fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel("")

            ax.text(
                0.05, 0.90, f"R²={r2:.1f}%\nRMSE={rmse:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none")
            )
            ax.grid(True, linestyle=":", alpha=0.5)

    for c_idx in range(5):
        axes[-1, c_idx].set_xlabel("Theoretical Quantiles", fontsize=10)

    plt.tight_layout()
    out_file = output_dir / f"qqplot_master_grid_5wilayah_{data_type}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Generate Q-Q Plot Residuals menggunakan sm.qqplot (line='s').")
    parser.add_argument("--output-dir", "-o", type=str, default="commands/plots/qqplots", help="Direktori output")
    args = parser.parse_args()

    base_out_dir = PROJECT_ROOT / args.output_dir

    print("=========================================================================")
    print(" GENERATE Q-Q PLOT DENGAN statsmodels sm.qqplot(residuals, line='s')")
    print(" (5 MODEL x 5 WILAYAH x 3 RASIO x DATA UJI & DATA LATIH)")
    print("=========================================================================")

    for ratio_name, r_cfg in RATIO_CONFIG.items():
        for d_type in ["uji", "latih"]:
            target_dir = base_out_dir / r_cfg["folder"] / f"data_{d_type}"
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--> Memproses Rasio {ratio_name} [{d_type.upper()}] ke {target_dir.name}")

            # 1. Plot per wilayah (masing-masing 1 gambar berisi 5 subplot model)
            for reg_name in REGION_CONFIG.keys():
                out_f = plot_qq_single_region(
                    region_name=reg_name,
                    ratio_name=ratio_name,
                    data_type=d_type,
                    output_dir=target_dir
                )
                print(f"  [OK] {reg_name} ({d_type}): {out_f.name}")

            # 2. Master Grid gabungan (1 gambar berisi 25 subplot: 5 baris wilayah x 5 kolom model)
            grid_f = plot_qq_master_grid(
                ratio_name=ratio_name,
                data_type=d_type,
                output_dir=target_dir
            )
            print(f"  [OK] Master Grid 5 Wilayah ({d_type}): {grid_f.name}")

    print("\n[SELESAI] Seluruh Q-Q plot dengan sm.qqplot(line='s') berhasil dibuat di:", base_out_dir)


if __name__ == "__main__":
    main()

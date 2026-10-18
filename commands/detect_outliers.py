"""
Skrip Deteksi Pencilan (Outlier) & Titik Pengaruh (Influential Points):
1. Leverage Values (Hii) dengan ambang batas 2p/n
2. DFFITS dengan ambang batas 2*sqrt(p/n)
3. Studentized Deleted Residuals (t_i)

Dilengkapi ekspor CSV hasil diagnostik dan grafik visualisasi resolusi tinggi (300 DPI).
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.outliers import detect_outliers, calculate_leverage, calculate_dffits

PLOTS_DIR = PROJECT_ROOT / "commands" / "plots" / "outlier"
SUMMARY_DIR = PROJECT_ROOT / "commands" / "summary" / "outlier"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def plot_outlier_diagnostics(
    df_diag: pd.DataFrame,
    summary: dict,
    dataset_base_name: str
):
    """Membuat visualisasi Leverage dan DFFITS dengan garis batas ambang menggunakan styling standar."""
    n = summary["n_samples"]
    lev_thresh = summary["leverage_threshold"]
    dffits_thresh = summary["dffits_threshold"]

    obs = df_diag["observation_id"].values
    lev = df_diag["leverage"].values
    is_lev = df_diag["leverage_outlier"].values
    dffits_vals = df_diag["dffits"].values
    is_dffits = df_diag["dffits_outlier"].values

    # -------------------------------------------------------------
    # 1. PLOT LEVERAGE VALUES (Hii)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(obs[~is_lev], lev[~is_lev], color='steelblue', alpha=0.7, s=25, label='Normal')
    ax.scatter(obs[is_lev], lev[is_lev], color='red', alpha=0.9, s=40, label=f'Leverage Outlier ({summary["total_leverage_outliers"]})')
    ax.axhline(y=lev_thresh, color='red', linestyle='--', label=f'Ambang Batas 2p/n ({lev_thresh:.4f})')

    ax.set_title(f"Plot Leverage (h_ii) - {dataset_base_name}")
    ax.set_xlabel("Observasi (i)")
    ax.set_ylabel("Nilai Leverage (h_ii)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    lev_plot_path = PLOTS_DIR / f"leverage_plot_{dataset_base_name}.png"
    plt.savefig(lev_plot_path, dpi=300)
    plt.close()
    print(f"  [SAVED] Grafik Leverage : {lev_plot_path}")

    # -------------------------------------------------------------
    # 2. PLOT DFFITS
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(obs[~is_dffits], dffits_vals[~is_dffits], color='steelblue', alpha=0.7, s=25, label='Normal')
    ax.scatter(obs[is_dffits], dffits_vals[is_dffits], color='orange', alpha=0.9, s=40, label=f'DFFITS Outlier ({summary["total_dffits_outliers"]})')
    ax.axhline(y=dffits_thresh, color='red', linestyle='--', label=f'Batas Atas +2*sqrt(p/n) ({dffits_thresh:.4f})')
    ax.axhline(y=-dffits_thresh, color='red', linestyle='--', label=f'Batas Bawah -2*sqrt(p/n) (-{dffits_thresh:.4f})')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)

    ax.set_title(f"Plot DFFITS - {dataset_base_name}")
    ax.set_xlabel("Observasi (i)")
    ax.set_ylabel("Nilai DFFITS")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    dffits_plot_path = PLOTS_DIR / f"dffits_plot_{dataset_base_name}.png"
    plt.savefig(dffits_plot_path, dpi=300)
    plt.close()
    print(f"  [SAVED] Grafik DFFITS   : {dffits_plot_path}")

    # -------------------------------------------------------------
    # 3. PLOT KOMBINASI INFLUENTIAL (Leverage vs Abs DFFITS)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    abs_dffits = df_diag["abs_dffits"].values

    is_both = df_diag["status"] == "Influential & Leverage"
    is_only_lev = df_diag["status"] == "Leverage Outlier"
    is_only_dff = df_diag["status"] == "DFFITS Outlier"
    is_normal = df_diag["status"] == "Normal"

    ax.scatter(lev[is_normal], abs_dffits[is_normal], color='gray', alpha=0.5, s=25, label='Normal')
    ax.scatter(lev[is_only_lev], abs_dffits[is_only_lev], color='blue', alpha=0.8, s=35, label='Leverage Outlier')
    ax.scatter(lev[is_only_dff], abs_dffits[is_only_dff], color='orange', alpha=0.8, s=35, label='DFFITS Outlier')
    ax.scatter(lev[is_both], abs_dffits[is_both], color='red', alpha=1.0, s=50, label=f'Kedua Kriteria ({summary["total_influential_both"]})')

    ax.axvline(x=lev_thresh, color='red', linestyle='--', label=f'Batas Lev ({lev_thresh:.4f})')
    ax.axhline(y=dffits_thresh, color='red', linestyle=':', label=f'Batas DFFITS ({dffits_thresh:.4f})')

    ax.set_title(f"Plot Pengaruh (Leverage vs |DFFITS|) - {dataset_base_name}")
    ax.set_xlabel("Leverage (h_ii)")
    ax.set_ylabel("|DFFITS|")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    inf_plot_path = PLOTS_DIR / f"influence_map_{dataset_base_name}.png"
    plt.savefig(inf_plot_path, dpi=300)
    plt.close()
    print(f"  [SAVED] Grafik Peta Pengaruh: {inf_plot_path}")


def run_outlier_detection(
    file_path: str = "Jakarta.xlsx",
    fit_intercept: bool = True
):
    path_obj = Path(file_path)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / file_path

    if not path_obj.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path_obj}")

    df = pd.read_excel(path_obj)
    feature_cols = ["CO_satelit", "NO2_satelit"]
    target_col = "O3"

    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise KeyError(f"Kolom '{col}' tidak ditemukan di {path_obj.name}")

    X = df[feature_cols].values
    y = df[target_col].values

    print("\n" + "=" * 80)
    print("         DETEKSI OUTLIER & TITIK PENGARUH (LEVERAGE & DFFITS)         ")
    print("=" * 80)
    print(f"Dataset          : {path_obj.name}")
    print(f"Jumlah Baris (n) : {len(y)}")
    print(f"Fitur (X)        : {feature_cols}")
    print(f"Target (y)       : {target_col}")
    print(f"Fit Intercept    : {fit_intercept}")
    print("-" * 80)

    # 1. Estimasi regresi OLS menggunakan objek OLSEstimator
    ols_model = OLSEstimator(fit_intercept=fit_intercept)
    ols_model.fit(X, y)

    # 2. Jalankan modul deteksi outlier dengan objek OLSEstimator
    df_diag, summary = detect_outliers(
        X=X,
        y=y,
        fit_intercept=fit_intercept,
        feature_names=feature_cols,
        target_name=target_col,
        model=ols_model
    )

    n = summary["n_samples"]
    p = summary["n_parameters"]
    lev_thresh = summary["leverage_threshold"]
    dffits_thresh = summary["dffits_threshold"]

    print(f"\n[FORMULA & AMBANG BATAS]")
    print(f"  1. Leverage Values (Hii):")
    print(f"     - Matriks Topi H = X (X^T X)^(-1) X^T")
    print(f"     - Jumlah Parameter (p) = {p}")
    print(f"     - Ambang Batas Cutoff  = 2p / n = 2 * {p} / {n} = {lev_thresh:.6f}")
    print(f"  2. DFFITS (Persamaan 2.13):")
    print(f"     - DFFITS_i = t_i * ( h_ii / (1 - h_ii) )^(1/2)")
    print(f"     - Residual Terstudentisasi t_i = e_i * sqrt( (n-p-1) / (SSE*(1-hii) - e_i^2) )")
    print(f"     - Ambang Batas Cutoff  = 2 * sqrt(p / n) = 2 * sqrt({p} / {n}) = {dffits_thresh:.6f}")
    print("-" * 80)

    print(f"\n[HASIL REKAPITULASI DETEKSI]")
    print(f"  * Total Observasi                     : {n}")
    print(f"  * Pencilan Leverage (h_ii > 2p/n)     : {summary['total_leverage_outliers']} titik")
    print(f"  * Pencilan DFFITS (|DFFITS| > 2*sqrt(p/n)): {summary['total_dffits_outliers']} titik")
    print(f"  * Titik Kritis (Kedua Kriteria)       : {summary['total_influential_both']} titik")
    print(f"  * Total Observasi Unik Terdeteksi     : {summary['total_unique_outliers']} titik")

    print("\n[DAFTAR INDEKS OBSERVASI]")
    print(f"  - Leverage Outliers : {summary['leverage_outlier_indices']}")
    print(f"  - DFFITS Outliers   : {summary['dffits_outlier_indices']}")
    print(f"  - Keduanya (Both)   : {summary['both_outlier_indices']}")

    # Filter hanya baris yang merupakan outlier
    outliers_only = df_diag[df_diag["status"] != "Normal"].copy()
    outliers_only.sort_values(by="abs_dffits", ascending=False, inplace=True)

    print("\n[DETAIL DATA OUTLIER (Diurutkan berdasarkan |DFFITS| tertinggi)]:")
    print("-" * 96)
    print(f"{'Obs':<6} {'CO_satelit':<12} {'NO2_satelit':<13} {'O3':<10} {'Leverage':<12} {'Stud Res':<12} {'DFFITS':<12} {'Status':<22}")
    print("-" * 96)
    for _, row in outliers_only.head(25).iterrows():
        print(f"#{int(row['observation_id']):<5} {row['CO_satelit']:<12.4f} {row['NO2_satelit']:<13.4f} {row[target_col]:<10.2f} {row['leverage']:<12.6f} {row['studentized_residual']:<12.4f} {row['dffits']:<12.4f} {row['status']:<22}")
    if len(outliers_only) > 25:
        print(f"... dan {len(outliers_only) - 25} baris lainnya (lihat file CSV).")
    print("-" * 96)

    # Simpan hasil lengkap ke CSV
    dataset_stem = path_obj.stem
    csv_filename = SUMMARY_DIR / f"outlier_detection_{dataset_stem}.csv"

    try:
        df_diag.to_csv(csv_filename, index=False)
        print(f"\n[EXPORT] File CSV (.csv) : {csv_filename}")
    except PermissionError:
        fallback_csv = SUMMARY_DIR / f"outlier_detection_{dataset_stem}_new.csv"
        df_diag.to_csv(fallback_csv, index=False)
        print(f"\n[PERINGATAN] '{csv_filename.name}' sedang dibuka di aplikasi lain.")
        print(f"[EXPORT] Disimpan ke : {fallback_csv}")

    # Buat grafik visualisasi
    print("\n[GENERATING PLOTS (300 DPI)]...")
    plot_outlier_diagnostics(df_diag, summary, dataset_stem)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi Outlier Leverage dan DFFITS")
    parser.add_argument("--file", type=str, default="Jakarta_Pusat.xlsx", help="Nama/Path file dataset")
    parser.add_argument("--no-intercept", action="store_true", help="Nonaktifkan intercept")
    args = parser.parse_args()

    run_outlier_detection(file_path=args.file, fit_intercept=not args.no_intercept)

"""
Skrip untuk menghasilkan grafik visualisasi hasil evaluasi regresi (300 DPI):
1. Boxplot Distribusi Metrik (R2, RMSE, MAE) dari 200 Replikasi Monte Carlo
2. Scatter Plot Aktual vs Prediksi & Plot Residual Model Representatif

Mendukung pemilihan dataset dan rasio test_size secara dinamis dari database PostgreSQL.
"""

import sys
import asyncio
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.database import engine
from app.modules.regression.algorithms.mm_estimator import MMEstimator
from commands.utils import split_data, predict_model
from app.modules.regression.metrics import calculate_r2, calculate_rmse, calculate_mae

OUTPUT_DIR = PROJECT_ROOT / "commands" / "plots" / "estimation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

async def generate_plots(
    dataset_name: str = "Jakarta_Selatan.xlsx",
    test_size: float = 0.3,
    fit_intercept: bool = True,
    best_seed: int = 4423
):
    path_obj = Path(dataset_name)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / dataset_name

    dataset_filename = path_obj.name

    print("=" * 76)
    print("                GENERATE PUBLICATION PLOTS (300 DPI)                ")
    print("=" * 76)
    print(f"Dataset          : {dataset_filename}")
    print(f"Rasio Test Size  : {test_size} ({int((1-test_size)*100)}% Train : {int(test_size*100)}% Test)")
    print(f"Fit Intercept    : {fit_intercept}")
    print(f"Representative   : Seed {best_seed}")
    print("-" * 76)

    print(f"Mengambil data evaluasi dari PostgreSQL (dataset='{dataset_filename}', test_size={test_size})...")
    async with engine.connect() as conn:
        res = await conn.execute(
            text("""
                SELECT random_state, estimator_type, test_r2, test_rmse, test_mae, train_r2, train_rmse, train_mae
                FROM regression_models
                WHERE dataset_name = :dataset_name
                  AND test_size = :test_size
                  AND fit_intercept = :fit_intercept
                ORDER BY id;
            """),
            {
                "dataset_name": dataset_filename,
                "test_size": test_size,
                "fit_intercept": fit_intercept
            }
        )
        df = pd.DataFrame(res.fetchall(), columns=[
            'random_state', 'estimator_type', 'test_r2', 'test_rmse', 'test_mae',
            'train_r2', 'train_rmse', 'train_mae'
        ])

    if df.empty:
        print(f"\n[PERINGATAN] Data evaluasi tidak ditemukan di database untuk:")
        print(f"  Dataset: '{dataset_filename}', test_size={test_size}, fit_intercept={fit_intercept}")
        print("Silakan jalankan batch_estimate.py terlebih dahulu untuk kombinasi ini.")
        return

    total_runs = len(df['random_state'].unique())
    print(f"Berhasil memuat {len(df)} baris data dari {total_runs} replikasi.")

    # Urutan estimator yang konsisten
    est_order = ["OLS", "S", "M", "MM"]
    palette = {
        "OLS": "#3498db",  # Biru
        "S": "#e74c3c",    # Merah
        "M": "#f39c12",    # Kuning/Oranye
        "MM": "#2ecc71"    # Hijau
    }

    # Set tema visualisasi
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

    # =========================================================================
    # GRAFIK 1: BOXPLOT PERBANDINGAN METRIK EVALUASI
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 1. Boxplot R2
    df['test_r2_pct'] = df['test_r2'] * 100
    sns.boxplot(
        data=df, x='estimator_type', y='test_r2_pct', order=est_order,
        hue='estimator_type', palette=palette, legend=False,
        ax=axes[0], width=0.45, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"}
    )
    axes[0].set_title(r"(a) Koefisien Determinasi ($R^2$ Test)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Algoritma Estimator", fontsize=11, fontweight='bold')
    axes[0].set_ylabel(r"$R^2$ (%)", fontsize=11, fontweight='bold')

    # 2. Boxplot RMSE
    sns.boxplot(
        data=df, x='estimator_type', y='test_rmse', order=est_order,
        hue='estimator_type', palette=palette, legend=False,
        ax=axes[1], width=0.45, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"}
    )
    axes[1].set_title("(b) Root Mean Squared Error (RMSE Test)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Algoritma Estimator", fontsize=11, fontweight='bold')
    axes[1].set_ylabel(r"RMSE ($O_3$)", fontsize=11, fontweight='bold')

    # 3. Boxplot MAE
    sns.boxplot(
        data=df, x='estimator_type', y='test_mae', order=est_order,
        hue='estimator_type', palette=palette, legend=False,
        ax=axes[2], width=0.45, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"}
    )
    axes[2].set_title("(c) Mean Absolute Error (MAE Test)", fontsize=13, fontweight='bold', pad=12)
    axes[2].set_xlabel("Algoritma Estimator", fontsize=11, fontweight='bold')
    axes[2].set_ylabel(r"MAE ($O_3$)", fontsize=11, fontweight='bold')

    fig.suptitle(f"Perbandingan Distribusi Metrik Evaluasi Data Uji ({total_runs} Replikasi Monte Carlo)", fontsize=15, fontweight='bold', y=1.03)

    plot1_path = OUTPUT_DIR / f"box_plot_{path_obj.stem}_{test_size}.png"
    plt.savefig(plot1_path, bbox_inches='tight')
    plt.close()
    print(f"Grafik 1 tersimpan di: {plot1_path}")

    # =========================================================================
    # GRAFIK 2: SCATTER PLOT AKTUAL VS PREDIKSI & RESIDUAL PLOT
    # =========================================================================
    if not path_obj.exists():
        print(f"\n[PERINGATAN] File excel '{path_obj}' tidak ditemukan, grafik scatter dilewati.")
        return

    df_excel = pd.read_excel(path_obj)
    feature_cols = ['CO_satelit', 'NO2_satelit']
    target_col = 'O3'

    X = df_excel[feature_cols].values
    y = df_excel[target_col].values

    # Model representatif
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=best_seed)

    mm = MMEstimator(fit_intercept=fit_intercept)
    mm.fit(X_train, y_train)

    y_pred_train = predict_model(mm, X_train, fit_intercept=fit_intercept)
    y_pred_test = predict_model(mm, X_test, fit_intercept=fit_intercept)
    res_test = y_test - y_pred_test

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Scatter Plot Aktual vs Prediksi
    train_pct = int((1 - test_size) * 100)
    test_pct = int(test_size * 100)
    axes[0].scatter(y_train, y_pred_train, color='#3498db', alpha=0.5, label=f'Data Latih (Train, {train_pct}%)', edgecolors='none', s=45)
    axes[0].scatter(y_test, y_pred_test, color='#2ecc71', alpha=0.85, label=f'Data Uji (Test, {test_pct}%)', edgecolors='black', linewidth=0.5, s=55)
    
    # Garis Ideal y = x
    lims = [
        min(np.min(y), np.min(y_pred_train), np.min(y_pred_test)) - 2,
        max(np.max(y), np.max(y_pred_train), np.max(y_pred_test)) + 2
    ]
    axes[0].plot(lims, lims, '--', color='#e74c3c', linewidth=2, label=r'Garis Ideal Prediksi ($y = \hat{y}$)')
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_title(f"Aktual vs Prediksi MM-Estimator (Seed {best_seed})", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel(r"Nilai Aktual $O_3$ (Observasi)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel(r"Nilai Prediksi $\hat{y}$ ($O_3$)", fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper left', frameon=True)

    # Anotasi metrik
    te_r2 = calculate_r2(y_test, y_pred_test) * 100
    te_rmse = calculate_rmse(y_test, y_pred_test)
    te_mae = calculate_mae(y_test, y_pred_test)
    tr_r2 = calculate_r2(y_train, y_pred_train) * 100

    annotation_text = (
        f"Data Uji (Test):\n"
        f"$R^2$ = {te_r2:.2f}%\n"
        f"RMSE = {te_rmse:.4f}\n"
        f"MAE = {te_mae:.4f}\n\n"
        f"Data Latih (Train):\n"
        f"$R^2$ = {tr_r2:.2f}%"
    )
    axes[0].text(0.68, 0.06, annotation_text, transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#2ecc71", alpha=0.9))

    # 2. Residual Plot
    axes[1].scatter(y_pred_test, res_test, color='#2ecc71', alpha=0.8, edgecolors='black', linewidth=0.5, s=50)
    axes[1].axhline(0, color='#e74c3c', linestyle='--', linewidth=2)
    axes[1].set_title(r"Plot Residual Data Uji ($e_i = y_i - \hat{y}_i$)", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel(r"Nilai Prediksi $\hat{y}$ ($O_3$)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel(r"Residual / Galat ($e_i$)", fontsize=11, fontweight='bold')

    plot2_path = OUTPUT_DIR / f"scatter_plot_{path_obj.stem}_{test_size}_{best_seed}.png"
    plt.savefig(plot2_path, bbox_inches='tight')
    plt.close()
    print(f"Grafik 2 tersimpan di: {plot2_path}")
    print("=" * 76)


if __name__ == "__main__":
    # =========================================================================
    # KONFIGURASI PARAMETER (Ubah nilai di bawah ini jika ingin mengganti langsung dari kode)
    # =========================================================================
    DATASET_NAME   = "Jakarta_Selatan.xlsx"  # Nama file dataset Excel
    TEST_SIZE      = 0.3                    # Rasio data uji (0.3 = 30% test, 70% train)
    FIT_INTERCEPT  = True                   # True jika model memakai konstanta, False jika tanpa intercept
    BEST_SEED      = 5124                   # Seed representatif terbaik untuk scatter plot
    # =========================================================================

    import argparse
    parser = argparse.ArgumentParser(description="Generate publication plots from regression evaluation database.")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME, help=f"Nama file dataset excel (default: {DATASET_NAME})")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help=f"Rasio data uji (default: {TEST_SIZE})")
    parser.add_argument("--fit-intercept", action="store_true", default=FIT_INTERCEPT, help="Gunakan intercept (default: True)")
    parser.add_argument("--no-intercept", action="store_false", dest="fit_intercept", help="Nonaktifkan intercept")
    parser.add_argument("--seed", type=int, default=BEST_SEED, help=f"Seed representatif terbaik untuk scatter plot (default: {BEST_SEED})")

    args = parser.parse_args()
    asyncio.run(generate_plots(
        dataset_name=args.dataset,
        test_size=args.test_size,
        fit_intercept=args.fit_intercept,
        best_seed=args.seed
    ))

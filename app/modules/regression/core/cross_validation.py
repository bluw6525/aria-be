"""
Modul Evaluasi Validasi Silang Monte Carlo (Monte Carlo Cross-Validation / Repeated Random Subsampling).
Menjalankan komputasi simulasi perbandingan performa estimator (OLS, S, M, MM).
"""

import random
from typing import Callable, List, Optional, Union
import numpy as np



from app.modules.regression.core.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_r2,
    calculate_f_significance,
)
from app.modules.regression.core.parser import parse_raw_seeds
from app.modules.regression.core.split import split_data
from app.modules.regression.schemas import (
    MCCVIterationResult,
    BenchmarkSummaryRow,
    MCCVBenchmarkResult,
)


def resolve_seeds(seeds: Optional[Union[str, List[int]]], n_splits: int) -> List[int]:
    """
    Menyiapkan daftar seed pengacakan sesuai ketentuan:
    - Jika seeds None/kosong: generate sebanyak n_splits secara acak.
    - Jika len(seeds) >= n_splits (contoh 201 dari 200): ambil tepat n_splits terdepan (200).
    - Jika 0 < len(seeds) < n_splits (contoh 1 dari 200): pertahankan seed yang dimasukkan,
      lalu lengkapi kekurangannya dengan seed acak unik hingga mencapai n_splits.
    """
    if isinstance(seeds, str):
        seeds = parse_raw_seeds(seeds)

    if not seeds:
        return random.sample(range(1, 100000), n_splits)

    cleaned_seeds: List[int] = []
    seen = set()
    for s in seeds:
        try:
            s_int = int(s)
            if s_int not in seen:
                seen.add(s_int)
                cleaned_seeds.append(s_int)
        except (ValueError, TypeError):
            continue

    if not cleaned_seeds:
        return random.sample(range(1, 100000), n_splits)

    if len(cleaned_seeds) >= n_splits:
        return cleaned_seeds[:n_splits]

    needed = n_splits - len(cleaned_seeds)
    pool = set()
    while len(pool) < needed:
        candidates = random.sample(range(1, 100000), min(needed * 2, 90000))
        for cand in candidates:
            if cand not in seen and cand not in pool:
                pool.add(cand)
                if len(pool) == needed:
                    break

    cleaned_seeds.extend(list(pool))
    return cleaned_seeds


def run_monte_carlo_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
    seeds: Optional[List[int]] = None,
    n_splits: int = 200,
    test_size: float = 0.3,
    fit_intercept: bool = True,
    estimator_types: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MCCVBenchmarkResult:
    """
    Menjalankan pengujian Monte Carlo Cross-Validation pada dataset dan kumpulan random state.

    Parameters
    ----------
    X : np.ndarray
        Matriks fitur input.
    y : np.ndarray
        Array nilai target.
    feature_cols : List[str]
        Nama-nama kolom fitur independen.
    seeds : Optional[List[int]], default=None
        Daftar seed pengacakan untuk iterasi split train/test. Jika None atau kurang dari n_splits,
        sisanya akan di-generate secara acak hingga tepat n_splits. Jika lebih, diambil n_splits pertama.
    n_splits : int, default=200
        Jumlah split acak yang digunakan.
    test_size : float, default=0.3
        Proporsi data uji.
    fit_intercept : bool, default=True
        Apakah estimator menghitung konstanta / intercept.
    estimator_types : Optional[List[str]]
        Daftar jenis algoritma yang diuji (default: ["OLS", "S", "M", "MM"]).
    progress_callback : Optional[Callable[[int, int], None]]
        Fungsi callback opsional yang dipanggil dengan parameter (current_idx, total_seeds).

    Returns
    -------
    MCCVBenchmarkResult
        Objek berisi daftar seluruh record evaluasi iteratif, tabel rangkuman agregat statistik,
        dan daftar seeds yang digunakan.
    """
    from app.modules.regression.algorithms.ols_estimator import OLSEstimator
    from app.modules.regression.algorithms.s_estimator import SEstimator
    from app.modules.regression.algorithms.m_estimator import MEstimator
    from app.modules.regression.algorithms.mm_estimator import MMEstimator

    seeds = resolve_seeds(seeds, n_splits)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    total_samples = len(y)

    if estimator_types is None:
        estimator_types = ["OLS", "M", "S", "MM (k=4.685)", "MM (k=1.547)"]

    records: List[MCCVIterationResult] = []
    summary_stats = {
        name: {
            "test_r2": [], "test_rmse": [], "test_mae": [],
            "train_r2": [], "train_rmse": [], "train_mae": [],
            "f_stat": [], "f_table": [], "is_sig": []
        }
        for name in estimator_types
    }

    total_seeds = len(seeds)
    for idx, seed in enumerate(seeds, start=1):
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=test_size, random_state=seed
        )
        cached_s_model = None

        for est_name in estimator_types:
            if est_name == "OLS":
                model = OLSEstimator(fit_intercept=fit_intercept)
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                scale_val = float(np.std(y_train - y_pred_train))
                n_iter = 1
                is_converged = True
            elif est_name == "M":
                model = MEstimator(fit_intercept=fit_intercept)
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                scale_val = float(model.scale_) if model.scale_ is not None else None
                n_iter = int(model.n_iter_)
                is_converged = bool(model.converged_)
            elif est_name == "S":
                model = SEstimator(fit_intercept=fit_intercept)
                model.fit(X_train, y_train)
                cached_s_model = model
                y_pred_train = model.predict(X_train)
                scale_val = float(model.scale_) if model.scale_ is not None else None
                n_iter = int(model.n_iter_)
                is_converged = bool(model.converged_)
            elif est_name in ("MM (k=4.685)", "MM_4.685", "MM", "MM-High", "MM_High"):
                model = MMEstimator(fit_intercept=fit_intercept, efficiency="high")
                model.fit(X_train, y_train, initial_s_estimator=cached_s_model)
                y_pred_train = model.predict(X_train)
                scale_val = float(model.scale_s_) if model.scale_s_ is not None else None
                n_iter = int(model.n_iter_)
                is_converged = bool(model.converged_)
            elif est_name in ("MM (k=1.547)", "MM_1.547", "MM-Low", "MM_Low"):
                model = MMEstimator(fit_intercept=fit_intercept, efficiency="low")
                model.fit(X_train, y_train, initial_s_estimator=cached_s_model)
                y_pred_train = model.predict(X_train)
                scale_val = float(model.scale_s_) if model.scale_s_ is not None else None
                n_iter = int(model.n_iter_)
                is_converged = bool(model.converged_)
            else:
                raise ValueError(f"Estimator tidak dikenali: {est_name}")

            y_pred_test = model.predict(X_test)

            tr_rmse = float(calculate_rmse(y_train, y_pred_train))
            tr_mae = float(calculate_mae(y_train, y_pred_train))
            tr_r2 = float(calculate_r2(y_train, y_pred_train))

            te_rmse = float(calculate_rmse(y_test, y_pred_test))
            te_mae = float(calculate_mae(y_test, y_pred_test))
            te_r2 = float(calculate_r2(y_test, y_pred_test))

            # Simpan metrik ke kolektor statistik
            summary_stats[est_name]["test_r2"].append(te_r2)
            summary_stats[est_name]["test_rmse"].append(te_rmse)
            summary_stats[est_name]["test_mae"].append(te_mae)
            summary_stats[est_name]["train_r2"].append(tr_r2)
            summary_stats[est_name]["train_rmse"].append(tr_rmse)
            summary_stats[est_name]["train_mae"].append(tr_mae)

            # Format intercept dan koefisien standar {"CO": ..., "NO2": ...}
            raw_coefs = model.coef_ if fit_intercept else model.params_
            val_co = 0.0
            val_no2 = 0.0
            for idx, col in enumerate(feature_cols):
                c_low = str(col).lower()
                if "co" in c_low and "no2" not in c_low and "score" not in c_low:
                    val_co = float(raw_coefs[idx])
                elif "no2" in c_low:
                    val_no2 = float(raw_coefs[idx])
            if val_co == 0.0 and len(raw_coefs) >= 1:
                val_co = float(raw_coefs[0])
            if val_no2 == 0.0 and len(raw_coefs) >= 2:
                val_no2 = float(raw_coefs[1])

            coef_dict = {
                "CO": val_co,
                "NO2": val_no2,
            }

            if fit_intercept:
                intercept_val = float(model.intercept_) if hasattr(model, "intercept_") and model.intercept_ is not None else None
            else:
                intercept_val = None

            f_sig = calculate_f_significance(
                X=X_test,
                y=y_test,
                intercept=intercept_val if intercept_val is not None else 0.0,
                coefs=raw_coefs,
            )

            record = MCCVIterationResult(
                estimator_type=est_name,
                random_state=seed,
                test_size=test_size,
                total_samples=total_samples,
                fit_intercept=fit_intercept,
                intercept=intercept_val,
                coefficients=coef_dict,
                scale=scale_val,
                iterations=n_iter,
                converged=is_converged,
                test_rmse=te_rmse,
                test_mae=te_mae,
                test_r2=te_r2,
                train_rmse=tr_rmse,
                train_mae=tr_mae,
                train_r2=tr_r2,
                f_statistic=f_sig["f_statistic"],
                f_table=f_sig["f_table"],
                p_value=f_sig["p_value"],
                is_significant=f_sig["is_significant"],
            )
            # Simpan metrik ke kolektor statistik
            summary_stats[est_name]["f_stat"].append(f_sig["f_statistic"])
            summary_stats[est_name]["f_table"].append(f_sig["f_table"])
            summary_stats[est_name]["is_sig"].append(1.0 if f_sig["is_significant"] else 0.0)

            records.append(record)

        if progress_callback:
            progress_callback(idx, total_seeds)

    summary_rows: List[BenchmarkSummaryRow] = []
    for est_name in estimator_types:
        r2_mean = float(np.mean(summary_stats[est_name]["test_r2"]) * 100)
        r2_std = float(np.std(summary_stats[est_name]["test_r2"]) * 100)
        rmse_mean = float(np.mean(summary_stats[est_name]["test_rmse"]))
        rmse_std = float(np.std(summary_stats[est_name]["test_rmse"]))
        mae_mean = float(np.mean(summary_stats[est_name]["test_mae"]))
        mae_std = float(np.std(summary_stats[est_name]["test_mae"]))

        tr_r2_mean = float(np.mean(summary_stats[est_name]["train_r2"]) * 100)
        tr_r2_std = float(np.std(summary_stats[est_name]["train_r2"]) * 100)
        tr_rmse_mean = float(np.mean(summary_stats[est_name]["train_rmse"]))
        tr_rmse_std = float(np.std(summary_stats[est_name]["train_rmse"]))
        tr_mae_mean = float(np.mean(summary_stats[est_name]["train_mae"]))
        tr_mae_std = float(np.std(summary_stats[est_name]["train_mae"]))

        f_mean = float(np.mean(summary_stats[est_name]["f_stat"])) if summary_stats[est_name]["f_stat"] else 0.0
        f_std = float(np.std(summary_stats[est_name]["f_stat"])) if summary_stats[est_name]["f_stat"] else 0.0
        f_tab = float(np.mean(summary_stats[est_name]["f_table"])) if summary_stats[est_name]["f_table"] else 0.0
        sig_rate = float(np.mean(summary_stats[est_name]["is_sig"]) * 100) if summary_stats[est_name]["is_sig"] else 0.0
        is_sig_overall = bool(f_mean > f_tab)

        summary_rows.append(
            BenchmarkSummaryRow(
                estimator=est_name,
                test_r2_mean=round(r2_mean, 2),
                test_r2_std=round(r2_std, 2),
                test_r2_display=f"{r2_mean:.2f}% (+/- {r2_std:.2f}%)",
                test_rmse_mean=round(rmse_mean, 4),
                test_rmse_std=round(rmse_std, 4),
                test_rmse_display=f"{rmse_mean:.4f} (+/- {rmse_std:.4f})",
                test_mae_mean=round(mae_mean, 4),
                test_mae_std=round(mae_std, 4),
                test_mae_display=f"{mae_mean:.4f} (+/- {mae_std:.4f})",
                train_r2_mean=round(tr_r2_mean, 2),
                train_r2_std=round(tr_r2_std, 2),
                train_r2_display=f"{tr_r2_mean:.2f}% (+/- {tr_r2_std:.2f}%)",
                train_rmse_mean=round(tr_rmse_mean, 4),
                train_rmse_std=round(tr_rmse_std, 4),
                train_rmse_display=f"{tr_rmse_mean:.4f} (+/- {tr_rmse_std:.4f})",
                train_mae_mean=round(tr_mae_mean, 4),
                train_mae_std=round(tr_mae_std, 4),
                train_mae_display=f"{tr_mae_mean:.4f} (+/- {tr_mae_std:.4f})",
                f_statistic_mean=round(f_mean, 4),
                f_statistic_display=f"{f_mean:.2f} (+/- {f_std:.2f})",
                f_table=round(f_tab, 4),
                is_significant=is_sig_overall,
                significance_rate=round(sig_rate, 2),
            )
        )

    return MCCVBenchmarkResult(
        records=records,
        summary_rows=summary_rows,
        seeds=seeds,
    )


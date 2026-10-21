"""
Modul Utilitas dan Fungsi Pembobot Tukey Bisquare untuk Regresi Robust.:
1. calculate_mad_scale: Penaksir skala awal MAD (Median Absolute Deviation).
2. weight_tukey: Fungsi pembobot Tukey Bisquare untuk estimasi parameter WLS.
"""

import numpy as np


def calculate_mad_scale(residuals, divisor=0.6745):
    """
    Menghitung skala MAD (Median Absolute Deviation):
    sigma = median(|e_i - median(e)|) / 0.6745 = 1.4826 * MAD

    Parameters:
    -----------
    residuals : array-like
        Array residual (e_i = y_i - y_hat_i).
    divisor : float, default=0.6745
        Konstanta normalisasi distribusi normal (1 / 1.4826 = 0.6745).

    Returns:
    --------
    float
        Nilai estimasi skala residual sigma.
    """
    # 1. Konversi data residual menjadi array float
    residuals = np.asarray(residuals, dtype=float)

    # 2. Hitung nilai median dari seluruh residual
    med_res = np.median(residuals)

    # 3. Hitung selisih mutlak setiap residual terhadap nilai median (|e_i - median(e)|)
    abs_deviations = np.abs(residuals - med_res)

    # 4. Ambil median dari selisih mutlak tersebut (MAD)
    mad = np.median(abs_deviations)

    # 5. Bagi nilai MAD dengan konstanta 0.6745 untuk memperoleh estimasi standar deviasi
    scale = mad / divisor

    # 6. Batasi nilai minimum skala (1e-8) agar tidak terjadi pembagian dengan nol
    return max(float(scale), 1e-8)


def weight_tukey(u, c=4.685):
    """
    Fungsi pembobot Tukey Bisquare untuk regresi terboboti (WLS):
    w(u) = [1 - (u / c)^2]^2  jika |u| <= c
    w(u) = 0                  jika |u| > c

    Digunakan pada:
    - Estimasi-M: c = 4.685 (efisiensi 95% pada distribusi normal)
    - Estimasi-S : c = 1.547 (pada iterasi awal WLS)
    - Estimasi-MM: c = 4.685 (pada tahapan pembaruan koefisien WLS)

    Parameters:
    -----------
    u : array-like
        Residual terstandarisasi (u_i = e_i / sigma).
    c : float, default=4.685
        Konstanta penala (tuning constant).

    Returns:
    --------
    np.ndarray
        Array nilai bobot w_i pada rentang [0, 1].
    """
    # 1. Pastikan u dalam bentuk array numerik bertipe float
    u = np.asarray(u, dtype=float)

    # 2. Hitung rasio residual terstandarisasi terhadap konstanta tuning (u / c)
    u_c = u / c

    # 3. Berikan bobot: jika |u| <= c gunakan [1 - (u/c)^2]^2, jika |u| > c beri bobot 0
    weights = np.where(
        np.abs(u) <= c,
        (1.0 - u_c**2)**2,
        0.0
    )

    # 4. Kembalikan array nilai bobot
    return weights
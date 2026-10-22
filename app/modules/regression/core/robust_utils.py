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


def weight_tukey(u, k=4.685):
    """
    Fungsi pembobot Tukey Bisquare untuk regresi terboboti (WLS):
    w(u) = [1 - (u / k)^2]^2  jika |u| <= k
    w(u) = 0                  jika |u| > k

    Digunakan pada:
    - Estimasi-M: k = 4.685 (efisiensi 95% pada distribusi normal)
    - Estimasi-S : k = 1.547 (pada iterasi awal WLS)
    - Estimasi-MM: k = 4.685 (pada tahapan pembaruan koefisien WLS)

    Parameters:
    -----------
    u : array-like
        Residual terstandarisasi (u_i = e_i / sigma).
    k : float, default=4.685
        Konstanta penala (tuning constant).

    Returns:
    --------
    np.ndarray
        Array nilai bobot w_i pada rentang [0, 1].
    """
    # 1. Pastikan u dalam bentuk array numerik bertipe float
    u = np.asarray(u, dtype=float)

    # 2. Hitung rasio residual terstandarisasi terhadap konstanta tuning (u / k)
    u_k = u / k

    # 3. Berikan bobot: jika |u| <= k gunakan [1 - (u/k)^2]^2, jika |u| > k beri bobot 0
    weights = np.where(
        np.abs(u) <= k,
        (1.0 - u_k**2)**2,
        0.0
    )

    # 4. Kembalikan array nilai bobot
    return weights

def weight_s_scale(u, k=1.547):
    """
    Fungsi pembobot skala Estimasi-S pada iterasi > 1:
    w(u) = rho(u) / u^2

    Dijabarkan langkah demi langkah per perkalian matematis untuk akurasi numerik.

    Parameters:
    -----------
    u : array-like
        Residual terstandarisasi (u_i = e_i / sigma).
    k : float, default=1.547
        Konstanta penala Estimasi-S (1.547 untuk breakdown point 50%).

    Returns:
    --------
    np.ndarray
        Array nilai bobot skala w(u) = rho(u) / u^2.
    """
    # 1. Konversi u ke tipe array float
    u = np.asarray(u, dtype=float)

    # 2. Ambil nilai mutlak dari u (|u|)
    abs_u = np.abs(u)

    # 3. Beri batas pengaman nilai minimum u (1e-12) agar tidak terjadi pembagian dengan nol
    u_safe = np.where(abs_u < 1e-12, 1e-12, abs_u)

    # 4. Langkah 1: Hitung (u / k)
    u_div_k = u_safe / k

    # 5. Langkah 2: Hitung kuadrat (u / k)^2
    u_div_k_sq = u_div_k ** 2

    # 6. Langkah 3: Hitung 1 - (u / k)^2
    satu_minus_sq = 1.0 - u_div_k_sq

    # 7. Langkah 4: Hitung pangkat tiga (1 - (u / k)^2)^3
    pangkat_tiga = satu_minus_sq ** 3

    # 8. Langkah 5: Hitung 1 - (1 - (u / k)^2)^3
    satu_minus_pangkat_tiga = 1.0 - pangkat_tiga

    # 9. Langkah 6: Hitung konstanta depan k^2 / 6
    konstanta_depan = (k ** 2) / 6.0

    # 10. Langkah 7: Hitung rho(u) = (k^2 / 6) * [1 - (1 - (u/k)^2)^3] jika |u| <= k, dan k^2/6 jika |u| > k
    rho_u = np.where(
        abs_u <= k,
        konstanta_depan * satu_minus_pangkat_tiga,
        konstanta_depan
    )

    # 11. Langkah 8: Hitung nilai u^2
    u_squared = u_safe ** 2

    # 12. Langkah 9: Hitung pembobot skala w(u) = rho(u) / u^2
    weights = rho_u / u_squared

    # 13. Kembalikan bobot skala hasil perhitungan
    return weights
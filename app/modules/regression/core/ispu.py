"""
Modul perhitungan dan klasifikasi Indeks Standar Pencemar Udara (ISPU)
berdasarkan Peraturan Menteri Lingkungan Hidup dan Kehutanan (Permen LHK) No. 14 Tahun 2020.
"""

from typing import Union


def get_ispu_category(ispu_val: Union[float, int]) -> str:
    """
    Menentukan kategori ISPU dari nilai indeks ISPU:
    - 0 s/d 50    : Baik
    - 51 s/d 100  : Sedang
    - 101 s/d 200 : Tidak Sehat
    - 201 s/d 300 : Sangat Tidak Sehat
    - > 300       : Berbahaya

    Parameters
    ----------
    ispu_val : float or int
        Nilai Indeks Standar Pencemar Udara.

    Returns
    -------
    str:
        Kategori ISPU ('Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya').
    """
    val = max(0.0, float(ispu_val))
    if val <= 50.0:
        return "Baik"
    elif val <= 100.0:
        return "Sedang"
    elif val <= 200.0:
        return "Tidak Sehat"
    elif val <= 300.0:
        return "Sangat Tidak Sehat"
    else:
        return "Berbahaya"

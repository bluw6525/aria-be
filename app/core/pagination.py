"""
Modul utilitas pagination untuk seluruh aplikasi.
Menangani konversi antara nomor halaman (page) dari frontend
menjadi offset database SQL untuk query database.
"""


def calculate_offset(page: int = 1, limit: int = 10) -> int:
    """
    Menghitung nilai offset SQL berdasarkan nomor halaman dan limit per halaman.

    Parameters:
    -----------
    page : int
        Nomor halaman manusiawi (1-indexed, mulai dari 1).
    limit : int
        Batas maksimum jumlah baris per halaman.

    Returns:
    --------
    int:
        Nilai offset baris untuk database (0-indexed).
        Contoh: page 1 limit 10 -> offset 0
                page 2 limit 10 -> offset 10
    """
    safe_page = max(1, page)
    safe_limit = max(1, limit)
    return (safe_page - 1) * safe_limit


def calculate_total_pages(total: int, limit: int = 10) -> int:
    """
    Menghitung total halaman berdasarkan jumlah total baris dan limit per halaman.
    """
    safe_limit = max(1, limit)
    return max(1, (total + safe_limit - 1) // safe_limit)

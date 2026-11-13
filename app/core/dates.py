"""
Utilitas kalkulasi dan manipulasi tanggal untuk seluruh aplikasi Aria.
Menyediakan fungsi murni (pure functions) untuk parsing, penambahan hari,
dan validasi rentang tanggal ISO (YYYY-MM-DD).
"""

from datetime import datetime, timedelta
from typing import Tuple, Union


def parse_date(date_val: Union[str, datetime], fmt: str = "%Y-%m-%d") -> datetime:
    """
    Mengonversi string tanggal atau objek datetime menjadi datetime.
    Jika format tidak sesuai, mengembalikan datetime saat ini (now).
    """
    if isinstance(date_val, datetime):
        return date_val
    try:
        return datetime.strptime(str(date_val), fmt)
    except (ValueError, TypeError):
        return datetime.now()


def add_days_to_date(
    date_val: Union[str, datetime],
    days: int,
    fmt: str = "%Y-%m-%d"
) -> str:
    """
    Menambahkan sejumlah hari (positif atau negatif) ke tanggal dan
    mengembalikannya dalam format string (default: YYYY-MM-DD).
    """
    dt = parse_date(date_val, fmt)
    return (dt + timedelta(days=days)).strftime(fmt)


def validate_date_range(
    start_date_str: str,
    end_date_str: str,
    max_days: int = 31,
    fmt: str = "%Y-%m-%d"
) -> Tuple[datetime, datetime]:
    """
    Validasi format tanggal (YYYY-MM-DD) dan memastikan rentang waktu tidak melebihi batas.

    Raises
    ------
    ValueError:
        Jika format tanggal salah, start_date > end_date, atau rentang waktu > max_days.
    """
    try:
        start_dt = datetime.strptime(start_date_str, fmt)
        end_dt = datetime.strptime(end_date_str, fmt)
    except ValueError as err:
        raise ValueError(f"Format tanggal tidak valid. Gunakan format {fmt}.") from err

    if start_dt > end_dt:
        raise ValueError("Tanggal awal (start_date) tidak boleh lebih besar dari tanggal akhir (end_date).")

    diff_days = (end_dt - start_dt).days + 1
    if diff_days > max_days:
        raise ValueError(
            f"Rentang tanggal ({diff_days} hari) melebihi batas maksimal {max_days} hari. "
            "Pilih rentang tanggal maksimal 1 bulan."
        )

    return start_dt, end_dt

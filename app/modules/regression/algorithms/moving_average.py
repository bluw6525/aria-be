"""
Modul algoritma Moving Average untuk peramalan time-series (H+1, H+2, ..., H+n).
Murni matematis, menerima deret numerik (Sequence[float]).
"""

from typing import List, Sequence


def forecast_moving_average(
    series: Sequence[float],
    k: int = 3,
    steps: int = 2,
) -> List[float]:
    """
    Melakukan peramalan deret waktu multi-langkah menggunakan Moving Average autoregresif.
    Nilai prediksi langkah sebelumnya digulung (rolled) ke dalam jendela observasi berikutnya.

    Args:
        series: Deret nilai numerik berurutan kronologis.
        k: Ukuran jendela waktu observasi terakhir (default: 3).
        steps: Jumlah langkah ke depan yang diramalkan (default: 2 untuk H+1 & H+2).

    Returns:
        List nilai ramalan sepanjang `steps`. Mengembalikan list kosong jika data valid tidak ada.
    """
    valid = [float(v) for v in series if v is not None]
    if not valid:
        return []

    forecasts: List[float] = []
    current_series = list(valid)

    for _ in range(steps):
        window = current_series[-k:]
        next_val = sum(window) / len(window)
        forecasts.append(next_val)
        current_series.append(next_val)

    return forecasts

"""
Modul pembantu (helper) untuk parsing dan validasi berkas data Excel (.xlsx, .xls) dan CSV.
Mengidentifikasi kolom fitur CO (Karbon Monoksida) dan NO2 (Nitrogen Dioksida) dari Sentinel-5P,
kolom target O3 (Ozon), serta kolom label waktu (tanggal/date) secara otomatis dan cerdas.
"""

import asyncio
import io
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from fastapi import UploadFile
from scipy.interpolate import CubicSpline

# Pola regex pencocokan kolom CO (Karbon Monoksida / Carbon Monoxide dari satelit)
CO_PATTERNS = [
    r"^co$",                           # "CO" persis
    r"\bco\b",                         # Kata utuh "CO", "CO (mol/m2)"
    r"^co[_.\s-]",                     # "CO_satelit", "co.surface"
    r"[_.\s-]co$",                     # "satelit_co"
    r"karbon[_\s-]?monoksida",         # "karbon monoksida", "karbon_monoksida"
    r"carbon[_\s-]?monoxide",          # "carbon monoxide", "carbon_monoxide"
    r"co[_\s-]?satelit",               # "co_satelit", "co satelit"
    r"satelit[_\s-]?co",               # "satelit_co", "satelit co"
]

# Pola regex pencocokan kolom NO2 (Nitrogen Dioksida / Nitrogen Dioxide dari satelit)
NO2_PATTERNS = [
    r"^no2$",                          # "NO2" persis
    r"\bno2\b",                        # Kata utuh "NO2", "NO2 (mol/m2)"
    r"^no2[_.\s-]",                    # "NO2_satelit", "no2.surface"
    r"[_.\s-]no2$",                    # "satelit_no2"
    r"nitrogen[_\s-]?dioksida",        # "nitrogen dioksida", "nitrogen_dioksida"
    r"nitrogen[_\s-]?dioxide",         # "nitrogen dioxide", "nitrogen_dioxide"
    r"no2[_\s-]?satelit",              # "no2_satelit", "no2 satelit"
    r"satelit[_\s-]?no2",              # "satelit_no2", "satelit no2"
]

# Pola regex pencocokan kolom target O3 (Ozon / Ozone dari stasiun SPKU)
O3_PATTERNS = [
    r"^o3$",                           # "O3" persis
    r"\bo3\b",                         # Kata utuh "O3", "O3 (ug/m3)"
    r"^o3[_.\s-]",                     # "O3_stasiun", "o3.ground", "o3_spku"
    r"[_.\s-]o3$",                     # "konsentrasi_o3", "spku_o3"
    r"o3[_\s-]?spku",                  # "o3_spku", "o3 spku"
    r"spku[_\s-]?o3",                  # "spku_o3", "spku o3"
    r"\bozon\b",                       # "ozon", "kadar ozon"
    r"\bozone\b",                      # "ozone", "surface ozone"
    r"^spku$",                         # nama kolom "spku"
]

# Pola regex pencocokan kolom penanda tanggal / waktu
LABEL_PATTERNS = [
    r"tanggal",
    r"date",
    r"waktu",
    r"time",
    r"tahun",
    r"year",
    r"timestamp",
    r"datetime",
]


def match_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    """
    Mencari nama kolom yang paling sesuai berdasarkan daftar pola regex prioritas.
    Mencegah false positive (misal kata 'score', 'record_id', 'constant' tidak boleh terdeteksi sebagai 'CO').
    """
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for col in columns:
            if regex.search(col):
                return col
    return None


def read_uploaded_dataframe(file: Union[UploadFile, str, Path]) -> pd.DataFrame:
    """
    Membaca berkas (.xlsx, .xls, .csv) baik dari UploadFile maupun path lokal (str/Path)
    secara sinkron dan mengembalikan pandas DataFrame yang bersih.
    """
    if isinstance(file, (str, Path)):
        path_obj = Path(file).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Berkas dataset tidak ditemukan: {path_obj}")
        ext = path_obj.suffix.lower().lstrip(".")
        target = path_obj
    else:
        filename = getattr(file, "filename", "") or ""
        ext = filename.lower().split(".")[-1]
        # file.file adalah file-like object (SpooledTemporaryFile) sinkron dari FastAPI / Starlette
        target = getattr(file, "file", file)
        if hasattr(target, "seek"):
            target.seek(0)

    try:
        if ext in ["xlsx", "xls"]:
            df = pd.read_excel(target)
        elif ext == "csv":
            df = pd.read_csv(target)
        else:
            raise ValueError(f"Format berkas '.{ext}' tidak didukung. Harap gunakan .csv, .xls, atau .xlsx.")
    except Exception as e:
        raise ValueError(f"Gagal membaca berkas: {str(e)}")

    df.columns = [str(col).strip() for col in df.columns]
    return df



def interpolate_natural_cubic_spline(series: pd.Series) -> pd.Series:
    """
    Melakukan interpolasi Natural Cubic Spline (bc_type='natural') pada kolom target O3
    jika terdapat nilai missing (NaN). Nilai hasil interpolasi dibatasi >= 0 karena
    konsentrasi gas di atmosfer tidak boleh bernilai negatif.
    """
    missing_mask = series.isna()
    if not missing_mask.any():
        return series

    valid_mask = ~missing_mask
    n_valid = int(valid_mask.sum())
    if n_valid < 2:
        raise ValueError(
            f"Jumlah data O3 SPKU valid hanya {n_valid} baris. "
            "Minimal dibutuhkan 2 baris valid untuk interpolasi Natural Cubic Spline."
        )

    # Indeks baris integer untuk interpolasi
    x_valid = series.index[valid_mask].to_numpy(dtype=float)
    y_valid = series.loc[valid_mask].to_numpy(dtype=float)

    # Natural Cubic Spline mensyaratkan turunan kedua di titik ujung adalah 0
    cs = CubicSpline(x_valid, y_valid, bc_type="natural", extrapolate=True)

    x_missing = series.index[missing_mask].to_numpy(dtype=float)
    interpolated_values = cs(x_missing)

    # Pastikan tidak ada nilai negatif untuk konsentrasi O3
    series_clean = series.copy()
    series_clean.loc[missing_mask] = np.clip(interpolated_values, a_min=0.0, a_max=None)
    return series_clean


async def parse_and_validate_file(
    file: Union[UploadFile, str, Path],
    require_target: bool = True,
    return_indices: bool = False,
) -> Union[
    Tuple[np.ndarray, Optional[np.ndarray], List[str], str],
    Tuple[np.ndarray, Optional[np.ndarray], List[str], str, np.ndarray],
]:
    """
    Membaca dan memvalidasi berkas data latih / uji untuk model regresi.
    - Mencari otomatis kolom CO satelit, NO2 satelit, dan O3 SPKU.
    - Hanya mengambil ketiga variabel tersebut (variabel lain diabaikan).
    - Jika terdapat missing value pada O3 SPKU, dilakukan interpolasi Natural Cubic Spline.
    - Hanya data yang memiliki O3 SPKU, NO2 satelit, dan CO satelit lengkap yang diambil.
    - Dapat mengembalikan original_indices untuk melacak nomor baris data sebelum prapemrosesan.

    Returns:
    --------
    X : np.ndarray (n_samples, 2)
    y : Optional[np.ndarray] (n_samples,) jika require_target=True, None jika False
    feature_cols : List[str] nama kolom fitur terdeteksi [co_col, no2_col]
    target_col : str nama kolom target terdeteksi
    original_indices : np.ndarray (n_samples,) jika return_indices=True (0-indexed baris asli)
    """
    df = await asyncio.to_thread(read_uploaded_dataframe, file)
    # Catat indeks baris asli dari file mentah sebelum pengurutan atau dropna (0-indexed)
    df["_raw_row_idx"] = np.arange(len(df), dtype=int)
    cols = [c for c in df.columns if c != "_raw_row_idx"]

    # 1. Deteksi Kolom Fitur CO & NO2 (Satelit Sentinel-5P)
    co_col = match_column(cols, CO_PATTERNS)
    no2_col = match_column(cols, NO2_PATTERNS)

    if not co_col or not no2_col:
        missing = []
        if not co_col:
            missing.append("CO satelit (Karbon Monoksida)")
        if not no2_col:
            missing.append("NO2 satelit (Nitrogen Dioksida)")
        raise ValueError(
            f"Kolom {', '.join(missing)} tidak ditemukan dalam berkas. "
            f"Kolom yang terdeteksi: {cols}. "
            f"Pastikan berkas memiliki kolom CO satelit dan NO2 satelit."
        )

    feature_cols = [co_col, no2_col]
    target_col = ""

    # 2. Urutkan secara kronologis jika ada kolom penanda waktu/tanggal
    label_col = match_column(cols, LABEL_PATTERNS)
    if label_col:
        try:
            parsed_dates = pd.to_datetime(df[label_col], errors="coerce")
            if parsed_dates.notna().sum() > 0.5 * len(df):
                df = (
                    df.assign(_temp_sort_date=parsed_dates)
                    .sort_values(by="_temp_sort_date", na_position="last")
                    .drop(columns=["_temp_sort_date"])
                )
        except Exception:
            pass

    df = df.reset_index(drop=True)

    # 3. Deteksi & Proses Kolom Target O3 SPKU jika diperlukan (Latih & Uji)
    if require_target:
        target_col = match_column(cols, O3_PATTERNS)

        if not target_col:
            raise ValueError(
                f"Kolom target O3 SPKU / Ozon tidak ditemukan dalam berkas. "
                f"Kolom yang terdeteksi: {cols}. "
                f"Pastikan berkas data latih atau uji menyertakan kolom O3 SPKU."
            )

        # Hanya ambil kolom CO satelit, NO2 satelit, O3 SPKU, dan indeks baris asli
        work_df = pd.DataFrame(index=df.index)
        work_df[co_col] = pd.to_numeric(df[co_col], errors="coerce")
        work_df[no2_col] = pd.to_numeric(df[no2_col], errors="coerce")
        work_df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        work_df["_raw_row_idx"] = df["_raw_row_idx"]

        # Interpolasi Natural Cubic Spline jika ada missing value pada target O3
        work_df[target_col] = interpolate_natural_cubic_spline(work_df[target_col])

        # Hanya ambil baris yang memiliki nilai lengkap O3 SPKU, NO2 satelit, dan CO satelit
        clean_df = work_df[[co_col, no2_col, target_col, "_raw_row_idx"]].dropna()
        clean_df = clean_df[
            np.isfinite(clean_df[co_col])
            & np.isfinite(clean_df[no2_col])
            & np.isfinite(clean_df[target_col])
        ]

        if len(clean_df) < 5:
            raise ValueError(
                f"Jumlah baris data valid terlalu sedikit ({len(clean_df)} baris). "
                f"Minimal 5 baris data yang memiliki O3 SPKU, NO2 satelit, dan CO satelit lengkap."
            )

        X = clean_df[feature_cols].to_numpy(dtype=float)
        y = clean_df[target_col].to_numpy(dtype=float)
        orig_indices = clean_df["_raw_row_idx"].to_numpy(dtype=int)

        if return_indices:
            return X, y, feature_cols, target_col, orig_indices
        return X, y, feature_cols, target_col

    else:
        # Jika hanya membutuhkan fitur (misal inferensi tanpa target)
        work_df = pd.DataFrame(index=df.index)
        work_df[co_col] = pd.to_numeric(df[co_col], errors="coerce")
        work_df[no2_col] = pd.to_numeric(df[no2_col], errors="coerce")
        work_df["_raw_row_idx"] = df["_raw_row_idx"]

        clean_df = work_df[[co_col, no2_col, "_raw_row_idx"]].dropna()
        clean_df = clean_df[
            np.isfinite(clean_df[co_col]) & np.isfinite(clean_df[no2_col])
        ]

        if len(clean_df) < 1:
            raise ValueError(
                "Tidak ada data baris valid yang memiliki nilai CO satelit dan NO2 satelit lengkap."
            )

        X = clean_df[feature_cols].to_numpy(dtype=float)
        orig_indices = clean_df["_raw_row_idx"].to_numpy(dtype=int)

        if return_indices:
            return X, None, feature_cols, "", orig_indices
        return X, None, feature_cols, ""


async def parse_batch_estimate_file(
    file: UploadFile
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Khusus untuk inferensi estimasi batch:
    Membaca berkas satelit dan mengekstrak array CO satelit, NO2 satelit, target O3 SPKU (opsional),
    serta label baris/waktu. 

    Validasi ketat:
    - Tidak melakukan interpolasi spline.
    - Jika terdapat missing value (kosong/NaN/non-numeric/inf) pada CO satelit, NO2 satelit,
      atau pada kolom O3 SPKU (jika kolom O3 disertakan), sistem akan langsung melempar error
      agar pengguna memastikan kebersihan dataset estimasi terlebih dahulu.

    Returns:
    --------
    co_vals : np.ndarray
    no2_vals : np.ndarray
    actual_vals : Optional[np.ndarray] (None jika kolom target O3 tidak ada di berkas)
    labels : List[str] daftar penanda baris (misal format YYYY-MM-DD atau nomor urut)
    """
    df = await asyncio.to_thread(read_uploaded_dataframe, file)
    if len(df) < 1:
        raise ValueError("Berkas data estimasi kosong, tidak ada baris data untuk diproses.")

    cols = list(df.columns)

    co_col = match_column(cols, CO_PATTERNS)
    no2_col = match_column(cols, NO2_PATTERNS)

    if not co_col or not no2_col:
        missing = []
        if not co_col:
            missing.append("CO satelit")
        if not no2_col:
            missing.append("NO2 satelit")
        raise ValueError(
            f"Kolom {', '.join(missing)} tidak ditemukan dalam berkas estimasi. Kolom terdeteksi: {cols}"
        )

    target_col = match_column(cols, O3_PATTERNS)
    label_col = match_column(cols, LABEL_PATTERNS)

    # 1. Validasi ketat kolom CO satelit
    co_series = pd.to_numeric(df[co_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_co = int(co_series.isna().sum())
    if missing_co > 0:
        raise ValueError(
            f"Ditemukan {missing_co} baris data kosong (missing value) atau bernilai non-numerik "
            f"pada kolom '{co_col}'. Harap pastikan seluruh baris fitur CO satelit telah terisi lengkap dan bersih."
        )

    # 2. Validasi ketat kolom NO2 satelit
    no2_series = pd.to_numeric(df[no2_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_no2 = int(no2_series.isna().sum())
    if missing_no2 > 0:
        raise ValueError(
            f"Ditemukan {missing_no2} baris data kosong (missing value) atau bernilai non-numerik "
            f"pada kolom '{no2_col}'. Harap pastikan seluruh baris fitur NO2 satelit telah terisi lengkap dan bersih."
        )

    # 3. Validasi ketat kolom O3 SPKU (jika pengguna menyertakannya di berkas untuk evaluasi)
    actual_vals: Optional[np.ndarray] = None
    if target_col:
        target_series = pd.to_numeric(df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_target = int(target_series.isna().sum())
        if missing_target > 0:
            raise ValueError(
                f"Ditemukan {missing_target} baris data kosong (missing value) atau bernilai non-numerik "
                f"pada kolom target '{target_col}'. Harap pastikan data O3 sudah bersih dan lengkap, "
                f"atau hapus kolom O3 dari berkas jika hanya ingin melakukan estimasi tanpa perbandingan nilai aktual."
            )
        actual_vals = target_series.to_numpy(dtype=float)

    co_vals = co_series.to_numpy(dtype=float)
    no2_vals = no2_series.to_numpy(dtype=float)

    labels: List[str] = []
    for idx in range(len(df)):
        if label_col:
            val_raw = df[label_col].iloc[idx]
            if pd.notnull(val_raw):
                if hasattr(val_raw, "strftime"):
                    labels.append(val_raw.strftime("%Y-%m-%d"))
                else:
                    labels.append(str(val_raw))
            else:
                labels.append(str(idx + 1))
        else:
            labels.append(str(idx + 1))

    return co_vals, no2_vals, actual_vals, labels


def parse_raw_seeds(seeds: Optional[Union[str, List[int]]]) -> Optional[List[int]]:
    """Mengekstrak list integer random seeds dari string (spasi/koma/JSON) atau list."""
    if not seeds:
        return None
    if isinstance(seeds, list):
        return [int(s) for s in seeds]
    if isinstance(seeds, str) and seeds.strip():
        raw_seeds = seeds.replace("[", "").replace("]", "").replace(",", " ").split()
        parsed = [int(s) for s in raw_seeds if s.lstrip("-").isdigit()]
        return parsed or None
    return None


def parse_raw_estimators(estimator_types: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """Mengekstrak list estimator dari string (JSON list / dipisahkan koma) atau list."""
    if not estimator_types:
        return None
    if isinstance(estimator_types, list):
        return [str(e) for e in estimator_types]
    if isinstance(estimator_types, str) and estimator_types.strip():
        try:
            parsed = json.loads(estimator_types)
            if isinstance(parsed, list):
                return [str(e) for e in parsed]
        except Exception:
            pass
        return [e.strip() for e in estimator_types.split(",") if e.strip()]
    return None


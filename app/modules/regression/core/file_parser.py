"""
Modul pembantu (helper) untuk parsing dan validasi berkas data Excel (.xlsx, .xls) dan CSV.
Mengidentifikasi kolom fitur CO (Karbon Monoksida) dan NO2 (Nitrogen Dioksida) dari Sentinel-5P,
kolom target O3 (Ozon), serta kolom label waktu (tanggal/date) secara otomatis dan cerdas.
"""

import io
import re
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from fastapi import UploadFile

# Pola regex pencocokan kolom CO (Karbon Monoksida / Carbon Monoxide)
CO_PATTERNS = [
    r"^co$",                           # "CO" persis
    r"\bco\b",                         # Kata utuh "CO", "CO (mol/m2)"
    r"^co[_.\s-]",                     # "CO_satelit", "co.surface"
    r"[_.\s-]co$",                     # "satelit_co"
    r"karbon[_\s-]?monoksida",         # "karbon monoksida", "karbon_monoksida"
    r"carbon[_\s-]?monoxide",          # "carbon monoxide", "carbon_monoxide"
]

# Pola regex pencocokan kolom NO2 (Nitrogen Dioksida / Nitrogen Dioxide)
NO2_PATTERNS = [
    r"^no2$",                          # "NO2" persis
    r"\bno2\b",                        # Kata utuh "NO2", "NO2 (mol/m2)"
    r"^no2[_.\s-]",                    # "NO2_satelit", "no2.surface"
    r"[_.\s-]no2$",                    # "satelit_no2"
    r"nitrogen[_\s-]?dioksida",        # "nitrogen dioksida", "nitrogen_dioksida"
    r"nitrogen[_\s-]?dioxide",         # "nitrogen dioxide", "nitrogen_dioxide"
]

# Pola regex pencocokan kolom target O3 (Ozon / Ozone)
O3_PATTERNS = [
    r"^o3$",                           # "O3" persis
    r"\bo3\b",                         # Kata utuh "O3", "O3 (ug/m3)"
    r"^o3[_.\s-]",                     # "O3_stasiun", "o3.ground"
    r"[_.\s-]o3$",                     # "konsentrasi_o3"
    r"\bozon\b",                       # "ozon", "kadar ozon"
    r"\bozone\b",                      # "ozone", "surface ozone"
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


async def read_uploaded_dataframe(file: Union[UploadFile, str, Path]) -> pd.DataFrame:
    """
    Membaca berkas (.xlsx, .xls, .csv) baik dari UploadFile maupun path lokal (str/Path)
    dan mengembalikan pandas DataFrame yang bersih.
    """
    if isinstance(file, (str, Path)):
        path_obj = Path(file).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Berkas dataset tidak ditemukan: {path_obj}")
        ext = path_obj.suffix.lower().lstrip(".")
        try:
            if ext in ["xlsx", "xls"]:
                df = pd.read_excel(path_obj)
            elif ext == "csv":
                df = pd.read_csv(path_obj)
            else:
                raise ValueError(f"Format berkas '.{ext}' tidak didukung. Harap gunakan .csv, .xls, atau .xlsx.")
        except Exception as e:
            raise ValueError(f"Gagal membaca berkas: {str(e)}")
    else:
        filename = getattr(file, "filename", "") or ""
        ext = filename.lower().split(".")[-1]

        contents = await file.read()
        if not contents:
            raise ValueError("Berkas yang diunggah kosong.")

        try:
            if ext in ["xlsx", "xls"]:
                df = pd.read_excel(io.BytesIO(contents))
            elif ext == "csv":
                df = pd.read_csv(io.BytesIO(contents))
            else:
                raise ValueError(f"Format berkas '.{ext}' tidak didukung. Harap unggah .csv, .xls, atau .xlsx.")
        except Exception as e:
            raise ValueError(f"Gagal membaca berkas: {str(e)}")

    df.columns = [str(col).strip() for col in df.columns]
    return df


async def parse_and_validate_file(
    file: Union[UploadFile, str, Path],
    require_target: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], str]:
    """
    Membaca dan memvalidasi berkas data latih / uji untuk model regresi.
    Mencari otomatis kolom CO (Karbon Monoksida), NO2 (Nitrogen Dioksida), dan O3 (Ozon).

    Returns:
    --------
    X : np.ndarray (n_samples, 2)
    y : Optional[np.ndarray] (n_samples,) jika require_target=True, None jika False
    feature_cols : List[str] nama kolom fitur terdeteksi [co_col, no2_col]
    target_col : str nama kolom target terdeteksi
    """
    df = await read_uploaded_dataframe(file)
    cols = list(df.columns)

    # 1. Deteksi Kolom Fitur CO & NO2
    co_col = match_column(cols, CO_PATTERNS)
    no2_col = match_column(cols, NO2_PATTERNS)

    if not co_col or not no2_col:
        missing = []
        if not co_col:
            missing.append("CO / Karbon Monoksida")
        if not no2_col:
            missing.append("NO2 / Nitrogen Dioksida")
        raise ValueError(
            f"Kolom {', '.join(missing)} tidak ditemukan dalam berkas. "
            f"Kolom yang terdeteksi: {cols}. "
            f"Pastikan berkas memiliki kolom CO dan NO2 (atau nama Karbon Monoksida dan Nitrogen Dioksida)."
        )

    feature_cols = [co_col, no2_col]
    target_col = ""

    # 2. Deteksi Kolom Target O3 (Ozon) jika diperlukan
    if require_target:
        target_col = match_column(cols, O3_PATTERNS)

        if not target_col:
            raise ValueError(
                f"Kolom target O3 / Ozon tidak ditemukan dalam berkas. "
                f"Kolom yang terdeteksi: {cols}. "
                f"Pastikan berkas data latih menyertakan kolom O3 / Ozon."
            )

        clean_df = df[feature_cols + [target_col]].dropna()
        if len(clean_df) < 5:
            raise ValueError(f"Jumlah baris data valid terlalu sedikit ({len(clean_df)} baris). Minimal 5 baris data.")

        X = clean_df[feature_cols].values.astype(float)
        y = clean_df[target_col].values.astype(float)
        return X, y, feature_cols, target_col
    else:
        clean_df = df[feature_cols].dropna()
        if len(clean_df) < 1:
            raise ValueError("Tidak ada data baris valid pada berkas yang diunggah.")
        X = clean_df[feature_cols].values.astype(float)
        return X, None, feature_cols, ""


async def parse_batch_estimate_file(
    file: UploadFile
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Khusus untuk inferensi estimasi batch:
    Membaca berkas satelit dan mengekstrak array CO, NO2, target O3 (opsional), dan label baris/waktu.

    Returns:
    --------
    co_vals : np.ndarray
    no2_vals : np.ndarray
    actual_vals : Optional[np.ndarray] (None jika kolom target O3 tidak ada di berkas)
    labels : List[str] daftar penanda baris (misal format YYYY-MM-DD atau nomor urut)
    """
    df = await read_uploaded_dataframe(file)
    cols = list(df.columns)

    co_col = match_column(cols, CO_PATTERNS)
    no2_col = match_column(cols, NO2_PATTERNS)

    if not co_col or not no2_col:
        raise ValueError(
            f"Kolom CO atau NO2 tidak ditemukan dalam berkas. Kolom terdeteksi: {cols}"
        )

    target_col = match_column(cols, O3_PATTERNS)
    label_col = match_column(cols, LABEL_PATTERNS)

    clean_df = df.dropna(subset=[co_col, no2_col]).copy()
    if len(clean_df) < 1:
        raise ValueError("Tidak ada baris data valid dengan nilai CO dan NO2.")

    co_vals = clean_df[co_col].to_numpy(dtype=float)
    no2_vals = clean_df[no2_col].to_numpy(dtype=float)
    actual_vals = clean_df[target_col].to_numpy(dtype=float) if target_col else None

    labels: List[str] = []
    for idx in range(len(clean_df)):
        if label_col:
            val_raw = clean_df[label_col].iloc[idx]
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

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
    # Kasus 1: Jika input berupa path berkas lokal di harddisk (str atau Path)
    if isinstance(file, (str, Path)):
        # Ubah menjadi objek Path absolut lengkap
        path_obj = Path(file).resolve()
        # Validasi keberadaan fisik file di harddisk
        if not path_obj.exists():
            raise FileNotFoundError(f"Berkas dataset tidak ditemukan: {path_obj}")
        # Ambil ekstensi berkas (misal .xlsx -> xlsx) dalam huruf kecil
        ext = path_obj.suffix.lower().lstrip(".")
        # Target pembacaan Pandas adalah path file itu sendiri
        target = path_obj
    else:
        # Kasus 2: Jika input berupa objek UploadFile dari request API FastAPI
        filename = getattr(file, "filename", "") or ""
        # Ambil ekstensi dari nama file yang diunggah
        ext = filename.lower().split(".")[-1]
        # file.file adalah stream biner di memori (SpooledTemporaryFile) dari Starlette/FastAPI
        target = getattr(file, "file", file)
        # Kembalikan kursor baca ke byte 0 agar data tidak terbaca kosong jika sudah pernah diakses
        if hasattr(target, "seek"):
            target.seek(0)

    try:
        # Baca berkas Excel (.xlsx / .xls) menggunakan engine Excel Pandas
        if ext in ["xlsx", "xls"]:
            df = pd.read_excel(target)
        # Baca berkas CSV menggunakan engine CSV Pandas
        elif ext == "csv":
            df = pd.read_csv(target)
        # Tolak format berkas selain spreadsheet/CSV
        else:
            raise ValueError(f"Format berkas '.{ext}' tidak didukung. Harap gunakan .csv, .xls, atau .xlsx.")
    except Exception as e:
        # Tangkap kegagalan parsing / berkas korup dan lempar sebagai ValueError yang informatif
        raise ValueError(f"Gagal membaca berkas: {str(e)}")

    # Bersihkan spasi kosong di awal/akhir nama kolom (misal ' CO ' -> 'CO') agar regex cocok
    df.columns = [str(col).strip() for col in df.columns]
    # Kembalikan tabel DataFrame Pandas yang sudah bersih
    return df



def interpolate_natural_cubic_spline(series: pd.Series) -> pd.Series:
    """
    Melakukan interpolasi Natural Cubic Spline (bc_type='natural') pada kolom target O3
    jika terdapat nilai missing (NaN). Nilai hasil interpolasi dibatasi >= 0 karena
    konsentrasi gas di atmosfer tidak boleh bernilai negatif.
    """
    # Buat boolean mask untuk mendeteksi baris-baris data yang bernilai hilang (NaN)
    missing_mask = series.isna()

    # Early exit: Jika data sudah lengkap tanpa ada nilai kosong sama sekali, langsung kembalikan series aslinya
    if not missing_mask.any():
        return series

    # Hitung jumlah titik data riil yang valid (non-NaN)
    valid_mask = ~missing_mask
    n_valid = int(valid_mask.sum())

    # Cubic Spline membutuhkan minimal 2 titik data referensi untuk membentuk kurva polinomial derajat 3
    if n_valid < 2:
        # Jika data valid < 2 titik, interpolasi tidak memungkinkan dilakukan secara matematis;
        # kembalikan series apa adanya agar baris yang valid tetap dapat diproses tanpa crash
        return series

    # Siapkan koordinat titik acuan interpolasi:
    # x_valid: indeks baris kronologis (waktu) dari data yang valid
    # y_valid: nilai riil konsentrasi target O3 pada baris valid tersebut
    x_valid = series.index[valid_mask].to_numpy(dtype=float)
    y_valid = series.loc[valid_mask].to_numpy(dtype=float)

    # Bentuk fungsi kurva Natural Cubic Spline f(x):
    # - bc_type="natural" mensyaratkan turunan kedua di kedua titik ujung adalah 0 (f''(x_0) = f''(x_n) = 0),
    #   mencegah kurva berosilasi liar di ujung data.
    # - extrapolate=True memungkinkan estimasi jika posisi NaN berada di baris awal sekali atau akhir sekali.
    cs = CubicSpline(x_valid, y_valid, bc_type="natural", extrapolate=True)

    # Ambil posisi baris-baris yang hilang (x_missing) dan hitung nilai estimasi berdasarkan kurva spline cs(x)
    x_missing = series.index[missing_mask].to_numpy(dtype=float)
    interpolated_values = cs(x_missing)

    # Buat salinan data agar tidak mengubah series asli secara langsung (immutability)
    series_clean = series.copy()

    # Batasan fisik atmosfer: kadar konsentrasi gas di udara tidak mungkin bernilai negatif.
    # Jika kurva spline melengkung ke bawah menembus nol, potong (clip) nilai minimalnya menjadi 0.0
    series_clean.loc[missing_mask] = np.clip(interpolated_values, a_min=0.0, a_max=None)

    # Kembalikan data deret waktu target O3 yang telah terisi penuh dan bersih
    return series_clean


async def parse_and_validate_file(
    file: Union[UploadFile, str, Path],
    require_target: bool = True,
    return_indices: bool = False,
    min_samples: int = 5,
    return_interpolation_info: bool = False,
) -> Union[
    Tuple[np.ndarray, Optional[np.ndarray], List[str], str],
    Tuple[np.ndarray, Optional[np.ndarray], List[str], str, np.ndarray],
    Tuple[np.ndarray, Optional[np.ndarray], List[str], str, int],
    Tuple[np.ndarray, Optional[np.ndarray], List[str], str, np.ndarray, int],
]:
    """
    Membaca dan memvalidasi berkas data latih / uji untuk model regresi.
    - Mencari otomatis kolom CO satelit, NO2 satelit, dan O3 SPKU.
    - Hanya mengambil ketiga variabel tersebut (variabel lain diabaikan).
    - Jika terdapat missing value pada O3 SPKU, dilakukan interpolasi Natural Cubic Spline.
    - Hanya data yang memiliki O3 SPKU, NO2 satelit, dan CO satelit lengkap yang diambil.
    - Dapat mengembalikan original_indices untuk melacak nomor baris data sebelum prapemrosesan.
    - Dapat mengembalikan interpolated_count (jumlah baris O3 hasil interpolasi spline).

    Returns:
    --------
    X : np.ndarray (n_samples, 2)
    y : Optional[np.ndarray] (n_samples,) jika require_target=True, None jika False
    feature_cols : List[str] nama kolom fitur terdeteksi [co_col, no2_col]
    target_col : str nama kolom target terdeteksi
    original_indices : np.ndarray (n_samples,) jika return_indices=True (0-indexed baris asli)
    interpolated_count : int jika return_interpolation_info=True
    """
    # Baca berkas secara asinkron di thread terpisah agar operasi I/O file tidak memblokir event loop FastAPI
    df = await asyncio.to_thread(read_uploaded_dataframe, file)

    # Catat indeks baris asli dari file mentah sebelum pengurutan atau dropna (0-indexed) untuk pelacakan baris data
    df["_raw_row_idx"] = np.arange(len(df), dtype=int)
    cols = [c for c in df.columns if c != "_raw_row_idx"]

    # 1. Deteksi Kolom Fitur CO & NO2 (Satelit Sentinel-5P) menggunakan pola regex
    co_col = match_column(cols, CO_PATTERNS)
    no2_col = match_column(cols, NO2_PATTERNS)

    # Validasi keberadaan kedua kolom fitur utama; jika salah satu tidak ada, tolak dengan pesan error informatif
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

    # 2. Urutkan data secara kronologis jika berkas menyertakan kolom penanda tanggal / waktu
    label_col = match_column(cols, LABEL_PATTERNS)
    if label_col:
        try:
            # Parse kolom tanggal; string yang tidak valid diubah menjadi NaT (Not a Time)
            parsed_dates = pd.to_datetime(df[label_col], errors="coerce")

            # Ambang batas > 50%: memastikan kolom tersebut benar-benar deret tanggal valid (mencegah false positive)
            if parsed_dates.notna().sum() > 0.5 * len(df):
                # Urutkan baris dari tanggal terlama ke terbaru (ascending), baris NaT ditempatkan di paling akhir
                df = (
                    df.assign(_temp_sort_date=parsed_dates)
                    .sort_values(by="_temp_sort_date", na_position="last")
                    .drop(columns=["_temp_sort_date"])
                )
        except Exception:
            # Fail-safe: jika parsing tanggal gagal, pertahankan urutan asli berkas tanpa menghentikan proses
            pass

    # Reset indeks baris agar berurutan kembali 0..n-1 setelah diurutkan
    # Wajib dilakukan agar CubicSpline pada langkah berikutnya tidak error (CubicSpline mensyaratkan sumbu x strictly increasing)
    df = df.reset_index(drop=True)

    # 3. Deteksi & Proses Kolom Target O3 SPKU jika diperlukan (untuk alur Training & Testing)
    if require_target:
        # Cari kolom target O3 SPKU berdasarkan pola regex
        target_col = match_column(cols, O3_PATTERNS)

        # Validasi keberadaan kolom target; wajib ada untuk proses latih/uji
        if not target_col:
            raise ValueError(
                f"Kolom target O3 SPKU / Ozon tidak ditemukan dalam berkas. "
                f"Kolom yang terdeteksi: {cols}. "
                f"Pastikan berkas data latih atau uji menyertakan kolom O3 SPKU."
            )

        # Buat DataFrame kerja dan konversi nilai kolom fitur & target menjadi numerik murni (string diubah jadi NaN)
        work_df = pd.DataFrame(index=df.index)
        work_df[co_col] = pd.to_numeric(df[co_col], errors="coerce")
        work_df[no2_col] = pd.to_numeric(df[no2_col], errors="coerce")
        work_df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        work_df["_raw_row_idx"] = df["_raw_row_idx"]

        # Tandai baris yang awalnya missing value pada kolom target O3 sebelum interpolasi
        was_o3_missing = work_df[target_col].isna()

        # Interpolasi Natural Cubic Spline untuk mengisi nilai hilang (missing value/NaN) pada target O3 secara mulus
        work_df[target_col] = interpolate_natural_cubic_spline(work_df[target_col])
        work_df["_was_o3_interpolated"] = was_o3_missing

        # Hapus baris yang masih mengandung nilai kosong (NaN/null) pada fitur atau target
        clean_df = work_df[[co_col, no2_col, target_col, "_raw_row_idx", "_was_o3_interpolated"]].dropna()

        # Pastikan seluruh nilai numerik berhingga (bukan Inf atau -Inf)
        clean_df = clean_df[
            np.isfinite(clean_df[co_col])
            & np.isfinite(clean_df[no2_col])
            & np.isfinite(clean_df[target_col])
        ]

        # Validasi kecukupan jumlah sampel data bersih terhadap batas minimum yang diminta
        if len(clean_df) < min_samples:
            raise ValueError(
                f"Jumlah baris data valid terlalu sedikit ({len(clean_df)} baris). "
                f"Minimal {min_samples} baris data yang memiliki O3 SPKU, NO2 satelit, dan CO satelit lengkap."
            )

        # Hitung jumlah baris data bersih yang nilai target O3-nya merupakan hasil interpolasi
        interpolated_count = int(clean_df["_was_o3_interpolated"].sum())

        # Konversi data bersih ke NumPy ndarray untuk komputasi matriks regresi
        X = clean_df[feature_cols].to_numpy(dtype=float)
        y = clean_df[target_col].to_numpy(dtype=float)
        orig_indices = clean_df["_raw_row_idx"].to_numpy(dtype=int)

        # Kembalikan tuple data sesuai parameter opsi yang diminta caller
        if return_indices and return_interpolation_info:
            return X, y, feature_cols, target_col, orig_indices, interpolated_count
        elif return_indices:
            return X, y, feature_cols, target_col, orig_indices
        elif return_interpolation_info:
            return X, y, feature_cols, target_col, interpolated_count
        return X, y, feature_cols, target_col

    else:
        # Alur khusus tanpa target (misal untuk inferensi estimasi masa depan / unlabelled data)
        work_df = pd.DataFrame(index=df.index)
        work_df[co_col] = pd.to_numeric(df[co_col], errors="coerce")
        work_df[no2_col] = pd.to_numeric(df[no2_col], errors="coerce")
        work_df["_raw_row_idx"] = df["_raw_row_idx"]

        # Hapus baris yang memiliki nilai kosong pada fitur input
        clean_df = work_df[[co_col, no2_col, "_raw_row_idx"]].dropna()
        clean_df = clean_df[
            np.isfinite(clean_df[co_col]) & np.isfinite(clean_df[no2_col])
        ]

        # Validasi minimal ada 1 baris data fitur yang valid untuk diestimasi
        if len(clean_df) < 1:
            raise ValueError(
                "Tidak ada data baris valid yang memiliki nilai CO satelit dan NO2 satelit lengkap."
            )

        # Konversi fitur ke NumPy ndarray
        X = clean_df[feature_cols].to_numpy(dtype=float)
        orig_indices = clean_df["_raw_row_idx"].to_numpy(dtype=int)

        # Kembalikan fitur X dengan target berupa None
        if return_indices and return_interpolation_info:
            return X, None, feature_cols, "", orig_indices, 0
        elif return_indices:
            return X, None, feature_cols, "", orig_indices
        elif return_interpolation_info:
            return X, None, feature_cols, "", 0
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


def parse_raw_seeds(seeds: Optional[Union[str, List[Union[int, str]]]]) -> Optional[List[int]]:
    """Mengekstrak list integer random seeds dari string (spasi/koma/JSON) atau list."""
    if not seeds:
        return None

    # 1. Jika berupa string teks, bersihkan format kurung/koma lalu pecah menjadi list
    if isinstance(seeds, str):
        if not seeds.strip():
            return None
        seeds = seeds.replace("[", "").replace("]", "").replace(",", " ").split()

    # 2. Parse elemen list/tuple secara aman dan abaikan nilai non-numerik
    if isinstance(seeds, (list, tuple)):
        cleaned = []
        for s in seeds:
            try:
                cleaned.append(int(s))
            except (ValueError, TypeError):
                continue
        return cleaned or None

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


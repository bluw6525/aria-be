from pathlib import Path
import pandas as pd
from scipy.interpolate import CubicSpline

# Path file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "Jakarta_Utara_New.xlsx"
OUTPUT_FILE = PROJECT_ROOT / "Jakarta_Utara_Interpolated.xlsx"

# 1. Baca data dari file Excel
df = pd.read_excel(INPUT_FILE)
columns = ["O3"]

# 2. Interpolasi Natural Cubic Spline untuk setiap parameter
for col in columns:
    valid = df[col].notna()
    missing = df[col].isna()

    # Fit natural cubic spline (bc_type='natural')
    cs = CubicSpline(df.index[valid], df.loc[valid, col], bc_type="natural")

    # Isi nilai yang kosong
    df.loc[missing, col] = cs(df.index[missing])

# 3. Simpan kembali ke file Excel
df.to_excel(OUTPUT_FILE, index=False)
print(f"Interpolasi selesai. File disimpan di: {OUTPUT_FILE}")

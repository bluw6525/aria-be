"""
CLI Command untuk membuat Plot Perbandingan Garis Regresi O3 terhadap CO dan NO2 Satelit.
Model dilatih (fit) HANYA pada Data Latih sesuai rasio split dan seed dari commands/seed.py.
Garis regresi dihitung pada rentang seluruh data dengan variabel lain dikunci pada median data latih.
Titik data latih dan data uji ditampilkan dengan warna berbeda (biru & oranye) untuk inspeksi visual.
"""

import argparse
import asyncio
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import core terlebih dahulu untuk mencegah circular import
from app.modules.regression.core.parser import parse_and_validate_file
from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.algorithms.m_estimator import MEstimator
from app.modules.regression.algorithms.s_estimator import SEstimator
from app.modules.regression.algorithms.mm_estimator import MMEstimator

import commands.seed as seed_module

# Mapping judul wilayah berdasarkan nama dataset
DATASET_REGION_TITLES = {
    "Pusat_New.xlsx": "Jakarta Pusat",
    "pusat_selatan.xlsx": "Jakarta Pusat dan Selatan",
    "Selatan_New2.xlsx": "Jakarta Selatan",
}


def get_region_title(file_path: str | Path) -> str:
    """Mengembalikan judul wilayah yang sesuai dari nama berkas dataset."""
    name = Path(file_path).name.lower()
    for pattern, title in DATASET_REGION_TITLES.items():
        if pattern.lower() == name or Path(pattern).stem.lower() == Path(name).stem:
            return title
    return Path(file_path).stem


def get_region_slug(file_path: str | Path) -> str:
    """Mengembalikan slug nama daerah dengan format lowercase dan underscore."""
    title = get_region_title(file_path).lower()
    return title.replace(" ", "_").replace("&", "dan")


def get_split_seed(file_path: str | Path, ratio: str = "80:20") -> int:
    """Mengambil seed resmi dari commands/seed.py berdasarkan wilayah dan rasio."""
    region = get_region_title(file_path).lower()
    ratio_code = ratio.replace(":", "_")

    if "pusat dan selatan" in region:
        prefix = "PUSAT_SELATAN_SEED"
    elif "pusat" in region:
        prefix = "PUSAT_SEED"
    else:
        prefix = "SELATAN_SEED"

    var_name = f"{prefix}_{ratio_code}"
    return int(getattr(seed_module, var_name, 9404))


RATIO_MAP = {
    "80:20": 0.2,
    "70:30": 0.3,
    "60:40": 0.4,
}

# 4 Metode Regresi (MM efficiency="low" -> k=1.547 tanpa menampilkan k di label)
ESTIMATORS = [
    ("OLS", "OLS", lambda: OLSEstimator(fit_intercept=True)),
    ("M", "M-Estimator", lambda: MEstimator(fit_intercept=True)),
    ("S", "S-Estimator", lambda: SEstimator(fit_intercept=True)),
    (
        "MM",
        "MM-Estimator",
        lambda: MMEstimator(fit_intercept=True, efficiency="low"),
    ),
]


def style_subplot(ax: plt.Axes, title: str):
    """Memberikan format seragam pada setiap subplot."""
    ax.set_title(title, fontsize=11, fontweight="bold", color="#0f172a", pad=8)
    ax.tick_params(axis="both", labelsize=9, colors="#334155")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


async def generate_garis_regresi_plots(
    file_path: str = "Selatan_New2.xlsx",
    ratio: str = "80:20",
    seed: int | None = None,
    output_dir: str | None = None,
):
    excel_path = Path(file_path)
    if not excel_path.is_absolute():
        excel_path = PROJECT_ROOT / excel_path

    if ratio not in RATIO_MAP:
        raise ValueError(f"Ratio '{ratio}' tidak valid. Pilihan: {list(RATIO_MAP.keys())}")

    test_size = RATIO_MAP[ratio]
    if seed is None:
        seed = get_split_seed(excel_path, ratio)

    region_title = get_region_title(excel_path)
    region_slug = get_region_slug(excel_path)

    train_pct = int(round((1.0 - test_size) * 100))
    test_pct = int(round(test_size * 100))
    ratio_code = f"{train_pct}{test_pct}"

    # 1. Baca dataset
    X, y, feature_cols, target_col, orig_indices = await parse_and_validate_file(
        excel_path, require_target=True, return_indices=True
    )
    n_samples = len(y)

    # 2. Pembagian data latih & uji
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = int(np.round(n_samples * test_size))
    n_train = n_samples - n_test
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Data fitur keseluruhan (untuk rentang grid garis)
    co_all = X[:, 0]
    no2_all = X[:, 1]

    # Data fitur data latih & data uji
    co_train = X_train[:, 0]
    no2_train = X_train[:, 1]
    co_test = X_test[:, 0]
    no2_test = X_test[:, 1]

    # Median dihitung HANYA dari data latih
    median_no2_train = float(np.median(no2_train))
    median_co_train = float(np.median(co_train))

    # Grid 200 titik dari min sampai max pada SELURUH data
    co_grid = np.linspace(float(co_all.min()), float(co_all.max()), 200)
    no2_grid = np.linspace(float(no2_all.min()), float(no2_all.max()), 200)

    # Input matriks untuk predict garis regresi
    X_co_line = np.column_stack([co_grid, np.full_like(co_grid, median_no2_train)])
    X_no2_line = np.column_stack([np.full_like(no2_grid, median_co_train), no2_grid])

    # Siapkan direktori output
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = PROJECT_ROOT / "commands" / "summary" / "garis_regresi"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"MEMBUAT PLOT GARIS REGRESI | Dataset: {excel_path.name} ({region_title}) | Rasio: {ratio}")
    print(f"Sampel: Latih {n_train} | Uji {n_test} | Seed: {seed}")
    print("=" * 72)

    # 3. Buat Grid Gambar: 4 Baris x 2 Kolom dengan sharex per kolom
    n_methods = len(ESTIMATORS)
    fig, axes = plt.subplots(
        n_methods,
        2,
        figsize=(13, 4.2 * n_methods),
        dpi=220,
        sharex="col",
        squeeze=False,
    )

    for r_idx, (_, label, model_factory) in enumerate(ESTIMATORS):
        # Latih model HANYA pada data latih
        model = model_factory()
        model.fit(X_train, y_train)

        # Prediksi garis regresi
        y_line_co = model.predict(X_co_line)
        y_line_no2 = model.predict(X_no2_line)

        # ----------------- Kolom Kiri: vs CO Satelit -----------------
        ax_co = axes[r_idx][0]
        # Titik data latih (biru)
        ax_co.scatter(
            co_train,
            y_train,
            color="#1f77b4",
            s=32,
            alpha=0.82,
            label=f"Data Latih ({n_train} sampel)",
            zorder=3,
        )
        # Titik data uji (oranye)
        ax_co.scatter(
            co_test,
            y_test,
            color="#f59e0b",
            s=32,
            alpha=0.85,
            label=f"Data Uji ({n_test} sampel)",
            zorder=4,
        )
        # Garis regresi (merah)
        ax_co.plot(
            co_grid,
            y_line_co,
            color="red",
            linewidth=2.2,
            label="Garis Regresi (Model Data Latih)",
            zorder=5,
        )
        style_subplot(ax_co, f"{label} (vs CO Satelit)")
        ax_co.set_ylabel("Konsentrasi O3 SPKU (ISPU)", fontsize=9.5, fontweight="medium", color="#0f172a")

        # ----------------- Kolom Kanan: vs NO2 Satelit -----------------
        ax_no2 = axes[r_idx][1]
        # Titik data latih (biru)
        ax_no2.scatter(
            no2_train,
            y_train,
            color="#1f77b4",
            s=32,
            alpha=0.82,
            label=f"Data Latih ({n_train} sampel)",
            zorder=3,
        )
        # Titik data uji (oranye)
        ax_no2.scatter(
            no2_test,
            y_test,
            color="#f59e0b",
            s=32,
            alpha=0.85,
            label=f"Data Uji ({n_test} sampel)",
            zorder=4,
        )
        # Garis regresi (merah)
        ax_no2.plot(
            no2_grid,
            y_line_no2,
            color="red",
            linewidth=2.2,
            label="Garis Regresi (Model Data Latih)",
            zorder=5,
        )
        style_subplot(ax_no2, f"{label} (vs NO2 Satelit)")
        ax_no2.set_ylabel("Konsentrasi O3 SPKU (ISPU)", fontsize=9.5, fontweight="medium", color="#0f172a")

        # Legenda hanya di panel pertama agar tidak menutupi titik pada panel lainnya
        if r_idx == 0:
            ax_co.legend(
                loc="upper right",
                frameon=True,
                facecolor="white",
                edgecolor="#cbd5e1",
                fontsize=8.5,
            )

    # Label sumbu X pada baris terbawah
    axes[-1][0].set_xlabel("Konsentrasi CO Satelit (mol/m²)", fontsize=10, fontweight="medium", color="#0f172a")
    axes[-1][1].set_xlabel("Konsentrasi NO2 Satelit (mol/m²)", fontsize=10, fontweight="medium", color="#0f172a")

    # Pastikan tick sumbu X tetap tampil di semua subplot agar mudah dibaca
    for r in range(n_methods):
        axes[r][0].tick_params(labelbottom=True)
        axes[r][1].tick_params(labelbottom=True)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.965])
    fig.suptitle(
        f"Perbandingan Garis Regresi O3 terhadap CO dan NO2 Satelit\n"
        f"{region_title} ({train_pct}:{test_pct})",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
        y=0.992,
    )

    out_file = out_dir / f"garis_regresi_4_metode_{region_slug}_{ratio_code}.png"
    fig.savefig(out_file, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"-> Tersimpan: {out_file}")
    print("=" * 72)
    return out_file


def parse_optional_seed(val: str | int | None) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = val.strip().lower()
    if s in ("none", "null", ""):
        return None
    return int(s)


def main():
    parser = argparse.ArgumentParser(
        description="Buat plot Perbandingan Garis Regresi O3 terhadap CO dan NO2 Satelit dari model yang dilatih HANYA pada data latih sesuai rasio split."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="Selatan_New2.xlsx",
        help="Path berkas dataset Excel/CSV (default: Selatan_New2.xlsx)",
    )
    parser.add_argument(
        "--ratio",
        "-r",
        type=str,
        default="80:20",
        choices=["80:20", "70:30", "60:40", "all"],
        help="Rasio data latih:uji (80:20, 70:30, 60:40, all). Default: 80:20",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=parse_optional_seed,
        default=None,
        help="Random state / seed split. Jika None, otomatis mengambil dari commands/seed.py",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Folder output penyimpanan gambar (default: commands/summary/garis_regresi)",
    )

    args = parser.parse_args()

    if args.ratio == "all":
        ratios_to_run = ["80:20", "70:30", "60:40"]
    else:
        ratios_to_run = [args.ratio]

    for rat in ratios_to_run:
        s_val = args.seed if args.seed is not None else get_split_seed(args.file, rat)
        asyncio.run(
            generate_garis_regresi_plots(
                file_path=args.file,
                ratio=rat,
                seed=s_val,
                output_dir=args.output_dir,
            )
        )


if __name__ == "__main__":
    main()


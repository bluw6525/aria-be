"""
Command script untuk membersihkan file GeoJSON mentah batas wilayah DKI Jakarta
dan menyimpannya kembali ke file individual masing-masing: jakarta_*.geojson.

Menghilangkan artefak MultiPoint (titik tetangga Bekasi, Bogor, Depok, Tangerang),
LineString (garis batas sungai/jalan), serta membongkar GeometryCollection
menjadi Polygon/MultiPolygon murni.

Penggunaan:
    uv run python commands/clean_geojson.py
    uv run python commands/clean_geojson.py --backup
    uv run python commands/clean_geojson.py --compact
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REGION_DEFINITIONS: List[Tuple[str, str]] = [
    ("jakarta_barat", "Jakarta Barat"),
    ("jakarta_pusat", "Jakarta Pusat"),
    ("jakarta_selatan", "Jakarta Selatan"),
    ("jakarta_timur", "Jakarta Timur"),
    ("jakarta_utara", "Jakarta Utara"),
]


def extract_polygon_geometry(feature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Mengekstrak geometri poligon murni dari sebuah feature GeoJSON."""
    geom = feature.get("geometry", {})
    geom_type = geom.get("type")

    if geom_type in ("Polygon", "MultiPolygon"):
        return geom

    if geom_type == "GeometryCollection":
        polys = [
            g["coordinates"]
            for g in geom.get("geometries", [])
            if g.get("type") == "Polygon"
        ]
        if polys:
            if len(polys) == 1:
                return {"type": "Polygon", "coordinates": polys[0]}
            return {"type": "MultiPolygon", "coordinates": polys}

    return None


def count_points_and_bbox(geom: Dict[str, Any]) -> Tuple[int, Tuple[float, float, float, float]]:
    """Menghitung total titik koordinat dan bounding box (min_lng, min_lat, max_lng, max_lat)."""
    coords = geom.get("coordinates", [])
    pts: List[Tuple[float, float]] = []

    def _collect(c: Any) -> None:
        if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
            pts.append((float(c[0]), float(c[1])))
        elif isinstance(c, (list, tuple)):
            for sub in c:
                _collect(sub)

    _collect(coords)

    if not pts:
        return 0, (0.0, 0.0, 0.0, 0.0)

    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)
    return len(pts), (min_x, min_y, max_x, max_y)


def clean_individual_geojson(
    data_dir: Path,
    backup: bool = False,
    indent: Optional[int] = 2,
) -> None:
    print("=" * 75)
    print("ARIA - Pembersih GeoJSON Wilayah DKI Jakarta (Per File: jakarta_*.geojson)")
    print("=" * 75)
    print(f"Direktori target : {data_dir.resolve()}")
    print(f"Backup file asli : {'Ya (.bak)' if backup else 'Tidak'}\n")

    if not data_dir.exists():
        print(f"[ERROR] Direktori '{data_dir}' tidak ditemukan!", file=sys.stderr)
        sys.exit(1)

    print(f"{'File':<24} | {'Wilayah':<16} | {'Tipe Geometri':<14} | {'Titik':<6} | {'Status'}")
    print("-" * 75)

    success_count = 0

    for region_key, region_name in REGION_DEFINITIONS:
        target_file = data_dir / f"{region_key}.geojson"
        if not target_file.exists():
            print(f"{target_file.name:<24} | {region_name:<16} | {'-':<14} | {'0':<6} | [SKIP] Tidak ada")
            continue

        try:
            raw_data = json.loads(target_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"{target_file.name:<24} | {region_name:<16} | {'-':<14} | {'0':<6} | [ERROR] Gagal parse JSON: {e}")
            continue

        raw_features = (
            raw_data.get("features", [raw_data])
            if raw_data.get("type") == "FeatureCollection"
            else [raw_data]
        )

        extracted_geom: Optional[Dict[str, Any]] = None

        # 1. Cari feature yang NAMOBJ-nya mengandung nama wilayah Jakarta yang bersangkutan
        for feat in raw_features:
            props = feat.get("properties", {}) or {}
            namobj = str(props.get("NAMOBJ", "") or props.get("WADMKK", "") or props.get("name", "")).lower()

            if region_name.lower() in namobj:
                geom = extract_polygon_geometry(feat)
                if geom:
                    extracted_geom = geom
                    break

        # Fallback jika tidak ada properti NAMOBJ yang cocok
        if not extracted_geom:
            for feat in raw_features:
                geom = extract_polygon_geometry(feat)
                if geom:
                    extracted_geom = geom
                    break

        if not extracted_geom:
            print(f"{target_file.name:<24} | {region_name:<16} | {'-':<14} | {'0':<6} | [GAGAL] Poligon tidak ditemukan")
            continue

        pts_count, bbox = count_points_and_bbox(extracted_geom)

        # Buat FeatureCollection bersih yang hanya berisi 1 fitur poligon murni
        clean_feature = {
            "type": "Feature",
            "id": region_key,
            "properties": {
                "id": region_key,
                "name": region_name,
                "NAMOBJ": f"Kota Administrasi {region_name}",
                "points_count": pts_count,
                "bbox": list(bbox),
            },
            "geometry": extracted_geom,
        }

        clean_collection = {
            "type": "FeatureCollection",
            "name": region_key,
            "features": [clean_feature],
        }

        # Backup file lama jika diminta
        if backup:
            bak_path = data_dir / f"{region_key}.geojson.bak"
            shutil.copy2(target_file, bak_path)

        target_file.write_text(
            json.dumps(clean_collection, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )

        size_kb = target_file.stat().st_size / 1024
        print(
            f"{target_file.name:<24} | {region_name:<16} | {extracted_geom['type']:<14} | {pts_count:<6} | [OK] {size_kb:.1f} KB"
        )
        success_count += 1

    print("-" * 75)
    print(f"[SUKSES] Berhasil membersihkan {success_count} file GeoJSON di {data_dir.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Membersihkan data GeoJSON mentah DKI Jakarta dan menyimpannya ke file individual jakarta_*.geojson."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("app/data/geojson"),
        help="Direktori file GeoJSON (default: app/data/geojson)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Simpan salinan cadangan (.bak) dari file mentah sebelum ditimpa",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Simpan tanpa indentasi untuk memperkecil ukuran berkas",
    )

    args = parser.parse_args()
    indent = None if args.compact else 2
    clean_individual_geojson(args.dir, backup=args.backup, indent=indent)


if __name__ == "__main__":
    main()

"""
Command script untuk mengonversi data GeoJSON batas wilayah DKI Jakarta (WGS84)
menjadi berkas visual SVG path yang digunakan oleh Frontend (JakartaMap.tsx).

Output berkas:
    aria-fe/src/data/jakarta-map-data.json

Format keluaran:
{
  "viewBox": "0 0 800 800",
  "regions": [
    {
      "id": "jakarta_barat",
      "name": "Jakarta Barat",
      "path": "M ... L ... Z",
      "center": [x, y]
    }, ...
  ]
}

Penggunaan:
    uv run python commands/generate_map_svg.py
    uv run python commands/generate_map_svg.py --fe-path ../aria-fe/src/data/jakarta-map-data.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Konfigurasi 5 Wilayah DKI Jakarta
REGIONS = [
    ("jakarta_barat", "Jakarta Barat"),
    ("jakarta_pusat", "Jakarta Pusat"),
    ("jakarta_selatan", "Jakarta Selatan"),
    ("jakarta_timur", "Jakarta Timur"),
    ("jakarta_utara", "Jakarta Utara"),
]

# Sentroid visual yang telah dioptimalkan agar badge label tidak saling bertumpukan
OPTIMIZED_CENTERS = {
    "jakarta_barat": [172.1, 255.6],
    "jakarta_pusat": [394.8, 306.2],
    "jakarta_selatan": [338.1, 550.1],
    "jakarta_timur": [581.6, 549.0],
    "jakarta_utara": [400.5, 127.1],
}


def compute_global_bbox(features: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """Menghitung bounding box global (min_lng, min_lat, max_lng, max_lat) dari seluruh wilayah."""
    all_pts: List[Tuple[float, float]] = []

    def _collect(coords: Any) -> None:
        if isinstance(coords, (list, tuple)) and len(coords) >= 2 and isinstance(coords[0], (int, float)):
            all_pts.append((float(coords[0]), float(coords[1])))
        elif isinstance(coords, (list, tuple)):
            for sub in coords:
                _collect(sub)

    for feat in features:
        geom = feat.get("geometry", {})
        _collect(geom.get("coordinates", []))

    if not all_pts:
        # Default bounding box DKI Jakarta jika kosong
        return (106.685803, -6.372626, 106.973393, -6.074098)

    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)
    return (min_x, min_y, max_x, max_y)


def generate_map_svg_data(
    input_dir: Path,
    output_file: Path,
    width: int = 800,
    height: int = 800,
    padding: int = 40,
) -> None:
    print("=" * 75)
    print("ARIA - Generator Data Peta SVG Frontend (jakarta-map-data.json)")
    print("=" * 75)
    print(f"Direktori input GeoJSON : {input_dir.resolve()}")
    print(f"File target keluaran    : {output_file.resolve()}\n")

    if not input_dir.exists():
        print(f"[ERROR] Direktori GeoJSON '{input_dir}' tidak ditemukan!", file=sys.stderr)
        sys.exit(1)

    loaded_features: List[Tuple[str, str, Dict[str, Any]]] = []

    for region_key, region_name in REGIONS:
        file_path = input_dir / f"{region_key}.geojson"
        if not file_path.exists():
            print(f"[ERROR] File '{file_path.name}' tidak ditemukan di {input_dir}", file=sys.stderr)
            sys.exit(1)

        data = json.loads(file_path.read_text(encoding="utf-8"))
        features = data.get("features", [data]) if data.get("type") == "FeatureCollection" else [data]
        if not features:
            print(f"[ERROR] File '{file_path.name}' tidak memiliki fitur poligon!", file=sys.stderr)
            sys.exit(1)

        feat = features[0]
        loaded_features.append((region_key, region_name, feat))

    # 1. Hitung Bounding Box Global
    min_lng, min_lat, max_lng, max_lat = compute_global_bbox([f[2] for f in loaded_features])
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding

    print(f"BBox Longitude : [{min_lng:.6f}, {max_lng:.6f}]")
    print(f"BBox Latitude  : [{min_lat:.6f}, {max_lat:.6f}]")
    print(f"Kanvas SVG     : {width} x {height} px (Padding: {padding} px)\n")

    def project(lng: float, lat: float) -> Tuple[float, float]:
        """Proyeksi koordinat GPS (WGS84) ke sistem koordinat layar 2D SVG."""
        x = padding + ((lng - min_lng) / (max_lng - min_lng)) * plot_w
        # Sumbu Y di layar komputer terbalik (latitude makin ke selatan makin besar nilai Y pikselnya)
        y = padding + ((max_lat - lat) / (max_lat - min_lat)) * plot_h
        return round(x, 1), round(y, 1)

    def ring_to_path(ring: List[Any]) -> str:
        pts = [project(p[0], p[1]) for p in ring]
        if not pts:
            return ""
        d = f"M {pts[0][0]} {pts[0][1]}"
        for pt in pts[1:]:
            d += f" L {pt[0]} {pt[1]}"
        d += " Z"
        return d

    regions_data: List[Dict[str, Any]] = []

    print(f"{'Region ID':<18} | {'Nama Wilayah':<16} | {'Pusat Label (X, Y)':<20} | {'Panjang Path'}")
    print("-" * 75)

    for region_key, region_name, feat in loaded_features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        path_d = ""
        all_pts: List[Tuple[float, float]] = []

        if gtype == "Polygon":
            for ring in coords:
                path_d += (" " if path_d else "") + ring_to_path(ring)
                all_pts.extend([project(p[0], p[1]) for p in ring])
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    path_d += (" " if path_d else "") + ring_to_path(ring)
                    all_pts.extend([project(p[0], p[1]) for p in ring])

        # Hitung sentroid geometris bawaan
        if all_pts:
            calc_cx = round(sum(p[0] for p in all_pts) / len(all_pts), 1)
            calc_cy = round(sum(p[1] for p in all_pts) / len(all_pts), 1)
        else:
            calc_cx, calc_cy = 400.0, 400.0

        # Gunakan sentroid yang telah dioptimalkan agar teks tidak tabrakan
        center = OPTIMIZED_CENTERS.get(region_key, [calc_cx, calc_cy])

        regions_data.append({
            "id": region_key,
            "name": region_name,
            "path": path_d,
            "center": center,
        })

        print(
            f"{region_key:<18} | {region_name:<16} | {str(center):<20} | {len(path_d):<12} char"
        )

    output_payload = {
        "viewBox": f"0 0 {width} {height}",
        "regions": regions_data,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    size_kb = output_file.stat().st_size / 1024
    print("-" * 75)
    print(f"[SUKSES] Berhasil menghasilkan berkas peta SVG:")
    print(f"[OUTPUT] {output_file.resolve()} ({size_kb:.1f} KB)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mengonversi GeoJSON DKI Jakarta menjadi berkas visual SVG path frontend (jakarta-map-data.json)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("app/data/geojson"),
        help="Direktori berisi file jakarta_*.geojson bersih (default: app/data/geojson)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../aria-fe/src/data/jakarta-map-data.json"),
        help="Lokasi berkas keluaran frontend (default: ../aria-fe/src/data/jakarta-map-data.json)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Lebar kanvas SVG (default: 800)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Tinggi kanvas SVG (default: 800)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=40,
        help="Padding tepi kanvas SVG dalam piksel (default: 40)",
    )

    args = parser.parse_args()
    generate_map_svg_data(
        input_dir=args.input_dir,
        output_file=args.output,
        width=args.width,
        height=args.height,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()

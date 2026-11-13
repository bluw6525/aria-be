"""
Modul pembacaan dan resolusi batas wilayah administrasi DKI Jakarta.
Menyediakan daftar wilayah resmi dan pembacaan poligon GeoJSON lokal.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Daftar 5 wilayah administrasi DKI Jakarta yang didukung secara resmi
SUPPORTED_REGIONS: Dict[str, str] = {
    "jakarta_barat": "Jakarta Barat",
    "jakarta_pusat": "Jakarta Pusat",
    "jakarta_selatan": "Jakarta Selatan",
    "jakarta_timur": "Jakarta Timur",
    "jakarta_utara": "Jakarta Utara",
}

# Direktori GeoJSON lokal: app/data/geojson/
GEOJSON_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "geojson"


def load_region_geojson(region_key: str) -> Optional[Dict[str, Any]]:
    """
    Membaca berkas GeoJSON lokal untuk wilayah tertentu dan mengembalikan kamus geometri poligonnya.
    """
    if region_key not in SUPPORTED_REGIONS:
        return None

    file_path = GEOJSON_DIR / f"{region_key}.geojson"
    if not file_path.is_file():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [data]) if data.get("type") == "FeatureCollection" else [data]
        target_name = SUPPORTED_REGIONS.get(region_key, "").lower()

        for feat in features:
            name = str(feat.get("properties", {}).get("NAMOBJ", "")).lower()
            if target_name and target_name not in name:
                continue

            geom = feat.get("geometry", feat)
            if geom.get("type") in ("Polygon", "MultiPolygon"):
                return geom
            if geom.get("type") == "GeometryCollection":
                polys = [g["coordinates"] for g in geom.get("geometries", []) if g.get("type") == "Polygon"]
                if polys:
                    return {"type": "MultiPolygon", "coordinates": polys}

    except Exception as err:
        logger.warning("Gagal membaca berkas GeoJSON '%s': %s", file_path, err)

    return None

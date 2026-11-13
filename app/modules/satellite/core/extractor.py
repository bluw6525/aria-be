"""
Modul ekstraksi data citra satelit Sentinel-5P dari Google Earth Engine.
Mendukung penarikan data OFFL (Offline konsolidasi) dengan fallback otomatis ke NRTI (Near Real-Time),
serta pemetaan poligon geometri dan reduksi spasial multi-band harian untuk gas CO dan NO2.
"""

import logging
from typing import Dict, List, Any
import ee
from injector import inject, singleton

from app.core.config import settings
from app.core.dates import validate_date_range
from app.core.gee import GEEClient
from app.core.geometry import SUPPORTED_REGIONS, load_region_geojson

logger = logging.getLogger(__name__)

# Konfigurasi Dataset Sentinel-5P Level-3 (OFFL - Konsolidasi presisi tinggi, latency 2-5 hari)
DATASET_CO_OFFL = "COPERNICUS/S5P/OFFL/L3_CO"
DATASET_NO2_OFFL = "COPERNICUS/S5P/OFFL/L3_NO2"

# Konfigurasi Dataset Sentinel-5P Level-3 (NRTI - Near Real-Time, latency ~3 jam)
DATASET_CO_NRTI = "COPERNICUS/S5P/NRTI/L3_CO"
DATASET_NO2_NRTI = "COPERNICUS/S5P/NRTI/L3_NO2"

BAND_CO = "CO_column_number_density"
BAND_NO2 = "tropospheric_NO2_column_number_density"


@singleton
class Sentinel5PExtractor:
    """
    Service singleton untuk ekstraksi data harian satelit Sentinel-5P dari GEE.
    Menyimpan cache poligon geometri di memori RAM dan mengelola koneksi GEE secara mandiri.
    Mendukung fallback otomatis dari OFFL ke NRTI untuk hari-hari terkini.
    """

    @inject
    def __init__(self, gee_client: GEEClient):
        self._gee_client = gee_client
        self._geometry_cache: Dict[str, ee.Geometry] = {}

    def get_geometry(self, region_key: str) -> ee.Geometry:
        """
        Mengambil objek ee.Geometry untuk wilayah tertentu.
        Jika sudah di memori cache, langsung dikembalikan dalam 0 ms.
        """
        if region_key not in SUPPORTED_REGIONS:
            valid_keys = ", ".join(SUPPORTED_REGIONS.keys())
            raise ValueError(f"Wilayah '{region_key}' tidak valid. Pilihan yang didukung: {valid_keys}")

        # 1. Cek cache memori RAM
        if region_key in self._geometry_cache:
            return self._geometry_cache[region_key]

        # 2. Baca GeoJSON lokal
        raw = load_region_geojson(region_key)
        if not raw:
            raise ValueError(f"Batas wilayah '{region_key}' tidak ditemukan di berkas GeoJSON lokal.")

        # 3. Pastikan GEE aktif sebelum instansiasi objek ee.Geometry
        self._gee_client.ensure_connected()
        geom = ee.Geometry(raw)
        self._geometry_cache[region_key] = geom
        return geom

    def extract(
        self,
        region_key: str,
        start_date_str: str,
        end_date_str: str,
    ) -> List[Dict[str, Any]]:
        """
        Mengekstrak data harian polutan CO dan NO2 dari Sentinel-5P L3 GEE untuk rentang tanggal tertentu.
        Memprioritaskan dataset OFFL, dan otomatis fallback ke NRTI jika produk OFFL belum rilis.

        Returns
        -------
        List[Dict[str, Any]]
            Daftar record harian: [{"region": str, "date": str, "co": float, "no2": float, "product_type": str}, ...]
        """
        validate_date_range(start_date_str, end_date_str)
        self._gee_client.ensure_connected()

        geometry = self.get_geometry(region_key)

        ee_start_date = ee.Date(start_date_str)
        # Tanggal akhir ditambah 1 hari karena filterDate bersifat eksklusif pada batas akhir
        ee_end_date = ee.Date(end_date_str).advance(1, "day")

        n_days = ee_end_date.difference(ee_start_date, "day")
        day_list = ee.List.sequence(0, n_days.subtract(1))
        scale = settings.SATELLITE_SCALE
        defaults = ee.Dictionary({"co": None, "no2": None})

        def process_day(day_offset: Any) -> ee.Feature:
            cur_date = ee_start_date.advance(day_offset, "day")
            next_date = cur_date.advance(1, "day")

            # 1. Komposit CO: prioritaskan OFFL, fallback ke NRTI
            co_offl = ee.ImageCollection(DATASET_CO_OFFL).filterDate(cur_date, next_date).select([BAND_CO], ["co"])
            co_nrti = ee.ImageCollection(DATASET_CO_NRTI).filterDate(cur_date, next_date).select([BAND_CO], ["co"])
            use_co_offl = co_offl.size().gt(0)
            img_co = ee.Image(ee.Algorithms.If(use_co_offl, co_offl.mean(), co_nrti.mean()))

            # 2. Komposit NO2: prioritaskan OFFL, fallback ke NRTI
            no2_offl = ee.ImageCollection(DATASET_NO2_OFFL).filterDate(cur_date, next_date).select([BAND_NO2], ["no2"])
            no2_nrti = ee.ImageCollection(DATASET_NO2_NRTI).filterDate(cur_date, next_date).select([BAND_NO2], ["no2"])
            use_no2_offl = no2_offl.size().gt(0)
            img_no2 = ee.Image(ee.Algorithms.If(use_no2_offl, no2_offl.mean(), no2_nrti.mean()))

            # 3. Tentukan tipe produk sumber ('OFFL' jika kedua gas ada di OFFL, selain itu 'NRTI')
            has_offl = use_co_offl.And(use_no2_offl)
            source_type = ee.String(ee.Algorithms.If(has_offl, "OFFL", "NRTI"))

            # 4. Gabungkan band CO dan NO2 menjadi 1 citra (efisiensi reduceRegion 2x lebih cepat)
            combined_img = img_co.addBands(img_no2)

            # 5. Hitung rata-rata spasial di atas poligon wilayah
            reduced = combined_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e9,
            )

            # Gabungkan dengan defaults agar hari yang tertutup awan tidak melempar error
            safe_props = defaults.combine(reduced, True).combine(
                ee.Dictionary({"tanggal": cur_date.format("YYYY-MM-dd"), "source_type": source_type}), True
            )

            return ee.Feature(None, safe_props)

        # Petakan proses harian untuk seluruh rentang hari
        daily_collection = ee.FeatureCollection(day_list.map(process_day))

        # Tarik hasil ke backend (termasuk hari awan tebal yang menghasilkan co/no2 None)
        features = daily_collection.getInfo().get("features", [])
        results: List[Dict[str, Any]] = []

        for feat in features:
            props = feat.get("properties", {})
            tanggal = props.get("tanggal")
            if not tanggal:
                continue

            raw_co = props.get("co")
            raw_no2 = props.get("no2")
            co_val = float(raw_co) if raw_co is not None else None
            no2_val = float(raw_no2) if raw_no2 is not None else None
            source_type = props.get("source_type", "OFFL")

            # Simpan product_type sumbernya ('OFFL' / 'NRTI') jika minimal salah satu prekursor (CO atau NO2) ada.
            # Jika kedua prekursor kosong (tutupan awan / tidak lolos QA), product_type bernilai None (NULL).
            has_data = (co_val is not None) or (no2_val is not None)
            prod_type = source_type if has_data else None

            results.append({
                "region": region_key,
                "date": str(tanggal),
                "co": co_val,
                "no2": no2_val,
                "product_type": prod_type,
            })

        return results

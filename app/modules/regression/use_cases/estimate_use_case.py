"""
Use Case untuk Estimasi Konsentrasi O3 dan Klasifikasi Kategori ISPU (Fitur /estimate).
Mendukung estimasi otomatis berbasis data satelit Sentinel-5P untuk 5 wilayah DKI Jakarta
serta peramalan time-series H+1 dan H+2 (Moving Average).
"""

from typing import Tuple, List, Optional, Union
from datetime import datetime, date
from injector import inject
import numpy as np

from app.core.dates import add_days_to_date, validate_date_range
from app.core.geometry import SUPPORTED_REGIONS
from app.modules.regression.core.models import PublicModel
from app.modules.regression.core.ispu import get_ispu_category
from app.modules.regression.repositories import PublicModelRepository
from app.modules.regression.schemas import (
    BatchEstimatePoint,
    BatchEstimateSummary,
    BatchEstimateResponse,
    SatelliteEstimateRequest,
    MapOverviewItem,
    MapOverviewResponse,
    ForecastItem,
)
from app.modules.satellite.repositories.satellite_repository import SatelliteRepository
from app.modules.regression.algorithms.moving_average import forecast_moving_average


def _calc_ispu(
    intercept: float,
    coef_co: float,
    coef_no2: float,
    co: float,
    no2: float,
) -> Tuple[float, str]:
    """
    Menghitung estimasi nilai ISPU non-negatif dan menentukan kategori mutunya.
    """
    est_val = intercept + (coef_co * co) + (coef_no2 * no2)
    ispu_round = round(max(0.0, float(est_val)), 1)
    category = get_ispu_category(ispu_round)
    return ispu_round, category


class EstimateUseCase:
    """
    Use case khusus untuk melakukan inferensi estimasi ISPU berbasis data satelit Sentinel-5P.
    """

    @inject
    def __init__(
        self,
        repository: PublicModelRepository,
        satellite_repository: SatelliteRepository,
    ):
        self.repository = repository
        self.satellite_repository = satellite_repository

    async def estimate_from_satellite(
        self,
        payload: SatelliteEstimateRequest
    ) -> BatchEstimateResponse:
        """
        Melakukan estimasi ISPU berdasarkan wilayah dan rentang tanggal satelit Sentinel-5P,
        serta memproyeksikan H+1 dan H+2 menggunakan Moving Average k=3.
        """
        model = await self.repository.get_active()
        if not model:
            models = await self.repository.get_all(limit=1)
            model = models[0] if models else None

        if not model:
            raise ValueError("Belum ada model tersimpan di sistem. Silakan latih model terlebih dahulu.")

        validate_date_range(payload.start_date, payload.end_date)

        # Ambil data satelit langsung dari repository (hanya data valid dengan kedua polutan)
        valid_items = await self.satellite_repository.get_by_date_range(
            region=payload.region,
            start_date=payload.start_date,
            end_date=payload.end_date,
            valid_only=True,
        )

        if not valid_items:
            region_label = SUPPORTED_REGIONS.get(payload.region, payload.region)
            raise ValueError(
                f"Tidak ditemukan data pengukuran satelit Sentinel-5P yang valid untuk {region_label} "
                f"pada rentang tanggal {payload.start_date} s/d {payload.end_date}. "
                "Hal ini biasanya terjadi karena tutupan awan mendung tebal merata atau data satelit tidak lengkap."
            )

        chart_data: List[BatchEstimatePoint] = []
        for idx, item in enumerate(valid_items):
            ispu_round, category = _calc_ispu(
                model.intercept, model.coef_co, model.coef_no2, item.co, item.no2
            )
            chart_data.append(
                BatchEstimatePoint(
                    index=idx + 1,
                    label=item.date,
                    co=round(float(item.co), 6),
                    no2=round(float(item.no2), 6),
                    actual_value=None,
                    estimated_value=ispu_round,
                    ispu_category=category,
                    is_forecast=False,
                    forecast_value=None,
                )
            )

        # Hitung ringkasan statistik historis
        all_estimated = [pt.estimated_value for pt in chart_data if pt.estimated_value is not None]
        avg_est = round(float(np.mean(all_estimated)), 1) if all_estimated else 0.0
        min_est = round(float(np.min(all_estimated)), 1) if all_estimated else 0.0
        max_est = round(float(np.max(all_estimated)), 1) if all_estimated else 0.0

        # Proyeksi H+1 dan H+2 hanya dihitung dan ditampilkan jika tanggal akhir adalah hari ini
        today_str = datetime.now().strftime("%Y-%m-%d")
        is_today_query = (payload.end_date == today_str)

        # Selalu hitung proyeksi 2 hari kedepan untuk kartu drawer berdasarkan 3 observasi valid terakhir hingga hari ini
        forecast_days: List[ForecastItem] = []
        forecast_records = await self.satellite_repository.get_recent_measurements(
            region=payload.region,
            up_to_date=today_str,
            limit=3
        )
        if forecast_records:
            co_series = [float(r.co) for r in forecast_records]
            no2_series = [float(r.no2) for r in forecast_records]
            co_preds = forecast_moving_average(co_series, k=3, steps=2)
            no2_preds = forecast_moving_average(no2_series, k=3, steps=2)

            if len(co_preds) >= 2 and len(no2_preds) >= 2:
                co_h1, co_h2 = co_preds[0], co_preds[1]
                no2_h1, no2_h2 = no2_preds[0], no2_preds[1]

                last_record = forecast_records[-1]
                last_ispu, _ = _calc_ispu(
                    model.intercept, model.coef_co, model.coef_no2, last_record.co, last_record.no2
                )

                ispu_h1, cat_h1 = _calc_ispu(
                    model.intercept, model.coef_co, model.coef_no2, co_h1, no2_h1
                )
                ispu_h2, cat_h2 = _calc_ispu(
                    model.intercept, model.coef_co, model.coef_no2, co_h2, no2_h2
                )

                date_h1 = add_days_to_date(today_str, 1)
                date_h2 = add_days_to_date(today_str, 2)

                forecast_days = [
                    ForecastItem(
                        day_offset=1,
                        date=date_h1,
                        label="Besok (H+1)",
                        co=round(co_h1, 6),
                        no2=round(no2_h1, 6),
                        estimated_value=ispu_h1,
                        ispu_category=cat_h1,
                        delta=round(ispu_h1 - last_ispu, 1),
                    ),
                    ForecastItem(
                        day_offset=2,
                        date=date_h2,
                        label="Lusa (H+2)",
                        co=round(co_h2, 6),
                        no2=round(no2_h2, 6),
                        estimated_value=ispu_h2,
                        ispu_category=cat_h2,
                        delta=round(ispu_h2 - last_ispu, 1),
                    ),
                ]

                # Sambungkan titik proyeksi ke chart data & tabel HANYA jika rentang observasi berakhir di hari ini
                if is_today_query and chart_data:
                    chart_data[-1].forecast_value = chart_data[-1].estimated_value
                    chart_data.append(
                        BatchEstimatePoint(
                            index=len(chart_data) + 1,
                            label=f"{date_h1} (H+1)",
                            co=round(co_h1, 6),
                            no2=round(no2_h1, 6),
                            actual_value=None,
                            estimated_value=None,
                            ispu_category=cat_h1,
                            is_forecast=True,
                            forecast_value=ispu_h1,
                        )
                    )
                    chart_data.append(
                        BatchEstimatePoint(
                            index=len(chart_data) + 1,
                            label=f"{date_h2} (H+2)",
                            co=round(co_h2, 6),
                            no2=round(no2_h2, 6),
                            actual_value=None,
                            estimated_value=None,
                            ispu_category=cat_h2,
                            is_forecast=True,
                            forecast_value=ispu_h2,
                        )
                    )

        summary = BatchEstimateSummary(
            total_samples=len(all_estimated),
            avg_estimated=avg_est,
            min_estimated=min_est,
            max_estimated=max_est,
        )

        return BatchEstimateResponse(
            model_id=model.id,
            model_name=model.name,
            estimator_type=model.estimator_type,
            chart_data=chart_data,
            summary=summary,
            forecast_days=forecast_days,
        )

    async def get_map_overview(
        self,
        start_date: Union[str, date],
        end_date: Union[str, date],
    ) -> MapOverviewResponse:
        """
        Menghitung estimasi nilai ISPU dan kategori untuk seluruh 5 wilayah DKI Jakarta
        pada tanggal observasi tertentu (menggunakan data observasi valid terdekat jika terjadi tutupan awan).
        """
        start_date_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, (date, datetime)) else str(start_date)
        end_date_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, (date, datetime)) else str(end_date)

        validate_date_range(start_date_str, end_date_str)
        model = await self.repository.get_active()
        if not model:
            models = await self.repository.get_all(limit=1)
            model = models[0] if models else None

        items: List[MapOverviewItem] = []
        for region_key, region_name in SUPPORTED_REGIONS.items():
            recent_records = await self.satellite_repository.get_recent_measurements(
                region=region_key,
                up_to_date=end_date_str,
                limit=1
            )

            ispu_val = None
            cat = None
            obs_date = None
            is_fallback = False

            if recent_records and model:
                latest_valid = recent_records[-1]
                obs_date = latest_valid.date
                is_fallback = (latest_valid.date != end_date_str)
                ispu_val, cat = _calc_ispu(
                    model.intercept, model.coef_co, model.coef_no2, latest_valid.co, latest_valid.no2
                )

            items.append(
                MapOverviewItem(
                    region=region_key,
                    region_name=region_name,
                    estimated_ispu=ispu_val,
                    ispu_category=cat,
                    observation_date=obs_date,
                    is_fallback=is_fallback,
                )
            )

        return MapOverviewResponse(
            start_date=start_date_str,
            end_date=end_date_str,
            regions=items,
        )

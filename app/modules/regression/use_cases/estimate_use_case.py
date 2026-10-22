"""
Use Case untuk Estimasi Konsentrasi O3 dan Klasifikasi Kategori ISPU (Fitur /estimate).
"""

from typing import Tuple, List
import numpy as np
from fastapi import UploadFile
from injector import inject

from app.modules.regression.repositories import TrainedModelRepository
from app.modules.regression.core.file_parser import parse_batch_estimate_file
from app.modules.regression.schemas import (
    SingleEstimateRequest,
    SingleEstimateResponse,
    BatchEstimatePoint,
    BatchEstimateSummary,
    BatchEstimateResponse,
)


class EstimateUseCase:
    """
    Use case khusus untuk melakukan inferensi estimasi O3 dan konversi indeks ISPU.
    """

    @inject
    def __init__(self, repository: TrainedModelRepository):
        self.repository = repository

    @staticmethod
    def calculate_ispu_o3(o3_ug_m3: float) -> Tuple[float, str]:
        """
        Menghitung Indeks Standar Pencemar Udara (ISPU) untuk Ozon (O3)
        berdasarkan Permen LHK No. 14 Tahun 2020.
        """
        thresholds = [
            (0, 120, 0, 50, "Baik"),
            (121, 235, 51, 100, "Sedang"),
            (236, 400, 101, 200, "Tidak Sehat"),
            (401, 800, 201, 300, "Sangat Tidak Sehat"),
            (801, 1200, 301, 500, "Berbahaya"),
        ]

        val = max(0.0, float(o3_ug_m3))

        for x_min, x_max, i_min, i_max, cat in thresholds:
            if x_min <= val <= x_max:
                ispu = ((i_max - i_min) / (x_max - x_min)) * (val - x_min) + i_min
                return round(ispu, 1), cat

        return 500.0, "Berbahaya"

    async def estimate_single(self, payload: SingleEstimateRequest) -> SingleEstimateResponse:
        """
        Melakukan estimasi konsentrasi O3 dan kategori ISPU dari nilai CO dan NO2 tunggal.
        """
        if payload.model_id:
            model = await self.repository.get_by_id(payload.model_id)
            if not model:
                raise ValueError(f"Model dengan ID {payload.model_id} tidak ditemukan.")
        else:
            model = await self.repository.get_default()
            if not model:
                models = await self.repository.get_all(limit=1)
                if not models:
                    raise ValueError("Belum ada model tersimpan di sistem. Silakan latih model terlebih dahulu.")
                model = models[0]

        estimated_o3 = model.intercept + (model.coef_co * payload.co) + (model.coef_no2 * payload.no2)
        ispu_val, category = self.calculate_ispu_o3(estimated_o3)
        eq_str = f"O3 = {model.intercept:.4f} + {model.coef_co:.4f}(CO) + {model.coef_no2:.4f}(NO2)"

        return SingleEstimateResponse(
            model_id=model.id,
            model_name=model.name,
            estimator_type=model.estimator_type,
            equation=eq_str,
            estimated_o3=round(float(estimated_o3), 2),
            ispu_value=ispu_val,
            ispu_category=category,
        )

    async def estimate_from_file(
        self,
        model_id: int,
        file: UploadFile
    ) -> BatchEstimateResponse:
        """
        Melakukan estimasi O3 dan klasifikasi ISPU untuk seluruh baris berkas satelit (.xlsx, .xls, .csv).
        Memanfaatkan parse_batch_estimate_file dari file_parser untuk parsing berkas.
        """
        model = await self.repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model dengan ID {model_id} tidak ditemukan.")

        co_vals, no2_vals, actual_vals, labels = await parse_batch_estimate_file(file)

        estimated_vals = model.intercept + (model.coef_co * co_vals) + (model.coef_no2 * no2_vals)

        chart_data: List[BatchEstimatePoint] = []
        ispu_counts = {
            "Baik": 0,
            "Sedang": 0,
            "Tidak Sehat": 0,
            "Sangat Tidak Sehat": 0,
            "Berbahaya": 0,
        }

        for idx, (co_val, no2_val, est_val, label_str) in enumerate(zip(co_vals, no2_vals, estimated_vals, labels)):
            est_round = round(float(est_val), 2)
            ispu_val, category = self.calculate_ispu_o3(est_round)
            if category in ispu_counts:
                ispu_counts[category] += 1

            actual_val = round(float(actual_vals[idx]), 2) if actual_vals is not None else None

            chart_data.append(
                BatchEstimatePoint(
                    index=idx + 1,
                    label=label_str,
                    co=round(float(co_val), 6),
                    no2=round(float(no2_val), 6),
                    actual_o3=actual_val,
                    estimated_o3=est_round,
                    ispu_value=ispu_val,
                    ispu_category=category,
                )
            )

        all_est = [pt.estimated_o3 for pt in chart_data]
        dominant_cat = max(ispu_counts, key=ispu_counts.get) if chart_data else "Baik"
        avg_est = round(float(np.mean(all_est)), 2)
        min_est = round(float(np.min(all_est)), 2)
        max_est = round(float(np.max(all_est)), 2)

        eq_str = f"O3 = {model.intercept:.4f} + {model.coef_co:.4f}(CO) + {model.coef_no2:.4f}(NO2)"

        summary = BatchEstimateSummary(
            total_samples=len(chart_data),
            avg_estimated_o3=avg_est,
            min_estimated_o3=min_est,
            max_estimated_o3=max_est,
            dominant_ispu_category=dominant_cat,
            ispu_distribution=ispu_counts,
        )

        return BatchEstimateResponse(
            model_id=model.id,
            model_name=model.name,
            estimator_type=model.estimator_type,
            equation=eq_str,
            chart_data=chart_data,
            summary=summary,
        )

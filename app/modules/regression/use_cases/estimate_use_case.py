"""
Use Case untuk Estimasi Konsentrasi O3 dan Klasifikasi Kategori ISPU (Fitur /estimate).
"""

from typing import Tuple, List
import numpy as np
from fastapi import UploadFile
from injector import inject

from app.modules.regression.repositories import TrainedModelRepository
from app.modules.regression.core.parser import parse_batch_estimate_file
from app.modules.regression.core.ispu import get_ispu_category
from app.modules.regression.schemas import (
    SingleEstimateRequest,
    SingleEstimateResponse,
    BatchEstimatePoint,
    BatchEstimateSummary,
    BatchEstimateResponse,
)


class EstimateUseCase:
    """
    Use case khusus untuk melakukan inferensi estimasi ISPU.
    """

    @inject
    def __init__(self, repository: TrainedModelRepository):
        self.repository = repository

    async def estimate_single(self, payload: SingleEstimateRequest) -> SingleEstimateResponse:
        """
        Melakukan estimasi nilai ISPU dan kategori secara langsung dari nilai CO dan NO2 tunggal.
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

        estimated_val = model.intercept + (model.coef_co * payload.co) + (model.coef_no2 * payload.no2)
        ispu_val = round(max(0.0, float(estimated_val)), 1)
        category = get_ispu_category(ispu_val)

        sign_co = "+" if model.coef_co >= 0 else "-"
        sign_no2 = "+" if model.coef_no2 >= 0 else "-"
        if abs(model.intercept) > 1e-9:
            eq_str = f"ISPU = {model.intercept:.4f} {sign_co} {abs(model.coef_co):.4f}(CO) {sign_no2} {abs(model.coef_no2):.4f}(NO2)"
        else:
            first_sign = "" if model.coef_co >= 0 else "-"
            eq_str = f"ISPU = {first_sign}{abs(model.coef_co):.4f}(CO) {sign_no2} {abs(model.coef_no2):.4f}(NO2)"

        return SingleEstimateResponse(
            model_id=model.id,
            model_name=model.name,
            estimator_type=model.estimator_type,
            equation=eq_str,
            estimated_value=ispu_val,
            ispu_category=category,
        )

    async def estimate_from_file(
        self,
        model_id: int,
        file: UploadFile
    ) -> BatchEstimateResponse:
        """
        Melakukan estimasi ISPU langsung untuk seluruh baris berkas satelit (.xlsx, .xls, .csv).
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
            ispu_round = round(max(0.0, float(est_val)), 1)
            category = get_ispu_category(ispu_round)
            if category in ispu_counts:
                ispu_counts[category] += 1

            actual_val = round(float(actual_vals[idx]), 1) if actual_vals is not None else None

            chart_data.append(
                BatchEstimatePoint(
                    index=idx + 1,
                    label=label_str,
                    co=round(float(co_val), 6),
                    no2=round(float(no2_val), 6),
                    actual_value=actual_val,
                    estimated_value=ispu_round,
                    ispu_category=category,
                )
            )

        all_estimated = [pt.estimated_value for pt in chart_data]
        dominant_cat = max(ispu_counts, key=ispu_counts.get) if chart_data else "Baik"
        avg_est = round(float(np.mean(all_estimated)), 1)
        min_est = round(float(np.min(all_estimated)), 1)
        max_est = round(float(np.max(all_estimated)), 1)

        sign_co = "+" if model.coef_co >= 0 else "-"
        sign_no2 = "+" if model.coef_no2 >= 0 else "-"
        if abs(model.intercept) > 1e-9:
            eq_str = f"ISPU = {model.intercept:.4f} {sign_co} {abs(model.coef_co):.4f}(CO) {sign_no2} {abs(model.coef_no2):.4f}(NO2)"
        else:
            first_sign = "" if model.coef_co >= 0 else "-"
            eq_str = f"ISPU = {first_sign}{abs(model.coef_co):.4f}(CO) {sign_no2} {abs(model.coef_no2):.4f}(NO2)"

        summary = BatchEstimateSummary(
            total_samples=len(chart_data),
            avg_estimated=avg_est,
            min_estimated=min_est,
            max_estimated=max_est,
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

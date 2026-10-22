"""
Use Case untuk Pelatihan Model Regresi Robust (MM, M, S, OLS).
"""

import asyncio
from fastapi import UploadFile
from injector import inject

from app.modules.regression.core.file_parser import parse_and_validate_file
from app.modules.regression.algorithms.mm_estimator import MMEstimator
from app.modules.regression.algorithms.m_estimator import MEstimator
from app.modules.regression.algorithms.s_estimator import SEstimator
from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.core.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_r2,
)
from app.modules.regression.schemas import (
    TrainModelResult,
    RegressionPoint,
)


class TrainModelUseCase:
    """
    Use case khusus untuk melatih model regresi robust dan mengevaluasi data latih.
    """

    @inject
    def __init__(self):
        pass

    async def train_model(
        self,
        file: UploadFile,
        estimator_type: str = "MM",
        fit_intercept: bool = True
    ) -> TrainModelResult:
        """
        Melatih model regresi robust pada data latih dan menghasilkan metrik evaluasi & grafik.
        """
        X, y, feature_cols, target_col = await parse_and_validate_file(file, require_target=True)
        if y is None:
            raise ValueError("Target y diperlukan untuk melatih model.")

        raw_type = estimator_type.lower().strip()

        if raw_type in ("mm", "mm-high", "mm_high"):
            model = MMEstimator(fit_intercept=fit_intercept, efficiency="high")
            resolved_type = "MM-High" if "high" in raw_type else "MM"
        elif raw_type in ("mm-low", "mm_low"):
            model = MMEstimator(fit_intercept=fit_intercept, efficiency="low")
            resolved_type = "MM-Low"
        elif raw_type == "m":
            model = MEstimator(fit_intercept=fit_intercept)
            resolved_type = "M"
        elif raw_type == "s":
            model = SEstimator(fit_intercept=fit_intercept)
            resolved_type = "S"
        elif raw_type == "ols":
            model = OLSEstimator(fit_intercept=fit_intercept)
            resolved_type = "OLS"
        else:
            raise ValueError(
                f"Estimator '{estimator_type}' tidak didukung. Pilih MM, MM-Low, MM-High, M, S, atau OLS."
            )

        await asyncio.to_thread(model.fit, X, y)
        y_pred = model.predict(X)

        rmse = round(calculate_rmse(y, y_pred), 4)
        mae = round(calculate_mae(y, y_pred), 4)
        r2 = round(calculate_r2(y, y_pred), 4)

        intercept_val = float(model.intercept_) if fit_intercept and model.intercept_ is not None else 0.0
        coef_co = float(model.coef_[0])
        coef_no2 = float(model.coef_[1])

        scale_val = None
        if hasattr(model, "scale_s_") and model.scale_s_ is not None:
            scale_val = round(float(model.scale_s_), 4)
        elif hasattr(model, "scale_") and model.scale_ is not None:
            scale_val = round(float(model.scale_), 4)

        sign_co = "+" if coef_co >= 0 else "-"
        sign_no2 = "+" if coef_no2 >= 0 else "-"
        equation = f"O3 = {intercept_val:.4f} {sign_co} {abs(coef_co):.4f}(CO) {sign_no2} {abs(coef_no2):.4f}(NO2)"

        chart_data = [
            RegressionPoint(
                index=i + 1,
                actual=round(float(y[i]), 2),
                predicted=round(float(y_pred[i]), 2),
            )
            for i in range(len(y))
        ]

        return TrainModelResult(
            estimator_type=resolved_type,
            equation=equation,
            intercept=round(intercept_val, 6),
            coef_co=round(coef_co, 6),
            coef_no2=round(coef_no2, 6),
            scale=scale_val,
            rmse=rmse,
            mae=mae,
            r2=r2,
            chart_data=chart_data,
            total_samples=len(y),
        )

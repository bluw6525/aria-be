"""
Use Case untuk Pengujian Model Tersimpan (Testing Saved Model) Terhadap Data Baru.
"""

from fastapi import UploadFile
from injector import inject

from app.modules.regression.repositories import TrainedModelRepository
from app.modules.regression.core.file_parser import parse_and_validate_file
from app.modules.regression.core.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_r2,
)
from app.modules.regression.schemas import (
    TestModelResult,
    RegressionPoint,
)


class TestModelUseCase:
    """
    Use case khusus untuk menguji performa model tersimpan dengan berkas data uji baru.
    """

    @inject
    def __init__(self, repository: TrainedModelRepository):
        self.repository = repository

    async def execute(
        self,
        model_id: int,
        file: UploadFile
    ) -> TestModelResult:
        """
        Menguji model tersimpan di database dengan file data uji baru.
        """
        model = await self.repository.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model dengan ID {model_id} tidak ditemukan di basis data.")

        X_test, y_test, feature_cols, target_col = await parse_and_validate_file(file, require_target=True)
        if y_test is None:
            raise ValueError("Data uji harus memiliki kolom target O3 untuk menghitung evaluasi.")

        y_pred = model.intercept + (model.coef_co * X_test[:, 0]) + (model.coef_no2 * X_test[:, 1])

        rmse = round(calculate_rmse(y_test, y_pred), 4)
        mae = round(calculate_mae(y_test, y_pred), 4)
        r2 = round(calculate_r2(y_test, y_pred), 4)

        sign_co = "+" if model.coef_co >= 0 else "-"
        sign_no2 = "+" if model.coef_no2 >= 0 else "-"
        equation = f"O3 = {model.intercept:.4f} {sign_co} {abs(model.coef_co):.4f}(CO) {sign_no2} {abs(model.coef_no2):.4f}(NO2)"

        chart_data = [
            RegressionPoint(
                index=i + 1,
                actual=round(float(y_test[i]), 2),
                predicted=round(float(y_pred[i]), 2),
            )
            for i in range(len(y_test))
        ]

        return TestModelResult(
            model_id=model.id,
            model_name=model.name,
            estimator_type=model.estimator_type,
            equation=equation,
            rmse=rmse,
            mae=mae,
            r2=r2,
            chart_data=chart_data,
            total_samples=len(y_test),
        )

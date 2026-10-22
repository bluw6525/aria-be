"""
Use Case untuk Diagnostik dan Deteksi Pencilan (Outlier Detection).
Menggunakan OutlierDetector (Leverage h_ii dan DFFITS).
"""

import asyncio
from fastapi import UploadFile
from injector import inject

from app.modules.regression.core.file_parser import parse_and_validate_file
from app.modules.regression.core.outliers import OutlierDetector
from app.modules.regression.schemas import OutlierDetectionResponse


class DetectOutliersUseCase:
    """
    Use case khusus untuk menganalisis dan mendeteksi titik pencilan data latih.
    """

    @inject
    def __init__(self):
        pass

    async def detect_outliers(
        self,
        file: UploadFile,
        fit_intercept: bool = True
    ) -> OutlierDetectionResponse:
        """
        Menjalankan deteksi outlier pada data latih menggunakan Leverage dan DFFITS secara non-blocking.
        """
        X, y, feature_cols, target_col = await parse_and_validate_file(file, require_target=True)
        if y is None:
            raise ValueError("Target y diperlukan untuk deteksi DFFITS.")

        detector = OutlierDetector(fit_intercept=fit_intercept)
        await asyncio.to_thread(detector.fit, X, y, method="both")
        res_dict = detector.to_dict(include_details=False)

        return OutlierDetectionResponse.from_detector_dict(res_dict)

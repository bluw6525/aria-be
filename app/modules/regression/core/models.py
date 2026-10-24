from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String

from app.core.database import Base


class RegressionModel(Base):
    """
    Tabel riwayat hasil estimasi regresi (OLS, M, S, MM-Estimator).
    """
    __tablename__ = "regression_models"

    # 1. Primary Key & Metadata
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    dataset_name = Column(String(100), nullable=True, index=True)

    # 2. Konfigurasi Estimator & Split
    estimator_type = Column(String(50), nullable=False, index=True)
    fit_intercept = Column(Boolean, default=True)
    random_state = Column(Integer, nullable=True)
    test_size = Column(Float, default=0.3)
    total_samples = Column(Integer, nullable=True)

    # 3. Parameter Hasil Estimasi
    intercept = Column(Float, nullable=True)
    coefficients = Column(JSON, nullable=False)
    scale = Column(Float, nullable=True)
    iterations = Column(Integer, nullable=True)
    converged = Column(Boolean, default=True)

    # 4. Metrik Evaluasi Data Uji (Test)
    test_rmse = Column(Float, nullable=False)
    test_mae = Column(Float, nullable=False)
    test_r2 = Column(Float, nullable=False)

    # 5. Metrik Evaluasi Data Latih (Train)
    train_rmse = Column(Float, nullable=True)
    train_mae = Column(Float, nullable=True)
    train_r2 = Column(Float, nullable=True)

    # 6. Timestamp
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return (
            f"<RegressionModel(id={self.id}, estimator='{self.estimator_type}', "
            f"test_r2={self.test_r2:.4f}, test_rmse={self.test_rmse:.4f})>"
        )


class TrainedModel(Base):
    """
    Tabel untuk menyimpan model regresi hasil pelatihan mandiri
    serta model bawaan (default) aplikasi Aria.
    """
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    estimator_type = Column(String(50), nullable=False)
    is_default = Column(Boolean, default=False)

    # Parameter Regresi: O3 = beta0 + beta1(CO) + beta2(NO2)
    intercept = Column(Float, nullable=False)
    coef_co = Column(Float, nullable=False)
    coef_no2 = Column(Float, nullable=False)
    scale = Column(Float, nullable=True)

    # Metrik Evaluasi Data Latih
    train_rmse = Column(Float, nullable=False)
    train_mae = Column(Float, nullable=False)
    train_r2 = Column(Float, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<TrainedModel(id={self.id}, name='{self.name}', estimator='{self.estimator_type}')>"



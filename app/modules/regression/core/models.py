from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from app.core.database import Base


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


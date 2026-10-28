from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class BenchmarkSession(Base):
    """
    Tabel induk (parent) untuk sesi pengujian Monte Carlo Cross-Validation (MCCV).
    Menyimpan metadata ringkasan run benchmark per dataset dan rasio split.
    """
    __tablename__ = "benchmark_sessions"

    # 1. Primary Key & Metadata
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_name = Column(String(150), nullable=True, index=True)
    dataset_name = Column(String(100), nullable=False, index=True)

    # 2. Konfigurasi Evaluasi
    test_size = Column(Float, default=0.2, nullable=False)
    total_seeds = Column(Integer, default=200, nullable=False)
    total_samples = Column(Integer, nullable=True)
    fit_intercept = Column(Boolean, default=True, nullable=False)

    # 3. Rangkuman Metrik Agregat (JSON: rata-rata R2, RMSE, MAE per estimator)
    summary_metrics = Column(JSON, nullable=True)

    # 4. Timestamp
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())

    # Relasi 1 : N ke iterasi per seed & estimator
    iterations = relationship(
        "MCCVIterationModel",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<BenchmarkSession(id={self.id}, session_name='{self.session_name}', "
            f"dataset='{self.dataset_name}', seeds={self.total_seeds})>"
        )


class MCCVIterationModel(Base):
    """
    Tabel rincian (child) hasil tiap iterasi seed dan algoritma regresi dalam suatu sesi MCCV.
    """
    __tablename__ = "mccv_iterations"

    # 1. Primary Key & Foreign Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(
        Integer,
        ForeignKey("benchmark_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 2. Konfigurasi Estimator & Iterasi
    estimator_type = Column(String(50), nullable=False, index=True)
    random_state = Column(Integer, nullable=True)

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

    # 5b. Uji Signifikansi F (ANOVA Regresi)
    f_statistic = Column(Float, nullable=True)
    f_table = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    is_significant = Column(Boolean, default=False, nullable=True)

    # 6. Timestamp
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())

    session = relationship("BenchmarkSession", back_populates="iterations")

    def __repr__(self) -> str:
        return (
            f"<MCCVIterationModel(id={self.id}, session_id={self.session_id}, "
            f"estimator='{self.estimator_type}', test_r2={self.test_r2:.4f}, test_rmse={self.test_rmse:.4f})>"
        )


class TrainedModel(Base):
    """
    Tabel untuk menyimpan model regresi hasil pelatihan / model terpilih
    serta model bawaan (default) aplikasi Aria.
    """
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    estimator_type = Column(String(50), nullable=False)
    is_default = Column(Boolean, default=False)

    # Parameter Regresi
    intercept = Column(Float, nullable=False)
    coefficients = Column(JSON, nullable=False)  # {"CO": float, "NO2": float, ...}
    scale = Column(Float, nullable=True)

    # Metrik Evaluasi Data Latih
    train_rmse = Column(Float, nullable=False)
    train_mae = Column(Float, nullable=False)
    train_r2 = Column(Float, nullable=False)

    # Uji Signifikansi F (ANOVA Regresi)
    f_statistic = Column(Float, nullable=True)
    f_table = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    is_significant = Column(Boolean, default=False, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())

    @property
    def coef_co(self) -> float:
        if isinstance(self.coefficients, dict):
            return float(self.coefficients.get("CO", 0.0))
        return 0.0

    @property
    def coef_no2(self) -> float:
        if isinstance(self.coefficients, dict):
            return float(self.coefficients.get("NO2", 0.0))
        return 0.0

    def __repr__(self) -> str:
        return f"<TrainedModel(id={self.id}, name='{self.name}', estimator='{self.estimator_type}')>"

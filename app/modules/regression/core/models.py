from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, func
from sqlalchemy.orm import relationship

from app.core.database import Base


class BatchSession(Base):
    """
    Tabel induk (parent) untuk sesi pengujian massal (batch evaluation).
    Menyimpan metadata ringkasan run pengujian per dataset dan rasio split.
    """
    __tablename__ = "batch_sessions"

    # 1. Primary Key & Metadata
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_name = Column(String(150), nullable=True, index=True)
    dataset_name = Column(String(100), nullable=False, index=True)

    # 2. Konfigurasi Evaluasi
    test_size = Column(Float, default=0.2, nullable=False)
    total_seeds = Column(Integer, default=200, nullable=False)
    total_samples = Column(Integer, nullable=True)
    fit_intercept = Column(Boolean, default=True, nullable=False)

    # 3. Rangkuman Metrik Agregat & Daftar Seeds (JSON)
    summary_metrics = Column(JSON, nullable=True)
    seeds = Column(JSON, nullable=True)

    # 4. Timestamp
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())

    # Relasi 1 : N ke iterasi per seed & estimator
    iterations = relationship(
        "BatchIteration",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<BatchSession(id={self.id}, session_name='{self.session_name}', "
            f"dataset='{self.dataset_name}', seeds={self.total_seeds})>"
        )


class BatchIteration(Base):
    """
    Tabel rincian (child) hasil tiap iterasi seed dan algoritma regresi dalam suatu sesi batch.
    """
    __tablename__ = "batch_iterations"

    # 1. Primary Key & Foreign Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(
        Integer,
        ForeignKey("batch_sessions.id", ondelete="CASCADE"),
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

    # 5. Metrik Evaluasi Data Latih (Train)
    train_rmse = Column(Float, nullable=True)
    train_mae = Column(Float, nullable=True)

    # 6. Timestamp
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now())

    session = relationship("BatchSession", back_populates="iterations")

    def __repr__(self) -> str:
        return (
            f"<BatchIteration(id={self.id}, session_id={self.session_id}, "
            f"estimator='{self.estimator_type}', test_rmse={self.test_rmse:.4f}, test_mae={self.test_mae:.4f})>"
        )


class PublicModel(Base):
    """
    Tabel untuk menyimpan model regresi definitif / model publik hasil pelatihan
    aplikasi Aria.
    """
    __tablename__ = "public_models"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    estimator_type = Column(String(50), nullable=False)

    # Parameter Regresi
    intercept = Column(Float, nullable=False)
    coefficients = Column(JSON, nullable=False)  # {"CO": float, "NO2": float, ...}
    scale = Column(Float, nullable=True)

    # Metrik Evaluasi Data Latih
    train_rmse = Column(Float, nullable=False)
    train_mae = Column(Float, nullable=False)

    # Status Model Aktif (hanya 1 model yang aktif untuk estimasi default)
    is_active = Column(Boolean, default=False, nullable=False, server_default="0", index=True)

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
        return f"<PublicModel(id={self.id}, name='{self.name}', estimator='{self.estimator_type}')>"

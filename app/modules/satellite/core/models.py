from datetime import datetime
from sqlalchemy import Column, Float, Integer, String, DateTime, UniqueConstraint
from app.core.database import Base


class SentinelMeasurement(Base):
    """
    Tabel penyimpanan data satelit Sentinel-5P (CO dan NO2) per wilayah dan tanggal.
    Data historis ini berfungsi sebagai data warehouse / cache lokal untuk mempercepat komputasi.
    Menyimpan pula baris dengan co & no2 kosong (null) jika tertutup awan,
    disertai product_type ('OFFL' / 'NRTI') dan updated_at untuk logika sinkronisasi ulang.
    """
    __tablename__ = "sentinel_measurements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    region = Column(String(50), nullable=False, index=True)   # contoh: 'jakarta_barat'
    date = Column(String(10), nullable=False, index=True)     # format: 'YYYY-MM-DD'
    co = Column(Float, nullable=True)                         # CO_satelit (mol/m^2), null jika awan tebal
    no2 = Column(Float, nullable=True)                        # NO2_satelit (mol/m^2), null jika awan tebal
    product_type = Column(String(10), nullable=True) # 'OFFL' | 'NRTI' | None (jika awan tebal)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("region", "date", name="uq_sentinel_region_date"),
    )

    def __repr__(self) -> str:
        co_str = f"{self.co:.6f}" if self.co is not None else "null"
        no2_str = f"{self.no2:.6f}" if self.no2 is not None else "null"
        return f"<SentinelMeasurement(region='{self.region}', date='{self.date}', co={co_str}, no2={no2_str}, type='{self.product_type}')>"

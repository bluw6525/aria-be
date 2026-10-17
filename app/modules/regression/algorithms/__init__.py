"""
Paket Algoritma Regresi Robust (OLS, M-Estimator, S-Estimator, MM-Estimator).
Mengimplementasikan metode Tukey Bisquare sesuai Susanti et al. (2014).
"""

from .ols_estimator import OLSEstimator, ols_estimator
from .m_estimator import MEstimator, m_estimator
from .s_estimator import SEstimator, s_estimator, solve_s_scale
from .mm_estimator import MMEstimator, mm_estimator
from .robust_utils import (
    calculate_mad_scale,
    weight_tukey,
    weight_s_scale,
    rho_tukey,
)

__all__ = [
    # Estimator Classes & Functions
    "OLSEstimator",
    "ols_estimator",
    "MEstimator",
    "m_estimator",
    "SEstimator",
    "s_estimator",
    "solve_s_scale",
    "MMEstimator",
    "mm_estimator",
    # Robust Utilities (Susanti et al., 2014)
    "calculate_mad_scale",
    "weight_tukey",
    "weight_s_scale",
    "rho_tukey",
]

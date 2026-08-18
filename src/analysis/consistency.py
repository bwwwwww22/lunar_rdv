"""Filter consistency utilities: NEES (state-error) and NIS (innovation)
statistics for validating that the EKF's claimed uncertainty (or internal
confidence) matches its actual error statistics.

1. NEES = δx^T P^-1 δx
   measures how accurately P reflects the true estimation error δx
   should be ~ dim(state)
2. NIS  = y^T S^-1 y
   measures how well S models the sensor measurement residual or innovation, y
   should be ~ dim(measurement)
Here NEES ≈ 12 consistent; > 12 overconfident (dangerous); < 12 underconfident (safe)
"""

import numpy as np
from scipy.stats import chi2
from src.utils.quaternions import quat_error


def compute_error_state(x_truth, x_est):
    """12-dim error state between truth and estimate at a single timestep."""
    dx = np.zeros(12)
    dx[0:3] = x_truth[0:3] - x_est[0:3]
    dx[3:6] = x_truth[3:6] - x_est[3:6]
    dx[6:9] = quat_error(x_truth[6:10], x_est[6:10])
    dx[9:12] = x_truth[10:13] - x_est[10:13]
    return dx


def nees(x_truth, x_est, P):
    """Normalized estimation error squared (at one timestep)"""
    dx = compute_error_state(x_truth, x_est)
    return float(dx @ np.linalg.solve(P, dx))

def nis(y, S):
    """Normalized innovation squared (at one timestep)"""
    return float(y @ np.linalg.solve(S, y))

def chi2_bounds(dof, N_runs=1, alpha=0.05):
    """Two-sided chi^2 bounds. For a single run time, use N_runs=1 (wide bounds). 
    For an average over N MC runs at each timestep, use N_runs=N (tight bounds; standard).
    alpha = 0.05 means we compare against 95% confidence bounds."""
    lo = chi2.ppf(alpha/2, dof * N_runs) / N_runs
    hi = chi2.ppf(1 - alpha/2, dof * N_runs) / N_runs
    return lo, hi
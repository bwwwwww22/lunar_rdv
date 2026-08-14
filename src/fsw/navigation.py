"""Navigation (state estimation)."""

import numpy as np
from src.fsw.ekf import MEKF

class PassthroughNav:
    """Just for testing: returns the sensor measurement as the estimate."""
    def estimate(self, z, dt):
        return np.asarray(z, dtype=float)

class EKFNav:
    """EKFNav: MEKF wrapper. Same estimate(z, dt) interface."""
    def __init__(self, params, x0, P0, Q, R):
        self.ekf = MEKF(params, x0, P0, Q, R)
        self._last_u = np.zeros(6)

    def set_last_control(self, u):
        """Runner tells nav what control was just applied (used in predict)."""
        self._last_u = np.asarray(u, dtype=float)

    def estimate(self, z, dt):
        # Predict forward using the last known control, then update with z
        self.ekf.predict(self._last_u, dt)
        # Only update when a measurement is available, no measurement means predict only
        if z is not None: self.ekf.update(z)
        return self.ekf.get_estimate()

    def get_covariance(self):
        # for consistency analysis
        return self.ekf.get_covariance()
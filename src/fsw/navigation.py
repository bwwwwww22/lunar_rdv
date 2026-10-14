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
        self._first = True   # x0 is the estimate at t_0, so the first call must not predict

    def set_last_control(self, u):
        """Runner tells nav what control was just applied (used in predict)."""
        self._last_u = np.asarray(u, dtype=float)

    def estimate(self, z, dt):
        # Predict t_{k-1} -> t_k using the last known control, then update with z (taken at t_k).
        # The first call skips predict: x0 is already at t_0, and predicting first would leave
        # the filter one step ahead of every measurement it fuses.
        if self._first:
            self._first = False
            self.ekf.w_m = np.asarray(z["gyro"], dtype=float)   # get_estimate() needs w_m
        else:
            self.ekf.predict(self._last_u, z["gyro"], dt)   # now feeds in gyro readings
        # Only update when a measurement is available, no measurement means predict only
        self.ekf.update(z)
        return self.ekf.get_estimate()

    def get_filter_state(self):
        # internal [r, v, q, b_hat], for logging gyro-bias estimate / NEES
        return self.ekf.get_filter_state()

    def get_covariance(self):
        # for consistency analysis
        return self.ekf.get_covariance()
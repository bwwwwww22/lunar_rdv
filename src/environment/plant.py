"""Plant: holds truth state, advances physics. Knows nothing about sensors, nav, or ctrl"""

import numpy as np
from src.environment.dynamics import rk4_step

class Plant:
    def __init__(self, params, x0):
        """params: dict with mass, inertia (3x3), inertia_inv (3x3), orbit_rate.
        x0: initial truth state (13,)"""
        self.params = params
        self._validate_params(params)
        self.x = np.array(x0, dtype=float)

    @staticmethod
    def _validate_params(p):
        """Check that params dict has required keys and that inertia_inv is correct."""
        for key in ("mass", "inertia", "inertia_inv", "orbit_rate"):
            if key not in p: raise KeyError(f"params missing '{key}'")
        if not np.allclose(p["inertia"] @ p["inertia_inv"], np.eye(3), atol=1e-6):
            raise ValueError("inertia_inv is not the inverse of inertia")

    def reset(self, x0):
        self.x = np.array(x0, dtype=float)

    def get_state(self):
        return self.x.copy() # COPY!

    def step(self, u, dt):
        """Advance truth state one step under control u (6,)."""
        self.x = rk4_step(self.x, np.asarray(u, dtype=float), dt, self.params)
        return self.get_state()

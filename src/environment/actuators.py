"""Actuator model: converts commanded F/torque into applied F/torque.
 Lives in the ENVIRONMENT (plant side)."""

import numpy as np

class Actuators:
    def __init__(self, max_force, max_torque):
        """max_force: [N], scalar or (3,) per-axis limit 
        max_torque: [N·m]"""
        self.max_force = np.broadcast_to(max_force, (3,)).astype(float).copy()
        self.max_torque = np.broadcast_to(max_torque, (3,)).astype(float).copy()

    def apply(self, u_cmd):
        """u_cmd (6,) = [Fx,Fy,Fz, taux,tauy,tauz] commanded.
        Returns u_applied (6,) after per-axis saturation."""
        u_cmd = np.asarray(u_cmd, dtype=float)
        F_cmd = u_cmd[0:3]
        tau_cmd = u_cmd[3:6]

        F_applied = np.clip(F_cmd, -self.max_force, self.max_force)
        tau_applied = np.clip(tau_cmd, -self.max_torque, self.max_torque)

        return np.concatenate([F_applied, tau_applied])

    def is_saturated(self, u_cmd):
        """Check which channels are saturated"""
        u_cmd = np.asarray(u_cmd, dtype=float)
        limits = np.concatenate([self.max_force, self.max_torque])
        return np.abs(u_cmd) >= limits

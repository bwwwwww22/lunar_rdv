"""Guidance: produces the reference (target) state the controller drives toward.
Lives in the fsw."""

import numpy as np

class Guidance:
    def __init__(self, dock_position, dock_attitude=None):
        """dock_position: (3,) tgt relative position at dock [m]
        dock_attitude: (4,) tgt attitude quaternion, scalar-last [x,y,z,w] (defaults to identity)"""
        self.dock_position = np.asarray(dock_position, dtype=float)
        if dock_attitude is None:
            dock_attitude = np.array([0.0, 0.0, 0.0, 1.0])
        self.dock_attitude = np.asarray(dock_attitude, dtype=float)

    def reference(self, t, x_est):
        """Return the 13-state reference at time t. Assumesconstant docking setpoint 
        (position, zero velocity, target attitude, zero angular rate)"""
        x_ref = np.zeros(13)
        x_ref[0:3] = self.dock_position   # target position
        x_ref[3:6] = 0.0                  # zero relative velocity at dock
        x_ref[6:10] = self.dock_attitude  # target attitude
        x_ref[10:13] = 0.0                # zero angular rate
        return x_ref

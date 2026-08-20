"""Fixed-rate scheduler. Assumes single-rate (sensor, nav, ctrl) are all at the same dt"""

class Scheduler:
    def __init__(self, dt, control_decimation=1, sensor_decimation=1):
        """dt: base timestep [s].
        *_decimation: run that block every N base steps (1 = every step)."""
        self.dt = dt
        self.control_decimation = control_decimation
        self.sensor_decimation = sensor_decimation

    def run_sensor(self, step_idx):
        return step_idx % self.sensor_decimation == 0

    def run_control(self, step_idx):
        return step_idx % self.control_decimation == 0

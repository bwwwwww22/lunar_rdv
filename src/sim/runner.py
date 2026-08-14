"""Closed-loop simulation runner. Where plant/environment and FSW meet"""

import numpy as np

class SimRunner:
    def __init__(self, plant, sensor, nav, guidance, controller, actuators, scheduler):
        self.plant = plant
        self.sensor = sensor
        self.nav = nav
        self.guidance = guidance
        self.controller = controller
        self.actuators = actuators
        self.scheduler = scheduler

    def run(self, t_end):
        dt = self.scheduler.dt
        n_steps = int(t_end / dt)

        log = {
            "t": np.zeros(n_steps + 1),
            "x_truth": np.zeros((n_steps + 1, 13)),    # real state of the spacecraft
            "x_est": np.zeros((n_steps + 1, 13)),      # estimated state from nav
            "x_ref": np.zeros((n_steps + 1, 13)),      # reference state from guidance
            "u_cmd": np.zeros((n_steps + 1, 6)),       # controller's intent (6d: force/torque)
            "u_applied": np.zeros((n_steps + 1, 6)),   # what's actually delivered post-saturation (actuators)
            "saturated": np.zeros((n_steps + 1, 6), dtype=bool),
        }

        # init
        log["x_truth"][0] = self.plant.get_state()
        u_applied = np.zeros(6)
        u_cmd = np.zeros(6)
        x_est = self.plant.get_state()

        for k in range(n_steps):
            t = k * dt
            x_truth = self.plant.get_state()

            # sensor (fsw sees only this)
            if self.scheduler.run_sensor(k):
                z = self.sensor.measure(x_truth)   # may be None during dropout
                x_est = self.nav.estimate(z, dt)

            # guidance + ctrl
            if self.scheduler.run_control(k):
                x_ref = self.guidance.reference(t, x_est)
                u_cmd = self.controller.command(x_est, x_ref)
                u_applied = self.actuators.apply(u_cmd)
                
                if hasattr(self.nav, "set_last_control"):
                    self.nav.set_last_control(u_applied)

            else:
                x_ref = self.guidance.reference(t, x_est)

            # plant advances under applied control
            self.plant.step(u_applied, dt)

            idx = k + 1
            log["t"][idx] = t + dt
            log["x_truth"][idx] = self.plant.get_state()
            log["x_est"][idx] = x_est
            log["x_ref"][idx] = x_ref
            log["u_cmd"][idx] = u_cmd
            log["u_applied"][idx] = u_applied
            log["saturated"][idx] = self.actuators.is_saturated(u_cmd)
        return log

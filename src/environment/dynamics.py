"""6-DOF relative dynamics
Translation: Clohessy-Wiltshire (Hill, chaser relative to target) + ctrl
Rotation:    rigid-body Euler

State x (13,):
    [0:3]  r   relative position (Hill) [m]
    [3:6]  v   relative velocity (Hill) [m/s]
    [6:10] q   attitude quaternion (ref->body, JPL passive), scalar-last [x,y,z,w]
    [10:13] w  angular velocity (body) [rad/s]

Control u (6,):
    [0:3] F  force (Hill) [N]
    [3:6] tau torque (body) [N·m]
"""

import numpy as np
from src.utils.quaternions import quat_derivative


def cw_acceleration(r, v, n):
    """Clohessy-Wiltshire relative acceleration (x, y and z double dot in textbook)
    n = mean orbital rate [rad/s], x = radial, y = along-track, z = cross-track.
    B. Wie: eqns 4.88-4.90
    """
    x, y, z = r
    vx, vy, vz = v
    ax = 3*n**2 * x + 2*n * vy  
    ay = -2*n * vx
    az = -n**2 * z
    return np.array([ax, ay, az])


def rigid_body_acceleration(omega, tau, I, I_inv):
    """Euler's equation: I·ω̇ = tau - ω x (I·ω), or ω̇ = I⁻¹ (tau - ω x (I·ω))
    B. Wie: eqn 6.4 (I=J, tau=M)"""
    return I_inv @ (tau - np.cross(omega, I @ omega))


def state_derivative(x, u, params):
    """xdot = f(x, u, params). Full 13-state derivative."""
    r     = x[0:3]
    v     = x[3:6]
    q     = x[6:10]
    omega = x[10:13]
    F   = u[0:3]
    tau = u[3:6]

    n     = params["orbit_rate"]     # mean motion
    m     = params["mass"]           # kg
    I     = params["inertia"]        # 3x3
    I_inv = params["inertia_inv"]    # 3x3

    # Translation
    a = cw_acceleration(r, v, n) + F / m
    # Rotation
    q_dot = quat_derivative(q, omega)
    omega_dot = rigid_body_acceleration(omega, tau, I, I_inv)

    xdot = np.empty(13)
    xdot[0:3]   = v
    xdot[3:6]   = a
    xdot[6:10]  = q_dot
    xdot[10:13] = omega_dot
    return xdot


def rk4_step(x, u, dt, params):
    """Fixed-step 4th-order Runge-Kutta. Could've used solve_ivp"""
    k1 = state_derivative(x, u, params)
    k2 = state_derivative(x + 0.5*dt*k1, u, params)
    k3 = state_derivative(x + 0.5*dt*k2, u, params)
    k4 = state_derivative(x + dt*k3, u, params)
    x_new = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # renormalize quaternion after integration otherwise it'd drfit
    x_new[6:10] /= np.linalg.norm(x_new[6:10]) 
    return x_new

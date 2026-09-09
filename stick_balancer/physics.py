"""
Physics of an N-link inverted pendulum ("stick with joints") on a cart.

This is the whole simulator: pure NumPy, no MuJoCo, no hidden magic.

Coordinates
-----------
q = [x, theta_1, ..., theta_N]

    x        : horizontal cart position (m), positive to the right
    theta_i  : absolute angle of link i measured from *straight up* (rad),
               positive when the link tip leans to the right.
               theta = 0 for every link  ==>  the stick is perfectly balanced.

Every link is a uniform rod of mass m_i and length l_i whose centre of mass
sits at a_i = l_i / 2 and whose moment of inertia about that centre is
I_i = m_i * l_i**2 / 12.  Link 1 is hinged on the cart, link i is hinged on
the tip of link i-1.  A horizontal force u (N) acts on the cart.

Equations of motion
-------------------
Using d'Alembert's principle (equivalent to Lagrange's equations) the system
takes the classic manipulator form

        M(q) * qdd  =  f(q, qd, u)

where
    M(q)   = m_0 e_0 e_0^T + sum_i ( m_i J_i^T J_i + I_i e_i e_i^T )
    f      = u e_0 + sum_i J_i^T ( F_grav_i - m_i * Jdot_i qd ) - damping

J_i is the 2x(N+1) Jacobian of the centre of mass of link i with respect to q
and Jdot_i qd is the "centripetal" acceleration that appears because the
Jacobian rotates with the links.  Because the centre of mass of link i is

    c_i = ( x + sum_{j<i} l_j sin(theta_j) + a_i sin(theta_i),
                sum_{j<i} l_j cos(theta_j) + a_i cos(theta_i) )

the Jacobian columns are trivial:  d c_i / d x = (1, 0)  and
d c_i / d theta_j = L_ij ( cos(theta_j), -sin(theta_j) ) with L_ij = l_j for
j < i, a_i for j == i and 0 for j > i.  Differentiating once more in time
gives  Jdot_i qd = sum_j L_ij thetadot_j**2 ( -sin(theta_j), -cos(theta_j) ).

Solving the (N+1)x(N+1) linear system gives the accelerations; we integrate
them with classic 4th-order Runge-Kutta.  With N = 1 this reproduces the
Gymnasium CartPole-v1 equations exactly (see test_physics.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81


@dataclass
class CartPendulumParams:
    """Physical constants of the cart + N-link stick."""

    n_links: int = 1
    cart_mass: float = 1.0
    link_masses: np.ndarray = field(default=None)   # kg, shape (N,)
    link_lengths: np.ndarray = field(default=None)  # m,  shape (N,)
    cart_damping: float = 0.0    # viscous friction on the cart (N s/m)
    joint_damping: float = 0.0   # viscous friction on each joint (N m s/rad)
    gravity: float = GRAVITY

    def __post_init__(self) -> None:
        n = self.n_links
        if self.link_masses is None:
            # Gymnasium CartPole uses a 0.1 kg pole; keep the same total mass
            # split evenly so that longer sticks do not get heavier.
            self.link_masses = np.full(n, 0.1 / n)
        if self.link_lengths is None:
            # A 1 m stick (Gymnasium's "length" is the half-length 0.5 m).
            self.link_lengths = np.full(n, 1.0 / n)
        self.link_masses = np.asarray(self.link_masses, dtype=float)
        self.link_lengths = np.asarray(self.link_lengths, dtype=float)
        assert self.link_masses.shape == (n,) and self.link_lengths.shape == (n,)

    @property
    def com_offsets(self) -> np.ndarray:
        """Distance from the lower hinge to the centre of mass of each link."""
        return 0.5 * self.link_lengths

    @property
    def inertias(self) -> np.ndarray:
        """Moment of inertia of each uniform rod about its own centre."""
        return self.link_masses * self.link_lengths**2 / 12.0

    @property
    def total_length(self) -> float:
        return float(self.link_lengths.sum())


def mass_matrix_and_forces(
    q: np.ndarray, qd: np.ndarray, u: float, p: CartPendulumParams
) -> tuple[np.ndarray, np.ndarray]:
    """Return M(q) and f(q, qd, u) such that M qdd = f."""
    n = p.n_links
    theta = q[1:]
    thetad = qd[1:]
    sin, cos = np.sin(theta), np.cos(theta)

    # L[i, j]: lever arm of theta_j in the position of the centre of mass of
    # link i (rows = links, cols = joints); lower triangular by construction.
    L = np.tril(np.tile(p.link_lengths, (n, 1)), k=-1)
    np.fill_diagonal(L, p.com_offsets)

    M = np.zeros((n + 1, n + 1))
    f = np.zeros(n + 1)

    # Cart contributions.
    M[0, 0] += p.cart_mass
    f[0] += u - p.cart_damping * qd[0]

    for i in range(n):
        # Jacobian of the centre of mass of link i: shape (2, N+1).
        J = np.zeros((2, n + 1))
        J[0, 0] = 1.0
        J[0, 1:] = L[i] * cos
        J[1, 1:] = -L[i] * sin

        # Jdot @ qd  (centripetal term)
        Jdot_qd = np.array([
            -np.sum(L[i] * thetad**2 * sin),
            -np.sum(L[i] * thetad**2 * cos),
        ])

        m = p.link_masses[i]
        M += m * J.T @ J
        M[i + 1, i + 1] += p.inertias[i]

        gravity_force = np.array([0.0, -m * p.gravity])
        f += J.T @ (gravity_force - m * Jdot_qd)

    # Viscous damping in each hinge acts on the *relative* joint velocity.
    if p.joint_damping:
        rel = np.diff(np.concatenate(([0.0], thetad)))  # theta_i' - theta_{i-1}'
        torque = -p.joint_damping * rel                  # torque on link i from hinge i
        # Newton's third law: hinge i also pushes back on link i-1.
        f[1:] += torque
        f[1:-1] -= torque[1:]

    return M, f


def accelerations(q: np.ndarray, qd: np.ndarray, u: float, p: CartPendulumParams) -> np.ndarray:
    M, f = mass_matrix_and_forces(q, qd, u, p)
    return np.linalg.solve(M, f)


def rk4_step(
    q: np.ndarray, qd: np.ndarray, u: float, dt: float, p: CartPendulumParams
) -> tuple[np.ndarray, np.ndarray]:
    """One classic Runge-Kutta 4 step of the second-order ODE (u held constant)."""

    def deriv(state: np.ndarray) -> np.ndarray:
        n1 = len(q)
        return np.concatenate((state[n1:], accelerations(state[:n1], state[n1:], u, p)))

    s0 = np.concatenate((q, qd))
    k1 = deriv(s0)
    k2 = deriv(s0 + 0.5 * dt * k1)
    k3 = deriv(s0 + 0.5 * dt * k2)
    k4 = deriv(s0 + dt * k3)
    s1 = s0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    n1 = len(q)
    return s1[:n1], s1[n1:]


def simulate(
    q: np.ndarray, qd: np.ndarray, u: float, dt: float, substeps: int, p: CartPendulumParams
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the system by `dt` using `substeps` RK4 sub-steps with a constant force."""
    h = dt / substeps
    for _ in range(substeps):
        q, qd = rk4_step(q, qd, u, h, p)
    return q, qd


def joint_positions(q: np.ndarray, p: CartPendulumParams) -> np.ndarray:
    """(N+1, 2) array with the cart hinge followed by the tip of every link."""
    pts = np.zeros((p.n_links + 1, 2))
    pts[0] = (q[0], 0.0)
    for i in range(p.n_links):
        pts[i + 1] = pts[i] + p.link_lengths[i] * np.array([np.sin(q[i + 1]), np.cos(q[i + 1])])
    return pts


def energy(q: np.ndarray, qd: np.ndarray, p: CartPendulumParams) -> float:
    """Total mechanical energy; conserved when u = 0 and there is no damping."""
    n = p.n_links
    theta, thetad = q[1:], qd[1:]
    L = np.tril(np.tile(p.link_lengths, (n, 1)), k=-1)
    np.fill_diagonal(L, p.com_offsets)
    kinetic = 0.5 * p.cart_mass * qd[0] ** 2
    potential = 0.0
    for i in range(n):
        vx = qd[0] + np.sum(L[i] * thetad * np.cos(theta))
        vy = -np.sum(L[i] * thetad * np.sin(theta))
        kinetic += 0.5 * p.link_masses[i] * (vx**2 + vy**2) + 0.5 * p.inertias[i] * thetad[i] ** 2
        potential += p.link_masses[i] * p.gravity * np.sum(L[i] * np.cos(theta))
    return float(kinetic + potential)

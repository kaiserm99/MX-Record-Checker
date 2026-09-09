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
them with classic 4th-order Runge-Kutta.  With N = 1 and dissipation switched
off this reproduces the Gymnasium CartPole-v1 equations exactly (see
test_physics.py).

Dissipation (the "realistic" part)
----------------------------------
Real sticks live in air and turn on real bearings, so the generalised force
vector also contains:

  * Aerodynamic drag on every link.  A rod in cross-flow feels a pressure drag
    per unit length  f = -1/2 * rho * C_d * d * |v_n| * v_n  where v_n is the
    velocity component perpendicular to the rod (C_d ~ 1.1 for a cylinder,
    d the rod diameter).  Because v_n varies along the rod (the tip moves
    faster than the hinge) we integrate J(s)^T f(s) along the rod with
    3-point Gauss-Legendre quadrature.  Skin friction along the rod is
    negligible and ignored.
  * Aerodynamic drag on the cart:  -1/2 * rho * (C_d A) * |x'| * x'.
  * Bearing friction in every hinge: viscous  -b * omega_rel  plus Coulomb
    -tau_c * sign(omega_rel), acting on the relative angular velocity and
    with the equal-and-opposite reaction on the link below.
  * Rail friction on the cart: viscous  -c * x'  plus Coulomb  -mu * m g *
    sign(x')  with the normal force approximated by the total weight.

The sign() in Coulomb friction is smoothed to tanh(v / v_eps) so the ODE stays
smooth for RK4 (the standard trick; v_eps = 1 cm/s or 0.01 rad/s here).
Every dissipative term can be zeroed individually; `IDEAL` and `REALISTIC`
presets are provided.
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
    gravity: float = GRAVITY

    # --- dissipation (all zero => frictionless stick in a vacuum) ---
    air_density: float = 1.225       # kg/m^3, sea level
    link_diameter: float = 0.02      # m; a 2 cm dowel
    link_drag_coeff: float = 1.1     # C_d of a cylinder in cross-flow (Re ~ 1e3..1e5)
    cart_drag_area: float = 0.01     # C_d * A of the cart (m^2), ~ a 10 cm box
    cart_damping: float = 0.0        # viscous rail friction (N s/m)
    cart_coulomb: float = 0.01       # dry rail friction coefficient mu (dimensionless)
    joint_damping: float = 0.002     # viscous bearing friction per hinge (N m s/rad); Glueck et al. 2013 measured 0.002 on their triple pendulum
    joint_coulomb: float = 0.0001    # dry bearing friction per hinge (N m); a small sealed ball bearing
    coulomb_smoothing: float = 0.01  # v_eps for the tanh() approximation of sign()

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

    f += dissipative_forces(q, qd, p)
    return M, f


# 3-point Gauss-Legendre nodes/weights on [0, 1] for integrating along a rod.
_GL_NODES = 0.5 * (1.0 + np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]))
_GL_WEIGHTS = 0.5 * np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


def dissipative_forces(q: np.ndarray, qd: np.ndarray, p: CartPendulumParams) -> np.ndarray:
    """Generalised forces from air drag and bearing/rail friction (all <= 0 power)."""
    n = p.n_links
    theta, thetad = q[1:], qd[1:]
    sin, cos = np.sin(theta), np.cos(theta)
    f = np.zeros(n + 1)
    smooth_sign = lambda v: np.tanh(v / p.coulomb_smoothing)  # noqa: E731

    # ---- cart: rail friction (viscous + Coulomb) and air drag -------------
    xd = qd[0]
    weight = (p.cart_mass + p.link_masses.sum()) * p.gravity
    f[0] -= p.cart_damping * xd
    f[0] -= p.cart_coulomb * weight * smooth_sign(xd)
    f[0] -= 0.5 * p.air_density * p.cart_drag_area * abs(xd) * xd

    # ---- hinges: bearing friction on the relative angular velocity ---------
    rel = np.diff(np.concatenate(([0.0], thetad)))           # omega_i - omega_{i-1}
    torque = -p.joint_damping * rel - p.joint_coulomb * smooth_sign(rel)
    f[1:] += torque                                           # on link i ...
    f[1:-1] -= torque[1:]                                     # ... and its reaction on link i-1

    # ---- links: quadratic air drag integrated along each rod --------------
    k = 0.5 * p.air_density * p.link_drag_coeff * p.link_diameter   # drag per length per v^2
    if k == 0.0:
        return f
    # velocity of the lower hinge of link i (starts at the cart)
    hinge_v = np.array([xd, 0.0])
    for i in range(n):
        li = p.link_lengths[i]
        normal = np.array([cos[i], -sin[i]])                  # unit vector perpendicular to rod i
        for node, w in zip(_GL_NODES, _GL_WEIGHTS):
            s_ = node * li                                    # distance along rod
            v = hinge_v + s_ * thetad[i] * normal             # velocity of that point
            v_n = v @ normal                                  # cross-flow component
            force = -k * abs(v_n) * v_n * normal              # drag force per unit length
            # Jacobian of that point: d/dx = (1,0); d/dtheta_j = L_j * normal_j, L_j = l_j (j<i), s (j=i)
            J = np.zeros((2, n + 1))
            J[0, 0] = 1.0
            J[0, 1:i + 1] = p.link_lengths[:i] * cos[:i]
            J[1, 1:i + 1] = -p.link_lengths[:i] * sin[:i]
            J[0, i + 1] = s_ * cos[i]
            J[1, i + 1] = -s_ * sin[i]
            f += w * li * (J.T @ force)
        hinge_v = hinge_v + li * thetad[i] * normal           # tip of link i = hinge of link i+1
    return f


def ideal(**kwargs) -> CartPendulumParams:
    """Frictionless stick in a vacuum: the textbook CartPole."""
    return CartPendulumParams(
        air_density=0.0, cart_drag_area=0.0, cart_damping=0.0, cart_coulomb=0.0,
        joint_damping=0.0, joint_coulomb=0.0, **kwargs,
    )


def realistic(**kwargs) -> CartPendulumParams:
    """Air at sea level, a 2 cm dowel, ball-bearing hinges and a linear rail (the defaults)."""
    return CartPendulumParams(**kwargs)


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

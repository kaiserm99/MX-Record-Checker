"""
Sanity checks for physics.py.  Run with:  python -m pytest stick_balancer/  (or plain python).

1. With one link the accelerations must match the textbook Gymnasium
   CartPole-v1 equations (Barto, Sutton & Anderson 1983) to machine precision.
2. With u = 0 and no damping the total energy must be conserved by the RK4
   integrator for 1..4 links (this exercises the whole mass matrix).
"""

import numpy as np

from physics import CartPendulumParams, accelerations, energy, simulate


def gymnasium_cartpole_accelerations(x_dot, theta, theta_dot, force):
    """Verbatim from gymnasium/envs/classic_control/cartpole.py."""
    gravity, masscart, masspole, length = 9.81, 1.0, 0.1, 0.5  # length = half-pole
    total_mass = masscart + masspole
    polemass_length = masspole * length
    costheta, sintheta = np.cos(theta), np.sin(theta)
    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    return xacc, thetaacc


def test_matches_gymnasium_cartpole():
    p = CartPendulumParams(n_links=1, cart_mass=1.0, link_masses=[0.1], link_lengths=[1.0])
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = np.array([rng.uniform(-2, 2), rng.uniform(-np.pi, np.pi)])
        qd = rng.uniform(-5, 5, size=2)
        u = rng.uniform(-20, 20)
        ours = accelerations(q, qd, u, p)
        ref = gymnasium_cartpole_accelerations(qd[0], q[1], qd[1], u)
        assert np.allclose(ours, ref, atol=1e-10), (ours, ref)


def test_energy_conserved():
    """Energy drift must be tiny and must shrink like h**4 when the step is halved
    (that is the signature of a correct ODE integrated with RK4)."""
    for n in range(1, 5):
        p = CartPendulumParams(n_links=n)
        rng = np.random.default_rng(n)
        q0 = np.concatenate(([0.0], rng.uniform(-0.5, 0.5, size=n)))
        qd0 = np.concatenate(([0.0], rng.uniform(-1, 1, size=n)))
        e0 = energy(q0, qd0, p)
        drifts = []
        for substeps in (4, 8):  # h = 2.5 ms and 1.25 ms
            q, qd = q0.copy(), qd0.copy()
            for _ in range(300):  # 3 simulated seconds
                q, qd = simulate(q, qd, 0.0, 0.01, substeps, p)
            drifts.append(abs(energy(q, qd, p) - e0) / max(abs(e0), 1e-9))
        assert drifts[1] < 1e-4, f"{n} links: relative energy drift {drifts[1]:.2e}"
        assert drifts[1] < drifts[0] / 8, f"{n} links: drift does not shrink like RK4 {drifts}"


if __name__ == "__main__":
    test_matches_gymnasium_cartpole()
    test_energy_conserved()
    print("all physics tests passed")

"""
Sanity checks for physics.py.  Run with:  python -m pytest stick_balancer/  (or plain python).

1. With one link and no dissipation the accelerations must match the textbook
   Gymnasium CartPole-v1 equations (Barto, Sutton & Anderson 1983) to machine
   precision.
2. With u = 0 and no dissipation the total energy must be conserved by the RK4
   integrator for 1..4 links (this exercises the whole mass matrix).
3. The air-drag torque on a single rod spinning about a fixed hinge must equal
   the closed-form integral  -k * w|w| * l^4 / 4.
4. With realistic drag and friction, mechanical energy must never increase.
"""

import numpy as np

from physics import accelerations, dissipative_forces, energy, ideal, realistic, simulate


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
    p = ideal(n_links=1, cart_mass=1.0, link_masses=[0.1], link_lengths=[1.0])
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
        p = ideal(n_links=n)
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


def test_rod_drag_torque_matches_closed_form():
    """Rod of length l spinning at rate w about a hinge on a stationary cart:
    torque = -int_0^l k (s w)^2 s ds = -k w|w| l^4 / 4, with k = rho C_d d / 2."""
    p = realistic(n_links=1, link_lengths=[0.8], joint_damping=0.0, joint_coulomb=0.0)
    k = 0.5 * p.air_density * p.link_drag_coeff * p.link_diameter
    for w in (-3.0, 0.7, 5.0):
        f = dissipative_forces(np.array([0.0, 0.3]), np.array([0.0, w]), p)
        expected = -k * w * abs(w) * 0.8**4 / 4
        assert np.isclose(f[1], expected, rtol=1e-9), (f[1], expected)


def test_realistic_stick_loses_energy():
    for n in (1, 2, 3):
        p = realistic(n_links=n)
        rng = np.random.default_rng(10 + n)
        q = np.concatenate(([0.0], rng.uniform(-0.5, 0.5, size=n)))
        qd = np.concatenate(([0.5], rng.uniform(-2, 2, size=n)))
        energies = [energy(q, qd, p)]
        for _ in range(400):
            q, qd = simulate(q, qd, 0.0, 0.01, 4, p)
            energies.append(energy(q, qd, p))
        diffs = np.diff(energies)
        assert np.all(diffs <= 1e-9), f"{n} links: energy increased by {diffs.max():.2e}"
        assert energies[-1] < 0.9 * energies[0], f"{n} links: too little dissipation {energies[0]:.3f}->{energies[-1]:.3f}"


if __name__ == "__main__":
    test_matches_gymnasium_cartpole()
    test_energy_conserved()
    test_rod_drag_torque_matches_closed_form()
    test_realistic_stick_loses_energy()
    print("all physics tests passed")

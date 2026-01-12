import numpy as np
import matplotlib.pyplot as plt

K = 8.9875517923e9
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
MEV = 1e6 * E_CHARGE

def derivs(state, m, q, Q, r_soft=1e-18):
    x, y, vx, vy = state
    r2 = x*x + y*y + r_soft*r_soft
    r = np.sqrt(r2)
    r3 = r2 * r

    ax = (K * Q * q / m) * x / r3
    ay = (K * Q * q / m) * y / r3

    return np.array([vx, vy, ax, ay], dtype=float)

def rk4_step(state, dt, m, q, Q):
    k1 = derivs(state, m, q, Q)
    k2 = derivs(state + 0.5*dt*k1, m, q, Q)
    k3 = derivs(state + 0.5*dt*k2, m, q, Q)
    k4 = derivs(state + dt*k3, m, q, Q)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(m, q, Q, x_start, b, v0, dt, t_max, r_min=1e-15, r_max=5e-14):
    state = np.array([x_start, b, v0, 0.0], dtype=float)
    n = int(t_max/dt)

    xs, ys = [], []
    last_v = None

    for _ in range(n):
        x, y, vx, vy = state
        xs.append(x); ys.append(y)
        last_v = (vx, vy)

        r = np.sqrt(x*x + y*y)
        if r < r_min or r > r_max:
            break

        state = rk4_step(state, dt, m, q, Q)

    xs = np.array(xs); ys = np.array(ys)

    vx, vy = last_v
    theta = np.degrees(np.arctan2(vy, vx))
    return xs, ys, theta

if __name__ == "__main__":
    q_alpha = 2 * E_CHARGE
    Q_gold = 79 * E_CHARGE
    m_alpha = 4 * AMU

    Ek = 4 * MEV
    v0 = np.sqrt(2 * Ek / m_alpha)

    x_start = -5e-14

    b_list = [0.2e-14, 0.4e-14, 0.6e-14, 0.8e-14, 1.0e-14]

    dt = 1e-23
    t_max = 5e-20

    plt.figure()
    for b in b_list:
        xs, ys, theta = simulate(
            m_alpha, q_alpha, Q_gold,
            x_start=x_start, b=b, v0=v0,
            dt=dt, t_max=t_max,
            r_min=2e-15, r_max=8e-14
        )
        plt.plot(xs, ys, label=f"b={b:.1e} m, θ≈{theta:.1f}°")

    plt.scatter([0.0], [0.0], s=60)
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X, m")
    plt.ylabel("Y, m")
    plt.title("Rutherford scattering: alpha particle (4 MeV) on Au nucleus")
    plt.legend()
    plt.savefig("traj_rutherford.png", dpi=200, bbox_inches="tight")
    plt.show()

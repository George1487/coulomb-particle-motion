import numpy as np
import matplotlib.pyplot as plt


def f(state, m, q, Q, r_soft=1e-6):
    x, y, vx, vy = state

    r2 = x*x + y*y + r_soft*r_soft
    r = np.sqrt(r2)
    r3 = r2 * r

    ax = -(Q * q / m) * x / r3
    ay = -(Q * q / m) * y / r3

    return np.array([vx, vy, ax, ay])


def rk4_step(state, dt, m, q, Q):
    k1 = f(state, m, q, Q)
    k2 = f(state + 0.5*dt*k1, m, q, Q)
    k3 = f(state + 0.5*dt*k2, m, q, Q)
    k4 = f(state + dt*k3, m, q, Q)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def simulate(m, q, Q, x0, v0, alpha_deg, dt, t_max,
             r_stop=0.02, r_max=5.0):

    alpha = np.deg2rad(alpha_deg)

    state = np.array([
        x0,
        0.0,
        v0*np.cos(alpha),
        v0*np.sin(alpha)
    ])

    xs, ys = [], []

    for _ in range(int(t_max/dt)):
        x, y, vx, vy = state
        xs.append(x)
        ys.append(y)

        r = np.sqrt(x*x + y*y)
        if r < r_stop or r > r_max:
            break

        state = rk4_step(state, dt, m, q, Q)

    return np.array(xs), np.array(ys)


def plot_case(title, m, q, Q, x0, v0, angles, dt, t_max, save_name):
    plt.figure()

    for a in angles:
        xs, ys = simulate(m, q, Q, x0, v0, a, dt, t_max)
        plt.plot(xs, ys, label=f"α = {a}°")

    plt.scatter([0], [0], s=60)

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X, m")
    plt.ylabel("Y, m")
    plt.title(title)
    plt.legend()
    
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    
    plt.show()


if __name__ == "__main__":

    m = 0.001
    q_abs = 1e-2
    Q_abs = 5e-2
    x0 = -1.0
    v0 = 0.1
    angles = [5, 15, 30, 45]

    plot_case(
        "Same sign (repulsion)",
        m, +q_abs, +Q_abs, x0, v0, angles,
        dt=1e-3, t_max=30, save_name="traj_repulsion.png"
    )

    plot_case(
        "Opposite sign (attraction)",
        m, -q_abs, +Q_abs, x0, v0, angles,
        dt=1e-3, t_max=30, save_name="traj_attraction.png"
    )

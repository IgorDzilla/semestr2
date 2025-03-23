from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import proto_pylab as pl

TOLERANCE = 1e-4
N_0 = 0.42  # random.uniform(0.2, 0.8)
PHI_0 = 0
START_TIME = 0
END_TIME = 10

print("n_0:\t", N_0)


def eqs_system(t, vars):
    n, phi = vars
    return [(1 - n**2) * np.sin(phi), (1 / n) * (1 - 3 * n**2) * np.cos(phi)]


def get_extremums(X, Y):
    vals: list[tuple] = [(x, y) for x, y in zip(X, Y)]
    extremums: list[tuple] = []

    derr_sign = 1 if vals[1][1] - vals[0][1] > 0 else -1

    for i in range(0, len(vals) - 1):
        itter_sign = 1 if vals[i + 1][1] - vals[i][1] > 0 else -1

        if itter_sign == derr_sign:
            continue
        else:
            extremums.append(vals[i])
            derr_sign = itter_sign

    return extremums


def get_function_period(extremums: list[tuple]):
    sorted_extrs = sorted(extremums, key=lambda x: x[1])
    print("SORTED EXTREMUMS")
    for i in sorted_extrs:
        print(i)

    return sorted_extrs[1][0] - sorted_extrs[0][0]


time = np.linspace(START_TIME, END_TIME, 100000, endpoint=True)
initial_conditions = [N_0, 0]
solution = solve_ivp(
    eqs_system,
    t_span=[START_TIME, END_TIME],
    y0=initial_conditions,
    method="BDF",
    t_eval=time,
)

extrs_n_t = get_extremums(solution.t, solution.y[0])
x_n_t_extrs = [x for x, _ in extrs_n_t]
y_n_t_extrs = [y for _, y in extrs_n_t]

extrs_phi_t = get_extremums(solution.t, solution.y[1])
x_phi_t_extrs = [x for x, _ in extrs_phi_t]
y_phi_t_extrs = [y for _, y in extrs_phi_t]

period_n_t = get_function_period(get_extremums(solution.t, solution.y[0]))
period_phi_t = get_function_period(get_extremums(solution.t, solution.y[1]))

print("\n\n---FUNCTION PERIODS---")
print("n(t) FUNCTION PERIOD:\t", period_n_t)
print("φ(t) FUNCTION PERIOD:\t", period_phi_t)


pane = pl.Pane(1, 2, spacing="constrained", sharex=True)


def n_vs_t_plot(ax: plt.Axes) -> None:
    global solution, x_n_t_extrs, y_n_t_extrs
    ax.plot(solution.t, solution.y[0])
    ax.plot(x_n_t_extrs, y_n_t_extrs, "o", label="Экстремумы")
    ax.set_title(r"$n(t)$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$n(t)$")


def phi_vs_t_plot(ax: plt.Axes) -> None:
    global solution, x_phi_t_extrs, y_phi_t_extrs, period_phi_t
    ax.plot(solution.t, solution.y[1])
    ax.plot(x_phi_t_extrs, y_phi_t_extrs, "o", label="Экстремумы")
    ax.set_title(r"$\varphi(t)$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\varphi(t)$")


pane.add_figure(0, 0, n_vs_t_plot)
pane.add_figure(0, 1, phi_vs_t_plot)
pane.show()

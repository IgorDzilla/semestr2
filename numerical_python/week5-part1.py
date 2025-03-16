from scipy.integrate import solve_ivp
import proto_pylab as lab
import matplotlib.pyplot as plt
import numpy as np


def analytic_solution(mu, lambda_, A):
    def eq(t):
        return -(A / (lambda_ - mu)) * np.exp(-lambda_ * t) + (
            A / (lambda_ - mu)
        ) * np.exp(-mu * t)

    return eq


def analytic_der_solution(mu, lambda_, A):
    def eq(t):
        return (lambda_ * A / (lambda_ - mu)) * np.exp(-lambda_ * t) - (
            mu * A / (lambda_ - mu)
        ) * np.exp(-mu * t)

    return eq


def der_equation(mu, lambda_, A):
    def eq(t, y):
        return -lambda_ * y + A * np.exp(-mu * t)

    return eq


def eq_1_plot_y_vs_t(ax: plt.Axes):
    time_range = np.linspace(0, 100, num=1000, endpoint=True)
    y = analytic_solution(1, 2, 3)
    ax.plot(time_range, y(time_range), label="Аналитическое")

    dy_dt = der_equation(1, 2, 3)
    sol = solve_ivp(dy_dt, t_span=[0, 100], y0=[0], t_eval=time_range)
    # Use sol.y[0] to get solution values
    ax.plot(time_range, sol.y[0], label="Численное")

    ax.set_title(r"$\mu = 1,\;\lambda = 2,\;A = 3$")
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")


def eq_2_plot_y_vs_t(ax: plt.Axes):
    time_range = np.linspace(0, 100, num=1000, endpoint=True)
    y = analytic_solution(44, 69, 47)
    ax.plot(time_range, y(time_range), label="Аналитическое")

    dy_dt = der_equation(44, 69, 47)
    sol = solve_ivp(dy_dt, t_span=[0, 100], y0=[0], t_eval=time_range)
    # Use sol.y[0] to get solution values
    ax.plot(time_range, sol.y[0], label="Численное")

    ax.set_title(r"$\mu = 44,\;\lambda = 69,\;A = 47$")
    ax.set_ylabel("$y$")
    ax.set_xlabel("$t$")


def eq_1_plot_y_vs_ydot(ax: plt.Axes):
    time_range = np.linspace(0, 100, num=1000, endpoint=True)
    y = analytic_solution(44, 69, 47)
    y_dot = analytic_der_solution(44, 69, 47)
    ax.plot(y(time_range), y_dot(time_range), label="Аналитическое")

    dy_dt = der_equation(1, 2, 3)
    sol = solve_ivp(dy_dt, t_span=[0, 100], y0=[0], t_eval=time_range)
    # Use sol.y[0] to get solution values
    ax.plot(sol.y[0], dy_dt(sol.t, sol.y[0]), label="Численное")

    ax.legend()
    ax.set_ylabel("$\dot{y}$")
    ax.set_xlabel("$y$")


def eq_1_plot_y_vs_ydot(ax: plt.Axes):
    time_range = np.linspace(0, 100, num=1000, endpoint=True)
    y = analytic_solution(1, 2, 3)
    y_dot = analytic_der_solution(1, 2, 3)
    ax.plot(y(time_range), y_dot(time_range), label="Аналитическое")

    dy_dt = der_equation(1, 2, 3)
    sol = solve_ivp(dy_dt, t_span=[0, 100], y0=[0], t_eval=time_range)
    # Use sol.y[0] to get solution values
    ax.plot(sol.y[0], dy_dt(sol.t, sol.y[0]), label="Численное")

    ax.legend()
    ax.set_ylabel("$\dot{y}$")
    ax.set_xlabel("$y$")


def eq_2_plot_y_vs_ydot(ax: plt.Axes):
    time_range = np.linspace(0, 100, num=1000, endpoint=True)
    y = analytic_solution(44, 69, 47)
    y_dot = analytic_der_solution(44, 69, 47)
    ax.plot(y(time_range), y_dot(time_range), label="Аналитическое")

    dy_dt = der_equation(44, 69, 47)
    sol = solve_ivp(dy_dt, t_span=[0, 100], y0=[0], t_eval=time_range)
    # Use sol.y[0] to get solution values
    ax.plot(sol.y[0], dy_dt(sol.t, sol.y[0]), label="Численное")

    ax.legend()
    ax.set_ylabel("$\dot{y}$")
    ax.set_xlabel("$y$")


pane = lab.Pane(2, 2, spacing="constrained")
pane.add_figure(0, 0, eq_1_plot_y_vs_t)
pane.add_figure(0, 1, eq_2_plot_y_vs_t)
pane.add_figure(1, 0, eq_1_plot_y_vs_ydot)
pane.add_figure(1, 1, eq_2_plot_y_vs_ydot)
pane.show()

import numpy as np
import matplotlib.pyplot as plt
import proto_pylab as lab
from scipy.integrate import solve_ivp
from scipy import constants

R0 = 6_371_000
v0 = 100  # м/с
h = 20  # м
alpha = 40  # градусы
time_range = np.linspace(0, 100, 1000, endpoint=True)


def dydt_eq_gen(v0, alpha, gconst=True):
    def eq(t, y):
        if gconst:
            return v0 * np.sin(alpha) - constants.g * t
        else:
            return v0 * np.sin(alpha) - (R0 / (R0 + y)) ** 2 * t

    return eq


def dxdt(v0, alpha):
    def eq(t, y):
        return v0 * np.cos(alpha)

    return eq


def analytic_y_vs_x(ax: plt.Axes):
    global time_range
    x = v0 * np.cos(alpha) * time_range
    y = [h + v0 * np.sin(alpha) * t - constants.g * t**2 / 2 for t in time_range]
    ax.plot(x, y)
    ax.set_title("Аналитическое решение")
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")


def analytic_y_der_vs_x_der(ax: plt.Axes):
    global time_range
    v_x = [v0 * np.cos(alpha)] * len(time_range)
    v_y = [v0 * np.sin(alpha) - constants.g * t for t in time_range]
    ax.plot(v_x, v_y)
    ax.set_title("Аналитическое решение")
    ax.set_ylabel(r"$\dot{y}$")
    ax.set_xlabel(r"$\dot{x}$")


def numerical_y_vs_x(ax: plt.Axes):
    global time_range
    dy_dt = dydt_eq_gen(v0, alpha, gconst=True)
    dx_dt = dxdt(v0, alpha)
    y_t = solve_ivp(dy_dt, t_span=[0, 100], y0=[h], t_eval=time_range)
    x_t = solve_ivp(dx_dt, t_span=[0, 100], y0=[0], t_eval=time_range)
    ax.plot(x_t.y[0], y_t.y[0])

    ax.set_title("Численное решение")
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")


def comparison(ax: plt.Axes):
    global time_range

    dy_dt = dydt_eq_gen(v0, alpha, gconst=True)
    dx_dt = dxdt(v0, alpha)
    y_t = solve_ivp(dy_dt, t_span=[0, 100], y0=[h], t_eval=time_range)
    x_t = solve_ivp(dx_dt, t_span=[0, 100], y0=[0], t_eval=time_range)
    ax.plot(x_t.y[0], y_t.y[0], label="g = const")

    dy_dt = dydt_eq_gen(v0, alpha, gconst=False)
    y_t = solve_ivp(dy_dt, t_span=[0, 100], y0=[h], t_eval=time_range)
    ax.plot(x_t.y[0], y_t.y[0], label=r"$g(y) = (\frac{R0}{R0 + y})^2$")

    ax.set_title("Сравнение приближений")
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")


pane = lab.Pane(2, 2, spacing="constrained")
pane.add_figure(0, 0, analytic_y_vs_x)
pane.add_figure(1, 0, analytic_y_der_vs_x_der)
pane.add_figure(0, 1, numerical_y_vs_x)
pane.add_figure(1, 1, comparison)
pane.show()

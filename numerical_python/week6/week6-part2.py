import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft

# Параметры потенциала Морза
U0 = 1.0  # Глубина потенциальной ямы
alpha = 1.0  # Крутизна потенциала
m = 1.0  # Масса частицы

# Функция потенциала Морза


def morse_potential(x):
    return U0 * np.exp(-2 * alpha * x) - 2 * U0 * np.exp(-alpha * x)


# Уравнение движения dx/dt = v, dv/dt = -dU/dx


def equations(t, y):
    x, v = y
    force = -2 * alpha * U0 * np.exp(-2 * alpha * x) + 2 * alpha * U0 * np.exp(
        -alpha * x
    )
    return [v, force / m]


# Численное решение уравнения движения
E = -0.2  # Энергия частицы (должна быть < 0)
x0 = np.log(-E / (2 * U0)) / alpha  # Найдем начальное положение
Tmax = 20  # Время интегрирования
time = np.linspace(0, Tmax, 100000)
sol = solve_ivp(equations, [0, Tmax], [x0, 0], t_eval=time, method="BDF")

# Построение графиков
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# График потенциала Морза
x_vals = np.linspace(-1, 2, 400)
ax[0].plot(x_vals, morse_potential(x_vals), label="U(x)")
ax[0].set_xlabel("x")
ax[0].set_ylabel("U(x)")
ax[0].legend()
ax[0].grid()

# График траектории x(t)
ax[1].plot(sol.t, sol.y[0], label="x(t)")
ax[1].set_xlabel("t")
ax[1].set_ylabel("x")
ax[1].legend()
ax[1].grid()

plt.show()

# Фурье-анализ
fourier_coeffs = fft(sol.y[0])
n_freqs = 10  # Количество гармоник
print("Амплитуды первых 10 гармоник:", np.abs(fourier_coeffs))

import proto_pylab
import numpy as np
import matplotlib.pyplot as plt

THERMAL_CONDUCTIVITY = {"ferrum": 449, "cuprum": 385}  # Дж / (кг * K)
THERMAL_CONDUCTIVITY_COEFF = {"ferrum": 92, "cuprum": 401}
A = 1e-4
H = 1e3

X = np.linspace(-100, 100, endpoint=True, num=100)


def thetha_function(x, t, Cp, D):
    return (
        H
        / (Cp * A * np.sqrt(D * t) * np.sqrt(4 * np.pi))
        * np.exp(-(x**2 / (4 * D * t)))
    )


def function_builder(t: float, material: str):
    def plot_build(ax: plt.Axes) -> None:
        global X
        Y = [
            thetha_function(
                i,
                t,
                THERMAL_CONDUCTIVITY[material],
                THERMAL_CONDUCTIVITY_COEFF[material],
            )
            for i in X
        ]
        color = ""
        plot_name = ""
        match material:
            case "ferrum":
                color = "r--"
                plot_name = "Железо, t=" + str(t)
            case "cuprum":
                color = "g"
                plot_name = "Медь, t=" + str(t)

        ax.plot(X, Y, color)
        ax.set_ylabel(r"$\theta(x, t)$")
        ax.set_xlabel(r"$\textbf{x}$")
        ax.set_title(plot_name)

    return plot_build


pane = proto_pylab.Pane(nrows=2, ncols=3, figsize=(16, 12), spacing="constrained")
pane.add_figure(0, 0, function_builder(1e-3, "ferrum"))
pane.add_figure(0, 1, function_builder(0.1, "ferrum"))
pane.add_figure(0, 2, function_builder(1, "ferrum"))
pane.add_figure(1, 0, function_builder(1e-3, "cuprum"))
pane.add_figure(1, 1, function_builder(0.1, "cuprum"))
pane.add_figure(1, 2, function_builder(1, "cuprum"))
pane.set_title("Распределение температуры в материале")
pane.save("gpraph.png")
pane.save("gpraph.pdf")
pane.show()

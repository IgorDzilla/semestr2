import matplotlib.pyplot as plt
from typing import Callable, Tuple

plt.rc("text", usetex=True)
# plt.rc("text.latex", unicode=True)
plt.rc("text.latex", preamble=r"\usepackage[utf8]{inputenc}")
plt.rc("text.latex", preamble=r"\usepackage[russian]{babel}")
plt.rcParams["font.family"] = "Arial"

LEGEND_PROPERTIES = {"size": 12}
TITLE_PROPERTIES = {"fontsize": 16, "fontweight": "bold"}
AXLABEL_PROPERTIES = {"fontsize": 14}


class Pane:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Tuple[float, float] = (6, 4),
        spacing: str = "tight",
        sharex: bool = False,
        sharey: bool = False,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.fig, self.axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey
        )
        self.figures: list[Figure] = []  # Store Figure objects
        self._apply_spacing(spacing)

    def _apply_spacing(self, spacing: str) -> None:
        if spacing == "tight":
            self.fig.tight_layout(rect=[0, 0.02, 1, 0.9])
        elif spacing == "constrained":
            self.fig.set_constrained_layout(True)

        self.fig.subplots_adjust(wspace=0.4, hspace=0.4)

    def add_figure(
        self, row: int, col: int, plot_func: Callable[[plt.Axes], None]
    ) -> None:
        ax = (
            self.axes[row, col]
            if self.nrows > 1 and self.ncols > 1
            else self.axes[max(row, col)]
        )
        plot_func(ax)
        self.figures.append(Figure(ax))

    def set_title(self, title: str) -> None:
        self.fig.suptitle(title, fontsize=20, fontweight="bold")

    def show(self) -> None:
        plt.show()

    def save(self, filename: str, dpi=300) -> None:
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"График сохранен как: {filename}")


class Figure:
    def __init__(self, ax: plt.Axes) -> None:
        self.ax = ax
        self.auto_format()

    def auto_format(self) -> None:
        # автоформатирование
        if len(self.ax.lines) > 1:
            self.ax.legend(prop=LEGEND_PROPERTIES)

        self.ax.set_title(self.ax.get_title(), fontdict=TITLE_PROPERTIES)
        self.ax.set_ylabel(self.ax.get_ylabel(), fontdict=AXLABEL_PROPERTIES)
        self.ax.set_xlabel(self.ax.get_xlabel(), fontdict=AXLABEL_PROPERTIES)

    def set_title(self, title: str) -> None:
        self.ax.set_title(title)

    def set_labels(self, xlabel: str, ylabel: str) -> None:
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)


if __name__ == "__main__":
    """Пример использования"""

    def example_plot(ax: plt.Axes) -> None:
        ax.plot([1, 2, 3], [4, 5, 6], marker="o")
        ax.set_title(r"\textbf{y}")

    pane = Pane(nrows=2, ncols=2, figsize=(8, 6), spacing="constrained")
    pane.add_figure(0, 0, example_plot)
    pane.add_figure(0, 1, example_plot)
    pane.add_figure(1, 0, example_plot)
    pane.add_figure(1, 1, example_plot)
    pane.show()

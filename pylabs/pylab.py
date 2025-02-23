import matplotlib.pyplot as plt

LAYOUTS = ["auto", "horizontal", "vertical"]

class Figure:
    def __init__(self, row=None, col=None, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.row = row
        self.col = col


class Pane:
    def __init__(self, figsize=(6, 4), spacing="tight", sharex=False, sharey=False):
        self.nrows = 0
        self.ncols = 0
        self.figures: list[Figure] = []

    def add_figure(self, row=None, col=None, *args, **kwargs):
        self.figures.append(Figure(row, col, args, kwargs))

    def set_layout(self, layout="auto"):
        if layout not in LAYOUTS:
            raise ValueError(f"Uknown layout type \"{layout}\"")

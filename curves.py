from matplotlib.figure import Figure
import numpy as np

# inheriting from matplotlib Figure
class Curve(Figure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ax = self.add_axes([0.08, 0.1, 0.86, 0.84])
        self.ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
        self.ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        self.ax.margins(x=0, y=0)

    def plot(self, x, y, **kwargs):
        return self.ax.plot(x, y, **kwargs)

    def get_color(self, idx: int) -> str:
        lines = self.ax.get_lines()
        return lines[idx].get_c()

class PC(Curve):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Predictiveness Curve"
        self.ax.set_title(self.name)
        self.ax.set_ylabel("Activity probability")
        self.ax.set_xlabel("Quantile")


class ROC(Curve):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Receiver Operating Characteristic"
        self.ax.set_title("ROC curve")
        self.ax.set_ylabel("Sensitivity")
        self.ax.set_xlabel("1 - specificity")
        self.ax.plot(
            np.linspace(0, 1),
            np.linspace(0, 1),
            linestyle="dashed",
            label="Random",
            color="red",
        )
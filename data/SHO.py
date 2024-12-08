from sklearn.discriminant_analysis import StandardScaler
from lib.datasets.datasets import SeqDataset
from lib.utils import *
from lib.data import Data, DataConfig
from sklearn.model_selection import KFold
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from lib.utils import _grad


@dataclass(kw_only=True)
class SHOConfig(DataConfig):
    ic: tuple = (1, 0)
    w_0: float = 1
    gamma: float = 0.1
    t_span: tuple = (0, 10)
    points: int = 1000
    fold_shuffle: bool = False


class SHO(Data):
    def __init__(self, c: DataConfig):
        super().__init__()
        self.save_config(c)

        t_eval = np.linspace(c.t_span[0], c.t_span[1], c.points)
        solution = solve_ivp(self.fo_eq, c.t_span, self.ic, t_eval=t_eval)

        Y = solution.y[0, :].reshape(-1, 1)
        X = t_eval.reshape(-1, 1)

        self.set_data(X, Y)
        self.set_folds(KFold(n_splits=6, shuffle=c.fold_shuffle))

    def fo_eq(self, t, Y):
        y, y_t = Y  # Y = [y, y_t]
        y_tt = -2 * self.gamma * y_t - self.w_0**2 * y
        return [y_t, y_tt]

    def eq(self, model, t_eval):
        y = model.forward(t_eval)
        y_t = _grad(y, t_eval)
        y_tt = _grad(y_t, t_eval)
        errs = y_tt - 2 * self.gamma * y_t - self.w_0**2 * y
        return errs

    def draw_points(self, points=None):
        """
        Plots the solution of the equation and a batch of points from self.sample_batch.
        """

        plt.figure()

        # Extract the actual solution (t_eval and Y)
        t_eval, Y = self.data

        # Plot the actual solution
        plt.plot(
            numpy_flatten(t_eval),
            numpy_flatten(Y),
            label="Actual Solution",
            color="blue",
        )

        if not points:
            # Get the batch points
            x, y = self.sample_batch()
        else:
            # Use provided points
            x, y = points

        # Ensure that the batch points' x values match the range of t_eval
        # You can also adjust the axes to make sure the full range is visible
        plt.xlim(
            [
                min(numpy_flatten(t_eval).min(), numpy_flatten(x).min()),
                max(numpy_flatten(t_eval).max(), numpy_flatten(x).max()),
            ]
        )

        # Scatter the batch points on the same plot
        plt.scatter(
            numpy_flatten(x),  # x values from batch points
            numpy_flatten(y),  # y values from batch points
            color="orange",
            label="Batch Points",
            zorder=5,
            s=1,
        )

        # Add labels, title, and legend
        plt.title("SHO Solution and Batch Points")
        plt.xlabel("Time (t)")
        plt.ylabel("y(t)")
        plt.legend()

        # Show the plot with grid enabled
        plt.grid(True)
        plt.show()

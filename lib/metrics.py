from abc import abstractmethod
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
from .Utils import *
import IPython.display

from enum import Enum


# Define the Enum with Train, Val, and Both values
class TrackModes(Enum):
    Val = False
    Train = True
    Both = 2


from torcheval.metrics.metric import Metric
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

# pyre-fixme[24]: Generic type `Metric` expects 1 type parameter.
TSelf = TypeVar("TSelf", bound="Metric")
TComputeReturn = TypeVar("TComputeReturn")
# pyre-ignore[33]: Flexible key data type for dictionary
TState = Union[torch.Tensor, List[torch.Tensor], Dict[Any, torch.Tensor], int, float]


# operator.add, operator.truediv?
class MeanMetric(Metric):
    def __init__(
        self,
        label,
        statistic: Callable[[object, object], object],
        compute: Callable[[object, object], object] = lambda total, count: total
        / count,
        reduce: Callable[[object, object], object] = lambda total, stat: total + stat,
        device: Optional[torch.device] = None,
        train: bool = False,
    ):
        self.save_attr()
        self.reset()
        self.label = "train_" + label if train else label

    @torch.inference_mode()
    def compute(self):
        return self.transform(self._total, self._count)

    @torch.inference_mode()
    def update(self, *args):
        self._total = self.reduce(self._total, self.statistic(*args))
        self._count += 1

    @torch.inference_mode()
    def reset(self):
        self._total = 0
        self._count = 0

    @torch.inference_mode()
    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        for metric in metrics:
            self._total = self.reduce(self._total, metric._total)
            self._count += metric._count


def from_tem(torcheval_metric, label, pred_funs=lambda x: x, train=False):
    class _subclass(torcheval_metric):
        def update(self, *args, **kwargs):
            super().update(
                *(f(a) for f, a in zip(self._pred_funs(len(args)), args)), **kwargs
            )

        def _pred_funs(n):
            return pred_funs + [lambda x: x] * (n - len(pred_funs))

    c = _subclass()
    c.label = "train_" + label if train else label
    c.train = train

    return c


def set_trackmodes(*columns, train=TrackModes.Both):
    if train == TrackModes.Both:
        new = []
        for c in columns:
            d = c.deepcopy()
            d.train = not c.train
            new.append(d)
        columns.extend(new)
    elif train == TrackModes.Val:
        for c in columns:
            c.train = False
    else:
        for c in columns:
            c.train = True
    return columns


# For simplicity, only plotted metrics are kept
class MetricsFrame(Base):
    def __init__(
        self,
        columns: List[Metric],
        compute_every=1,
        plot_every=5,
        index_fn=lambda batch_num, batches_per_epoch: np.float16(
            batch_num / batches_per_epoch
        ),
        plot_on_record=False,
        name=None,
        xlabel="epoch",
        board=None,
    ):
        """_summary_

        Args:
            columns (List[Metric]): Instance of torcheval.metrics.metric.Metric with a label property set
            batch_per_epoch (np.float16): _description_
            compute_every (int, optional): _description_. Defaults to 1.
            plot_every (int, optional): _description_. Defaults to 5. Updates between plots. 0 for never.
            index_fn (_type_, optional): _description_. Defaults to lambdabatch_num.
            plot_on_record (bool, optional): _description_. Defaults to False.
            name (_type_, optional): _description_. Defaults to None.
        """
        self.save_attr()
        self._count = 0

        self.dict = {col.label: [] for col in columns}
        self.dict[xlabel] = []  # Assuming xlabel is a variable defined elsewhere

        if self.name is None:
            self.name = ", ".join([col.label for col in columns])

    def flush(self, index):
        """Computes and resets all columns

        Args:
            index (int): index to associate with row
        """
        for c in self.columns:
            self.dict[c.label].append(c.compute())
            self.dict[self.xlabel].append(index)
            c.reset()

    # def _compute(self, train=TrackModes.Both):
    #     """Computes columns

    #     Args:
    #         index (int): index to associate with row
    #     """

    #     if train == True:
    #         (c.compute() for c in filter(lambda x: x.train == train, self.columns))
    #     elif train == False:
    #         (c.compute() for c in filter(lambda x: x.train == train, self.columns))
    #     else:
    #         (c.compute() for c in self.columns)

    # todo: compute_batch, probably not worth it

    def update(self, outputs, Y, train=True, batch_num=None, batches_per_epoch=None):
        for c in self.columns:
            if c.train == train:
                c.update(outputs, Y)
        self._count += 1

        index = (
            self._count
            if batch_num is None
            else self.index_fn(batch_num, batches_per_epoch)
        )  # maybe should not be optional

        if self.flush_every != 0:
            div, mod = divmod(self._count, self.flush_every)
            if mod == 0:
                self.flush(index)
                if div % self.plot_every == 0:
                    self.plot()

    def _init_plot(self):
        if self.board is None:
            self.board = ProgressBoard(xlabel=self.xlabel)

    def plot(self, df=False):
        """Plots"""
        self._init_plot()
        if df == True:
            self.board.draw(self.df, self.name)
        else:
            logging.info(f"Displaying dictionary of {self.name}")
            self.board.draw(self.dict, self.name)

    def record(self):
        if getattr(self, "df") is None:
            self.df = pl.DataFrame(self.dict)

        new_df = pl.DataFrame(self.dict)
        self.df = self.df.extend(new_df)
        self.dict.clear()
        if self.plot_on_record:
            self.plot(plot_df=True)

    def reset(self):
        for c in self.columns:
            c.reset()


class ProgressBoard(Base):
    def __init__(
        self,
        width=800,
        height=600,
        xlim=None,
        ylim=None,
        xlabel="X",
        ylabel="Y",
        xscale="linear",
        yscale="linear",
        labels=None,
        display=True,
        update_every=5,
        interactive=None,
        save=False,
    ):
        # Initialize parameters and data structures
        self.save_attr()

        self.fig, self.ax = plt.subplots(
            figsize=(width / 100, height / 100)
        )  # Adjust size for Matplotlib

        if not isinstance(interactive, bool):
            self.interactive = is_notebook()
        if self.interactive:
            plt.ion()
            self.dh = IPython.display.display(self.fig, display_id=True)

        assert update_every >= 2

        if labels:
            self.schema = pl.Schema(
                [(xlabel, pl.Float64), (ylabel, pl.Float64), ("Label", pl.Enum(labels))]
            )

        else:
            self.schema = pl.Schema(
                [(xlabel, pl.Float64), (ylabel, pl.Float64), ("Label", pl.String())]
            )

        self.data = pl.DataFrame(
            schema=self.schema, data=[]
        )  # To store processed data (mean of every n)

        self.raw_points = (
            OrderedDict()
        )  # OrderedDict for each label, storing raw points as (n, 2) arrays

        # Set log scaling based on the provided xscale and yscale
        if xscale == "log":
            self.ax.set_xscale("log")
        if yscale == "log":
            self.ax.set_yscale("log")

        # legend_labels = []
        # for orbit in self.data['Label'].unique():
        #     legend_labels.append(f"{orbit}")

        # handles, _ = self.ax.get_legend_handles_labels()
        # self.ax.legend(handles, legend_labels, loc="lower left", bbox_to_anchor=(1.01, 0.29), title="Orbit")
        # plt.close()

    def draw(self, data):
        if not self.display:
            return
        self.ax.clear()

        if isinstance(data, pl.DataFrame):
            for col in data.columns:
                if col != self.xabel:
                    sns.lineplot(x=self.xlabel, y=data[col], label=col, data=data)
        elif isinstance(data, Dict):
            sns.lineplot(data=data)
            plt.xticks(labels=data[self.xlabel])
        else:
            raise TypeError

        self.iupdate()

    def draw_points(self, x, y, label, every_n=1, force=False, clear=False):
        """Update plot with new points (arrays) and redraw. todo: Choice of dictionary and df storage as well as batching methods."""

        # Todo: clarify aggregation logic

        raw = self.raw_points.setdefault(label, np.empty((0, 2)))

        new_points = np.column_stack(
            (x, y)
        )  # Create a (n, 2) array with x and y columns
        raw = np.vstack((raw, new_points))

        if len(raw) < every_n * self.update_every and not force:
            self.raw_points[label] = raw
            return

        def mean(chunk):
            # mean = np.mean(chunk, 0)
            # return mean[0], mean[1]
            return chunk[-1][0], np.mean(chunk[:, 1], 0)

        # Smooth by taking every_n points
        new_rows = []
        end = len(raw) - every_n  # don't smooth the final value(s)

        for i in range(0, end, every_n):
            chunk = raw[i : i + every_n]
            x, y = mean(chunk)
            # Add the new row to the new_rows list
            new_rows.append((x, y, label))

        raw = raw[end + every_n :]
        if force:
            # for row in raw:
            #     new_rows.append((row[0], row[1], label))
            if len(raw) > 0:
                new_rows.append((*mean(chunk), label))
            raw = np.empty((0, 2))

        self.raw_points[label] = raw

        new_df = pl.DataFrame(new_rows, schema=self.schema, orient="row")
        self.data = self.data.extend(new_df)

        if not self.display:
            return

        # # X-axis values (common for all lines)
        # x_values = [0, 1, 2, 3, 4]

        # # Plot using the dictionary directly
        # sns.lineplot(data=data, palette='tab10')

        # # Setting x-values explicitly
        # plt.xticks(ticks=range(len(x_values)), labels=x_values)
        # Redraw the plot

        if True:
            if clear:
                self.ax.clear()
                sns.scatterplot(x=self.xlabel, y=self.ylabel, hue="Label", data=new_df)
            else:
                self.ax.clear()
                sns.scatterplot(
                    x=self.xlabel, y=self.ylabel, hue="Label", data=self.data
                )

        else:
            for label in self.labels:
                label_data = self.data.filter(pl.col("Label") == label)
                sns.lineplot(
                    x="X",
                    y="Y",
                    data=label_data,
                    ax=self.ax,
                    label=label,
                    linestyle=self.line_styles[label],
                    color=self.line_colors[label],
                )

        self.iupdate()

    def flush(self):
        for key in self.raw_points.keys():
            self.draw([], [], key, force=True)
        if self.save:
            plt.savefig("updated_plot.png")

    def iupdate(self):
        if self.interactive:
            self.dh.update(self.fig)

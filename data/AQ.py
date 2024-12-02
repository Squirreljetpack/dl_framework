from sklearn.discriminant_analysis import StandardScaler
from lib.datasets.datasets import SeqDataset
from lib.utils import *
from lib.data import *
import pandas as pd
from sklearn.model_selection import KFold

class AQ(DataModule):
    def __init__(self, y_len=1, seq_len=9, **kwargs):
        super().__init__(**kwargs)
        self.save_attr()

        target = "C6H6(GT)"

        # Read csv into Polars DataFrame <- Works but somehow cannot train
        # df = pl.read_csv(
        #     "resources/AirQualityUCI.csv",
        #     null_values=["-200"],
        #     separator=";",
        #     has_header=True,
        #     decimal_comma=True,
        # )

        # not_null_cols = filter(lambda x: x.null_count() != df.height, df)
        # not_null_col_names = map(lambda x: x.name, not_null_cols)
        # df = df.select(not_null_col_names)

        # df.glimpse()

        # # Drop Date and Time columns
        # df = df.drop(["Date", "Time"])  # "NMHC(GT)"

        # df = df.interpolate().drop_nulls()  # drop uninterpolated row with null

        # X = df.drop(target)
        # y = df.select(target)

        df = pd.read_excel("resources/AirQualityUCI.xlsx")
        df = df.drop(["Date", "Time"], axis=1)

        df = df.replace(-200, np.nan)  # Replace -200 with NaN
        df = df.interpolate(method="linear")  # Interpolate missing values
        X, y = df.drop(target, axis=1), df[[target]]

        self.data = (X, y)
        self.set_folds(KFold(n_splits=6))

    def _to_dataset(self, data):
        tensors = tuple(to_tensors(x) for x in data)
        return SeqDataset(tensors, self.seq_len, y_len=self.y_len)

    def _fit_transforms(self, tensors):
        def transform(t):
            q = StandardScaler()
            q.fit_transform(t)
            return lambda v: q.transform(v)

        return tuple(transform(t) for t in tensors)

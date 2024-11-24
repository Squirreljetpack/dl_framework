from sklearn.discriminant_analysis import StandardScaler
from torch.nn import functional as F
from lib.Utils import *
from lib.data import *
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split


class AQ(DataModule):
    def __init__(self, y_len=1, seq_len=9, batch_size=32, num_workers=1):  # 2 days
        self.save_attr()

        target = "C6H6(GT)"

        # Read csv into Polars DataFrame
        # df = pl.read_csv(
        #     "air+quality/AirQualityUCI.csv",
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

        df = pd.read_excel("air+quality/AirQualityUCI.xlsx")
        df = df.drop(["Date", "Time"], axis=1)

        df = df.replace(-200, np.nan)  # Replace -200 with NaN
        df = df.interpolate(method="linear")  # Interpolate missing values
        X, y = df.drop(target, axis=1), df[[target]]

        self.data = (X, y)
        self.configure_folds("KFold", n_splits=6)

        super().__init__()

    def _dataset(self, tensors):
        return SeqDataset(tensors, self.seq_len, y_len=self.y_len)

    def fit_transforms(self, tensors):
        def transform(t):
            q = StandardScaler()
            q.fit_transform(t)
            return lambda v: q.transform(v)

        return tuple(transform(t) for t in tensors)

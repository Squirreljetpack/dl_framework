from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from lib.utils import *
from lib.data import *
import polars as pl
import polars.selectors as cs
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(kw_only=True)
class ClimbConfig(DataConfig):
    target: Tuple[str] = ("boulder grade #",)
    omit: Tuple[str] = (
        "boulder grade #",
        "route grade # (IRCRA)",
        #   "partcipant"
    )


# see ipynb
class ClimbData(Data):
    def __init__(self, c: DataConfig):
        super().__init__()
        self.save_config(c)

        # Preprocessing
        df = pl.read_excel(
            "resources/climb_symlink.xlsx",
            # schema=dtype,
            infer_schema_length=50,
        )
        df = df.with_columns((~cs.numeric()).custom.clean_str())

        # drop nulls
        # target = ["route grade # (IRCRA)"]
        # target = ["boulder grade #", "route grade # (IRCRA)"]

        df = drop_rows(
            df,
            count_null=lambda row: sum(1 if row[k] in [None] else 0 for k in c.target),
        )

        y = df[c.target]

        df = df.drop(c.omit)

        transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    SimpleImputer(missing_values=np.nan, strategy="mean"),
                    df.select(cs.numeric()).columns,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        indices = tuple(
            df.columns.index(c) for c in df.select(cs.by_dtype(pl.Float64)).columns
        )

        transformer.set_output(transform="polars")

        df = transformer.fit_transform(df)
        df = process_categoricals(df)

        self.processor = ColumnTransformer(
            transformers=[
                ("real", StandardScaler(), indices),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )  # this operates on tensors and so moves the transformed columns to the front

        if vb(10):
            print(df.describe())
        self.set_data(df, y)
        self.set_folds(KFold(n_splits=10))

    def shuffle_col(self, column):
        self.unshuffle()
        self.data_backup = self.data.clone()
        df = self.data[0]
        self.set_data(df.with_columns(pl.col(column).shuffle()), self.data[1])
        pass

    def unshuffle(self):
        b = getattr(self, "data_backup", None)
        if b is None:
            return
        else:
            self.data = b
            self.data_backup = None

    def _fit_transforms(self, tensors):
        tensors[0] = self.processor.fit_transform(tensors[0])
        return [lambda x: self.processor.transform(x)]

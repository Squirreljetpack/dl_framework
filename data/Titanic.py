from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lib.utils import *
from lib.data import *
import polars as pl
from lib import dfs
import polars.selectors as cs
from sklearn.model_selection import KFold


class Titanic(ClassifierData):
    def __init__(self, c: DataConfig):
        super().__init__()
        self.save_config(c)

        # Preprocessing
        df = pl.read_csv(
            "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv",
            has_header=True,
        )
        target = "Survived"

        # df = dfs.drop_null_rows(df, 50, target)
        X = df.drop([target, "Name"])
        y = df[target]

        # self.classes = int(df.select(pl.col(target).n_unique()).to_numpy())

        categorical_cols = ["Sex", "Pclass"]
        X = dfs.process_categoricals(X, categorical_cols)

        numerical_cols = [
            "Age",
            "Siblings/Spouses Aboard",
            "Parents/Children Aboard",
            "Fare",
        ]

        if vb(5):
            preview_df(X)

        # Numerical transformers
        indices = tuple(X.columns.index(c) for c in numerical_cols)
        print("Numerical indices:", indices)

        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        self.processor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, indices),
            ]
        )

        self.set_data(X, y)
        self.set_folds(KFold(n_splits=6))

    def _fit_transforms(self, tensors):
        def transform_X(t):
            q = self.processor
            q.fit_transform(t)
            return lambda v: q.transform(v)

        return [transform_X(tensors[0])]

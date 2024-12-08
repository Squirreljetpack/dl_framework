from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from lib.utils import *
from lib.data import *
import polars as pl
import polars.selectors as cs
from sklearn.model_selection import KFold, train_test_split


# see ipynb
class CellViabilityClassification(ClassifierData):
    def __init__(self, c: DataConfig):
        super().__init__()
        self.save_config(c)

        # Preprocessing
        df = pl.read_csv(
            "resources/Cell viability and extrusion dataset V1.csv",
            has_header=True,
            null_values=[""],
            # schema=dtype,
            infer_schema_length=1000,
        )

        # drop nulls
        transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
                    ["Syringe_Temperature_(°C)", "Substrate_Temperature_(°C)"],
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        transformer.set_output(transform="polars")

        df = transformer.fit_transform(df)

        # Drop "effect" variables
        # Since we don't have the same dataset, we try to follow the spirit instead of the letter of https://scholarscompass.vcu.edu/cgi/viewcontent.cgi?article=7979&context=etd
        # The paper only mentions dropping Fiber diameter for extrusion but since it's an "effect" variable, we decided to drop it as well
        df = df.drop(
            [
                "Fiber_Diameter_(µm)",
                "Reference",
                "DOI",
                "Acceptable_Viability_(Yes/No)",
                "Acceptable_Pressure_(Yes/No)",
                "Final_PEGTA_Conc_(%w/v)",
                "Final_PEGMA_Conc_(%w/v)",
            ]
        )

        df = drop_cols(
            df,
            drop_criterion=lambda col: sum((1 if x is None else 0 for x in col))
            >= df.shape[0] // 2,
        )

        df = drop_rows(
            df,
            count_null=lambda row: 1
            if row["Viability_at_time_of_observation_(%)"] in [None, 0]
            else 0,
        )

        # Imputing Values

        # Threshold to binary val
        target = ["Viability_at_time_of_observation_(%)"]

        y = df.select(
            pl.when(pl.col("Viability_at_time_of_observation_(%)") > 70)
            .then(pl.lit("Y"))
            .otherwise(pl.lit("N"))
            .alias("Acceptable_Viability")
        )["Acceptable_Viability"]

        df = df.drop(target)

        # Numeric KNN
        # Not sure if it's right to impute on validation, but the paper seems to do it
        numeric_cols = df.select(cs.numeric()).columns
        transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    KNNImputer(n_neighbors=30, weights="uniform"),
                    numeric_cols,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        transformer.set_output(transform="polars")
        df = transformer.fit_transform(df)

        # categorical mode impute + one hot
        df = process_categoricals(df)

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.15, random_state=42
        )

        self.set_data(X_train, y_train)
        self.set_folds(KFold(n_splits=10))
        self.set_data(X_test, y_test, test=True)


# same as above but with y value unchanged
class CellViability(Data):
    def __init__(self, c: DataConfig):
        super().__init__()
        self.save_config(c)

        # Preprocessing
        df = pl.read_csv(
            "resources/Cell viability and extrusion dataset V1.csv",
            has_header=True,
            null_values=[""],
            # schema=dtype,
            infer_schema_length=1000,
        )

        # drop nulls
        transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
                    ["Syringe_Temperature_(°C)", "Substrate_Temperature_(°C)"],
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        transformer.set_output(transform="polars")

        df = transformer.fit_transform(df)

        # Drop "effect" variables
        # Since we don't have the same dataset, we try to follow the spirit instead of the letter of https://scholarscompass.vcu.edu/cgi/viewcontent.cgi?article=7979&context=etd
        # The paper only mentions dropping Fiber diameter for extrusion but since it's an "effect" variable, we decided to drop it as well
        df = df.drop(
            [
                "Fiber_Diameter_(µm)",
                "Reference",
                "DOI",
                "Acceptable_Viability_(Yes/No)",
                "Acceptable_Pressure_(Yes/No)",
                "Final_PEGTA_Conc_(%w/v)",
                "Final_PEGMA_Conc_(%w/v)",
            ]
        )

        df = drop_cols(
            df,
            drop_criterion=lambda col: sum((1 if x is None else 0 for x in col))
            >= df.shape[0] // 2,
        )

        df = drop_rows(
            df,
            count_null=lambda row: 1
            if row["Viability_at_time_of_observation_(%)"] in [None, 0]
            else 0,
        )

        # Imputing Values

        # Threshold to binary val
        target = ["Viability_at_time_of_observation_(%)"]

        y = df[target[0]]
        df = df.drop(target)

        # Numeric KNN
        numeric_cols = df.select(cs.numeric()).columns
        transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    KNNImputer(n_neighbors=30, weights="uniform"),
                    numeric_cols,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        transformer.set_output(transform="polars")
        df = transformer.fit_transform(df)

        # categorical mode impute + one hot
        df = process_categoricals(df)

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.15, random_state=42
        )

        self.set_data(X_train, y_train)
        self.set_folds(KFold(n_splits=10))
        self.set_data(X_test, y_test, test=True)

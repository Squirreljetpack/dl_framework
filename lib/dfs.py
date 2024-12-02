import logging
from typing import List, Union
import polars as pl
import pandas as pd
import polars.selectors as cs


def drop_nulls(df):
    not_null_cols = filter(lambda x: x.null_count() != df.height, df)
    not_null_col_names = map(lambda x: x.name, not_null_cols)
    return df.select(not_null_col_names)


def preview_df(frame, head=True, glimpse=True, tail=True, describe=True):
    if isinstance(frame, (pl.DataFrame)):  # Check for polars DataFrame
        print("-" * 50)
        if head:
            print(frame)
        if glimpse:
            print("\nGlimpse:")
            frame.glimpse()

        print("\nDescription:")
        print(frame.describe())

        print("\nTail:")
        print(frame.tail())

        print("-" * 50)
    elif isinstance(frame, (pd.DataFrame)):
        print("-" * 50)
        if head:
            print(frame)
        if glimpse:
            print("\nGlimpse:")
            print(frame.info())

        print("\nDescription:")
        print(frame.describe())

        print("\nTail:")
        print(frame.tail())

        print("-" * 50)


def process_categoricals(
    df: Union[pl.DataFrame, pd.DataFrame],
    categorical_cols: List[str],
    prefix_sep: str = "_",
    impute=None,
    drop_first=True,
) -> pl.DataFrame:
    """
    Polars version of impute_and_encode function.
    Mimics sklearn's SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")

    Args:
        df: Input DataFrame (Polars or Pandas)
        prefix_sep: Separator between column name and category in new column names
        impute: Impute expr, default: mode

    Returns:
        Polars DataFrame with imputed and encoded categorical columns
    """

    # Store modes and unique values for each column
    modes = {}
    if not categorical_cols:
        categorical_cols = df.select(cs.categorical() | cs.string()).columns
        logging.info("Guessing Categorical columns: ", categorical_cols)

    if impute is None:
        for col in categorical_cols:
            modes[col] = df.select(pl.col(col).mode().first()).item()
        impute = [pl.col(col).fill_null(modes[col]) for col in categorical_cols]

    # Apply imputation
    processed_df = df.with_columns(impute)

    processed_df = processed_df.to_dummies(
        categorical_cols, separator=prefix_sep, drop_first=drop_first
    )

    return processed_df


def process_nulls():
    return


def drop_null_rows(df: pl.DataFrame, min_percent: float, target=[]) -> pl.DataFrame:
    def count_null(row: dict) -> int:
        y = sum(1 for x in row.values() if x is not None)
        return y

    target_columns = [target] if isinstance(target, str) else target
    threshold = (len(df.columns) * min_percent) // 100
    df = df.drop_nulls(subset=target_columns)
    df = df.filter(
        pl.sum_horizontal(
            pl.struct(pl.all()).map_elements(count_null, return_dtype=pl.Int64)
        )
        >= threshold
    )
    return df


def drop_null_cols(df):
    return df.select(
        col.name
        for col in df.select(
            pl.all().map_batches(
                lambda d: pl.Series([any(e is not None and e != 0 for e in d)])
            )
        )
        if col.all()
    )


@pl.api.register_expr_namespace("dfs")
class Dfs:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def clean_numeric(self) -> pl.Expr:
        """Converts string to numeric column by dropping non-numeric characters

        Returns:
            pl.Expr

        Example:
            df.with_columns(
                pl.col("Fiber_Diameter_(µm)").my.clean_numeric()
            )
        """
        return (
            self._expr.str.replace_all(
                r"[^0-9.]", ""
            ).cast(  # Replace non-digit and non-dot characters
                pl.Float64
            )  # Cast to Float64
        )

def drop_nulls(df):
    not_null_cols = filter(lambda x: x.null_count() != df.height, df)
    not_null_col_names = map(lambda x: x.name, not_null_cols)
    return df.select(not_null_col_names)
def clean_categorical(df, categorical_columns):
    df[categorical_columns] = df[categorical_columns].fillna("unknown")

    min_count = 50
    for col in categorical_columns:
        counts = df[col].value_counts()
        df[col] = df[col].apply(lambda x: x if counts[x] >= min_count else 'other')

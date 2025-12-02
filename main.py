"""
Main Data Cleaning Pipeline

Loads raw Excel files, applies cleaning transformations, and outputs clean_data.csv.
This is the original data cleaning pipeline - the starting point before ML feature engineering.
"""

import pandas as pd
from src.load_raw_data import load_raw_data
from src.clean_numeric import clean_numeric
from src.clean_categorical import clean_categorical
from src.column_types import (
    ignored_columns,
    numerical_columns,
    categorical_columns,
    location_columns,
    date_columns,
    year_columns
)


def main():
    """Run the data cleaning pipeline."""
    print("Loading raw data from Excel files...")
    df = load_raw_data("../data/excel_sheets")
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    print("\nDropping ignored columns...")
    df = df.drop(columns=ignored_columns)
    print(f"Retained {len(df.columns)} columns after dropping")

    print("\nCleaning numeric columns...")
    clean_numeric(df)

    print("Cleaning categorical columns...")
    clean_categorical(df, categorical_columns)

    print("\nSaving cleaned data...")
    df.to_csv('../data/clean_data.csv', index=False)
    print("Saved to ../data/clean_data.csv")

    print("\n" + "="*50)
    print("DATA CLEANING SUMMARY")
    print("="*50)
    df.info(verbose=True, show_counts=True)

    print(f"""
    Column Breakdown:
    -----------------
    Numerical Columns:   {len(numerical_columns)}
    Categorical Columns: {len(categorical_columns)}
    Location Columns:    {len(location_columns)}
    Date Columns:        {len(date_columns + year_columns)}
    ------------------------
    Total Columns:       {len(df.columns)}
    Total Rows:          {len(df)}
    """)


if __name__ == "__main__":
    main()


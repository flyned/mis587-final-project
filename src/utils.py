"""
Utility Functions

Helper functions for data processing, visualization, and common operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_data_path():
    """Get the path to the data directory."""
    project_root = Path(__file__).parent.parent
    return project_root.parent / "data"


def load_clean_data():
    """Load the cleaned data CSV file."""
    data_path = get_data_path()
    return pd.read_csv(data_path / "clean_data.csv")


def load_train_data():
    """Load the pre-made training split."""
    data_path = get_data_path()
    return pd.read_csv(data_path / "train.csv")


def load_val_data():
    """Load the pre-made validation split."""
    data_path = get_data_path()
    return pd.read_csv(data_path / "val.csv")


def load_test_data():
    """Load the pre-made test split."""
    data_path = get_data_path()
    return pd.read_csv(data_path / "test.csv")


def load_data_splits():
    """
    Load all three data splits at once.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    return load_train_data(), load_val_data(), load_test_data()


def save_data_splits(train, val, test):
    """
    Save train/validation/test splits to CSV files.

    Args:
        train: Training DataFrame
        val: Validation DataFrame
        test: Test DataFrame
    """
    data_path = get_data_path()
    train.to_csv(data_path / "train.csv", index=False)
    val.to_csv(data_path / "val.csv", index=False)
    test.to_csv(data_path / "test.csv", index=False)
    print(f"Saved data splits to {data_path}/")
    print(f"  Train: {len(train)} rows")
    print(f"  Validation: {len(val)} rows")
    print(f"  Test: {len(test)} rows")

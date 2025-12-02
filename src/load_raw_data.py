import pandas as pd
from pathlib import Path

def load_raw_data(data_dir):
    data_dir = Path(data_dir)

    dfs = [pd.read_excel(f) for f in data_dir.iterdir() if f.is_file() and f.suffix == '.xlsx']
    df = pd.concat(dfs, ignore_index=True)

    return df

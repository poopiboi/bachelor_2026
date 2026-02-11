from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def read_data(file_in: Path) :
    '''
    Simple reader for data. Returns the pandas dataframe for input, depending on format.
    '''
    # We get the file extension
    suffix = file_in.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_in)
    elif suffix == ".xlsx":
        return pd.read_excel(file_in)
    elif suffix == ".parquet":
        return pd.read_parquet(file_in)
    else:
        raise ValueError(f"Unsupported file format detected! Expecting .csv, .xlsx or .parquet, but got {suffix}")
    



if __name__ == "__main__":

    
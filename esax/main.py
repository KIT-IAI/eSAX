"""
MIT License
Copyright (c) 2021 KIT-IAI Jan Ludwig, Oliver Neumann, Marian Turowski
"""

import os
import sys

import numpy as np
import pandas as pd

import esax.get_motif as mot
import esax.plots as plots
import esax.get_subsequences as subs


def load_data():
    """
    This method loads data from the data folder.
    NOTE: The UCI ElectricityLoadDiagrams20112014 Data Set serves as an examples for data loading here (source:
    https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014).

    :return: the univariate time series of interest
    :rtype: pandas.Series
    """
    # Example: UCI building load dataset
    path = os.path.join("data", "LD2012.txt")
    df = pd.read_csv(
        path, index_col=0, parse_dates=True,
        delimiter=";", decimal=",", header=0
    )
    data = df.iloc[:, 198]

    return data


def main():
    # Load data
    data = load_data()

    # Calculate the measurement intervals
    try:
        dates = pd.arrays.DatetimeArray(data.index, dtype=np.dtype("<M8[ns]"), freq=None, copy=False)
        time_dif_in_sec = (dates[1] - dates[0]).seconds
    except:
        print("Unexpected error:", sys.exc_info()[0])
        time_dif_in_sec = 60
        raise

    if time_dif_in_sec == 0:
        time_dif_in_sec = 60

    # Get subsequences
    ts_subs = subs.get_subsequences(data, time_dif_in_sec)

    # Get motifs
    if ts_subs:
        found_motifs = mot.get_motifs(data, ts_subs)
        if found_motifs:
            plots.plot_motifs(data, found_motifs)


if __name__ == "__main__":
    main()

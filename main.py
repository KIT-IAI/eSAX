"""
MIT License
Copyright (c) 2021 KIT-IAI Jan Ludwig, Oliver Neumann, Marian Turowski
"""

import os
import sys

import numpy as np
import pandas as pd

import get_motif as mot
import plots
import get_subsequences as subs


def load_data():
    """
    This method loads data from the data folder.
    NOTE: Two examples for data loading exist: TODO Bitte vervollständingen: Wo gibt es die Beispieldaten? Was unterscheidet die beiden Beispiele?
    1) UCI building load dataset
    2) Weather data

    :return: TODO
    :rtype: TODO
    """
    # Example 1: UCI building load dataset
    path = os.path.join("data", "LD2011_2014_1.txt")
    df = pd.read_csv(
        path, index_col=0, parse_dates=True,
        delimiter=";", decimal=",", header=0
    )
    data = df.iloc[:, 198]

    #####
    # Example 2: Weather data
    # path = os.path.join("data", "dataSunTime.txt")
    # df = pd.read_csv(
    #    path, index_col=0, parse_dates=True,
    #    delimiter=",", decimal=".", header=0)
    # data = df.iloc[:, 0]
    # moq = np.repeat(400, 1686)
    # data = data-moq
    # data[data < 0]=0
    #####

    return data


def main():
    # Load data
    data = load_data()

    # Calculate the measurement intervals
    # NOTE: Find a way to pass time_dif_in_sec to module in pyWATTS (e.g. by user parameter) TODO Kann diese Note gelöscht werden?
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
    time_dif_in_sec = 3600
    ts_subs = subs.get_subsequences(data, time_dif_in_sec)

    # Get motifs
    if ts_subs:
        found_motifs = mot.get_motifs(data, ts_subs)
        if found_motifs:
            plots.plot_motifs(data, found_motifs)


if __name__ == "__main__":
    main()

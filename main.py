import os
import sys

import numpy as np
import pandas as pd

import get_motif as mot
import plots
import get_subsequences as subs


def load_data():
    ## NOTE: Two examples for data loading
    # UCI (building load data) dataset using 198 column
    path = os.path.join("data", "LD2011_2014_1.txt")
    df = pd.read_csv(
        path, index_col=0, parse_dates=True,
        delimiter=";", decimal=",", header=0
    )
    data = df.iloc[:, 198]
    #####
    # Example time-series from the internet with some weather data
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
    # print(strftime("%Y-%m-%d %H:%M:%S"))
    data = load_data()

    # Here the measurement intervals are calculated
    # TODO find method to scale the window size right (atm: window = round(24 * ((60 * 60) / measuring_intervall))
    # NOTE: Find a way to pass time_dif_in_sec to module in pyWATTS (e.g. by user parameter)
    try:
        dates = pd.arrays.DatetimeArray(data.index, dtype=np.dtype("<M8[ns]"), freq=None, copy=False)
        time_dif_in_sec = (dates[1] - dates[0]).seconds
    except:
        print("Unexpected error:", sys.exc_info()[0])
        time_dif_in_sec = 60
        raise

    if time_dif_in_sec == 0:
    # TODO find default that is making sense
        time_dif_in_sec = 60

    # Get subsequences
    time_dif_in_sec = 3600
    ts_subs = subs.get_subsequences(data, time_dif_in_sec)

    if ts_subs:
        found_motifs = mot.get_motif(data, ts_subs)
        if found_motifs:
            plots.motif_plots(data, found_motifs)


if __name__ == "__main__":
    main()

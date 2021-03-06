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
import logging

logger = logging.getLogger(__name__)

def load_data():
    """
    This method loads data from the data folder.
    NOTE: The UCI ElectricityLoadDiagrams20112014 Data Set serves as an examples for data loading here (source:
    https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014).

    :return: the univariate time series of interest
    :rtype: pandas.Series
    """
    # Example: UCI building load dataset
    path = os.path.abspath(os.path.join("./data/", "LD2012.txt"))
    df = pd.read_csv(
        path, index_col=0, parse_dates=True,
        delimiter=";", decimal=",", header=0
    )
    data = df.MT_019

    return data


def main():

    # Load data
    data = load_data()

    # Calculate the measurement intervals
    try:
        dates = pd.arrays.DatetimeArray(data.index, dtype=np.dtype("<M8[ns]"), freq=None, copy=False)
        resolution = (dates[1] - dates[0]).seconds / 3600
    except:
        logging.warning("Unexpected error:", sys.exc_info()[0])
        resolution = 1
        raise

    if resolution == 0:
        resolution = 1

    # Get subsequences
    ts_subs, startpoints, indexes_subs = subs.get_subsequences(data, resolution)

    # Get motifs
    if ts_subs:
        found_motifs = mot.get_motifs(data, ts_subs, breaks=5, word_length=10, num_iterations=0,
                                      mask_size=2, mdr=2.5, cr1=5, cr2=1.5)
        if found_motifs:
            plots.plot_ecdf(found_motifs['ecdf'], './run')
            plots.plot_motifs(data.index, found_motifs['motifs_raw'], found_motifs['indexes'], './run')
            plots.plot_repr_motif(found_motifs['motifs_raw'], './run')


if __name__ == "__main__":
    main()

"""
MIT License
Copyright (c) 2021 KIT-IAI Jan Ludwig, Oliver Neumann, Marian Turowski
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *


def plot_ecdf(ecdf):
    """
    This method plots the ECDF as a pdf image

    :param ecdf: Euclidean Cumulative Distribution Function
    :type: tuple (x,y) of numpy.ndarrays
    """
    plt.plot(ecdf[0], ecdf[1])
    plt.xlabel("Sample Quantiles")
    # Latex code for axis declarations
    plt.ylabel('$\^{F}_n(x)$')
    plt.title("Empirical Cumulative Distribution Function")
    plt.savefig("ecdf_Power.pdf")


def plot_time_series(data, filepath, xlabel="Time", ylabel="Power"):
    """
    This method plots the original time series

    :param data: the original time series
    :type: pandas.Series
    :param filepath: path to the file containing the data
    :type: string
    :param xlabel: label of the x-axis
    :type: string
    :param ylabel: label of the y-axis
    :type: string
    """
    # plot the complete time series
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.xticks([])
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filepath)


def plot_subsequences(sequences, filepath, xlabel="Time", ylabel="Power"):
    """
    This method plots the detected subsequences of a time series

    :param sequences: the subsequences of the subsequence detection step
    :type: list of numpy.ndarrays
    :param filepath: path to the file containing the data
    :type: string
    :param xlabel: label of the x-axis
    :type: string
    :param ylabel: label of the y-axis
    :type: string
    """
    # Create a data frame with one y variable (all sequences after each other) and a x variable (time steps), add the
    # name of the sequence as third column by repeating the name according to the length of the sequence
    dat = []
    for i in sequences:
        dataframe = pd.DataFrame()
        dataframe["x"] = list(range(len(i)))
        dataframe["y"] = i
        dat.append(dataframe)

    # Set names for the individual sequences
    list_names = []
    for i, _ in enumerate(sequences):
        list_names.append("Seq " + str(i))

    # Store the length off the individual sequences
    lns = np.array([len(i) for i in dat])

    try:
        data = pd.concat(dat)
    except ValueError:
        print("No minima found!")
        return False

    data["Sequence"] = np.repeat(list_names, lns)
    data.rename({"x": xlabel, "y": ylabel}, axis=1, inplace=True)
    p = ggplot(data, aes(x=xlabel, y=ylabel, colour="Sequence")) + theme_bw() + \
        geom_step() + facet_wrap("Sequence") + theme(legend_position="none")
    p.save(filepath, width=14, height=10)


def plot_motifs(data, found_motifs):
    """
    This method generates the result plots of eSAX. Subsequences with a similar appearance are grouped (motifs) and
    plotted into the same pdf file.

    :param data: the original time series
    :type: pandas.Series
    :param found_motifs: the result list of get_motif
    :type: dict
    """
    motif_raw = found_motifs.get("motif_raw")
    dates = pd.arrays.DatetimeArray(
        data.index, dtype=np.dtype('<M8[ns]'), freq=None, copy=False)
    # Plot all instances of one motif
    for m in range(0, len(motif_raw)):
        startpoints = dates[found_motifs.get("indices")[m]]

        wd = [x.day_name() for x in startpoints]
        d = [x.day for x in startpoints]
        mth = [x.month for x in startpoints]

        identifier = ["{} {}.{}.".format(wd, d, mth)
                      for wd, d, mth in zip(wd, d, mth)]

        # Transform the list of raw motifs to a list with x,y data
        dat = (np.empty(shape=(0, 2)))
        dat_lengths = []

        for i in motif_raw[m]:
            seq = np.linspace(start=1, stop=len(i), num=len(i), dtype="int64")
            zipped = np.array(list(zip(seq, i)))
            dat_lengths.append(zipped)
            dat = np.concatenate((dat, zipped), axis=0)

        # Set names for the individual sequences
        list_names = identifier

        # Store the length off the individual sequences
        lns = [len(s) for s in dat_lengths]

        # Create a data frame with one y variable (all sequences after each other) and a x variable (time steps), add
        # the name of the sequence as third column by repeating the name according to the length of the sequence TODO Das ist derselbe Kommentar wie in Zeile 62f. Passt der an beiden Stellen?
        dat = pd.DataFrame(dat, columns=["Timesteps", "Load"])
        dat["Sequence"] = np.repeat(list_names, lns, axis=0)

        # NOTE: This is a plot with the package Plotnine to be able to use the same code as in R.
        p = ggplot(dat,
                   aes(x="Timesteps", y="Load", colour="Sequence")) + theme_bw() + geom_line() + facet_wrap(
            "Sequence") + theme(legend_position="none")
        p.save("eMotif_{}.pdf".format(m), width=14, height=10)

    print("All motifs plotted ...")

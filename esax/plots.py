"""
MIT License
Copyright (c) 2021 KIT-IAI Jan Ludwig, Oliver Neumann, Marian Turowski
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from plotnine import *


def plot_ecdf(ecdf, filepath):
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
    plt.savefig(os.path.join(filepath, "ecdf_Power.pdf"))
    plt.close()


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
    plt.savefig(os.path.join(filepath, "full_ts.png"))
    plt.close()


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
    p.save(os.path.join(filepath, "all_sequences.png"), width=14, height=10)


def plot_motifs(data, motifs_raw, indexes, filepath):
    """
    This method generates the result plots of eSAX. Subsequences with a similar appearance are grouped (motifs) and
    plotted into the same pdf file.

    :param data: the original time series
    :type: pandas.Series
    :param found_motifs: the result list of get_motif
    :type: dict
    """
    motifs_raw = motifs_raw
    # If the time-series does not have a timestamp column, the motif plots are described with a number.
    # In case there is timestamp column, the dates of the motifs are added to the plots.
    if isinstance(data.index.to_series()[0], datetime.date):
        dates = data.index
        # plot all instances of one motif
        for m in range(0, len(motifs_raw)):
            startpoints = dates[list(indexes[m])]

            wd = [x.day_name() for x in startpoints]
            d = [x.day for x in startpoints]
            mth = [x.month for x in startpoints]

            identifier = ["{} {}.{}.".format(wd, d, mth)
                          for wd, d, mth in zip(wd, d, mth)]

            # transform the list of raw motifs to a list with x,y data
            dat = (np.empty(shape=(0, 2)))
            dat_lengths = []

            for i in motifs_raw[m]:
                seq = np.linspace(start=1, stop=len(i), num=len(i), dtype="int64")
                zipped = np.array(list(zip(seq, i)))
                dat_lengths.append(zipped)
                dat = np.concatenate((dat, zipped), axis=0)

            # set names for the individual sequences
            if len(set(identifier)) > 1:
                list_names = identifier
            else:
                list_names = [y + "(" + str(x) + ")" for x, y in enumerate(identifier)]

            # store the length off the individual sequences
            lns = [len(s) for s in dat_lengths]

            # create a data frame with one y variable (all sequences after each other)
            # and a x variable (time steps)
            # add the name of the sequence as third column
            # by repeating the name according to the length of the sequence
            dat = pd.DataFrame(dat, columns=["Timesteps", "Load"])
            dat["Sequence"] = np.repeat(list_names, lns, axis=0)

            # NOTE This is a plot from the package Plotnine. You can use the same code as in R.
            p = ggplot(dat,
                       aes(x="Timesteps", y="Load", colour="Sequence")) + theme_bw() + geom_line() + facet_wrap(
                "Sequence") + theme(legend_position="none")
            p.save(os.path.join(filepath, "eMotif_{}.png".format(m)), width=14, height=10)

    else:
        for m in range(0, len(motifs_raw)):
            startpoints = found_motifs.get("indices")[m]
            identifier = startpoints

            dat = (np.empty(shape=(0, 2)))
            dat_lengths = []
            for i in motifs_raw[m]:
                seq = np.linspace(start=1, stop=len(i), num=len(i), dtype="int64")
                zipped = np.array(list(zip(seq, i)))
                dat_lengths.append(zipped)
                dat = np.concatenate((dat, zipped), axis=0)

            # set names for the individual sequences
            list_names = identifier

            # store the length off the individual sequences
            lns = [len(s) for s in dat_lengths]

            # create a data frame with one y variable (all sequences after each other)
            # and a x variable (time steps)
            # add the name of the sequence as third column
            # by repeating the name according to the length of the sequence
            dat = pd.DataFrame(dat, columns=["Timesteps", "Load"])
            dat["Sequence"] = np.repeat(list_names, lns, axis=0)

            # NOTE This is a plot from the package Plotnine. You can use the same code as in R.
            p = ggplot(dat,
                       aes(x="Timesteps", y="Load", colour="Sequence")) + theme_bw() + geom_line() + facet_wrap(
                "Sequence") + theme(legend_position="none")
            p.save(os.path.join(filepath, "eMotif_{}.png".format(m)), width=14, height=10)

    print("All motifs plotted ...")


def plot_repr_motif(motifs_raw, filepath):
    """
    This methods calculates the median period of all the periods in one cluster
    :param motifs_raw: list containing one ore more motifs
    :param path: name of the
    :return:
    """
    for idx,motif in enumerate(motifs_raw):
        motif_df = pd.DataFrame(motif)
        # mean of all sequences in one moment at each point of time
        repr = []
        # std of all measurements at each point of time
        std_seq = []

        # Hint of Nicole: In case there is more than one motif, the distance ratio parameters have to be adjusted
        repr_motif = motif_df.median(axis=0)
        std_seq = motif_df.std(axis=0)
        up_quantile = motif_df.quantile(q=0.75)
        low_quantile = motif_df.quantile(q=0.25)

        # plot actual load curve
        plt.plot(repr_motif, '-k')
        plt.plot(std_seq, '-b')
        plt.plot(up_quantile, '--r')
        plt.plot(low_quantile, '--r')

        # set some plot parameters
        plt.xlabel('Timesteps')
        plt.xticks(rotation=45)
        plt.ylabel('Load [MW]')
        plt.legend(['repr_period', 'std_deviation', 'quartiles (25,75)'], loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(filepath, "repr_motif_{}.png".format(idx)))
        plt.clf()
    plt.close()
    
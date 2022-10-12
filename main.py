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
import argparse

logger = logging.getLogger(__name__)


def parse_hparams(args=None):
    """
    Parses command line statement

    :return: the parsed arguments
    """
    # prepare argument parser
    parser = argparse.ArgumentParser(
        description="Anomaly detection benchmark pipeline."
    )
    # csv path file
    parser.add_argument("csv_path", type=str, help="Path to the data CSV file.")
    # time_index
    parser.add_argument("time", type=str, help="Name of the time index.")
    # data_index
    parser.add_argument("target", type=str, help="Name of the target index.")
    # delimiter
    parser.add_argument(
        "--csv_separator",
        type=str,
        default=";",
        help="CSV file column separator (default ;).",
    )
    # decimal
    parser.add_argument(
        "--csv_decimal",
        type=str,
        default=",",
        help="CSV file decimal delimiter (default ,).",
    )
    # enable plots
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Enable plots to be generated.",
    )

    # convert argument strings
    parsed_hparams = parser.parse_args(args=args)

    return parsed_hparams


def load_data(csv_path: str, time: str, target: str, sep: str, dec: str):
    """
    This method loads data.

    :return: the univariate time series of interest
    :rtype: pandas.Series
    """
    df = pd.read_csv(
        csv_path, index_col=time, parse_dates=True, delimiter=sep, decimal=dec, header=0
    )
    data = df[target]

    return data


def main(hparams: argparse.Namespace):

    # Load data
    data = load_data(
        csv_path=hparams.csv_path,
        time=hparams.time,
        target=hparams.target,
        sep=hparams.csv_separator,
        dec=hparams.csv_decimal,
    )

    # Calculate the measurement intervals
    try:
        dates = pd.arrays.DatetimeArray(
            data.index, dtype=np.dtype("<M8[ns]"), freq=None, copy=False
        )
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
        found_motifs = mot.get_motifs(
            data,
            ts_subs,
            breaks=5,
            word_length=10,
            num_iterations=0,
            mask_size=2,
            mdr=2.5,
            cr1=5,
            cr2=1.5,
        )
        if found_motifs and hparams.plots:
            if not os.path.exists("./run"):
                os.makedirs("./run")

            plots.plot_ecdf(found_motifs["ecdf"], "./run")
            plots.plot_motifs(
                data.index, found_motifs["motifs_raw"], found_motifs["indexes"], "./run"
            )
            plots.plot_repr_motif(found_motifs["motifs_raw"], "./run")


if __name__ == "__main__":
    # parse command line statement
    hparams = parse_hparams()

    main(hparams)

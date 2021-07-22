"""
MIT License
Copyright (c) 2021 KIT-IAI Jan Ludwig, Oliver Neumann, Marian Turowski
"""

import numpy as np
import plots


def determine_subsequences(data, event, window, custom_event=0.00, window_size=100):
    """
    This method finds minima in the time series that could indicate where a motif starts.
    NOTE: The minima are not included in the subsequences (that's why the points in localmin are always n+1)

    :param window: custom window length in case 'event' equals 'none'
    :type: int
    :param data: the time series of interest
    :type: pandas.Series
    :param event: (none, zero, minimum, custom) subsequences are either determined by a
    minimum search or through the points where they are zero or another
    specified value. If none is selected the subsequences are predefined by
    the window length
    :type: string
    :param custom_event: the customized value for the event start (default = 0.06)
    :type: float
    :param window_size: indicates the window size for the minimum search
    :type: int

    :return: a list of np.ndarrays containing the subsequences and list of startpoints
    """
    dmin = []
    localmin = []

    # the subsequences in dmin always start with the minimum
    if event == "minimum":
        print("Searching for minima ...\n")
        # initialise __vector__ for minima

        # Loop that finds all minima occuring in each run
        w = window_size

        # find the minima in the window (use the first one found if more than one)

        for i in range(1, int(len(data) / w) + 1):
            k = i * w
            j = (i * w) - w
            vectorPart = data[j:k]
            localmin.append(np.where(vectorPart == min(vectorPart))[0][0] + ((i - 1) * w) + 1)

        print("Preparing list ...\n")

        dmin.append(data[0:localmin[0]].to_numpy())

        for i in range(0, len(localmin) - 1):
            if i == 0:
                dmin.append(data[localmin[i] - 1:(localmin[i + 1])].to_numpy())
            else:
                dmin.append(data[localmin[i]:(localmin[i + 1])].to_numpy())
        dmin.append(data[localmin[len(localmin) - 1]:len(data)].to_numpy())

    elif event == "zero":
        print("Searching for zeros ...\n")
        zeros = np.where(data == 0)[0]

        for i in range(0, len(zeros)):
            if data[zeros[i] + 1] != 0:
                localmin.append(zeros[i] + 1)
                # next point where it is zero again
                if i + 1 < len(zeros):
                    localmin.append(zeros[i + 1])
                else:
                    localmin.append(len(data) - 1)

        print("Preparing list ...\n")

        for i in range(0, len(localmin), 2):
            dmin.append(data[localmin[i]:localmin[i + 1]].to_numpy())

    elif event == "custom":
        print("Searching for custom event ...\n")

        start = np.where(data == custom_event)[0]

        for i in range(0, len(start)):
            if data[start[i] + 1] != custom_event:
                localmin.append(start[i] + 1)
                # next point where it is custom again
                if i + 1 < len(start):
                    localmin.append(start[i + 1])
                else:
                    localmin.append(len(data) - 1)

        print("Preparing list ...\n")

        for i in range(0, len(localmin), 2):
            dmin.append(data[localmin[i]:localmin[i + 1]].to_numpy())

    elif event == "none":
        print("Preparing subsequences ...\n")

        # store the subsequences of size window length for motif discovery in dmin

        for i in range(0, round(len(data) / window)):
            if ((i + 1) * window) < len(data):
                dmin.append(data[(i * window):((i + 1) * window)].to_numpy())
            else:
                dmin.append(data[(i * window):len(data)-1].to_numpy())

        # save the startpoints(window length distance)
        for i in range(0, len(dmin)):
            localmin.append(i * window)

        print("Preparing list ...\n")

    return dmin, localmin


def get_subsequences(data, measuring_interval):
    """
    This method separates the time series into subsequences.
    ASSUME: All measurements must have a timestamp and there should be no NaN values in it.

    :param measuring_interval: time difference in seconds between the measurements of the time series
    :type: int
    :param data: original time series of interest
    :type: pandas.Series

    :return: The subsequences as a list of np.ndarrays
    """
    # create the subsequences with the day or subday patterns

    # calculate how many measuring intervalls fit in one day (in seconds)
    window = round(24 * ((60 * 60) / measuring_interval))

    # get sequences and store the startpoints and sequences separately to not have lists of lists
    sequences, _ = determine_subsequences(data=data, event="minimum", window=window)

    # Plot input (whole time-series) and output (sequences) data
    # NOTE: 'data' variable need unchanged!
    plots.plot_time_series(data, "CompleteTimeSeries.pdf")
    plots.plot_subsequences(sequences, "All_Sequences.pdf")

    print("Done")

    return sequences

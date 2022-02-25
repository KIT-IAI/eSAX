"""
MIT License
Copyright (c) 2021 KIT-IAI Jan Ludwig, Oliver Neumann, Marian Turowski
"""

from itertools import combinations
import numpy as np
import pandas as pd
import random
from statistics import median
import string


def get_ecdf(data):
    """
    This method provides the empirical cumulative distribution function (ECDF) of a time series.

    :param data: a numeric vector representing the univariate time series
    :type data: pandas.Series
    :return: ECDF function of the time series
    :rtype: (x,y) -> x and y numpy.ndarrays
    """
    # Drop all values = 0
    #data = data[np.array(data, dtype=np.int64) != 0]
    ecdf = calculate_ecdf(data)
    return ecdf


def calculate_ecdf(data):
    """
    This method calculates the empirical cumulative distribution function (ECDF) of a time series.
    Warning: This method is equal to stats::ecdf in R. The ECDF in
    statsmodels.distributions.empirical_distribution.ECDF does not calculate the same ECDF as stats::ecdf does.

    :param data: numeric vector representing the univariate time series
    :type: pandas.Series
    :return: ECDF for the time series as tuple of numpy.ndarrays
    :rtype: (x,y) -> x and y numpy.ndarrays
    """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return x, y


def create_esax(x, b, w):
    """
    This method creates eSAX symbols for a univariate time series.

    :param x: numeric vector representing the univariate time series
    :type: numpy.array
    :param b: breakpoints used for the eSAX representation
    :type: numpy.array
    :param w: word size used for the eSAX transformation
    :type: int
    :return: eSAX representation of x
    :rtype: [list, numpy.array, numpy.array, np.array]
    """
    # Perform the piecewise aggregation
    indices = ((np.linspace(start=0, stop=len(x) - 1, num=w + 1)).round(0).astype(int))
    aggr_period = np.zeros(shape=w)

    # Aggregation of the sequence into w values (by using downsampling: mean of all values in one range)
    if len(x) == w:
        aggr_period = x
    else:
        splits = np.array_split(x, w)
        aggr_period = np.array([j.mean() for j in splits])

    # Create an alphabet with double and triple letter combinations (a, aa, aaa) (number of elements = 78)
    letters = list(string.ascii_lowercase)
    alphabet = letters + [x + x for x in letters] + [x + x + x for x in letters]

    # Assign the alphabet
    let = alphabet[0:len(b) - 1]

    # Add symbols to sequence according to breakpoints
    sym = []
    for val in aggr_period:
        sym.append(let[np.argmax(b >= val) - 1])

    return [sym, aggr_period, indices, b]


def create_esax_time_series(ts_subs, w, per):
    """
    This method creates the eSAX representation for each subsequence and puts them row wise into a dataframe.

    :param ts_subs: a list of np arrays with the subsequences of the time series
    :type ts_subs: list of np arrays
    :param w: word size used for the eSAX transformation
    :type w: int
    :param per: percentiles depending on the ECDF of the time series
    :type per: np.quantile
    :return: dataframe with the symbolic representations of the subsequences (row wise) and the non-symbolic
    subsequences in pieces_all
    :rtype: pandas.DataFrame, list of numpy.ndarrays
    """
    # Create eSAX time series
    print("Creating the eSAX pieces")
    # Create list to access the SAX pieces later
    pieces_all = []

    # Initialize empty vector for the results
    ts_sax = []

    # Transformation of every subsequence in ts.subs into a symbolic aggregation
    startpoints = [0]

    # Store the start point of each sequence in the original time series:
    # the start point is the sum of the length of all previous sequences + 1
    # for the first sequence there are no previous sequences, thus start = 1.

    for i in range(0, len(ts_subs) - 1):
        sax_temp = create_esax(x=ts_subs[i], w=w, b=per)
        startpoints.append(startpoints[i] + len(ts_subs[i]))

        # Store the sax pieces
        pieces = sax_temp[1]
        pieces_all.append(pieces)
        ts_sax.append(create_esax(x=ts_subs[i], w=w, b=per)[0])

    ts_sax.append(create_esax(x=ts_subs[len(ts_subs) - 1], w=w, b=per)[0])

    ts_sax_df1 = pd.DataFrame(startpoints)
    ts_sax_df1.rename(columns={0: "StartP"}, inplace=True)
    ts_sax_df2 = pd.DataFrame(ts_sax)
    ts_sax_df = pd.concat([ts_sax_df1, ts_sax_df2], axis=1)
    ts_sax_df = ts_sax_df.set_index('StartP')

    print("Searching for Motifs")

    return ts_sax_df, pieces_all


def perform_random_projection(ts_sax_df, num_iterations, mask_size, seed):
    """
    This method carries out the random projection by randomly choosing columns of ts_sax_df (pairwise) and a generating
    a collision matrix.

    :param ts_sax_df: dataframe with the symbolic representation of the subsequences (rowwise)
    :type: pandas.Dataframe
    :param num_iterations: number of iterations for the random projection (the higher that number is, the
    approximate result gets closer to the "true" result
    :type: int
    :param mask_size: size of the sample of columns from the collision matrix
    :type: int
    :param seed: the seed for the random projection
    :type: int
    :return: a collision matrix for identifying motif candidates
    :rtype: pandas.DataFrame
    """
    # Perform the random projection
    col_mat = np.zeros((ts_sax_df.shape[0], ts_sax_df.shape[0]))
    col_mat = pd.DataFrame(col_mat).astype(int)
    for i in range(0, num_iterations):
        random.seed(i + seed)
        col_pos = sorted(random.sample(list(ts_sax_df.columns.values), mask_size))
        sax_mask = pd.DataFrame(ts_sax_df.iloc[:, col_pos])
        # elimintate duplicate rows from the two randomly selected columns
        unique_lab = sax_mask.drop_duplicates()


        mat = []
        for j in range(0, len(unique_lab.index)):
            indices = []
            for k in range(0, len(sax_mask.index)):
                indices.append(sax_mask.iloc[k, :].equals(unique_lab.iloc[j, :]))
            mat.append(indices)

        # mat is a matrix where collisions are stored as boolean values
        # therefore, a vector (length = number of sequences) is stored row-wise
        ## e.g. if the value of the vector is 'True' at position 10 --> the 11th sequence
        # is equal to another sequence at the indexes k and j (we have a collision)
        mat = pd.DataFrame(mat)

        if len(mat) != 0:
            for k in range(0, len(mat) - 1):
                true_idx = np.where(mat.iloc[k, ])
                # the length must be greater than 1 because there is always one collision
                # (unique_lab is a subset of sax_mask)
                if len(true_idx[0]) > 1:
                    com = [n for n in combinations(true_idx[0], 2)]
                    for m in com:
                        col_mat.iloc[m[0], m[1]] += 1

    return col_mat


def extract_motif_pair(ts_sax_df, col_mat, ts_subs, num_iterations, count_ratio_1=5.0,
                       count_ratio_2=1.5, max_dist_ratio=2.5):
    """
    This method extracts the motif pairs with the highest number of collisions in the collision matrix.

    :param ts_sax_df: dataframe with the symbolic representation of the subsequences (row wise)
    :type: pandas.Dataframe
    :param col_mat: collision matrix
    :type: pandas.Dataframe
    :param ts_subs: subsequences from the subsequence detection
    :type: list of numpy.ndarrays
    :param num_iterations: number of iterations for the random projection
    :type: int
    :param count_ratio_1: influences if a collision matrix entry becomes a candidate
    (higher count_ratio_1 lowers the threshold)
    :type: float
    :param count_ratio_2: second count ratio
    :type: float
    :param max_dist_ratio: maximum distance ratio for determining if the euclidean distance between two motif candidates
    is smaller than a threshold
    :type: float
    :return: a list of numpy.ndarrays with the starting indices of the motifs in the original time series
    :rtype: list of numpy.ndarrays
    """
    # Extract the tentative motif pair
    counts = np.array([], dtype=np.int64)
    for i in range(0, col_mat.shape[1]):
        temp = col_mat.iloc[:, i]
        counts = np.concatenate((counts, temp), axis=None)
    counts = -np.sort(-counts)
    counts_sel = np.where(counts >= (num_iterations / count_ratio_1))[0]
    counts_sel = [counts[sel] for sel in counts_sel]
    counts_sel_no_dupl = sorted(set(counts_sel), reverse=True)

    motif_pair = []
    for value in counts_sel_no_dupl:
        temp = np.where(col_mat == value)
        for x, y in zip(temp[0], temp[1]):
            motif_pair.append([x, y])

    motif_pair = pd.DataFrame(motif_pair)
    if motif_pair.shape == (0, 0):
        print("No motif candidates")
        return []

    indices = []
    for x, y in zip(motif_pair.iloc[:, 0], motif_pair.iloc[:, 1]):

        pair = np.array([ts_sax_df.index[x], ts_sax_df.index[y]])
        cand_1 = np.array(ts_subs[x])
        cand_2 = np.array(ts_subs[y])

        # Dynamic time warping can be used for candidates of different length
        dist_raw = np.linalg.norm(cand_1 - cand_2)

        col_no = col_mat.iloc[x, :]
        ind_cand = np.where(col_no > (max(col_no) / count_ratio_2))[0]
        ind_final = None

        if len(ind_cand) > 1:
            # Delete all the indexes from ind_cand, which are equal to y
            ind_temp = np.delete(ind_cand, np.where(ind_cand == y)[0])
            if (len(ind_temp) == 1) & (np.linalg.norm(cand_1 - ts_subs[ind_temp[0]]) <= max_dist_ratio * dist_raw):
                ind_final = np.array([ts_sax_df.index[ind_temp[0]]])
            elif len(ind_temp) > 1:
                cand_sel = []
                dist_res = []
                for j in ind_temp:
                    # check which of the preliminary candidates has the shortest distance to x
                    dist_res.append(np.linalg.norm(cand_1 - ts_subs[j]))
                    cand_sel.append(ts_subs[j])
                # final starting indexes of similar subsequences to x are derived in this step
                ind_final = ts_sax_df.index[
                    ind_temp[[i for i, v in enumerate(dist_res) if v <= max_dist_ratio * dist_raw]]].to_numpy()
        else:
            pass

        if ind_final is not None:
            pair = np.concatenate((pair, ind_final), axis=0)
            pair = np.unique(pair, axis=0)
        ind_final = None
        indices.append(pair)

    # Combine the indices if there is any overlap
    indices = index_merge(indices)

    return indices


def get_motifs(data, ts_subs, breaks, word_length, num_iterations, mdr, cr1, cr2, mask_size=2, seed=42):
    """
    This method combines all previous steps to extract the motifs.

    :param data: the univariate time series
    :type data: pandas.Series
    :param ts_subs: subsequences from the subsequence detection
    :type ts_subs: list of numpy.ndarrays
    :param breaks: number of breakpoints in the alphabet, 10 = all quantiles of the ecdf
    :type breaks: int
    :param word_length: word size (always the same if sequences are of equal length)
    :type word_length: int
    :param num_iterations: number of iterations of the random projection algorithm (note: the motif candidate search
    depends on it together with count_ratio_1
    :type num_iterations: int
    :param mdr: final distance allowed between occurrences in one motif
    :type mdr: float
    :param cr1: controls when entries in the collision matrix become candidate motifs
    :type cr1: float
    :param cr2: controls whether a candidate motif becomes a motif
    :type cr2: float
    :param mask_size: mask size for random projection
    :type mask_size: int
    :param seed: the seed for the random projection
    :type: int
    :return: dict with subsequences, SAX dataframe, motifs (symbolic, non-symbolic), collision matrix, indices where the
    motifs start, and non-symbolic subsequences
    :rtype: {list of numpy.ndarrays, pandas.DataFrame, list of np.ndarrays, list of pandas.DataFrames, pandas.DataFrame,
    list of numpy.ndarrays, list of numpy.ndarrays}
    """

    # Calculate the ECDF for the alphabet
    ecdf = get_ecdf(data)
    ecdf_df = pd.DataFrame()
    ecdf_df["x"] = ecdf[0]
    ecdf_df["y"] = ecdf[1]


    # Set parameters for the eSAX algorithm
    # NOTE: According to Nicole Ludwig, these parameters were set based on experience and turned out to be the best
    # working ones across 2-3 data sets (e.g. count ratios have high influence but she found a good trade-off)
    # The parameters can be adapted for optimizing the algorithm's quality

    lengths = [len(i) for i in ts_subs]
    if word_length == 0:
        word_length = round(median(lengths) + 0.5)

    # Set parameters for the random projection

    # Calculate the breakpoints for the eSAX algorithm
    # Set the number of breakpoints (percentiles)
    qq = np.linspace(start=0, stop=1, num=breaks + 1)

    # Store the percentiles
    per = np.quantile(ecdf_df["x"], qq)

    # Use only unique percentiles for the alphabet distribution
    per = np.unique(per)

    # Add the minimum as the lowest letter
    minimum = min([i.min() for i in ts_subs])
    per[0] = minimum

    # Set parameters for the random projection and motif candidates
    max_length = (max(lengths) * 0.1).__round__()

    if num_iterations == 0:
        num_iterations = min(max_length, round(word_length / 10))

    # Create eSAX time Series
    ts_sax_df, pieces_all = create_esax_time_series(ts_subs, word_length, per)

    # Perform the random projection
    col_mat = perform_random_projection(ts_sax_df, num_iterations, mask_size, seed)

    # Extract motif candidates
    indexes = extract_motif_pair(ts_sax_df, col_mat, ts_subs, num_iterations, cr1, cr2, mdr)

    motifs_raw = []
    motifs_sax = []
    ts_raw_df = pd.DataFrame(ts_subs, index=ts_sax_df.index, dtype=float)

    for val in indexes:
        motifs_raw.append(ts_raw_df.loc[val])
        motifs_sax.append(ts_sax_df.loc[val])

    motifs_raw.sort(key=lambda x: x.shape[0], reverse=True)
    motifs_sax.sort(key=lambda x: x.shape[0], reverse=True)

    # update list of index tuples after sorting
    indexes = [val.index for val in motifs_raw]

    found_motifs = {'ts_subs': ts_subs, 'ts_sax_df': ts_sax_df, 'motifs_raw': motifs_raw,
                    'motifs_sax': motifs_sax, 'col_mat': col_mat, 'indexes': indexes,
                    'pieces_all': pieces_all, 'ecdf': ecdf}

    print("Done")

    return found_motifs


def index_merge(lsts):
    """
    Merging algorithm that merges lists if they are not disjoint.
    Returns a list of disjoint lists.
    :param lsts: list of lists
    :type: list
    :return: list of disjoint lists
    :rtype: list
    """
    newsets, sets = [set(lst) for lst in lsts], []
    while len(sets) != len(newsets):
        sets, newsets = newsets, []
        for aset in sets:
            for eachset in newsets:
                if not aset.isdisjoint(eachset):
                    eachset.update(aset)
                    break
            else:
                newsets.append(aset)

    return newsets
<h1>Energy Time Series Motif Discovery using Symbolic Aggregated Approximation (eSAX)</h1>

This Python implementation of eSAX is based on the [original eSAX implementation in R](https://github.com/mlsustainableenergy/eSAX) from [Nicole Ludwig](https://github.com/nicoleludwig).
Thereby, this implementation is based on the corresponding paper:

>N. Ludwig, L. Barth, D. Wagner, and V. Hagenmeyer (2019). “Industrial Demand-Side Flexibility: A Benchmark Data Set”. In: Proceedings of the Ninth International Conference on Future Energy Systems - e-Energy ’19. The Association for Computing Machinery, pp. 460–473. doi: [10.1145/3307772.3331021](https://doi.org/10.1145/3307772.3331021)

<h2>Input requirements</h2> 

This implementation assumes the input time series to be in the data folder. The input time series cannot contain NaN-values. Furthermore, the input data must have a timestamp in the index column (to be able to determine the window size by comparing the first two timestamps) and a variable of interest such as power in the second column. If no timestamp column exist, sequences automatically have a length of 24 measurements.

<h2>Files and parameters</h2>

<code>get_subsequences.py:</code> Extracts the subsequences from the original time series. The series is seperated into the subsequences depening on the method selected by the user.

Available methods for subsequence determination:
- "minimum": The minima in the time series define the start- and endpoints of the subsequences (Please note: The minimum itself is not part of the subsequence because the algorithm was originally designed for time series with zeros as minimum)
- "zero": The minimum value in the time series is 0 and the subsequences are values greater than 0 between two zeros.
- "custom": The same method as zero except the fact that the user can choose a custom value for separating the time series
- "none": The time series is separated into parts of equal length. For daily subsequences, this length is defined by the measuring interval (in seconds): window_length = 24 * ((60 * 60) / measuring_interval).

Currently, only the "none" method can be used in eSAX. For using the alternatives, the Euclidean distance calculation has to be swapped with dynamic time warping.

The method get_subsequences(data, measuring_interval) returns a list of numpy.ndarray where each contains one subsequence.

<code>get_motifs.py:</code> Extracts eSAX motifs from the identified subsequences. First, the Empirical Cumulative Distribution Function (ECDF) of the time series is calculated to find the right percentiles for the allocation of the SAX letters.
After the allocation of these letter to the subsequences, the resulting SAX representations of the subsequences are extracted with a random projection and a collision matrix is generated. The pairs of subsequences that have a high collision value are declared as motif candidates.
To prepare the decision whether two candidates belong to the same motif, the Euclidean distance between the two subsequences has to be smaller than a threshold value. In case of two candidates with unequal length, dynamic time warping could be used to calculate the distance. Finally, the indexes of the found motifs are checked for overlap to decide whether they belong to the same motif:
For example, if the start indices of two motif pair are (50,150) and the start indices of another pair are (150,300), all three subsequences will belong to the same motif.

The results are returned in a dict with 
* all the subsequences (ts_subs), 
* the SAX representation of the subsequences (ts_sax_df), 
* the motifs with real values (motif_raw), 
* the motifs with SAX representation (motif_sax), 
* the colision matrix (col_mat), 
* the indices of the time steps where motifs occured (indices) as well as 
* the piecewise approximation (pieces, equal to ts_subs if the length of the subsequences is equal).

As described by Nicole Ludwig, "there are several parameters in this file which can change the outcome of the motif search. There are three categories of parameters: parameters for the alphabet distribution in eSAX, parameters for random projection and parameters for motif discovery". Since these parameters have a high influence on the results, the following table lists the default values that Nicole Ludwig worked with most often and that have been shown to be the best working ones across 2-3 datasets.

| Parameter  | Default  | Description |
| :------------ |:---------------:| :-----|
| breaks      | 10 | number of breakpoints in alphabet, 10 = all quantiles of the ecdf |
| w      | median(length(subsequences))        | word size (note: always the same if sequences are of equal length) |
| iter | min(max(length(subsequences)*0.1), w/10)  | number of iterations of the random projection algorithm (note: the motif candidate search depends on it together with count_ratio_1) |
|mask_size|2|mask size for random projection (the mask size can currently not be varied)|
|max_dist_ratio|2.5|final scalar for the distance allowed between occurrences in one motif|
|count_ratio_1|5|controls when entries in the collision matrix become candidate motifs (if >= word size/10, all entries are considered as a candidate motif)|
|count_ratio_2|1.5|controls whether a candidate motif becomes a motif|

<code>plots.py:</code> Contains methods for plotting interim and final results into the "run"-folder.

<code>main.py:</code> Loads the data and calls the other methods step by step.


<h2>Comparison to the R implementation</h2>

Since this Python implementation of eSAX is based on the original R implemention of eSAX, we want to enable an easy comparison of both that can be used to verify that both implementations work equally. For this purpose, we list aspects in the following that are required to consider when comparing both implementations.
* The initial application for the R implementation was machine data so only the loads greater than 0 are used to find motifs. In the Python version this deletion of zeros is not implemented.
* The R implementation was designed for analysing multiple time series in one run. When comparing both implementations, the loop in the R implementation that loads multiple time series has to be removed.
* In the R implementation, the algorithm can be executed step-by-step. For comparing the interim results between Python and R, the debugger function of the IDE is helpful.
* Currently, both versions are not designed for sequences with different lengths. In case of subsequences with different lengths, the R implementation thus extends the shorter sequence to the length of the longer sequence by repeating the shorter sequence in order to calculates the Euclidean distance between two sequences of different lengths. In the Python version only the "none" method of the get_subsequences-method can be used. This method returns subsequences of equal length.
* The calculate_ecdf() method in get_motifs.py is equal to stats::ecdf in R. The ecdf() function in statsmodels.distributions.empirical_distribution.ECDF does not calculate the same ecdf like stats::ecdf
* In the Python implmentation, the subsequence and the motif plots are plotted with a function from the package plotnine. This way, the plots implemented in R can be used in Python and the plots look the same.
* An upgrade in terms of runtime was implemented in the python version. The algorithm for merging the indexes of candidate motifs works more efficiently in the Python version and thus enables processing of large time series.
* Additionally, the Python implementation contains test cases which test the main functionalities of the algorithm.


<h2>Funding</h2>

This project is funded by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI and the Helmholtz Association under the Program "Energy System Design".


<h2>License</h2>

This code is licensed under the [MIT License](LICENSE).

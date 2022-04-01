import random
import string
import unittest
import pandas as pd
import numpy as np
import esax.get_motif as mot
import matplotlib.pyplot as plt


class MotifTests(unittest.TestCase):

    def test_get_ecdf(self):
        data = np.repeat(np.NaN, 100)
        self.assertRaises(TypeError, mot.get_ecdf(data))
        data = np.repeat([0,1], 100)
        ecdf = mot.get_ecdf(data)

    def test_index_merge(self):
        lists = [[1,2,3,4,5], [5,6,7,8,9], [9,10,11,12,13], [13,14,15,16,17]]
        result = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}]
        self.assertCountEqual(mot.index_merge(lists), result)
        lists = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        result = [{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}]
        self.assertCountEqual(mot.index_merge(lists), result)
        lists = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        result = [{1, 2, 3, 4}]
        self.assertCountEqual(mot.index_merge(lists), result)

    def test_create_esax(self):
        data = np.array(range(1,25))
        word_length = 2
        per = np.linspace(0, 1, 24)
        quantiles = np.quantile(data, per)

        splits = np.array_split(data, word_length)
        repr = [x.mean() for x in splits]
        alphabet = list(string.ascii_lowercase[0:len(per)])

        sax = []
        for x in repr:
            i = np.argmax(quantiles > x)
            sax.append(alphabet[i-1])

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(data)
        levels = np.repeat(repr, int(len(data)/word_length))
        ax.plot(levels)
        # just inserted the letters in sax for visualization purposes
        ax.text(5, 10, sax[0])
        ax.text(18, 20, sax[1])
        plt.show()

        result = mot.create_esax(data, quantiles, word_length)[0]
        self.assertEqual(len(result), word_length)
        self.assertCountEqual(result, sax)

    def test_create_esax_time_series(self):
        data = np.array(range(0,1000))
        ts_subs = []
        ecdf = mot.get_ecdf(data)
        qq = np.linspace(start=0, stop=1, num=10 + 1)
        quantiles = np.quantile(ecdf[0], qq)

        ts_subs = np.array_split(data, 10)
        w = 11
        alphabet = list(string.ascii_lowercase[0:10])

        startpoints = [0,100,200,300,400,500,600,700,800,900]

        esax = []
        for i in range(0,10):
            esax.append(np.repeat(alphabet[i], w))

        columns_index = pd.Index([0,1,2,3,4,5,6,7,8,9,10], dtype=object)
        esax_ts_df_comp = pd.DataFrame(index=startpoints, data=esax, columns=columns_index)

        esax_ts_df_comp.index.name = 'StartP'
        esax_ts_df, _ = mot.create_esax_time_series(ts_subs, w, quantiles)

        pd.testing.assert_frame_equal(esax_ts_df, esax_ts_df_comp)


    def test_random_projection(self):
        letters = list(string.ascii_lowercase)
        length = 20
        disjoint = {}
        num_seq = 6
        for i in range(0, num_seq-1):
            disjoint[i] = np.repeat(letters[i], length)

        disjoint[num_seq] = np.repeat(letters[1], length)

        sim_ts_sax_df = pd.DataFrame(disjoint.values(), index=disjoint.keys(),
                                     columns=list(range(0, length)), dtype=object)

        col_mat = mot.perform_random_projection(sim_ts_sax_df, num_iterations=2, mask_size=2, seed=42)

        sample_indices = []
        random.seed(42 + 0)
        sample_indices.append(random.sample(list(range(0, length)), 2))
        random.seed(42 + 1)
        sample_indices.append(random.sample(list(range(0, length)), 2))

        ref_col_mat = np.zeros((num_seq, num_seq), dtype=int)

        # set the values at the sample indices in row 1 and row 6 to one because those two are similar
        ref_col_mat[1, 5] = 2
        ref_col_mat = pd.DataFrame(ref_col_mat, dtype=int)

        pd.testing.assert_frame_equal(ref_col_mat, col_mat)


    def test_extract_motif_pair(self):
        letters = list(string.ascii_lowercase)
        length = 2
        ts_sax_df = {}
        num_seq_a = 10
        num_seq_b = 10
        ts_subs = []

        for ind in range(0,20):
            ts_subs.append(np.array([ind]))

        for i in range(0, num_seq_a):
            ts_sax_df[i] = np.repeat(letters[0], length)

        for j in range(10, 10+num_seq_b):
            ts_sax_df[j] = np.repeat(letters[1], length)

        ts_sax_df = pd.DataFrame(ts_sax_df.values(), index=ts_sax_df.keys(), columns=list(range(0, length)), dtype=object)
        col_mat = mot.perform_random_projection(ts_sax_df, num_iterations=2, mask_size=2, seed=42)

        indexes = mot.extract_motif_pair(ts_sax_df, col_mat, ts_subs, num_iterations=2)

        comp = [{0,1,2,3,4,5,6,7,8,9},{10,11,12,13,14,15,16,17,18,19}]
        self.assertEqual(indexes, comp)

        # Trying out different length of col_mat than number of periods
        ts_sax_df = {}
        for i in range(0, num_seq_a):
            ts_sax_df[i] = np.repeat(letters[0], length)

        ts_sax_df = pd.DataFrame(ts_sax_df.values(), index=ts_sax_df.keys(), columns=list(range(0, length)), dtype=object)
        col_mat = mot.perform_random_projection(ts_sax_df, num_iterations=2, mask_size=2, seed=42)

        indexes = mot.extract_motif_pair(ts_sax_df, col_mat, ts_subs, num_iterations=2)
        self.assertNotEqual(indexes, comp)


if __name__ == '__main__':
    unittest.main()
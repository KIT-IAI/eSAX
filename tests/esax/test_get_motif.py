import string
import unittest
import pandas as pd
import numpy as np
import esax.get_motif as mot
import matplotlib.pyplot as plt

class Motif_tests(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_ecdf(self):
        self.data = np.repeat(np.NaN, 100)
        self.assertRaises(TypeError, mot.get_ecdf(self.data))

    def test_index_merge(self):
        self.lists = [[1,2,3,4,5], [5,6,7,8,9], [9,10,11,12,13], [13,14,15,16,17]]
        self.result = [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}]
        self.assertCountEqual(mot.index_merge(self.lists), self.result)

    def test_create_esax(self):
        self.data = np.array(range(1,25))
        #self.data0 = np.zeros(6)
        #self.data1 = np.ones(6)
        #self.data = np.concatenate((self.data0, self.data1))
        #self.data = np.concatenate((self.data, self.data))
        self.word_length = 2
        self.per = np.linspace(0, 1, 24)
        self.quantiles = np.quantile(self.data, self.per)

        self.splits = np.array_split(self.data, self.word_length)
        self.repr = [x.mean() for x in self.splits]
        self.alphabet = list(string.ascii_lowercase[0:len(self.per)])

        self.sax = []
        for x in self.repr:
            i = np.argmax(self.quantiles > x)
            self.sax.append(self.alphabet[i-1])

        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(self.data)
        levels = np.repeat(self.repr, int(len(self.data)/self.word_length))
        ax.plot(levels)
        plt.show()

        self.result = mot.create_esax(self.data, self.quantiles, self.word_length)[0]
        self.assertEqual(len(self.result), self.word_length)
        self.assertCountEqual(self.result, self.sax)


    def test_random_projection(self):
        self.data = np.array(list(string.ascii_lowercase[0:2]))
        self.data = self.data.repeat(125)
        self.data = self.data.reshape(5, 50)
        self.data = pd.DataFrame(self.data)
        self.num_iterations = 2
        self.mask_size = 2
        self.result = mot.perform_random_projection(self.data, self.num_iterations, self.mask_size)

    def test_extract_motif_pair(self):
        pass
        #self.result = mot.extract_motif_pair(self.data, col_mat, ts_subs, num_iterations, count_ratio_1=5.0,
        #              count_ratio_2=1.5, max_dist_ratio=2.5):

if __name__ == '__main__':
    unittest.main()
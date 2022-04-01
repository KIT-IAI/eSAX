import os
import random
import string
import unittest
import pandas as pd
import numpy as np
import esax.get_motif as mot
import esax.get_subsequences as subs
import esax.plots as plots
import matplotlib.pyplot as plt


class TestESAX(unittest.TestCase):

    def test_eSAX(self):
        two_days = np.repeat([0,1], 24)
        data = np.array([])
        # generates data consisting of 200 days (alternating between 0 and 1)
        for i in range(0,100):
            data = np.concatenate((data, two_days))

        data = pd.Series(data)
        resolution = 1

        # Get subsequences
        ts_subs, startpoints, indexes_subs = subs.get_subsequences(data, resolution)

        # Get motifs
        if ts_subs:
            found_motifs = mot.get_motifs(data, ts_subs, breaks=5, word_length=10, num_iterations=0,
                                          mask_size=2, mdr=30, cr1=20, cr2=20)
            if found_motifs:
                plots.plot_ecdf(found_motifs['ecdf'], '../../run')
                plots.plot_motifs(data.index, found_motifs['motifs_raw'], found_motifs['indexes'], '../../run')
                plots.plot_repr_motif(found_motifs['motifs_raw'], '../../run')
                self.assertEqual(found_motifs['motifs_raw'][0].shape[0], 200)


if __name__ == '__main__':
    print(os.getcwd())
    unittest.main()

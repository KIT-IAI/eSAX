import esax.get_subsequences as subs
import numpy as np
import pytest
import unittest
import os
import pandas as pd


class Subsequence_Tests(unittest.TestCase):

    def setUp(self):
        pass


    def test_local_min_none(self):
        index = np.array(range(0, 8760, 24))
        data = np.ones(8760)
        data[index] = 0
        data = pd.Series(data)

        window = 24
        dmin, localmin, indexes_subs = subs.determine_subsequences(data, 'none', window)
        self.assertIsInstance(dmin, list)
        self.assertEqual(len(dmin), 365)
        self.assertIsInstance(localmin, list)
        self.assertIsInstance(indexes_subs, list)

        for sequence in dmin:
            self.assertTrue(len(sequence), 24)
            self.assertEqual(sequence[0], 0)


    def test_local_min_with_range(self):
        data = np.array(range(0,8760))
        data = pd.Series(data)
        window = 24
        dmin, localmin, indexes_subs = subs.determine_subsequences(data, 'minimum', window)

        # The parameter window_size in the head of determine_subsequences is 100 per default (maybe not reasonable)
        # Also the minima are part of the period before, which is why the startpoints of each period
        # begins 1 position after the minimum
        self.assertEqual(len(dmin), 88)


        for idx, seq in enumerate(dmin):
            if idx > 0:
                self.assertEqual(seq[0], (idx - 1) * 100 + 1)
            else:
                pass


    def test_local_min_with_range_extra(self):
        self.data = np.array(range(0, 8760))
        self.index = list(range(50, 8750, 100))
        self.index.append(8759)
        self.data[self.index] = -1
        self.data = pd.Series(self.data)
        self.window = 24
        self.dmin, self.localmin, self.indexes_subs = subs.determine_subsequences(self.data, 'minimum', self.window)

        # The parameter window_size in the head of determine_subsequences is 100 per default (maybe not reasonable)
        # Also the minima are part of the period before, which is why the startpoints of each period
        # begins 1 position after the minimum
        self.assertEqual(len(self.dmin), 88)

        for idx, seq in enumerate(self.dmin):
            self.assertEqual(seq[len(seq)-1], -1)


    def test_get_subsequences(self):
        self.data = np.empty(1000)
        self.data[:] = np.NaN
        self.data = pd.Series(self.data)
        self.resolution = 1

        self.assertRaises(TypeError, subs.get_subsequences(self.data, self.resolution))



if __name__ == '__main__':
    unittest.main()
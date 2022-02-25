import random
import string
import unittest
import pandas as pd
import numpy as np
import esax.get_motif as mot
import matplotlib.pyplot as plt


class Test_eSAX(unittest.TestCase):

    def test_eSAX(self):
        two_days = np.repeat([0,1], 24)
        data = np.array([])
        for i in range(0,100):
        data = data.concatenate(data)
        print()
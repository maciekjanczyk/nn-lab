import unittest
from bcm import BCM
import numpy as np


class TestBCM(unittest.TestCase):

    def test_createCorrectMatrix(self):
        w1 = [1, 1, 0, 0, 0, 0]
        w2 = [0, 1, 0, 0, 0, 1]
        bcm = BCM(6)
        bcm.remember_vector(w1)
        bcm.remember_vector(w2)
        expected = np.zeros([6, 6])
        expected[0] = [1, 1, 0, 0, 0, 0]
        expected[1] = [1, 1, 0, 0, 0, 1]
        expected[5] = [0, 1, 0, 0, 0, 1]
        for i in range(0, 6):
            for j in range(0, 6):
                self.assertEqual(bcm.w[i][j], expected[i][j])

    def test_returnsCorrectCheckingAnswer(self):
        w1 = [1, 1, 0, 0, 0, 0]
        w2 = [0, 1, 0, 0, 0, 1]
        bcm = BCM(6)
        bcm.remember_vector(w1)
        bcm.remember_vector(w2)
        a1 = bcm.check_if_known([0, 1, 0, 0, 0, 1])
        a2 = bcm.check_if_known([0, 1, 0, 1, 0, 0])
        b1 = [0, 1, 0, 0, 0, 1]
        b2 = [0, 0, 0, 0, 0, 0]
        for i in range(0, 6):
            self.assertEqual(a1[i], b1[i])
            self.assertEqual(a2[i], b2[i])


if __name__ == '__main__':
    unittest.main()
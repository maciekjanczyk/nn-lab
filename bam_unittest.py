import unittest
from bam import BAM


class Test(unittest.TestCase):

    def test_makeCorrectMatrix(self):
        bam = BAM(6, 4)
        bam.remember([0, 1, 1, 1, 0, 1], [1, 1, 0, 0])
        bam.remember([1, 0, 0, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(bam.w[0].tolist(), [0, -2, 2, 0])
        self.assertEqual(bam.w[1].tolist(), [0, 2, -2, 0])
        self.assertEqual(bam.w[2].tolist(), [0, 2, -2, 0])
        self.assertEqual(bam.w[3].tolist(), [2, 0, 0, -2])
        self.assertEqual(bam.w[4].tolist(), [-2, 0, 0, 2])
        self.assertEqual(bam.w[5].tolist(), [0, 2, -2, 0])

    def test_returnsCorrectNameForGivenImage(self):
        bam = BAM(6, 4)
        bam.remember([0, 1, 1, 1, 0, 1], [1, 1, 0, 0])
        bam.remember([1, 0, 0, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(bam.recover_image([1, 1, 0, 0]).tolist(), [0, 1, 1, 1, 0, 1])
        self.assertEqual(bam.recover_image([1, 0, 1, 0]).tolist(), [1, 0, 0, 1, 0, 0])

    def test_returnsCorrectImageForGivenName(self):
        bam = BAM(6, 4)
        bam.remember([0, 1, 1, 1, 0, 1], [1, 1, 0, 0])
        bam.remember([1, 0, 0, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(bam.recover_name([0, 1, 1, 1, 0, 1]).tolist(), [1, 1, 0, 0])
        self.assertEqual(bam.recover_name([1, 0, 0, 1, 0, 0]).tolist(), [1, 0, 1, 0])
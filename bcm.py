from __future__ import print_function
import numpy as np


class BCM:

    def __init__(self, size):
        self.w = self.w = np.zeros([size, size])

    def remember_vector(self, wek):
        if len(wek) != len(self.w):
            return None

        w_tmp = np.zeros([len(wek), len(wek)])
        for i in range(0, len(wek)):
            for j in range(0, len(wek)):
                if wek[i] and wek[j]:
                    w_tmp[i][j] = 1
                w_tmp[i][j] = w_tmp[i][j] or self.w[i][j]

        self.w = w_tmp

    def check_if_known(self, wek):
        wtrans = self.w.transpose().T
        a = np.zeros(len(wek))
        for i in range(0, len(wek)):
            if wek[i] == 1:
                a += wtrans[i]

        for i in range(0, len(a)):
            if a[i] > 1:
                a[i] = 1
            else:
                a[i] = 0

        return a

if __name__ == '__main__':
    None
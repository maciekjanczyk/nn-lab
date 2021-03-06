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


def czytaj_z_pliku(name):
    f = open(name, 'r')
    ret = []
    for line in f:
        splitted = line.split('\t')
        ret2 = []
        for ch in splitted:
            ret2.append(int(ch))
        ret.append(ret2)
    return ret


if __name__ == '__main__':
    x = czytaj_z_pliku('./data/BCM_simple6.txt')
    bcm = BCM(len(x[0]))
    for xx in x:
        print(str.format("Uczenie wektora {0}...", xx))
        bcm.remember_vector(xx)
    for xx in x:
        print(str.format("Trwa odtwarzanie wektora {0}...", xx))
        print(str.format("Wynik: {0}...", bcm.check_if_known(xx).tolist()))

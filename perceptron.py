import math
import matplotlib.pyplot as plt
import random


def NormalizujWektory(vs):
    max_v = 0.0
    for v in vs:
        for x in v:
            if math.fabs(x) > max_v:
                max_v = math.fabs(x)
    for v in vs:
        v[0] /= max_v
        v[1] /= max_v


class Perceptron2D:
    def __init__(self, w1, w2, theta):
        self.x0 = 1.0
        self.x1 = 0.0
        self.x2 = 0.0
        self.w1 = w1
        self.w2 = w2
        self.theta = theta

    @staticmethod
    def f(s):
        if s > 0.0:
            return 1.0
        else:
            return -1.0

    def PodajNaWejscie(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def Wyjscie(self):
        suma = self.x1 * self.w1 + self.x2 * self.w2
        return Perceptron2D.f(suma + self.theta)

    def Uczenie(self, wektory, wzorce, max_iter):
        nauczone = False
        iter = -1
        w_tmp = []
        for w in wektory:
            w_tmp.append(w)
        #NormalizujWektory(w_tmp)
        while (not nauczone) and (iter <= max_iter):
            iter += 1
            ii = -1
            for w in w_tmp:
                ii += 1
                self.PodajNaWejscie(w[0], w[1])
                wyjscie = self.Wyjscie()
                if wyjscie != wzorce[ii]:
                    self.w1 += w[0] * wzorce[ii]
                    self.w2 += w[1] * wzorce[ii]
                    self.theta += self.x0 * wzorce[ii]
            test = []
            for w in w_tmp:
                self.PodajNaWejscie(w[0], w[1])
                test.append(self.Wyjscie())
            ile_zgadza = 0
            ii = 0
            for v in test:
                if v == wzorce[ii]:
                    ile_zgadza += 1
                ii += 1
            if ile_zgadza == len(wzorce):
                nauczone = True

    def WyznaczPunktDlaProstej(self, x11):
        return - self.w1 / self.w2 * x11 - self.theta / self.w2

    def Klasyfikuj(self, vecs):
        max_v = 0.0
        xs = []
        ys = []
        for v in vecs:
            if math.fabs(v[0]) > max_v:
                max_v = math.fabs(v[0])
            xs.append(v[0])
            ys.append(v[1])
        max_v += max_v / 10.0
        x2 = [-1.0 * math.fabs(max_v), math.fabs(max_v)]
        y2 = [self.WyznaczPunktDlaProstej(x2[0]), self.WyznaczPunktDlaProstej(x2[1])]
        plt.plot(xs, ys, "o", x2, y2, "-")
        plt.show()


if __name__ == '__main__':
    #perc = Perceptron2D(0.5, 0.1, 0.5)
    perc = Perceptron2D(random.uniform(0.01, 0.4), random.uniform(0.01, 0.4), 0.03)
    vecs = [[2.0, 1.0], [2.0, 2.0], [0.0, 6.0], [-2.0, 8.0], [-2.0, 0.0], [0.0, 0.0], [4.0, -20.0]]
    wzorc = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    perc.Uczenie(vecs, wzorc, 1000)
    perc.Klasyfikuj(vecs)

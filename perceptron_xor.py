import math
import matplotlib.pyplot as plt
import random
import time


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

    def Uczenie(self, wektory, wzorce):
        nauczone = False
        w_tmp = []
        for w in wektory:
            w_tmp.append(w)
        # NormalizujWektory(w_tmp)
        while not nauczone:
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
        suma = self.x1 * self.w1 + self.x2 * self.w2 + self.theta
        return Perceptron2D.f(suma)

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


class SiecXOR:
    def __init__(self, theta1, theta2, theta3):
        self.theta = theta1
        self.warstwa1 = []
        self.warstwa1.append(Perceptron2D(random.uniform(0.0, 0.1), random.uniform(0.0, 0.1), theta1))
        #self.warstwa1.append(Perceptron2D(-0.89, -0.95, theta))
        self.warstwa1.append(Perceptron2D(random.uniform(-0.1, -0.001), random.uniform(-0.1, -0.001), theta2))
        #self.warstwa1.append(Perceptron2D(0.38, 0.33, theta))
        self.warstwa2 = Perceptron2D(random.uniform(0.1, 0.4), random.uniform(0.1, 0.4), theta3)
        #self.warstwa2 = Perceptron2D(-2.3, -1.8, theta)
        self.kieszonka = []
        self.kieszonka_c = 0

    def PodajNaWejscie(self, x1, x2):
        w1 = self.warstwa1[0]
        w2 = self.warstwa1[1]
        w1.PodajNaWejscie(x1, x2)
        w2.PodajNaWejscie(x1, x2)
        self.warstwa2.PodajNaWejscie(Perceptron2D.f(w1.Wyjscie()), Perceptron2D.f(w2.Wyjscie()))

    def Wyjscie(self):
        x1 = self.warstwa1[0].Wyjscie()
        x2 = self.warstwa1[1].Wyjscie()
        self.warstwa2.PodajNaWejscie(x1, x2)
        return self.warstwa2.Wyjscie()

    def Uczenie(self, wektory, wzorce, epoki):
        nauczone = False
        _epoki = 0
        w_tmp = []
        for w in wektory:
            w_tmp.append(w)
        # NormalizujWektory(w_tmp)
        while (not nauczone) and (_epoki <= epoki):
            _epoki += 1
            print _epoki
            ii = -1
            for w in w_tmp:
                ii += 1
                self.PodajNaWejscie(w[0], w[1])
                wyjscie = self.Wyjscie()
                if wyjscie != wzorce[ii]:
                    self.warstwa1[0].w1 += w[0] * wzorce[ii]
                    self.warstwa1[1].w1 += w[0] * wzorce[ii]
                    #self.warstwa2.w1 += Perceptron2D.f(self.warstwa1[0].Wyjscie()) * wzorce[ii]
                    self.warstwa2.w1 += w[0] * wzorce[ii]
                    self.warstwa1[0].w2 += w[1] * wzorce[ii]
                    self.warstwa1[1].w2 += w[1] * wzorce[ii]
                    #self.warstwa2.w2 += Perceptron2D.f(self.warstwa1[1].Wyjscie()) * wzorce[ii]
                    self.warstwa2.w2 += w[1] * wzorce[ii]
                    self.theta += self.theta * wzorce[ii]
                    self.warstwa1[0].theta = self.warstwa1[0].theta * wzorce[ii]
                    self.warstwa1[1].theta = self.warstwa1[1].theta * wzorce[ii]
                    self.warstwa2.theta = self.warstwa2.theta * wzorce[ii]
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
            if len(self.kieszonka) == 0:
                self.kieszonka.append(self.warstwa1[0].w1)
                self.kieszonka.append(self.warstwa1[0].w2)
                self.kieszonka.append(self.warstwa1[1].w1)
                self.kieszonka.append(self.warstwa1[1].w2)
                self.kieszonka.append(self.warstwa2.w1)
                self.kieszonka.append(self.warstwa2.w2)
                self.kieszonka_c = ile_zgadza
                '''elif ile_zgadza < self.kieszonka_c:
                    self.warstwa1[0].w1 = self.kieszonka[0]
                    self.warstwa1[0].w2 = self.kieszonka[1]
                    self.warstwa1[1].w1 = self.kieszonka[2]
                    self.warstwa1[1].w2 = self.kieszonka[3]
                    self.warstwa2.w1 = self.kieszonka[4]
                    self.warstwa2.w2 = self.kieszonka[5]'''
            else:
                self.kieszonka[0] = self.warstwa1[0].w1
                self.kieszonka[1] = self.warstwa1[0].w2
                self.kieszonka[2] = self.warstwa1[1].w1
                self.kieszonka[3] = self.warstwa1[1].w2
                self.kieszonka[4] = self.warstwa2.w1
                self.kieszonka[5] = self.warstwa2.w2
                self.kieszonka_c = ile_zgadza
            if ile_zgadza == len(wzorce):
                nauczone = True
            self.Klasyfikuj(wektory, _epoki)

    def Klasyfikuj(self, vecs, epos):
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
        y2 = [self.warstwa1[0].WyznaczPunktDlaProstej(x2[0]), self.warstwa1[0].WyznaczPunktDlaProstej(x2[1])]
        y3 = [self.warstwa1[1].WyznaczPunktDlaProstej(x2[0]), self.warstwa1[1].WyznaczPunktDlaProstej(x2[1])]
        y4 = [self.warstwa2.WyznaczPunktDlaProstej(x2[0]), self.warstwa2.WyznaczPunktDlaProstej(x2[1])]
        plt.plot(xs, ys, "o", x2, y2, "-", x2, y3, "-", x2, y4, "-")
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.savefig('./results/{0}.png'.format(epos))
        plt.clf()
        return plt


if __name__ == '__main__':
    perc = SiecXOR(-0.02, 0.1, 0.03)
    #vecs = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]]
    vecs = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    #wzorc = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
    wzorc = [-1.0, 1.0, 1.0, -1.0]
    perc.Uczenie(vecs, wzorc, 1000)
    perc.Klasyfikuj(vecs, "last")
    for vec in vecs:
        perc.PodajNaWejscie(vec[0], vec[1])
        print("[%f, %f] => %f" % (vec[0], vec[1], perc.warstwa2.Wyjscie()))

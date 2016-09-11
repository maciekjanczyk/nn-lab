import math
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


class Perceptron:
    def __init__(self, w1, w2):
        self.x1 = 0.0
        self.x2 = 0.0
        self.w1 = w1
        self.w2 = w2

    @staticmethod
    def f(s):
        if s > 0.0:
            return 1.0
        else:
            return -1.0

    @staticmethod
    def sigmoid(s):
        t = float(s)
        try:
            return 1.0 / (1.0 + math.exp(-1.0 * t))
        except Exception:
            if s > 0:
                return 1.0
            else:
                return 0.0

    def PodajNaWejscie(self, vec):
        self.x1 = vec[0]
        self.x2 = vec[1]

    def Wyjscie(self):
        suma = self.x1 * self.w1 + self.x2 * self.w2
        return Perceptron.sigmoid(suma)


class SiecXOR:
    def __init__(self, theta):
        self.warstwa = []
        self.warstwa.append(Perceptron(0.1, 0.8))
        self.warstwa.append(Perceptron(0.4, 0.6))
        self.output = Perceptron(0.3, 0.9)
        self.EPOKI = 0
        self.theta = theta

    def PodajNaWejscie(self, w):
        for war in self.warstwa:
            war.PodajNaWejscie(w)
        wyj1 = self.warstwa[0].Wyjscie()
        wyj2 = self.warstwa[1].Wyjscie()
        self.output.PodajNaWejscie([wyj1, wyj2])

    def Wyjscie(self):
        return self.output.Wyjscie()

    def Uczenie(self, vecs, wzorc, iters):
        self.EPOKI = iters
        for k in range(0, self.EPOKI):
            for i in range(0, len(vecs)):
                self.PodajNaWejscie(vecs[i])
                # warstwa wyjsciowa
                outO = self.output.Wyjscie()
                if outO == wzorc[i]:
                    continue
                sigmaO = outO * (1.0 - outO) * (wzorc[i] - outO)
                self.output.w1 += self.theta * sigmaO * self.warstwa[0].Wyjscie()
                self.output.w2 += self.theta * sigmaO * self.warstwa[1].Wyjscie()
                # warstwa ukryta
                outA = self.warstwa[0].Wyjscie()
                sigmaA = outA * (1.0 - outA) * (sigmaO * self.output.w1)
                outB = self.warstwa[1].Wyjscie()
                sigmaB = outB * (1.0 - outB) * (sigmaO * self.output.w2)
                self.warstwa[0].w1 += self.theta * sigmaA * vecs[i][0]
                self.warstwa[0].w2 += self.theta * sigmaA * vecs[i][1]
                self.warstwa[1].w1 += self.theta * sigmaB * vecs[i][0]
                self.warstwa[1].w2 += self.theta * sigmaB * vecs[i][1]
        print("OK")

    def Klasyfikuj(self, vecs, res):
        ret = []
        i = 0
        for v in vecs:
            self.PodajNaWejscie(v)
            wyjscie = self.Wyjscie()
            ret.append(wyjscie)
            print(str.format("Vec: {0}, Test: {1}, Result: {2}", v, wyjscie, res[i]))
            i += 1
        return ret



if __name__ == '__main__':
    perc = SiecXOR(1.0)
    vecs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    wzorc = [0.0, 1.0, 1.0, 0.0]
    print("Trwa uczenie standardowego zestawu...")
    perc.Uczenie(vecs, wzorc, 40000)
    perc.Klasyfikuj(vecs, wzorc)
    perc = SiecXOR(1.0)
    vecs = [[1.0, 1.0], [1.0, 3.0], [3.0, 1.0], [3.0, 3.0],
            [0.0, 0.0], [0.0, 4.0], [4.0, 0.0], [4.0, 4.0]]
    wzorc = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    print("Trwa uczenie rozszerzonego zestawu...")
    perc.Uczenie(vecs, wzorc, 20000)
    perc.Klasyfikuj(vecs, wzorc)

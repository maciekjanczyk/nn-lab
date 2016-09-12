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
    def __init__(self, wags):
        self.w = []
        self.x = []
        for wag in wags:
            self.w.append(wag)

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
        self.x = vec

    def Wyjscie(self):
        suma = 0
        for i in range(0, len(self.x)):
            suma += self.x[i] * self.w[i]
        return Perceptron.sigmoid(suma)


class SiecXOR:
    def __init__(self, theta, dim=2):
        self.warstwa = []
        self.DIM = dim
        for i in range(0, self.DIM):
            wags = []
            for j in range(0, self.DIM):
                wags.append(random.uniform(0.05, 0.2))
            self.warstwa.append(Perceptron(wags))
        # self.warstwa.append(Perceptron(0.4, 0.6))
        wags = []
        for j in range(0, self.DIM):
            wags.append(random.uniform(0.05, 0.2))
        self.output = Perceptron(wags)
        self.EPOKI = 0
        self.theta = theta

    def PodajNaWejscie(self, w):
        wyj = []
        for war in self.warstwa:
            war.PodajNaWejscie(w)
            wyj.append(war.Wyjscie())
        self.output.PodajNaWejscie(wyj)

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
                for ii in range(0, self.DIM):
                    self.output.w[ii] += self.theta * sigmaO * self.warstwa[ii].Wyjscie()
                # warstwa ukryta
                sigmas = []
                for ii in range(0, self.DIM):
                    outW = self.warstwa[ii].Wyjscie()
                    sigmas.append(outW * (1.0 - outW) * (sigmaO * self.output.w[ii]))
                for ii in range(0, self.DIM):
                    for j in range(0, self.DIM):
                        self.warstwa[ii].w[j] += self.theta * sigmas[ii] * vecs[i][j]

    def Klasyfikuj(self, vecs, res):
        ret = []
        i = 0
        przeszlo = 0
        for v in vecs:
            self.PodajNaWejscie(v)
            wyjscie = self.Wyjscie()
            ret.append(wyjscie)
            print(str.format("Vec: {0}, Test: {1}, Result: {2}", v, wyjscie, res[i]))
            if (res[i] == 1.0 and wyjscie > 0.5) or (res[i] == 0.0 and wyjscie <= 0.5):
                przeszlo += 1
            i += 1
        print("Wynik testu: {0}%".format(float(przeszlo) / float(len(vecs)) * 100.0))
        return ret


class ZestawDanych:
    def __init__(self, plik1, plik2):
        # pozycja[0] to id
        # pozycja[10] to klasa
        # plik1 - zestaw treningowy
        # plik2 - zestaw testowy
        self.x_trening = None
        self.y_trening = None
        self.x_testy = None
        self.y_testy = None
        with open(plik1, 'r') as f:
            linie = f.readlines()
            xt = []
            yt = []
            j = 0
            for linia in linie:
                rozwalone = linia.split(',')
                lista = []
                for i in range(1, 10):
                    lista.append(float(rozwalone[i]))
                xt.append(lista)
                if float(rozwalone[10][0]) == 2.0:
                    yt.append(0.0)
                else:
                    yt.append(1.0)
                j += 1
            self.x_trening = xt
            self.y_trening = yt
        with open(plik2, 'r') as f:
            linie = f.readlines()
            xt = []
            yt = []
            j = 0
            for linia in linie:
                rozwalone = linia.split(',')
                lista = []
                for i in range(1, 10):
                    lista.append(float(rozwalone[i]))
                xt.append(lista)
                if float(rozwalone[10][0]) == 2.0:
                    yt.append(0.0)
                else:
                    yt.append(1.0)
                j += 1
            self.x_testy = xt
            self.y_testy = yt

    def zwroc_dane(self):
        return (self.x_trening, self.y_trening, self.x_testy, self.y_testy)


if __name__ == '__main__':
    perc = SiecXOR(0.3, 9)
    # vecs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    # wzorc = [0.0, 1.0, 1.0, 0.0]
    #vecs = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    #wzorc = [0.0, 1.0, 1.0, 0.0]
    dane = ZestawDanych('./data/trening.data', './data/test.data')
    vecs, wzorc, xt, yt = dane.zwroc_dane()
    print("Trwa uczenie standardowego zestawu...")
    perc.Uczenie(vecs, wzorc, 100)
    perc.Klasyfikuj(xt, yt)

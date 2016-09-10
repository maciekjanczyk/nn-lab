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
        v[2] /= max_v
        v[3] /= max_v


class Perceptron2D:
    def __init__(self, w1, w2, w3, w4, theta ,wuczenia):
        self.x1 = 0.0
        self.x2 = 0.0
        self.x3 = 0.0
        self.x4 = 0.0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.theta = theta
        self.p = wuczenia
        self.kieszonka = []
        self.kieszonka_c = -1.0

    @staticmethod
    def f(s):
        if s > 0.0:
            return 1.0
        else:
            return -1.0

    '''def PodajNaWejscie(self, x1, x2, x3, x4):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4'''

    def PodajNaWejscie(self, vec):
        self.x1 = vec[0]
        self.x2 = vec[1]
        self.x3 = vec[2]
        self.x4 = vec[3]

    def Wyjscie(self):
        suma = self.x1 * self.w1 + self.x2 * self.w2 + self.x3 * self.w3 + self.x4 * self.w4
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
                self.PodajNaWejscie(w)
                wyjscie = self.Wyjscie()
                if wyjscie != wzorce[ii]:
                    self.w1 += w[0] * wzorce[ii] * self.p
                    self.w2 += w[1] * wzorce[ii] * self.p
                    self.w3 += w[2] * wzorce[ii] * self.p
                    self.w4 += w[3] * wzorce[ii] * self.p
                    self.theta += wzorce[ii]
            test = []
            for w in w_tmp:
                #self.PodajNaWejscie(w[0], w[1])
                self.PodajNaWejscie(w)
                test.append(self.Wyjscie())
            ile_zgadza = 0
            ii = 0
            '''for v in test:
                if v == wzorce[ii]:
                    ile_zgadza += 1
                ii += 1
            if len(self.kieszonka) == 0:
                self.kieszonka.append(self.w1)
                self.kieszonka.append(self.w2)
                self.kieszonka.append(self.w3)
                self.kieszonka.append(self.w4)
                self.kieszonka.append(self.theta)
            elif ile_zgadza < self.kieszonka_c:
                self.w1 = self.kieszonka[0]
                self.w2 = self.kieszonka[1]
                self.w3 = self.kieszonka[2]
                self.w4 = self.kieszonka[3]
                self.theta = self.kieszonka[4]
            else:
                self.kieszonka[0] = self.w1
                self.kieszonka[1] = self.w2
                self.kieszonka[2] = self.w3
                self.kieszonka[3] = self.w4
                self.kieszonka[4] = self.theta
                self.kieszonka_c = ile_zgadza'''
            if ile_zgadza == len(wzorce):
                nauczone = True

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


def czytaj_z_pliku(name):
    f = open(name, 'r')
    ret = []
    ret2 = []
    for line in f:
        splitted = line.split('\t')
        vec = []
        for i in range(0, len(splitted) - 1):
            vec.append(float(splitted[i]))
        ret.append(vec)
        ret2.append(float(splitted[len(splitted) - 1]))
    return ret, ret2


def score(test, res):
    cnt1 = len(test)
    cnt2 = 0
    for i in range(0, len(test)):
        if test[i] == res[i]:
            cnt2 += 1
    return float(cnt2) / float(cnt1) * 100.0


if __name__ == '__main__':
    for letter in ['A', 'B', 'C', 'D', 'E']:
        perc = Perceptron2D(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1),
                            random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 1.0, 1.0)
        vecs, wzorc = czytaj_z_pliku('./data/iris_2vs3_{0}_tr.txt'.format(letter))
        print(str.format("Trwa uczenie zestawu iris_2vs3_{0}...".format(letter)))
        perc.Uczenie(vecs, wzorc, 1000)
        test, test_y = czytaj_z_pliku('./data/iris_2vs3_{0}_te.txt'.format(letter))
        klas = perc.Klasyfikuj(test, test_y)
        print(str.format("Wynik testu: {0}%", score(test_y, klas)))


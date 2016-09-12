import numpy.random as random
import math
import matplotlib.pyplot as plt


class Wezel:
    def __init__(self, x, y, p):
        self.x = x
        self.y = y
        self.p = p

    def __getitem__(self, item):
        if item == 0:
            return self.x
        if item == 1:
            return self.y

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y

    def X(self):
        return self.x

    def Y(self):
        return self.y

    def wyjscie(self, v):
        return self.x * v[0] + self.y * v[1]


class SOM:
    def __init__(self, points_count):
        self.N = points_count
        self.W = []
        self.winner = None
        self.winner_idx = None
        self.currentVec = None
        self.eta = 0.005
        self.p_min = 0.9999
        #self.promien = 0.03
        self.promien = 5
        self.poletr = self.ptr_pkt([0.0, 0.0], [0.5, 1.0], [1.0, 0.0])
        self.T = 1
        self.EPOKI = 0
        self.K = 0.4
        self.wuczenia = 0.2
        self.vecSet = []

    def glownaProcedura(self, epoki):
        self.EPOKI = epoki
        self.losujWektoryWejsciowe()
        fig, ax = plt.subplots()
        self.dodajPunkty()
        x=[]
        y=[]
        for w in self.W:
            x.append(w[0])
            y.append(w[1])
        points, = ax.plot(x, y, "o-")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.pause(0.0000001)
        for i in range(0, epoki, 1):
            self.pojedynczaIteracja()
            if self.T % 10 != 0:
                continue
            x=[]
            y=[]
            for w in self.W:
                x.append(w.x)
                y.append(w.y)
            points.set_data(x, y)
            plt.pause(0.0000001)
        plt.plot(x, y, "o-")
        plt.show()

    def pojedynczaIteracja(self):
        self.winner = None
        self.currentVec = None
        self.currentVec = self.vecSet.pop()
        self.wylonWygranego()
        self.modyfikujWagiZwyciezcy()
        self.T += 1
        self.K *= 0.99995
        if self.promien > 1.0:
            self.promien *= 0.99995
        elif self.promien < 1.0:
            self.promien = 1.0
        if self.T % 500 == 0:
            print("Epoka: {0}, K: {1}, Promien: {2}".format(self.T, self.K, self.promien))

    def ptr_pkt(self, p1, p2, p3):
        p1p2 = self.__d(p1, p2)
        p2p3 = self.__d(p2, p3)
        p3p1 = self.__d(p3, p1)
        return self.p_trojkata(p1p2, p2p3, p3p1)

    def czyNalezyDoTr(self, p):
        p1 = self.ptr_pkt([0.0, 0.0], [0.5, 1.0], p)
        p2 = self.ptr_pkt([0.0, 0.0], [1.0, 0.0], p)
        p3 = self.ptr_pkt([0.5, 1.0], [1.0, 0.0], p)
        if (p1 + p2 + p3) == self.ptr_pkt([0.0, 0.0], [0.5, 1.0], [1.0, 0.0]):
            return True
        else:
            return False

    def dodajPunkty(self):
        for i in range(0, self.N, 1):
            v = [random.uniform(0.375, 0.625), random.uniform(0.25, 0.5)]
            self.W.append(Wezel(v[0], v[1], 1.0))

    def fsasiedztwa(self, x, i):
        if self.__d(x, i) <= self.promien:
            return 1
        else:
            return 0

    def neuron_d(self, n1, n2):
        ret = 0
        naliczanie = False
        its = None
        for i in range(0, len(self.W)):
            if naliczanie:
                if self.W[i] != its:
                    ret += 1
                else:
                    break
                continue
            if (self.W[i] == n1):
                its = n2
                naliczanie = True
            elif (self.W[i] == n2):
                its = n1
                naliczanie = True
        return ret

    def fsasiedztwa2(self, x, i):
        if self.neuron_d(x, i) <= self.promien:
            return 1
        else:
            return 0

    def wylonWygranego(self):
        curr_d = float("inf")
        for w in self.W:
            d = math.sqrt(math.pow(self.currentVec[0] - w.x, 2) + math.pow(self.currentVec[1] - w.y, 2))
            if d < curr_d and w.p > self.p_min:
                curr_d = d
                self.winner = w
        for w in self.W:
            if w != self.winner:
                w.p += 1.0 / float(self.N)

    def p_trojkata(self, a, b, c):
        p = float((a + b + c) / 2.0)
        return math.sqrt(p * (p - a) * (p - b) * (p - c))

    def __d(self, a1, a2):
        return math.sqrt(math.pow(a1[0] - a2[0], 2) + math.pow(a1[1] - a2[1], 2))

    def losujWektorWejscia(self):
        self.currentVec = [-1.0, -1.0]
        while not self.czyNalezyDoTr(self.currentVec):
            self.currentVec = [random.rand(1, 1)[0][0], random.rand(1, 1)[0][0]]

    def losujWektoryWejsciowe(self):
        self.vecSet = []
        for i in range(0, self.EPOKI):
            self.losujWektorWejscia()
            self.vecSet.append(self.currentVec)
        random.shuffle(self.vecSet)

    def modyfikujWagiZwyciezcy(self):
        self.eta = 0.0001 * (self.EPOKI - self.T) * self.wuczenia
        self.winner.setX(self.winner.x + self.K * (self.currentVec[0] - self.winner.x))
        self.winner.setY(self.winner.y + self.K * (self.currentVec[1] - self.winner.y))
        ss = 0
        for w in self.W:
            if w == self.winner:
                continue
            G = self.fsasiedztwa2(self.winner, w)
            if G == 1:
                ss += 1
                d = self.__d(w, self.winner)
                w.setX(w.x + self.eta * G * self.K * (self.currentVec[0] - w.x))
                w.setY(w.y + self.eta * G * self.K * (self.currentVec[1] - w.y))


if __name__ == '__main__':
    som = SOM(100)
    som.glownaProcedura(25000)


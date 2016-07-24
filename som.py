import numpy.random as random
import math
import matplotlib.pyplot as plt


class Wezel:
    def __init__(self, x, y, p):
        self.x = x
        self.y = y
        self.p = p

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
        self.currentVec = None
        self.dodajPunkty()
        self.eta = 0.4
        self.p_min = 0.9999

    def glownaProcedura(self, epoki):
        fig, ax = plt.subplots()
        self.dodajPunkty()
        x=[]
        y=[]
        for w in self.W:
            x.append(w.x)
            y.append(w.y)
        points, = ax.plot(x, y, marker='o', linestyle='None')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.pause(0.0001)
        for i in range(0, epoki, 1):
            self.pojedynczaIteracja()
            x=[]
            y=[]
            for w in self.W:
                x.append(w.x)
                y.append(w.y)
            points.set_data(x, y)
            plt.pause(0.0001)
        plt.plot(x, y, "o")
        plt.show()

    def pojedynczaIteracja(self):
        self.winner = None
        self.currentVec = None
        self.losujWektorWejscia()
        self.wylonWygranego()
        self.modyfikujWagiZwyciezcy()

    def dodajPunkty(self):
        for i in range(0, self.N, 1):
            self.W.append(Wezel(random.rand(1, 1)[0][0], random.rand(1, 1)[0][0], 1.0))

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
        self.winner.p -= self.p_min

    def losujWektorWejscia(self):
        wariant = random.randint(0, 4)
        if wariant == 0:
            self.currentVec = [0, random.rand(1, 1)[0][0]]
        if wariant == 1:
            self.currentVec = [1, random.rand(1, 1)[0][0]]
        if wariant == 2:
            self.currentVec = [random.rand(1, 1)[0][0], 0]
        if wariant == 3:
            self.currentVec = [random.rand(1, 1)[0][0], 1]

    def modyfikujWagiZwyciezcy(self):
        self.winner.setX(self.winner.x + self.eta * (self.currentVec[0] - self.winner.x))
        self.winner.setY(self.winner.y + self.eta * (self.currentVec[1] - self.winner.y))


if __name__ == '__main__':
    som = SOM(50)
    som.glownaProcedura(10000)


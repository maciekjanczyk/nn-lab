import numpy as np
import random
import sys
import PySide
from PySide.QtCore import Slot
from PySide.QtGui import QApplication
from PySide.QtGui import QMessageBox
from PySide.QtGui import QWidget


class Hopfield:
    def __init__(self, rozmiar, biasy):
        if len(biasy) != rozmiar:
            raise Exception('Biasow musi byc tyle samo co neuronow.')
        self.bias = biasy
        self.N = rozmiar
        self.wagi = np.ndarray(shape=(self.N, self.N))
        self.__winit()
        self.__lastOutputs = []
        self.__backendOutputs = []
        self.__loinit()

    def __loinit(self):
        for i in range(0, self.N):
            self.__lastOutputs.append(0.0)
            self.__backendOutputs.append(0.0)

    def __winit(self):
        for i in range(0, self.N):
            for j in range(0, self.N):
                if i == j:
                    self.wagi[i][j] = 0.0
                    self.wagi[j][i] = 0.0
                else:
                    self.wagi[i][j] = random.uniform(0.01, 0.1)
                    self.wagi[j][i] = random.uniform(0.01, 0.1)

    def __sgn(self, a, k):
        if a > 0:
            return 1.0
        elif a < 0:
            return -1.0
        else:
            return self.__lastOutputs[k]

    def Output(self, k):
        if k not in range(0, self.N):
            raise Exception('Nieprawidlowy numer neuronu.')
        suma = 0.0
        for j in range(0, self.N):
            suma += self.wagi[k][j] * self.__lastOutputs[j]
        suma += self.bias[k]
        out = self.__sgn(suma, k)
        self.__backendOutputs[k] = out
        return out

    def Outputs(self):
        ret = []
        for k in range(0, self.N):
            ret.append(self.Output(k))
        self.__lastOutputs = self.__backendOutputs
        return ret

    def SetVector(self, vec):
        if len(vec) != self.N:
            raise Exception('Rozmiar vectora musi byc rowny ilosci neuronow.')
        self.__lastOutputs = vec

    def Training(self, vec_set):
        for vec in vec_set:
            if len(vec) != self.N:
                raise Exception('Rozmiar vectora musi byc rowny ilosci neuronow.')
        for k in range(0, self.N):
            for j in range(0, self.N):
                if k != j:
                    suma = 0.0
                    for vec in vec_set:
                        suma += vec[k] * vec[j]
                    self.wagi[k][j] = 1.0/self.N * suma
                    #self.wagi[k][j] = suma


class Widget(QWidget):
    def __init__(self, hopfield):
        QWidget.__init__(self)
        self.treningSet = []
        self.currVec = []
        self.hopfield = hopfield
        self.setMinimumSize(550, 170)
        self.setWindowTitle('Hopfield demo')
        self.__lay = PySide.QtGui.QVBoxLayout()
        self.przyciski1 = []
        self.przyciski2 = []
        duzylay = PySide.QtGui.QHBoxLayout()
        duzylay1 = PySide.QtGui.QVBoxLayout()
        duzylay1.setSpacing(0)
        duzylay2 = PySide.QtGui.QVBoxLayout()
        duzylay2.setSpacing(0)
        duzylayLabels = PySide.QtGui.QHBoxLayout()
        duzylayLabels.addWidget(PySide.QtGui.QLabel('Input:'))
        duzylayLabels.addWidget(PySide.QtGui.QLabel('Output:'))
        duzylayKom = PySide.QtGui.QHBoxLayout()
        duzylayKom1 = PySide.QtGui.QVBoxLayout()
        duzylayKom1.addWidget(PySide.QtGui.QLabel('Pojemnosc zestawu treningowego:'))
        duzylayKom1.addWidget(PySide.QtGui.QLabel('Ilosc zapamietanych wzorcow:'))
        duzylayKom2 = PySide.QtGui.QVBoxLayout()
        self.__iloscTrening = PySide.QtGui.QLabel('0')
        self.__iloscUczone = PySide.QtGui.QLabel('0')
        duzylayKom2.addWidget(self.__iloscTrening)
        duzylayKom2.addWidget(self.__iloscUczone)
        duzylayKom.addLayout(duzylayKom1)
        duzylayKom.addLayout(duzylayKom2)
        for i in range(0, 25):
            buttonek = PySide.QtGui.QPushButton('', self)
            buttonek.setMinimumWidth(1)
            buttonek.setMinimumHeight(50)
            buttonek.setStyleSheet("background-color: white")
            buttonek.clicked.connect(self.on_click)
            self.przyciski1.append(buttonek)
        for i in range(0, 25):
            buttonek = PySide.QtGui.QPushButton('', self)
            buttonek.setMinimumWidth(1)
            buttonek.setMinimumHeight(50)
            buttonek.setStyleSheet("background-color: white")
            self.przyciski2.append(buttonek)
        offset = 0
        for i in range(0, 5):
            minilay = PySide.QtGui.QHBoxLayout()
            for j in range(0, 5):
                minilay.addWidget(self.przyciski1[offset + j])
                self.przyciski1[offset + j].mojTag = offset + j
            offset += 5
            duzylay1.addLayout(minilay)
        offset = 0
        for i in range(0, 5):
            minilay = PySide.QtGui.QHBoxLayout()
            for j in range(0, 5):
                minilay.addWidget(self.przyciski2[offset + j])
            self.przyciski2[offset + j].mojTag = offset + j
            offset += 5
            duzylay2.addLayout(minilay)
        duzylay.setContentsMargins(10, 0, 10, 10)
        duzylay.addLayout(duzylay1)
        duzylay.addLayout(duzylay2)
        # Buttonki do obslugi GUI
        duzylaybutt = PySide.QtGui.QHBoxLayout()
        self.__dodButton = PySide.QtGui.QPushButton('Dodaj do uczenia')
        self.__uczButton = PySide.QtGui.QPushButton('Ucz')
        self.__sprawdzButton = PySide.QtGui.QPushButton('Sprawdz')
        self.__rozwButton = PySide.QtGui.QPushButton('Rozwijaj')
        self.__czyscButton = PySide.QtGui.QPushButton('Czysc')
        self.__czyscButton.clicked.connect(self.clear)
        self.__dodButton.clicked.connect(self.uczenie)
        self.__uczButton.clicked.connect(self.ucz)
        self.__sprawdzButton.clicked.connect(self.wydobadz)
        self.__rozwButton.clicked.connect(self.rozwijaj)
        duzylaybutt.addWidget(self.__dodButton)
        duzylaybutt.addWidget(self.__uczButton)
        duzylaybutt.addWidget(self.__sprawdzButton)
        duzylaybutt.addWidget(self.__rozwButton)
        duzylaybutt.addWidget(self.__czyscButton)
        self.__lay.addLayout(duzylayLabels)
        self.__lay.addLayout(duzylay)
        self.__lay.addLayout(duzylaybutt)
        self.__lay.addLayout(duzylayKom)
        self.setLayout(self.__lay)
        self.clear()

    def getBtnIndex(self, btn):
        for i in range(0, len(self.przyciski1)):
            if btn == self.przyciski1[i]:
                return i

    @Slot()
    def clear(self):
        self.currVec = []
        for i in range(0, 25):
            self.currVec.append(-1.0)
        for btn in self.przyciski1:
            btn.setStyleSheet("background-color: white")

    @Slot()
    def uczenie(self):
        self.treningSet.append(self.currVec)
        self.__iloscTrening.setText(str(int(self.__iloscTrening.text()) + 1))
        self.clear()

    @Slot()
    def ucz(self):
        self.hopfield.Training(self.treningSet)
        self.__iloscUczone.setText(str(len(self.treningSet)))
        self.treningSet = []
        self.__iloscTrening.setText('0')

    @Slot()
    def on_click(self):
        sender = self.sender()
        color = sender.styleSheet().split(' ')[1]
        idx = self.getBtnIndex(sender)
        if color == 'white':
            sender.setStyleSheet("background-color: black")
            self.currVec[idx] = 1.0
        else:
            sender.setStyleSheet("background-color: white")
            self.currVec[idx] = -1.0
        print self.currVec

    @Slot()
    def wydobadz(self):
        self.hopfield.SetVector(self.currVec)
        self.rozwijaj()

    @Slot()
    def rozwijaj(self):
        out = self.hopfield.Outputs()
        for i in range(0, 25):
            if out[i] == 1.0:
                self.przyciski2[i].setStyleSheet("background-color: black")
            else:
                self.przyciski2[i].setStyleSheet("background-color: white")


if __name__ == '__main__':
    rozm = 25
    biasy = []
    for i in range(0, rozm):
        biasy.append(random.uniform(0.01, 0.2))
    hopf = Hopfield(rozm, biasy)
    app = QApplication(sys.argv)
    widget = Widget(hopf)
    widget.show()
    app.exec_()

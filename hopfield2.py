import numpy as np
import random
import sys
import PySide
from PySide.QtCore import Slot
from PySide.QtGui import QApplication
from PySide.QtGui import QMessageBox
from PySide.QtGui import QWidget


class Hopfield:
    def __init__(self, dim, biases):
        if len(biases) != dim:
            raise Exception('Bias array length must be equal net dimension.')
        self.bias = biases
        self.N = dim
        self.weights = np.ndarray(shape=(self.N, self.N))
        self.__winit()
        self.__last_outputs = []
        self.__backend_outputs = []
        self.__low_init()

    def __low_init(self):
        for i in range(0, self.N):
            self.__last_outputs.append(0.0)
            self.__backend_outputs.append(0.0)

    def __winit(self):
        for i in range(0, self.N):
            for j in range(0, self.N):
                if i == j:
                    self.weights[i][j] = 0.0
                    self.weights[j][i] = 0.0
                else:
                    self.weights[i][j] = random.uniform(0.01, 0.1)
                    self.weights[j][i] = random.uniform(0.01, 0.1)

    def __sgn(self, a, k):
        if a > 0:
            return 1.0
        elif a < 0:
            return -1.0
        else:
            return self.__last_outputs[k]

    def output(self, k):
        if k not in range(0, self.N):
            raise Exception('Invalid neuron number.')
        suma = 0.0
        for j in range(0, self.N):
            suma += self.weights[k][j] * self.__last_outputs[j]
        suma += self.bias[k]
        out = self.__sgn(suma, k)
        self.__backend_outputs[k] = out
        return out

    def outputs(self):
        ret = []
        for k in range(0, self.N):
            ret.append(self.output(k))
        self.__last_outputs = self.__backend_outputs
        return ret

    def set_vector(self, vec):
        if len(vec) != self.N:
            raise Exception('The length of vector must be equal net dimension.')
        self.__last_outputs = vec

    def training(self, vec_set):
        for vec in vec_set:
            if len(vec) != self.N:
                raise Exception('The length of vector must be equal net dimension.')
        for k in range(0, self.N):
            for j in range(0, self.N):
                if k != j:
                    suma = 0.0
                    for vec in vec_set:
                        suma += vec[k] * vec[j]
                    self.weights[k][j] = 1.0 / self.N * suma


class Widget(QWidget):
    def __init__(self, hopfield, dim):
        QWidget.__init__(self)
        self.dim = dim
        self.setWindowTitle('Hopfield nerual network')
        self.net = hopfield
        self.training_set = []
        self.current_image = PySide.QtGui.QImage()
        # pixmap labels
        self.input_label = PySide.QtGui.QLabel('')
        self.input_label.setFrameStyle(PySide.QtGui.QFrame.Panel)
        self.input_label.setMinimumHeight(self.dim)
        self.input_label.setMinimumWidth(self.dim)
        self.input_label.setMaximumHeight(self.dim)
        self.input_label.setMaximumWidth(self.dim)
        self.output_label = PySide.QtGui.QLabel('')
        self.output_label.setFrameStyle(PySide.QtGui.QFrame.Panel)
        self.output_label.setMinimumHeight(self.dim)
        self.output_label.setMinimumWidth(self.dim)
        self.output_label.setMaximumHeight(self.dim)
        self.output_label.setMaximumWidth(self.dim)
        self.pixmap_labels_layout = PySide.QtGui.QHBoxLayout()
        self.pixmap_labels_layout.setSpacing(5)
        self.pixmap_labels_layout.addWidget(self.input_label)
        self.pixmap_labels_layout.addWidget(self.output_label)
        # buttons
        self.buttons_layout = PySide.QtGui.QHBoxLayout()
        self.open_button = PySide.QtGui.QPushButton('Open')
        self.add_noise_button = PySide.QtGui.QPushButton('Add noise')
        self.add_vector_button = PySide.QtGui.QPushButton('Add vector')
        self.train_button = PySide.QtGui.QPushButton('Train')
        self.first_iteration_button = PySide.QtGui.QPushButton('First iteration')
        self.next_iteration_button = PySide.QtGui.QPushButton('Next iteration')
        self.buttons_layout.addWidget(self.open_button)
        self.buttons_layout.addWidget(self.add_noise_button)
        self.buttons_layout.addWidget(self.add_vector_button)
        self.buttons_layout.addWidget(self.train_button)
        self.buttons_layout.addWidget(self.first_iteration_button)
        self.buttons_layout.addWidget(self.next_iteration_button)
        # add buttons slots
        self.open_button.clicked.connect(self.open)
        self.add_noise_button.clicked.connect(self.add_noise)
        self.add_vector_button.clicked.connect(self.add_vector)
        self.train_button.clicked.connect(self.train)
        self.first_iteration_button.clicked.connect(self.first_iteration)
        self.next_iteration_button.clicked.connect(self.one_iteration)
        # communicates
        self.communicates_layout = PySide.QtGui.QVBoxLayout()
        self.communicates_layout_1 = PySide.QtGui.QHBoxLayout()
        self.communicates_layout_1.addWidget(PySide.QtGui.QLabel('Vector set length:'))
        self.vector_set_length_label = PySide.QtGui.QLabel('0')
        self.communicates_layout_1.addWidget(self.vector_set_length_label)
        self.communicates_layout_2 = PySide.QtGui.QHBoxLayout()
        self.communicates_layout_2.addWidget(PySide.QtGui.QLabel('Remembered vectors count:'))
        self.remembered_vectors_label = PySide.QtGui.QLabel('0')
        self.communicates_layout_2.addWidget(self.remembered_vectors_label)
        self.communicates_layout.addLayout(self.communicates_layout_1)
        self.communicates_layout.addLayout(self.communicates_layout_2)
        # constructing main layout
        self.main_layout = PySide.QtGui.QVBoxLayout()
        self.main_layout.addLayout(self.pixmap_labels_layout)
        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addLayout(self.communicates_layout)
        self.setLayout(self.main_layout)

    @staticmethod
    def make_binary_image(image, dim, threshold):
        ret = PySide.QtGui.QImage(image)
        for i in range(0, dim):
            for j in range(0, dim):
                color = PySide.QtGui.QColor(image.pixel(i, j))
                red = float(color.red())
                blue = float(color.blue())
                green = float(color.green())
                value = (red + blue + green) / 3.0
                if value < threshold:
                    ret.setPixel(i, j, PySide.QtGui.QColor(0, 0, 0).rgb())
                else:
                    ret.setPixel(i, j, PySide.QtGui.QColor(255, 255, 255).rgb())
        return ret

    def convert_pixmap_to_vector(self):
        ret = []
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                color = PySide.QtGui.QColor(self.current_image.pixel(i, j))
                red = float(color.red())
                if red == 255:
                    ret.append(-1.0)
                else:
                    ret.append(1.0)
        return ret

    def add_noise(self):
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                choice = random.randint(0, 1)
                color = PySide.QtGui.QColor(self.current_image.pixel(i, j))
                if choice == 1:
                    self.current_image.setPixel(i, j, PySide.QtGui.QColor(255, 255, 255).rgb())
        self.input_label.setPixmap(PySide.QtGui.QPixmap.fromImage(self.current_image))

    def convert_vector_to_pixmap(self, vector):
        image = PySide.QtGui.QImage(self.dim, self.dim, PySide.QtGui.QImage.Format_ARGB32)
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if vector[self.dim * i + j] == 1.0:
                    color = PySide.QtGui.QColor(0, 0, 0)
                else:
                    color = PySide.QtGui.QColor(255, 255, 255)
                image.setPixel(i, j, color.rgb())
        self.output_label.setPixmap(PySide.QtGui.QPixmap.fromImage(image))

    @Slot()
    def open(self):
        file_dialog = PySide.QtGui.QFileDialog()
        dialog_filter = 'Images (*.png *.bmp *.jpg)'
        dialog_title = 'Open images'
        file_name = file_dialog.getOpenFileName(self, dialog_title, '', dialog_filter)
        if file_name[0] != '':
            self.current_image = self.make_binary_image(PySide.QtGui.QImage(file_name[0]).scaled(self.dim, self.dim),
                                                        self.dim, 150)
            pixmap = PySide.QtGui.QPixmap.fromImage(self.current_image)
            self.input_label.setPixmap(pixmap)

    @Slot()
    def add_vector(self):
        self.training_set.append(self.convert_pixmap_to_vector())
        self.vector_set_length_label.setText(str(int(self.vector_set_length_label.text()) + 1))

    @Slot()
    def train(self):
        print("Training in progress...")
        self.net.training(self.training_set)
        self.remembered_vectors_label.setText(str(len(self.training_set)))
        self.training_set = []
        self.vector_set_length_label.setText('0')

    @Slot()
    def one_iteration(self):
        vector = self.net.outputs()
        self.convert_vector_to_pixmap(vector)

    @Slot()
    def first_iteration(self):
        self.net.set_vector(self.convert_pixmap_to_vector())
        self.one_iteration()

    @Slot()
    def about(self):
        QMessageBox.information(self, "About author",
                                "Author: Maciej Janczyk\njanczyk@linux.pl\nInstitute of Physics\nNicolaus Copernicus University in Torun, Poland")


if __name__ == '__main__':
    random.seed(1337)
    dim = 80
    biases = []
    for i in range(0, dim * dim):
        biases.append(random.uniform(0.01, 0.2))
    print("Hopfield network init process...")
    net = Hopfield(dim * dim, biases)
    print("App init process...")
    app = QApplication(sys.argv)
    widget = Widget(net, dim)
    widget.show()
    app.exec_()
    print("Ok.")

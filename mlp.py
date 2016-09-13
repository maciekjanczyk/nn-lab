import math
import random
import numpy as np


class Perceptron:
    def __init__(self, weights):
        self.w = []
        self.x = []
        for weight in weights:
            self.w.append(weight)

    @staticmethod
    def bipolar_activation(s):
        if s > 0.0:
            return 1.0
        else:
            return -1.0

    @staticmethod
    def sigmoid_activation(s):
        t = float(s)
        try:
            return 1.0 / (1.0 + math.exp(-1.0 * t))
        except Exception:
            if s > 0:
                return 1.0
            else:
                return 0.0

    def set_input(self, vector):
        self.x = vector

    def output(self):
        sum = 0
        for i in range(0, len(self.x)):
            sum += self.x[i] * self.w[i]
        return Perceptron.sigmoid_activation(sum)


class MLP:
    def __init__(self, learn_rate, dim=2):
        self.current_err = 0.0
        self.layers = []
        self.current_input = []
        self.dim = dim
        for i in range(0, self.dim):
            weights = []
            for j in range(0, self.dim):
                weights.append(random.uniform(0.05, 0.2))
            self.layers.append(Perceptron(weights))
        weights = []
        for j in range(0, self.dim):
            weights.append(random.uniform(0.05, 0.2))
        self.output_layer = Perceptron(weights)
        self.epochs = 0
        self.learn_rate = learn_rate

    def set_input(self, vector):
        self.current_input = vector
        outs = []
        for war in self.layers:
            war.set_input(vector)
            outs.append(war.output())
        self.output_layer.set_input(outs)

    def output(self):
        return self.output_layer.output()

    def output_error(self):
        return self.current_err

    def learn(self, x_vectors, y_vectors, epochs):
        self.epochs = epochs
        for k in range(0, self.epochs):
            for i in range(0, len(x_vectors)):
                self.current_err = 0.0
                self.set_input(x_vectors[i])
                out_o = self.output_layer.output()
                if out_o == y_vectors[i]:
                    continue
                sigma_o = out_o * (1.0 - out_o) * (y_vectors[i] - out_o)
                for ii in range(0, self.dim):
                    self.output_layer.w[ii] += self.learn_rate * sigma_o * self.layers[ii].output()
                sigmas = []
                for ii in range(0, self.dim):
                    outW = self.layers[ii].output()
                    sigmas.append(outW * (1.0 - outW) * (sigma_o * self.output_layer.w[ii]))
                for ii in range(0, self.dim):
                    self.current_err += math.fabs(sigmas[ii])
                    for j in range(0, self.dim):
                        self.layers[ii].w[j] += self.learn_rate * sigmas[ii] * x_vectors[i][j]
                self.current_err += math.fabs(sigma_o)

    def do_classification(self, x_vectors, y_vectors):
        i = 0
        errors = 0.0
        for v in x_vectors:
            self.set_input(v)
            output = self.output()
            print(str.format("Vector: {0}, Output: {1}, Expected: {2}", v, output, y_vectors[i]))
            errors += math.fabs(output - y_vectors[i])
            i += 1
        errors /= float(len(x_vectors))
        score = 1.0 - errors
        print("Test score: {0}% +/- {1}%".format(score * 100.0, self.output_error() * 100.0))
        return score


class DataLoader:
    def __init__(self, file_name_training, file_name_test='', fill_missing=0.0, ignore_fields=None, class_field=-1, separator=','):
        if ignore_fields is None:
            ignore_fields = []
        self.has_test_set = False
        self.x_training = None
        self.y_training = None
        self.x_tests = None
        self.y_tests = None
        self.class_field = class_field
        self.xt = []
        self.yt = []
        self.class_values_training = []
        self.class_values_test = []
        self.first_iteration = True
        with open(file_name_training, 'r') as f:
            lines = f.readlines()
            self.xt = []
            self.yt = []
            j = 0
            for line in lines:
                separated = line.split(separator)
                if self.class_field == -1 and self.first_iteration:
                    self.class_field = len(separated) - 1
                if self.first_iteration:
                    self.first_iteration = False
                values = []
                for i in range(0, len(separated)):
                    if i not in ignore_fields and i != self.class_field:
                        try:
                            values.append(float(separated[i]))
                        except ValueError:
                            values.append(float(fill_missing))
                self.xt.append(values)
                self.yt.append(float(separated[self.class_field]))
                if float(separated[self.class_field]) not in self.class_values_training:
                    self.class_values_training.append(float(separated[self.class_field]))
                j += 1
            self.x_training = self.xt
            self.y_training = self.yt
        if file_name_test != '':
            self.has_test_set = True
            with open(file_name_test, 'r') as f:
                lines = f.readlines()
                self.xt = []
                self.yt = []
                j = 0
                for line in lines:
                    separated = line.split(separator)
                    values = []
                    for i in range(0, len(separated)):
                        if i not in ignore_fields and i != self.class_field:
                            try:
                                values.append(float(separated[i]))
                            except ValueError:
                                values.append(float(fill_missing))
                    self.xt.append(values)
                    self.yt.append(float(separated[self.class_field]))
                    if float(separated[self.class_field]) not in self.class_values_test:
                        self.class_values_test.append(float(separated[self.class_field]))
                    j += 1
                self.x_tests = self.xt
                self.y_tests = self.yt

    @staticmethod
    def normalize(vector):
        min_class_value = min(vector)
        if min_class_value < 0.0:
            for i in range(0, len(vector)):
                vector[i] += min_class_value
        max_class_value = max(vector)
        if max_class_value > 1.0:
            for i in range(0, len(vector)):
                vector[i] /= max_class_value

    def get_data(self, classification=False, normalize_outputs=False, normalize_inputs=False):
        if classification:
            self.class_values_training.sort()
            space_training = np.linspace(0.0, 1.0, len(self.class_values_training))
            space_test = []
            if self.has_test_set:
                self.class_values_test.sort()
                space_test = np.linspace(0.0, 1.0, len(self.class_values_test))
            for i in range(0, len(self.y_training)):
                for j in range(0, len(self.class_values_training)):
                    if self.y_training[i] == self.class_values_training[j]:
                        self.y_training[i] = space_training[j]
                        continue
            if self.has_test_set:
                for i in range(0, len(self.y_tests)):
                    for j in range(0, len(self.class_values_test)):
                        if self.y_tests[i] == self.class_values_test[j]:
                            self.y_tests[i] = space_test[j]
                            continue
        if normalize_outputs:
            self.normalize(self.y_training)
            if self.has_test_set:
                self.normalize(self.y_tests)
        if normalize_inputs:
            self.normalize(self.x_training)
            if self.has_test_set:
                self.normalize(self.x_tests)
        if self.has_test_set:
            return self.x_training, self.y_training, self.x_tests, self.y_tests
        else:
            return self.x_training, self.y_training


if __name__ == '__main__':
    print("Load data vectors...")
    data_loader = DataLoader('./data/tic-tac-toe.data')
    x, y = data_loader.get_data(classification=True)
    net = MLP(0.03, 9)
    print("Learning on progress...")
    net.learn(x, y, 100)
    net.do_classification(x, y)
    '''inp = ''
    while input != 'quit()':
        print("Input: ")
        inp = raw_input()
        separated = str(inp).split(',')
        vector = []
        for item in separated:
            vector.append(float(item))
        net.set_input(vector)
        print("Net answer: {0}".format(net.output()))'''

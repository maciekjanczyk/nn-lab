import numpy as np


class BAM:

    def __init__(self, im_size, n_size):
        self.w = self.w = np.zeros([im_size, n_size])

    def remember(self, image, name):
        for i in range(0, len(image)):
            for j in range(0, len(name)):
                if name[j] == image[i]:
                    self.w[i][j] += 1
                else:
                    self.w[i][j] += -1

    def print_w(self):
        for i in range(0, len(self.w)):
            print self.w[i]

    def get_image(self, name):
        ret = np.array(name * np.matrix(self.w).T)[0]
        for i in range(0, len(ret)):
            if ret[i] > 0:
                ret[i] = 1
            else:
                ret[i] = 0
        return ret

    def get_name(self, image):
        ret = np.array(image * np.matrix(self.w))[0]
        for i in range(0, len(ret)):
            if ret[i] > 0:
                ret[i] = 1
            else:
                ret[i] = 0
        return ret

    def recover_image(self, name):
        im = self.get_image(name)
        nm = self.get_name(im)
        while nm.tolist() != name:
            im = self.get_image(name)
            nm = self.get_name(im)
        return im

    def recover_name(self, image):
        nm = self.get_name(image)
        im = self.get_image(nm)
        while im.tolist() != image:
            nm = self.get_name(im)
            im = self.get_image(nm)
        return nm


def czytaj_z_pliku(name):
    f = open(name, 'r')
    ret = []
    for line in f:
        splitted = line.split(' ')
        ret2 = []
        for ch in splitted:
            ret2.append(int(ch))
        ret.append(ret2)
    return ret


if __name__ == '__main__':
    x = czytaj_z_pliku('./data/BAM_X_simple.txt')
    y = czytaj_z_pliku('./data/BAM_Y_simple.txt')
    bam = BAM(len(x[0]), len(y[0]))
    for i in range(0, len(x)):
        print(str.format("Uczenie zestawu {0} {1}...", x[i], y[i]))
        bam.remember(x[i], y[i])
    for im in x:
        print(str.format("Odtwarzanie nazwy na podstawie obrazu {0}...", im))
        print(str.format("Wynik: {0}", bam.recover_name(im).tolist()))
    for name in y:
        print(str.format("Odtwarzanie obrazu na podstawie nazwy {0}...", name))
        print(str.format("Wynik: {0}", bam.recover_image(name).tolist()))
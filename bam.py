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


if __name__ == '__main__':
    None
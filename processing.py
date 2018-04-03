from numpy import array, zeros, asarray
from math import exp
from random import shuffle
from PIL import Image
from time import time

class IntegralImage():
    def __init__(self, image):
        self.h = image.shape[0] - 1
        self.w = image.shape[1] - 1
        self.img = image
        self.ss = None
        self.ii = None

    def process(self):
        """Preprocess the integral image from top-left corner."""
        image = self.img
        s = zeros(image.shape)

        for y, row in enumerate(image):
            for x, cell in enumerate(row):
                if x == 0:
                    s[y][x] = cell
                else:
                    s[y][x] = cell + s[y][x - 1]

        ii = zeros(image.shape)

        for y, row in enumerate(s):
            for x, cell in enumerate(row):
                if y == 0:
                    ii[y][x] = cell
                else:
                    ii[y][x] = ii[y - 1][x] + cell

        self.ss = s
        self.ii = ii

        return self

    def sum_square(self, x, y, w, h):
        x = int(x * self.w)
        y = int(y * self.h)
        w = int(w * self.w)
        h = int(h * self.h)

        tl_x, tl_y = (x, y)
        tr_x, tr_y = (x + w, y)
        bl_x, bl_y = (x, y + h)
        br_x, br_y = (x + w, y + h)

        return self.ii[br_y][br_x] + self.ii[tl_y][tl_x] - self.ii[tr_y][tr_x] - self.ii[bl_y][bl_x]




def load_image(name):
    """Load image for classification."""
    img = Image.open(name)
    img.load()

    return asarray(img, dtype="B")


def grey_scale(image):
    """Make image monochannel :D."""

    if len(image.shape) == 2:
        return image

    x, y, = image.shape
    grey = zeros((y, x))
    for row in range(y):
        for col in range(x):
            grey[y][x] = 0.25 * image[y][x][0]\
                    + 0.5  * image[y][x][1]\
                    + 0.25 * image[y][x][2]

    return grey


def bulk_II(images):
    """Process images to integral images."""
    timestamp = time()

    IIs = []
    for image in images:
        IIs.append(IntegralImage(image).process())

    print("Processing all integral images: {}".format(time() - timestamp))

    return IIs

def cross_validate(IIs, labels, percent):
    """Get cross validation data."""
    data = list(zip(IIs, labels))
    shuffle(data)

    train = int(len(data) * (1 - percent))

    train_data = data[:train]
    validate_data = data[train:]

    td = []
    tl = []

    for ii, label in train_data:
        td.append(ii)
        tl.append(label)

    vd = []
    vl = []
    for ii, label in validate_data:
        vd.append(vd)
        vd.append(vl)

    return td, array(tl), vd, array(vl)



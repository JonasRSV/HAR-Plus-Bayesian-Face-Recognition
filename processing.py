from numpy import array, zeros, asarray
from PIL import Image

class IntegralImage():
    def __init__(self, image):
        self.h = image.shape[0] # Height first for np shape
        self.w = image.shape[1]
        self.img = image

    def process(self):
        """Preprocess the integral image from top-left corner."""
        image = self.img
        s = zeros(image.shape)
        ii = zeros(image.shape)

        for indeY, row in enumerate(image):
            for indeX, cell in enumerate(row):
                    
                if indeX == 0:
                    s[indeY][indeX] = cell
                else:
                    s[indeY][indeX] = cell + s[indeY][indeX - 1]

                if indeY == 0:
                    ii[indeY][indeX] = s[indeY][indeX]
                else:
                    ii[indeY][indeX] = ii[indeY - 1][indeX] + s[indeX][indeY]

        self.ii = ii

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

from numpy import array, zeros, asarray
from PIL import Image

def integral_image(image):
    """Preprocess the integral image from top-left corner."""
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

    return ii


def seg_ii(ii, top_left, top_right, bottom_left, bottom_right):
    """Get segment from integral image given segment corners."""
    tl_x, tl_y = top_left
    tr_x, tr_y = top_right
    br_x, br_y = bottom_right
    bl_x, bl_y = bottom_left

    return ii[br_y][br_x] + ii[tl_y][tl_x] - ii[tr_y][tr_x] - ii[bl_y][bl_x]


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

from processing import IntegralImage
from matplotlib.patches import Rectangle

class Window(object):

    def __init__(self, start, shape, data):
        self.start = start
        self.shape = shape
        self.data = data

        self.distance = None
        self.reason = None

    def get_classifiable(self):
        """Get classifiable dimension."""
        return IntegralImage(self.data).process()

    def add_rectangle_to_image(self, axis):
        """Add rectangle to a patch."""
        (width, height) = self.shape
        (x, y) = self.start

        axis.add_patch(
            Rectangle((y, x), width, height, linewidth=1, edgecolor="r", fill=False))


def slide(image, shape, stride=10):
    """Slide through the image and return windows."""
    mx, my = image.shape
    x, y = shape

    sx, sy = 0, 0

    windows = []
    while sy + y < my:
        while sx + x < mx:
            from_x = sx
            to_x = sx + x

            from_y = sy
            to_y = sy + y

            data = image[from_x:to_x, from_y:to_y]

            windows.append(Window((sx, sy), shape, data))

            sx += stride

        sx = 0
        sy += stride

    return windows



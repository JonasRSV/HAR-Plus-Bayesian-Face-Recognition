from matplotlib.patches import Rectangle

class FeatureSize():
    """As percentages of image size."""
    def __init__(self, x, y, w, h):
        """Initialize object."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return (self.x, self.y, self.w, self.h).__str__()


def generate_all_sizes():
    h = 1
    while h <= 32:
        w = 1
        while w <= 32:
            for y in range(0, 32, h):
                for x in range(0, 32, w):
                    yield FeatureSize(x, y, w, h);
            w = w * 2
        h = h * 2

def A(ps):
    return [(ps.x           , ps.y           , ps.w / 2, ps.h / 2),
            (ps.x + ps.w / 2, ps.y           , ps.w / 2, ps.h / 2),
            (ps.x + ps.w / 2, ps.y + ps.h / 2, ps.w / 2, ps.h / 2),
            (ps.x           , ps.y + ps.h / 2, ps.w / 2, ps.h / 2)]

def B1(ps):
    return [(ps.x           , ps.y, ps.w / 2, ps.h),
            (ps.x + ps.w / 2, ps.y, ps.w / 2, ps.h)]

def B2(ps):
    return [(ps.x, ps.y           , ps.w, ps.h / 2),
            (ps.x, ps.y + ps.h / 2, ps.w, ps.h / 2)]

def C1(ps):
    return [(ps.x + 0 * ps.w / 3, ps.y, ps.w / 3, ps.h),
            (ps.x + 1 * ps.w / 3, ps.y, ps.w / 3, ps.h),
            (ps.x + 2 * ps.w / 3, ps.y, ps.w / 3, ps.h)]

def C2(ps):
    return [(ps.x, ps.y + 0 * ps.h / 3, ps.w, ps.h / 3),
            (ps.x, ps.y + 1 * ps.h / 3, ps.w, ps.h / 3),
            (ps.x, ps.y + 2 * ps.h / 3, ps.w, ps.h / 3)]


class Feature():
    def __init__(self, size, ptrn):
        self.size = size # A FeatureSize
        self.ptrn = ptrn

    def print_feature(self, ii, plt):
        color = ["green", "red"]
        plt.imshow(ii.img)
        for i, val in enumerate(self.ptrn(self.size)):
            plt.add_patch(Rectangle((val[0] * ii.w, val[1] * ii.h), val[2] * ii.w, val[3] * ii.h, facecolor=color[i%2]))

    def calculate(self, ii):
        sign = [1, -1]
        s = 0
        for i, val in enumerate(self.ptrn(self.size)):
            s += sign[i%2] * ii.sum_square(*val)
        return s
        


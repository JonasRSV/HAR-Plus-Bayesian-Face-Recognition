from ada_boost import boosted_classifier
from processing import IntegralImage
from time import time


class cascade(object):
    """Cascade of integral images."""

    def __init__(self):
        pass

    def process_bulk(self, images_and_labels):
        """Process a list of tuples of data and labels."""
        tpp = time()

        IIs = []
        IIlabels = []
        for images, labels in images_and_labels:
            for index, image in enumerate(images):
                IIs.append(IntegralImage(image).process())
                IIlabels.append(labels[index])

        print("Proccessing all images to II: {}"
              .format(time() - tpp))

        return IIs


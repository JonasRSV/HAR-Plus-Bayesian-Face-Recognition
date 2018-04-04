from sliding_window import slide, Window
from cascade import cascade


class detector(object):
    """
    Object detector using some classifier and sliding window.
    """

    def __init__(self, window_start, window_end, sizeup, stride, classifier):
        """
        Window start shape
        Window end shape
        factor of which its resized every slide
        stride
        classifier
        """

        self.window_start = window_start
        self.window_end = window_end
        self.sizeup = sizeup
        self.stride = stride
        self.classifier = classifier

    def __get_windows(self, image):
        """Get pyramide of windows."""

        slide_shape = self.window_start

        x_diff = self.window_end[0] - slide_shape[0]
        y_diff = self.window_end[1] - slide_shape[1]
        IIWs = []
        while x_diff >= 0 and y_diff >= 0:
            IIWs.extend(slide(image, slide_shape, self.stride))

            slide_shape = (slide_shape[0] * self.sizeup, slide_shape[1] * self.sizeup)

            x_diff = self.window_end[0] - slide_shape[0]
            y_diff = self.window_end[1] - slide_shape[1]

        return IIWs


    def __find_objects(self, IIWs):
        """Classify all windows."""

        objects = []
        for IIW in IIWs:
            if self.classifier.predict(IIW.get_classifiable()):
                objects.append(IIW)

        return objects

    
    def detect(self, image, axis):
        """Return detected objects and add patches to image."""
        windows = self.__get_windows(image)
        objects = self.__find_objects(windows)

        for object_ in objects:
            object_.add_rectangle_to_image(axis)

        return objects


        





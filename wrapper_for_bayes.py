from numpy import zeros

class wrapper(object):
    """
    This is a wrapper so that we can
    test the naive bayes classifier solo
    on the Integral Images, it just stores
    the features and calculates them before
    classification each time.
    """


    def __init__(self, classifier, features):
        self.classifier = classifier
        self.feautres = features
        self.num_of_features = len(features)


    def predict(self, II):
        """Predict on a II."""
        features = zeros(self.num_of_features)

        for index, feature in enumerate(self.feautres):
            features[index] = feature.calculate(II)

        return self.classifier.predict(features)



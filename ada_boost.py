from naive_bayes import naive_bayes
from numpy import array, zeros, ones, log
from time import time

INT_MAX = 10000000000


class boosted_classifier(object):
    """
    Applying speical boost for feature extraction.

    IMPORTANT! Because this Adaboosting is done
    abit differently it only supports binary
    classification.

    Make
    positive class label = 1
    Negative class label = 0
    """

    def __init__(self, num_of_features, classifier=naive_bayes):
        """Constructor."""
        self.num_of_features = num_of_features
        self.classifier = classifier

        self.classifiers = []
        self.feature_extracters = []

        self.alphas = zeros(num_of_features)
        self.alpha_sum = 0
        self.bias = 1

    def set_bias(self, bias):
        self.bias = bias

    def train(self, feature_matrix, feature_extracters, labels):
        """Boost classifiers on features."""
        print("Training boosted classifier with {} features"
              .format(self.num_of_features))
        timestamp = time()

        weigths = ones(labels.shape) / len(labels)

        for index in range(self.num_of_features):

            best_classifier = None
            feature_extracter = None
            lowest_error = INT_MAX
            classifications = []

            """
            Create classifier for each feature and
            calculate error for that classifier,
            choose classifier with least error
            """
            for feature_index, feature_on_images in enumerate(feature_matrix):
                """For each row in feature matrix."""
                classifier = self.classifier()
                classifier.train(feature_on_images, labels, weigths)

                error = 0
                classifications_ = []
                for index, image in enumerate(feature_on_images):
                    classification = 1 if classifier.predict(image, weigths)\
                        != labels[index] else 0

                    error += classification * weigths[index]
                    classifications_.append(classification)

                if error < lowest_error:
                    best_classifier = classifier
                    feature_extracter = feature_extracters[feature_index]

                    lowest_error = error
                    classifications = classifications_

            """
            Update weights with the info
            from the best classifier.
            Add feature extracter, alpha
            and classifier to boosted
            classifier.
            """
            Bt = lowest_error / (1 - lowest_error)

            """
            Contrary to the other distribution
            we used in Lab3, this distribution
            reduces the weights of the correctly
            classified, rather than increase the
            weight of the wrongly classified.
            """
            intermediary = zeros(weigths.shape)
            for w_index, weigth in enumerate(weigths):
                intermediary[w_index] =\
                    weigth * pow(Bt, 1 - classifications[w_index])

            weigths = intermediary

            """
            The paper does not mention normalizing the weights
            but i'll do that anyway, it seems like a reasonable thing
            to do
            """
            weigths = weigths / sum(weigths)

            self.alphas[index] = log(1 / Bt)
            self.classifiers.append(best_classifier)
            self.feature_extracters.append(feature_extracter)

        self.alpha_sum = sum(self.alphas)

        return time() - timestamp

    def test(self, iis, labels):
        total_pass = 0
        false_pass = 0
        total_false = 0
        for label, ii in zip(labels, iis):
            face = self.predict(ii)
            total_pass += face
            total_false += (label == 0)
            false_pass += (face == 1 && label == 0)
        return (total_pass / len(labels), false_pass / (1e-20 + total_false)

    def predict(self, X):
        """Predict belongance of X."""
        prediction = 0
        for index, alpha in self.alphas:
            """feature = self.feature_extracters[index].get(X)."""
            feature = 5
            prediction += alpha * self.classifiers[index].predict(feature)

        """
        If half of the boosted classifiers thinks it a thing, it's a thing.
        """

        return 1 if prediction > self.bias * (self.alpha_sum / 2) else 0


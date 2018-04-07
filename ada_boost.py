from naive_bayes import naive_bayes
from numpy import array, zeros, ones, log
from sys import stdout
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

        self.alphas = []
        self.alpha_sum = 0
        self.bias = 1

    def set_bias(self, bias):
        self.bias = bias

    def __best_classifier(self, feature_matrix, feature_extracters, labels, weigths):
        """Get best classifier."""
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

            """Don't add same feature multiple times."""
            if feature_extracters[feature_index] in self.feature_extracters:
                continue

            classifier = self.classifier()
            feature_on_images = feature_on_images.reshape(-1, 1)
            classifier.train(feature_on_images, labels, weigths)

            error = 0
            classifications_ = []
            for i_index, image in enumerate(feature_on_images):
                classification = 1 if classifier.predict(image)\
                    != labels[i_index] else 0

                error += classification * weigths[i_index]
                classifications_.append(classification)

            if error < lowest_error:
                best_classifier = classifier
                feature_extracter = feature_extracters[feature_index]

                lowest_error = error
                classifications = classifications_

        return best_classifier, feature_extracter, lowest_error, classifications


    def __update_weigths(self, weigths, lowest_error, classifications):
        """Update weights from boosting."""
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
            intermediary[w_index] = weigths[w_index] * pow(Bt, 1 - classifications[w_index])

        weigths = intermediary

        """
        The paper does not mention normalizing the weights
        but i'll do that anyway, it seems like a reasonable thing
        to do
        """
        weigths = weigths / sum(weigths)

        return weigths, Bt

    def train(self, feature_matrix, feature_extracters, labels, weigths=None, memory=None):
        """Boost classifiers on features."""
        stdout.write("\rTraining boosted classifier with {} features\r"
              .format(self.num_of_features))
        timestamp = time()

        if weigths is None:
            weigths = ones(labels.shape) / len(labels)

        if not memory is None:
            self.classifiers, self.alphas, self.feature_extracters = memory


        for _ in range(self.num_of_features - len(self.classifiers)):

            best_classifier, feature_extracter, lowest_error, classifications =\
                    self.__best_classifier(feature_matrix, feature_extracters, labels, weigths)



            weigths, Bt = self.__update_weigths(weigths, lowest_error, classifications) 

            self.alphas.append(log(1 / Bt))
            self.classifiers.append(best_classifier)
            self.feature_extracters.append(feature_extracter)

            # Added some space to clear row
            stdout.write("\rclassifiers left: {}                                       \r"
                         .format(self.num_of_features - len(self.classifiers)))

        self.alpha_sum = sum(self.alphas)

        return weigths, (self.classifiers, self.alphas, self.feature_extracters)


    def test(self, iis, labels):
        total_pass = 0
        false_pass = 0
        total_false = 0
        for label, ii in zip(labels, iis):
            face = self.predict(ii)
            total_pass += face
            total_false += (label == 0)
            false_pass += (face == 1 and label == 0)

        return (total_pass / len(labels), false_pass / total_false)

    def predict(self, X):
        """Predict belongance of X."""
        prediction = 0
        for index, alpha in enumerate(self.alphas):
            feature = self.feature_extracters[index].calculate(X)

            prediction += alpha * self.classifiers[index].predict(feature)

        """
        If half of the boosted classifiers thinks it a thing, it's a thing.
        """

        return 1 if prediction > self.bias * (self.alpha_sum / 2) else 0


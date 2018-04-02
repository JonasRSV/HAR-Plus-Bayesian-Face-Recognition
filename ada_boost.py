from naive_bayes import naive_bayes
from numpy import array, zeros, ones, log

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
        self.features = []

        self.alphas = zeros(num_of_features)
        self.alpha_sum = 0

    def train(self, feature_matrix, features_extracters, labels):
        """Boost classifiers on features."""
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
            for feature_index, features in enumerate(feature_matrix):
                classifier = self.classifier()
                classifier.train(features, labels, weigths)

                error = 0
                classifications_ = []
                for index, feature in enumerate(features):
                    classification = 1 if classifier.predict(feature, weigths)\
                        != labels[index] else 0

                    error += classification * weigths[index]
                    classifications_.append(classification)

                if error < lowest_error:
                    best_classifier = classifier
                    feature_extracter = features_extracters[feature_index]

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
            self.features.append(feature_extracter)

        self.alpha_sum = sum(self.alphas)
        return True

    def predict(self, X):
        """Predict belongance of X."""
        prediction = 0
        for index, alpha in self.alphas:
            """feature = self.features.get(X)."""
            feature = 5
            prediction += alpha * self.classifiers[index].predict(feature)

        """
        If half of the boosted classifiers thinks it a thing, it's a thing.
        """

        return 1 if prediction > (self.alpha_sum / 2) else 0






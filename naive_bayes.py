from numpy import array, zeros, mean, log, ones, square
from sys import exit, stdout

MINUS_INF = -100000000000
PLUS_INF = 100000000000


class naive_bayes(object):
    """
    Naive Bayes classifier.

    Labels are expected to be numbers,
    enumerate from 0...N
    where N is the number of classes.

    Because the class will be used as index
    <key> for covariance, prior and expected value
    belonging to that class.
    """

    def __init__(self):
        """Constructor."""
        self.priors = None
        self.covariances = None
        self.expected_values = None
        self.labels = None

        self.classes = None
        self.num_of_classes = None

        self.features = None
        self.samples = None

    def __calculate_priori(self, labels, boost_weigths=None):
        """Calculate the priori for each class."""
        if boost_weigths is None:
            boost_weigths = ones(self.samples)

        priors = zeros(self.num_of_classes)
        label_ones = ones(self.samples)
        for _class in self.classes:
            stdout.write("\r\r\rCalculating Priori for class {}\
                          \r\r\r".format(_class))
            _class = int(_class)

            occurrences = label_ones[_class == labels]
            weights = boost_weigths[_class == labels]

            priors[_class] = (weights @ occurrences) / sum(boost_weigths)

        return priors

    def __calculate_maximum_likelyhood(self, data, labels,
                                       boost_weigths=None):
        """Calculate covariance and expected value for each class."""
        if boost_weigths is None:
            boost_weigths = ones(self.samples)

        expected_values = zeros((self.num_of_classes,
                                 self.features)
                                )

        covariances = zeros((self.num_of_classes,
                             self.features,
                             self.features)
                            )

        for _class in self.classes:
            stdout.write("\r\r\rCalculating Maximum likelyhood for class {}\
                          \r\r\r".format(_class))
            _class = int(_class)

            class_data = data[_class == labels]
            weights = boost_weigths[_class == labels]

            expected_values[_class] = (weights.T @ class_data) / sum(weights)
            covariance = zeros((self.features, self.features))

            for feature in range(self.features):
                variance = square(class_data[:, feature]
                                  - expected_values[_class][feature])

                covariance[feature][feature] =\
                    (weights @ variance) / sum(weights)

            covariances[_class] = covariance

        return covariances, expected_values

    def __normal_dist_max_aposteori(self, X):
        """Classify X using the generalized normal distribution."""
        class_belongance = -1
        max_aposteori = MINUS_INF
        for class_ in self.classes:
            class_ = int(class_)

            covariance = self.covariances[class_]
            expected_value = self.expected_values[class_]
            priori = self.priors[class_]

            quadratic_form = (X - expected_value).T @ iodm(covariance) @ (X - expected_value)
            log_likelyhood = -0.5 * (log(dodm(covariance) + 1e-10) + quadratic_form)
            log_priori = log(priori)

            belongance = log_likelyhood + log_priori

            if belongance > max_aposteori:
                max_aposteori = belongance
                class_belongance = class_

        return class_belongance

    def train(self, data, labels, boost_weigths=None):
        """Train the classifier."""
        print("Training a nb classifier\n")

        self.samples, self.features = data.shape
        self.classes = set(labels)
        self.num_of_classes = len(self.classes)
        self.priors = self.__calculate_priori(labels, boost_weigths)
        self.covariances, self.expected_values =\
            self.__calculate_maximum_likelyhood(data, labels, boost_weigths)

        print("\nTraining done.")
        return self

    def predict(self, X):
        """Predict new data."""
        return self.__normal_dist_max_aposteori(X)


def dodm(matrix):
    """Get determinant of a diagonalized matrix (Or assumed to be)."""
    (x, y) = matrix.shape

    if x != y:
        print("Determinant of a non-square matrix is always 0")
        return 0

    det = 1
    for i in range(x):
        det *= matrix[i][i]

    return det


def iodm(matrix):
    """Get inverse of a diagnoalized matrix (Or assumed to be)."""
    (x, y) = matrix.shape

    if x != y:
        print("Non Square matrix not invertible.")
        exit(1)

    penis = zeros(matrix.shape)
    for i in range(x):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
        else:
            penis[i][i] = 1 / matrix[i][i]

    return penis


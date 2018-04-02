from numpy import array, zeros, mean, log
from sys import exit

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
        """Constructor. """
        self.priors = None
        self.covariances = None
        self.expected_values = None
        self.labels = None

        self.unique_classes = None

    def __calculate_priori(self, classes):
        """Calculate the priori for each class."""
        cardinality = len(classes)
        priors = zeros(len(self.unique_classes))
        for index, _class in enumerate(self.unique_classes):
            occurrences = classes[_class == classes]

            priors[index] = occurrences / cardinality

        return priors

    def __calculate_maximum_likelyhood(self, data, classes):
        """Calculate covariance and expected value for each class."""
        nof_classes = len(self.unique_classes)
        (cardinality, features) = data.shape

        expected_values = zeros((nof_classes, features))
        covariances = zeros((nof_classes, features, features))
        for index, _class in enumerate(self.unique_classes):
            class_data = data[_class == classes]

            self.expected_values[index] = mean(class_data, axis=0)
            self.covariances[index] = class_data @ class_data.T

        return covariances, expected_values

    def __normal_dist_max_aposteori(self, X):
        """Classify X using the generalized normal distribution."""
        class_belongance = -1
        max_aposteori = -1
        for class_ in self.unique_classes:
            covariance = self.covariances[class_]
            expected_value = self.expected_values[class_]
            priori = self.priors[class_]

            quadratic_form =  (X - expected_value) @ iodm(covariance) @ (X - expected_value).T
            log_likelyhood = -0.5 * (log(dodm(covariance)) + quadratic_form)
            log_priori = log(priori)

            belongance = log_likelyhood + log_priori

            if belongance > max_aposteori:
                max_aposteori = belongance
                class_belongance = class_

        return class_belongance

    def train(self, data, classes):
        """Train the classifier."""
        self.unique_classes = set(classes)
        self.priors = self.__calculate_priori(classes)
        self.covariances, self.expected_values =\
            self.__calculate_maximum_likelyhood(data, classes)

        """Not sure if this is needed, but it does impact result."""
        intermediary = zeros(self.covariances.shape)
        for index, covariance in enumerate(self.covariances):
            intermediary[index] = ignore_feature_dependence(covariance)

        self.covariances = intermediary

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

    det = 0
    for i in range(x):
        det += matrix[i][i]

    return det


def iodm(matrix):
    """Get inverse of a diagnoalized matrix (Or assumed to be)."""
    return 1 / matrix


def ignore_feature_dependence(matrix):
    """Remove feature dependant factors of covmatrix."""
    (x, y) = matrix.shape
    ifdm = zeros((x, y))

    if x != y:
        print("Invalid matrix shape in ignore_feature_dependence")
        exit(1)

    for i in range(x):
        ifdm[i][i] = matrix[i][i]

    return ifdm



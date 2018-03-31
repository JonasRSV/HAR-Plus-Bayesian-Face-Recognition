from numpy import array, zeros, mean
from sys import exit

class naive_bayes(object):

    def __init__(self):
        self.priors = None
        self.covariances = None
        self.expected_values = None
        self.labels = None


    def __calculate_priori(self, classes):
        """Calculate the priori for each class."""
        unique_classes = set(classes)

        cardinality = len(classes)
        self.priors = zeros(len(unique_classes))
        for index, _class in enumerate(unique_classes):
            occurrences = classes[_class == classes]

            self.priors[index] = occurrences / cardinality

        return self.priors

    def __calculate_maximum_likelyhood(self, data, classes):
        """Calculate covariance and expected value for each class."""
        unique_classes = set(classes)

        nof_classes = len(unique_classes)
        (cardinality, features) = data.shape

        self.expected_values = zeros((nof_classes, features))
        self.covariances = zeros((nof_classes, features, features))
        for index, _class enumerate(unique_classes):
            class_data = data[_class == classes]

            self.expected_values[index] = mean(class_data, axis=0)
            self.covariances[index] = class_data @ class_data.T

        return self.covariances, self.expected_values

    def __normal_dist_max_aposteori(self, X):
        """Classify X using the generalized normal distribution."""
        pass
    


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



    



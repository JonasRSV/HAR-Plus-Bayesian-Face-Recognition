from ada_boost import boosted_classifier
from processing import cross_validate
from features import generate_all_features, get_feature_matrix
from numpy import array


class cascade(object):
    def __init__(self, accepted_false_positive, min_accepted_detection):
        self.f = accepted_false_positive # Maximum percenteage off false positives to make it throug
        self.d = min_accepted_detection # Minimum amount of things that should pass through

    def train(self, iis, labels, fail_target):
        """Train the cascade."""
        # Positive samples
        # Negative samples
        F_old = 1.0
        F_new = F_old
        D_old = 1.0
        D_new = D_old

        # Generate feature_matrix

        classifyer_list = []
        i = 0

        print("Starting training")

        while F_new > fail_target:
            print("Round we go!! Currently:", F_new, " Target is: ", fail_target)

            iis_train, labels_train, iis_test, labels_test = cross_validate(iis, labels, 0.8)
            all_features = generate_all_features()
            feature_matrix = get_feature_matrix(iis_train, all_features)

            i += 1
            n_i = 0
            F_new = F_old
            b = None
            while F_new > F_old * self.f:
                n_i += 1
                print("Generating classifiers with", n_i, "features")
                print("Feature matrix shape", feature_matrix.shape)
                b = boosted_classifier(n_i)
                b.train(feature_matrix, all_features, labels_train)
                D_new, F_new = b.test(iis_test, labels_test)

                if D_new < self.d:
                    print("Here we go binsearching again")
                    hi = 1.0
                    lo = 0.0
                    for i in range(10):
                        mid = (hi + lo) / 2
                        b.set_bias(mid)
                        D_new, F_new = b.test(iis_test, labels_test)
                        if D_new < self.d:
                            hi = mid
                        else:
                            lo = mid

            classifyer_list.append(b)

            new_iis = []
            new_labels = []
            for label, ii in zip(labels, iis):
                if label:
                    new_iis.append(ii)
                    new_labels.append(label)
                else:
                    if b.predict(ii):
                        new_iis.append(ii)
                        new_labels.append(0)
            iis = new_iis
            labels = array(new_labels)

        self.classifyers = classifyer_list

                

    def predict(self, ii):
        for classifyer in self.classifyers:
            if not classifyer.predict(ii):
                return False
        return True


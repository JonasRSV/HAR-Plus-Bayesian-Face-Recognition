from ada_boost import boosted_classifier
from processing import cross_validate
from features import generate_all_features, get_feature_matrix
from numpy import array
from sys import stdout


class cascade(object):
    def __init__(self, accepted_false_positive, min_accepted_detection, max_features):
        self.f = accepted_false_positive # Maximum percenteage off false positives to make it throug
        self.d = min_accepted_detection # Minimum amount of things that should pass through
        self.mf = max_features

    def train(self, iis, labels, fail_target):
        """Train the cascade."""
        # Positive samples
        # Negative samples
        total_labels = labels
        F_old = 1.0
        F_new = F_old
        D_old = 1.0
        D_new = D_old

        # Generate feature_matrix

        classifyer_list = []
        i = 0

        n_i = 0
        print("Starting training")
        all_features = generate_all_features()
        while F_new > fail_target:
            i += 1
            print("Round: {}, Currently: {},  Target is: {}" .format(i, F_new, fail_target))

            iis_train, labels_train, iis_test, labels_test = cross_validate(iis, labels, 0.8)

            feature_matrix = get_feature_matrix(iis_train, all_features)

            b = None
            F_old = F_new
            D_old = D_new
            inner_target = max(F_old * self.f, fail_target)

            weights, memory = None, None
            while F_new > inner_target:
                n_i += 1
                print("Current {}, Goal {}, Number Of Features {}"
                        .format(F_new, inner_target, n_i))
                b = boosted_classifier(n_i)
                weights, memory = b.train(feature_matrix, all_features, labels_train, weights, memory)
                d, f = b.test(iis_test, labels_test)

                if d < self.d:
                    hi = 1.0
                    lo = 0.0
                    for _ in range(10):
                        mid = (hi + lo) / 2
                        b.set_bias(mid)
                        d, f = b.test(iis_test, labels_test)
                        if d < self.d:
                            hi = mid
                        else:
                            lo = mid

                D_new = d * D_old
                F_new = f * F_old

                if self.mf < n_i:
                    F_new = 0.0

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

def calc_true_fp(current_labels, percentage, total_labels):
    total_false = len(total_labels[total_labels == 0])
    current_false = len(current_labels[current_labels == 0])

    return (current_false * percentage) / total_false


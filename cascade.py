from ada_boost import boosted_classifier
from processing import cross_validate
from features import generate_all_features, get_feature_matrix
from numpy import array
from sys import stdout


class cascade(object):
    def __init__(self, accepted_false_positive, min_accepted_detection):
        self.f = accepted_false_positive # Maximum percenteage off false positives to make it throug
        self.d = min_accepted_detection # Minimum amount of things that should pass through

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

        print("Starting training")
        """
        Instead of having F_new > fail_target i believe this is better.

        Having F_new > fail_target is just mean. Imagine incrementally succeeding in the middle
        loop so that the number of false positives constantly is reduced.. But you'll never get closer
        to fail_target target since you need to be below 0.05 percent false positive on the CURRENT classification.
        Imagine there only being 2 false positives left, y'd have to get 0 false positives for that to be
        below 0.05... And perhaps one of those two false positives is a cloud or something that practially
        looks like a face..
        """
        while F_new > fail_target:
            i += 1
            print("Round: {}, Currently: {},  Target is: {}" .format(i, F_new, fail_target))

            iis_train, labels_train, iis_test, labels_test = cross_validate(iis, labels, 0.8)

            all_features = generate_all_features()
            feature_matrix = get_feature_matrix(iis_train, all_features)

            n_i = 0
            b = None
            F_old = F_new
            D_old = D_new
            inner_target = max(F_old * self.f, fail_target)
            while F_new > inner_target:
                n_i += 1
                print("Current {}, Goal {}, Number Of Features {}"
                        .format(F_new, inner_target, n_i))
                b = boosted_classifier(n_i)
                b.train(feature_matrix, all_features, labels_train)
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

def calculate_true_false_positive(current_labels, percentage, total_labels):
    total_false = len(total_labels[total_labels == 0])
    current_false = len(current_labels[current_labels == 0])

    return (current_false * percentage) / total_false


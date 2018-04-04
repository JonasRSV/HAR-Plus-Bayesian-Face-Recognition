from pandas import read_csv
from random import choice, shuffle
from numpy import array, zeros, fromstring, sqrt, ones, concatenate, save, load
from time import time
import processing
import features
import os
import pickle

def read_filip_csv_format(file_path):
    """Read the awesome filip csv format."""
    negatives = []
    with open(file_path, "r") as filip_csv:
        images = filip_csv.readlines()
        for image in images:
            np_im = fromstring(image, dtype=int, sep=", ")
            dim = np_im.shape

            new_dim = int(sqrt(dim))

            negatives.append(np_im.reshape((new_dim, new_dim)))

    return negatives


def training_validation():
    """Read and reformat traning data."""
    tpcsv = time()

    train_positives = read_csv("training.csv", sep=',', header='infer')
    train_negatives = read_filip_csv_format("file.csv")
    train_test = read_csv("test.csv", sep=",", header="infer")

    print("reading csv: {}".format(time() - tpcsv))

    tfcsv = time()
    negative_half = int(len(train_negatives) / 2)

    negative_images_1 = train_negatives[:negative_half]
    negative_images_2 = train_negatives[negative_half:]

    positive_images_1 = []
    positive_images_2 = []
    for index, img in enumerate(train_positives["Image"]):
        positive_images_1.append(
            fromstring(img, dtype=int, sep=" ").reshape((96, 96)))

    for index, img in enumerate(train_test["Image"]):
        positive_images_2.append(
            fromstring(img, dtype=int, sep=" ").reshape((96, 96)))

    print("formatting to numpy matrices: {}".format(time() - tfcsv))

    tpd = time()

    n1random = choose_images(negative_images_1,
                                 cardinality=negative_half - 1)

    n2random = choose_images(negative_images_2,
                                cardinality=negative_half)

    p1random = choose_images(positive_images_1)
    p2random = choose_images(positive_images_2)

    print("picking random data: {}".format(time() - tpd))

    fdt = time()

    p1label = [1] * (len(p1random))
    p2label = [1] * (len(p2random))

    n1label = [0] * (len(n1random))
    n2label = [0] * (len(n2random))

    p1random.extend(n1random)
    p1label.extend(n1label)

    p2random.extend(n2random)
    p2label.extend(n2label)

    train_images = p1random
    train_labels = p1label

    test_images = p2random
    test_labels = p2label

    print("Formatting data: {}".format(time() - fdt))

    return (train_images, array(train_labels)),\
           (test_images, array(test_labels))


def choose_images(images, cardinality=500):
    """Pick 'cardinality' random images."""
    shuffle(images)
    return images[:cardinality]


def get_trainable_data(training_images, test_images):
    """Handle caching and loading of formatted data."""
    outTrain = None
    outTest = None

    training = None
    test = None
    IItraining = None
    IItest = None
    all_features = features.generate_all_features()

    if not os.path.isfile("training_feature_matrix.npy"):
    
        print("Stage 1")
        IItraining = processing.bulk_II(training_images)
        IItest = processing.bulk_II(test_images)
        
        outTrain = open('IITraining.pkl', 'wb')
        outTest = open('IITesting.pkl', 'wb')
        pickle.dump(IItraining, outTrain)
        pickle.dump(IItest, outTest)
        outTrain.close()
        outTest.close()
        
        print("Stage 2")
        training = features.get_feature_matrix(IItraining, all_features)
        test = features.get_feature_matrix(IItest, all_features)

        """ _ is same as above. """

        save("training_feature_matrix", training)
        save("test_feature_matrix", test)
    else:
        training = load("training_feature_matrix.npy")
        test = load("test_feature_matrix.npy")
        outTrain = open('IITraining.pkl', 'rb')
        outTest = open('IITesting.pkl', 'rb')
        IItraining = pickle.load(outTrain)
        IItest = pickle.load(outTest)
        outTrain.close()
        outTest.close()


    return training, test, IItraining, IItest, all_features, outTrain, outTest

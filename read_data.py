from pandas import read_csv
from random import choice, shuffle
from numpy import array, zeros, fromstring
from time import time


def training_validation():
    """Read and reformat traning data."""

    tpcsv = time()

    train_positives = read_csv("training.csv", sep=',', header='infer')
    train_negatives = read_csv("file.csv", sep=",", header="infer")
    train_test = read_csv("test.csv", sep=",", header="infer")

    print("reading csv: {}".format(time() - tpcsv))

    tfcsv = time()

    images = zeros((7049, 96, 96))
    test_images = zeros((1783, 96, 96))
    for index, img in enumerate(train_positives["Image"]):
        images[index] = fromstring(img, dtype=int, sep=" ").reshape((96, 96))

    for index, img in enumerate(train_test["Image"]):
        test_images[index] = fromstring(img, dtype=int, sep=" ").reshape((96, 96))

    print("formatting to numpy matrices: {}".format(time() - tfcsv))

    tpd = time()

    print(train_test[0])

    training, validation = choose_images(images), choose_images(images)

    print("picking random data: {}".format(time() - tpd))

    return training, validation


def choose_images(images, cardinality=500):
    """Pick 'cardinality' random images."""
    shuffle(images)
    return images[:cardinality]


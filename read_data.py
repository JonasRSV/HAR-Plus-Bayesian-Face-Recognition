from pandas import read_csv
from random import choice, shuffle
from numpy import array, zeros, fromstring, sqrt, ones, concatenate
from time import time

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

    positive_images_1 = zeros((7049, 96, 96))
    positive_images_2 = zeros((1783, 96, 96))
    for index, img in enumerate(train_positives["Image"]):
        positive_images_1[index] =\
            fromstring(img, dtype=int, sep=" ").reshape((96, 96))

    for index, img in enumerate(train_test["Image"]):
        positive_images_1[index] =\
            fromstring(img, dtype=int, sep=" ").reshape((96, 96))

    print("formatting to numpy matrices: {}".format(time() - tfcsv))

    tpd = time()

    tr_negatives = choose_images(negative_images_1,
                                 cardinality=negative_half - 1)

    t_negatives = choose_images(negative_images_2,
                                cardinality=negative_half)

    tr_positives = choose_images(positive_images_1)
    t_positives = choose_images(positive_images_2)

    print("picking random data: {}".format(time() - tpd))

    fpd = time()

    nsize1 = len(tr_negatives)
    nsize2 = len(t_negatives)

    psize1 = len(tr_positives)
    psize2 = len(t_positives)

    nlabels1 = zeros(nsize1)
    nlabels2 = zeros(nsize2)

    plabels1 = ones(psize1)
    plabels2 = ones(psize2)

    training_ = concatenate([tr_positives, tr_negatives])
    training_labels = concatenate([plabels1, nlabels1])

    test_ = concatenate([t_positives, t_negatives])
    test_labels = concatenate([plabels2, nlabels2])

    print("Formatting data: {}".format(time() - fpd))

    return (training_, training_labels), (test_, test_labels)


def choose_images(images, cardinality=500):
    """Pick 'cardinality' random images."""
    shuffle(images)
    return images[:cardinality]


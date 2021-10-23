import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import random


# Sample SIFT descriptors, cluster them using k-means, and return the fitted k-means model
def build_vocabulary(image_paths, vocab_size):
    n_image = len(image_paths)

    # Since want to sample tens of thousands of SIFT descriptors from different images, we
    # calculate the number of SIFT descriptors we need to sample from each image.
    n_each = int(np.ceil(10000 / n_image))

    # Initialize an array of features, which will store the sampled descriptors
    descriptors = np.zeros((n_image * n_each, 128))
    descriptorsList = descriptors.tolist()

    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',', dtype=float)
        sift_descriptors = features[:, 2:]

        # choose a random sample from sift_descriptors without replacement
        index = 0
        alreadyPicked = []
        siftSize = len(sift_descriptors)

        while index < n_each:
            rSift = random.randint(0, siftSize - 1)
            if rSift not in alreadyPicked:
                alreadyPicked.append(rSift)
                descriptorsList.append(sift_descriptors[rSift])
                index += 1

    # perform k-means clustering to cluster sampled sift descriptors into vocab_size regions.
    kmeans = KMeans(n_clusters=vocab_size, random_state=0).fit(descriptorsList)

    return kmeans


# Represent each image as bags of SIFT features histogram.
def get_bags_of_sifts(image_paths, kmeans):
    n_image = len(image_paths)
    vocab_size = kmeans.cluster_centers_.shape[0]

    image_feats = np.zeros((n_image, vocab_size))

    # create histogram for every image
    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',', dtype=float)
        sift_descriptors = features[:, 2:]

        # Assign each feature to the closest cluster center
        labelPredictionList = kmeans.predict(sift_descriptors)
        histogram = np.bincount(labelPredictionList, minlength=vocab_size)

        # Build a histogram normalized by the number of descriptors
        numberDescriptor = sift_descriptors.shape[0]
        image_feats[i] = list(map(lambda x: float(x / numberDescriptor), histogram))

    return image_feats


# Load from the training/testing dataset
def load(ds_path):
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.txt"))
    n_files = len(files)
    image_paths = np.asarray(files)
 
    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_files)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0,0]

    # Randomize the order
    idx = np.random.choice(n_files, size=n_files, replace=False)
    image_paths = image_paths[idx]
    labels = labels[idx]

    return image_paths, labels


# Make histogram
def showAverageHistogram(image_feats, labels, image_paths):
    # create list of 15 histograms and keep track of number of labels each
    averageHistogram = np.zeros((15, image_feats.shape[1]))
    count = np.zeros(15)
    names = ["" for x in range(15)]

    # add up histograms into their label group
    index = 0
    for label in labels:
        label = int(label)
        averageHistogram[label] += image_feats[index]
        count[label] += 1
        head, tail = os.path.split(image_paths[index])
        names[label] = head
        index += 1

    # average each 15 histogram and display them
    featurePlot = range(image_feats.shape[1])
    for i in range(15):
        # average each histogram
        averageHistogram[i] = list(map(lambda x: float(x / count[i]), averageHistogram[i]))

        # display histogram
        plt.bar(featurePlot, averageHistogram[i], width=1)
        plt.title(names[i])
        plt.xlabel('features')
        plt.ylabel('Proportion of all features(normalised to 1)')
        plt.grid(axis='x', alpha=0.75)
        plt.show()


# Calculate accuracy
def accuracyScore(testLabels, predictionLabels):
    count = 0
    success = 0
    for tLabel, pLabel in zip(testLabels, predictionLabels):
        count += 1
        if tLabel == pLabel:
            success += 1
    score = success / count * 100

    return score


# Make colorful confusion matrix
def heat_confusion_matrix(paths, labels, predictions, title):
    #  match the label numbers with the label names in ascending order to suit confusion_matrix ordering system
    labelList = []
    #  convert from ndarray to list
    test_image_paths_list = paths.tolist()
    test_labels_list = labels.tolist()

    #  match labels 0-14
    for n in range(15):
        # find index 0-14
        index = test_labels_list.index(n)
        # find matching path
        path = test_image_paths_list[index]
        # get folder name
        name = path.split(os.sep)[2:-1][0]
        # add to label list from 0-14
        labelList.append(name)

    # "If label = None is given, those that appear at least once in y_true or y_pred are used in sorted order.""
    #  so we know that the order of the labels will match the order from ascending order
    matrix = confusion_matrix(labels, predictions, normalize="true", labels=None)

    #  create dataframe
    df_cm = pd.DataFrame(matrix, index=labelList, columns=labelList)
    # fix size
    plt.figure(figsize=(15, 10))
    # add title
    ax = plt.axes()
    ax.set_title(title)
    # for label size
    sn.set(font_scale=1.4)
    # create heatmap and set font size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})
    # display
    plt.show()


if __name__ == "__main__":
    paths, labels = load("sift/train")
    build_vocabulary(paths, 10)

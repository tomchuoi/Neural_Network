import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_dataset(dataset, path):
    #scan all the directories and put them in labels
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []

    for label in labels:
        label_path = os.path.join(path, dataset, label)
        if not os.path.isdir(label_path):
            continue  #skip files that are not directories

        for file in os.listdir(os.path.join(path, dataset, label)):
            #read the image data
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    #convert the data to arrays
    return np.array(X), np.array(y).astype('uint8')

def create_dataset(path):
    X, y = load_dataset('train', path)
    X_test, y_test = load_dataset('test', path)

    return X, y, X_test, y_test


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


NUM_CLASSES = 3

# Set the dimensions of the images
SHAPE = 256
DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data")
DATASET_PATH = os.path.join(DATA_PATH, "Dataset_BUSI_with_GT")
TRAINIG_PATH = DATASET_PATH + "_Train"
TESTING_PATH = DATASET_PATH + "_Test"
SAMPLE_TYPES = ["normal", "benign", "malignant"]
NORMAL = 0
BENIGN = 1
MALIGNANT = 2


def get_segmentation_data():
    segmentations_train = []
    diagnoses_train = []

    segmentations_test = []
    diagnoses_test = []

    sample_types = ["normal", "benign", "malignant"]

    for sample_type in sample_types:
        train_folder = os.path.join(TRAINIG_PATH, sample_type)
        test_folder = os.path.join(TESTING_PATH, sample_type)
        get_folder_segmentation_data(
            train_folder, segmentations_train, diagnoses_train, sample_type
        )
        get_folder_segmentation_data(
            test_folder, segmentations_test, diagnoses_test, sample_type
        )

    return (
        np.array(segmentations_train),
        np.array(diagnoses_train),
        np.array(segmentations_test),
        np.array(diagnoses_test),
    )


def get_folder_segmentation_data(folder, segmentations, diagnoses, sample_type):
    samples = os.listdir(folder)

    for sample in samples:
        if not re.search("segmentation", sample):
            continue

        file = plt.imread(os.path.join(folder, sample))
        file_resized = cv2.resize(file, (SHAPE, SHAPE))

        segmentations.append(file_resized)

        if sample_type == "normal":
            diagnoses.append(NORMAL)
        elif sample_type == "benign":
            diagnoses.append(BENIGN)
        elif sample_type == "malignant":
            diagnoses.append(MALIGNANT)


if __name__ == "__main__":
    # Load the training and test data
    # You will need to provide the paths to your own dataset
    X_train, y_train, X_test, y_test = get_segmentation_data()

    # # Normalize the data
    # X_train = X_train.astype("float32") / 255
    # X_test = X_test.astype("float32") / 255

    # One-hot encode the labels
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Define the model architecture
    model = Sequential()
    model.add(
        Conv2D(
            128,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(SHAPE, SHAPE, 4),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    # Compile the model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=10,
        epochs=100,
        verbose=2,
        validation_data=(X_test, y_test),
    )

    # Evaluate the model on the test set
    score = model.evaluate(X_test, y_test, verbose=0)

    # Print test accuracy
    print("Test accuracy:", score[1])

    model.save("SegmentationClassificationModel.h5")

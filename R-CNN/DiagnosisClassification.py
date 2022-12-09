import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
import constants
import ClassificationModel


SAMPLE_TYPES = ["benign", "malignant"]
BENIGN = 0
MALIGNANT = 1
NUM_CLASSES = 2


def get_segmentation_data():
    segmentations_train = []
    diagnoses_train = []

    segmentations_test = []
    diagnoses_test = []

    for sample_type in SAMPLE_TYPES:
        train_folder = os.path.join(constants.TRAINIG_PATH, sample_type)
        test_folder = os.path.join(constants.TESTING_PATH, sample_type)
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
        file_resized = cv2.resize(file, (constants.SHAPE, constants.SHAPE))

        segmentations.append(file_resized)

        if sample_type == "benign":
            diagnoses.append(BENIGN)
        elif sample_type == "malignant":
            diagnoses.append(MALIGNANT)


if __name__ == "__main__":
    # Load the training and test data
    # You will need to provide the paths to your own dataset
    X_train, y_train, X_test, y_test = get_segmentation_data()

    # # Normalize the data
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # One-hot encode the labels
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model = ClassificationModel.get_image_classification_model(num_classes=2)

    data_generator = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0,
        height_shift_range=0,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=True,
    )

    X_train_augmented = data_generator.flow(X_train, y_train, batch_size=8)

    # Train the model
    model.fit(
        X_train_augmented,
        epochs=200,
        verbose=2,
        validation_data=(X_test, y_test),
    )

    # Evaluate the model on the test set
    score = model.evaluate(X_test, y_test, verbose=0)

    model.save(os.path.join(constants.MODELS_PATH, "DiagnosisClassificationModel.h5"))

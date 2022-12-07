import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from SonogramSegmentation import get_sonogram_data


def save_predictions(folder_name, sample_type, images, masks):
    destination_folder = os.path.join(folder_name, sample_type)

    # Create a directory for saving the images
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for i, image in enumerate(images):
        prediction = unet_model.predict(np.array([image]))

        plt.imsave(f"{destination_folder}/{sample_type} ({i}).png", image)
        plt.imsave(
            f"{destination_folder}/{sample_type} ({i})_segmentation.png",
            prediction[0][:, :, 0],
        )


if __name__ == "__main__":
    # Load the saved detection segmentation model
    unet_model = tf.keras.models.load_model("DetectionSegmentationModel.h5")

    # Initialize dictionary to store image and mask data
    norm_X_train, norm_y_train, norm_X_test, norm_y_test = get_sonogram_data(
        ["normal"], shape=256
    )
    benign_X_train, benign_y_train, benign_X_test, benign_y_test = get_sonogram_data(
        ["benign"], shape=256
    )
    (
        malignant_X_train,
        malignant_y_train,
        malignant_X_test,
        malignant_y_test,
    ) = get_sonogram_data(["malignant"], shape=256)

    segmentation_trainig_folder = os.path.join(os.getcwd(), "Segmentation Data Train")
    segmentation_test_folder = os.path.join(os.getcwd(), "Segmentation Data Test")

    # Save the training data
    save_predictions(segmentation_trainig_folder, "normal", norm_X_train, norm_y_train)
    save_predictions(
        segmentation_trainig_folder, "benign", benign_X_train, benign_y_train
    )
    save_predictions(
        segmentation_trainig_folder, "malignant", malignant_X_train, malignant_y_train
    )

    # Save the testing data
    save_predictions(segmentation_test_folder, "normal", norm_X_test, norm_y_test)
    save_predictions(segmentation_test_folder, "benign", benign_X_test, benign_y_test)
    save_predictions(
        segmentation_test_folder, "malignant", malignant_X_test, malignant_y_test
    )

    # data_dir = ""
    # num_classes = 3

    # # Set the dimensions of the images
    # IMG_WIDTH = 128
    # IMG_HEIGHT = 128

    # # Load the training and test data
    # # You will need to provide the paths to your own dataset
    # x_train = np.load(os.path.join(data_dir, "x_train.npy"))
    # y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    # x_test = np.load(os.path.join(data_dir, "x_test.npy"))
    # y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    # # Resize the images
    # x_train = np.array([cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) for img in x_train])
    # x_test = np.array([cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) for img in x_test])

    # # Normalize the data
    # x_train = x_train.astype("float32") / 255
    # x_test = x_test.astype("float32") / 255

    # # One-hot encode the labels
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    # # Define the model architecture
    # model = Sequential()
    # model.add(
    #     Conv2D(
    #         32,
    #         kernel_size=(3, 3),
    #         activation="relu",
    #         input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    #     )
    # )
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(128, activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation="softmax"))

    # # Compile the model
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # # Train the model
    # model.fit(
    #     x_train,
    #     y_train,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     verbose=1,
    #     validation_data=(x_test, y_test),
    # )

    # # Evaluate the model on the test set
    # score = model.evaluate(x_test, y_test, verbose=0)

    # # Print test accuracy
    # print("Test accuracy:", score[1])

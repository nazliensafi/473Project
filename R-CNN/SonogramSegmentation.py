import os
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import math
import shutil

DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), "data")
DATASET_PATH = os.path.join(DATA_PATH, "Dataset_BUSI_with_GT")
TRAINIG_PATH = DATASET_PATH + "_Train"
TESTING_PATH = DATASET_PATH + "_Test"


def split_sonogram_data():
    """
    This function takes the original ultrasound image dataset and splits it into training and testing data.
    The two subsets are then saved in their appropriate folders
    """
    normal_folder = os.path.join(DATASET_PATH, "normal")
    benign_folder = os.path.join(DATASET_PATH, "benign")
    malignant_folder = os.path.join(DATASET_PATH, "malignant")

    split_folder("normal", normal_folder)
    split_folder("benign", benign_folder)
    split_folder("malignant", malignant_folder)


def split_folder(sample_type, folder):
    """
    This functions splits the given folder in a 80-20 training testing split.
    The training and testing images are saved in their appropriate folders.

    Args:
    - sample_type: the class of tumor in the folder
    - folder: the file path to the folder to split
    """
    samples = os.listdir(folder)

    # Get the list of unique sample numbers
    sample_numbers = []

    for sample in samples:
        number = re.search("(\d+)", sample).group()
        sample_numbers.append(int(number))

    sample_numbers = list(set(sample_numbers))
    sample_numbers.sort()

    # Get the number of training samples
    num_training = math.floor(len(sample_numbers) * 0.8)

    # Create directories for saving the images
    training_folder = os.path.join(TRAINIG_PATH, sample_type)
    test_folder = os.path.join(TESTING_PATH, sample_type)

    if not os.path.exists(training_folder):
        os.makedirs(training_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Copy over the appropriate samples to training and testing folders
    for i in sample_numbers:
        destination_folder = test_folder if i > num_training else training_folder

        image_name = f"{sample_type} ({i}).png"
        mask_name = f"{sample_type} ({i})_mask.png"

        image_source = os.path.join(folder, image_name)
        image_destination = os.path.join(destination_folder, image_name)

        mask_source = os.path.join(folder, mask_name)
        mask_destination = os.path.join(destination_folder, mask_name)

        shutil.copy(image_source, image_destination)
        shutil.copy(mask_source, mask_destination)


def get_sonogram_data(shape):
    """
    This function returns the training and testing images and masks.
    Args:
    - shape: the desired width and height of the images.
    Returns:
    - X_train, y_train, X_test, y_test, the training and testing dataa
    """
    images_train = []
    masks_train = []

    images_test = []
    masks_test = []

    normal_train_folder = os.path.join(TRAINIG_PATH, "normal")
    normal_test_folder = os.path.join(TESTING_PATH, "normal")
    get_folder_data(normal_train_folder, images_train, masks_train, shape)
    get_folder_data(normal_test_folder, images_test, masks_test, shape)

    benign_train_folder = os.path.join(TRAINIG_PATH, "benign")
    benign_test_folder = os.path.join(TESTING_PATH, "benign")
    get_folder_data(benign_train_folder, images_train, masks_train, shape)
    get_folder_data(benign_test_folder, images_test, masks_test, shape)

    malignant_train_folder = os.path.join(TRAINIG_PATH, "malignant")
    malignant_test_folder = os.path.join(TESTING_PATH, "malignant")
    get_folder_data(malignant_train_folder, images_train, masks_train, shape)
    get_folder_data(malignant_test_folder, images_test, masks_test, shape)

    return (
        np.array(images_train),
        np.array(masks_train),
        np.array(images_test),
        np.array(masks_test),
    )


def get_folder_data(folder, images, masks, shape):
    samples = os.listdir(folder)

    for sample in samples:
        file = plt.imread(os.path.join(folder, sample))
        file_resized = cv2.resize(file, (shape, shape))

        if re.search("mask", sample):
            if file_resized.shape == (shape, shape, 3):
                file_resized = np.zeros((shape, shape))

            masks.append(file_resized)
        else:
            images.append(file_resized)


def convolutional_block(input_tensor, num_filters, kernel_size=3, use_batch_norm=True):
    """
    This function applies a convolutional block to the input tensor.
    A convolutional block consists of two convolutional layers,
    each followed by an optional batch normalization layer
    and an activation layer.

    Args:
    - input_tensor: a tensor, the input to the convolutional block.
    - num_filters: an integer, the number of filters in the convolutional layers.
    - kernel_size: an integer, the size of the convolutional kernels.
    - use_batch_norm: a boolean, whether to apply batch normalization after the convolutional layers.

    Returns:
    - a tensor, the output of the convolutional block.
    """
    tensor = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)

    if use_batch_norm:
        tensor = tf.keras.layers.BatchNormalization()(tensor)

    tensor = tf.keras.layers.Activation("relu")(tensor)

    tensor = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(tensor)

    if use_batch_norm:
        tensor = tf.keras.layers.BatchNormalization()(tensor)

    tensor = tf.keras.layers.Activation("relu")(tensor)

    return tensor


def create_unet_model(
    input_image, num_filters=16, dropout_rate=0.1, use_batch_norm=True
):
    """
    This function creates a U-Net model.
    A U-Net model consists of a sequence of convolutional blocks,
    followed by max pooling layers, dropout layers,
    and up-sampling layers.

    Args:
    - input_image: a tensor, the input image to the U-Net model.
    - num_filters: an integer, the number of filters in the convolutional layers.
    - dropout_rate: a float, the rate of dropout to apply after the max pooling layers.
    - use_batch_norm: a boolean, whether to apply batch normalization after the convolutional layers.

    Returns:
    - a keras Model, the U-Net model.
    """
    # Apply a series of convolutional blocks to the input image
    conv_block1 = convolutional_block(
        input_image, num_filters * 1, kernel_size=3, use_batch_norm=use_batch_norm
    )
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block1)
    pool1 = tf.keras.layers.Dropout(dropout_rate)(pool1)

    conv_block2 = convolutional_block(
        pool1, num_filters * 2, kernel_size=3, use_batch_norm=use_batch_norm
    )
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block2)
    pool2 = tf.keras.layers.Dropout(dropout_rate)(pool2)

    conv_block3 = convolutional_block(
        pool2, num_filters * 4, kernel_size=3, use_batch_norm=use_batch_norm
    )
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block3)
    pool3 = tf.keras.layers.Dropout(dropout_rate)(pool3)

    conv_block4 = convolutional_block(
        pool3, num_filters * 8, kernel_size=3, use_batch_norm=use_batch_norm
    )
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv_block4)
    pool4 = tf.keras.layers.Dropout(dropout_rate)(pool4)

    conv_block5 = convolutional_block(
        pool4, num_filters * 16, kernel_size=3, use_batch_norm=use_batch_norm
    )

    # Apply up-sampling and concatenation to the output of the convolutional blocks
    up6 = tf.keras.layers.Conv2DTranspose(
        num_filters * 8, (3, 3), strides=(2, 2), padding="same"
    )(conv_block5)
    up6 = tf.keras.layers.concatenate([up6, conv_block4])
    up6 = tf.keras.layers.Dropout(dropout_rate)(up6)
    conv_block6 = convolutional_block(
        up6, num_filters * 8, kernel_size=3, use_batch_norm=use_batch_norm
    )

    up7 = tf.keras.layers.Conv2DTranspose(
        num_filters * 4, (3, 3), strides=(2, 2), padding="same"
    )(conv_block6)
    up7 = tf.keras.layers.concatenate([up7, conv_block3])
    up7 = tf.keras.layers.Dropout(dropout_rate)(up7)
    conv_block7 = convolutional_block(
        up7, num_filters * 4, kernel_size=3, use_batch_norm=use_batch_norm
    )

    up8 = tf.keras.layers.Conv2DTranspose(
        num_filters * 2, (3, 3), strides=(2, 2), padding="same"
    )(conv_block7)
    up8 = tf.keras.layers.concatenate([up8, conv_block2])
    up8 = tf.keras.layers.Dropout(dropout_rate)(up8)
    conv_block8 = convolutional_block(
        up8, num_filters * 2, kernel_size=3, use_batch_norm=use_batch_norm
    )

    up9 = tf.keras.layers.Conv2DTranspose(
        num_filters * 1, (3, 3), strides=(2, 2), padding="same"
    )(conv_block8)
    up9 = tf.keras.layers.concatenate([up9, conv_block1])
    up9 = tf.keras.layers.Dropout(dropout_rate)(up9)
    conv_block9 = convolutional_block(
        up9, num_filters * 1, kernel_size=3, use_batch_norm=use_batch_norm
    )

    output = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(conv_block9)
    unet = tf.keras.Model(inputs=[input_image], outputs=[output])

    return unet


def save_predictions(folder, sample_type, unet, shape):
    """
    This function generates the predictions of images found inside given folder.
    This function then saves the predictions of a specific sample type in a given folder.
    These predictions will be used to train and test and classification model.
    Args:
    - folder: the folder in which the predictions will be saved.
    - sample_type: the class of the sample (normal, benign, malignant).
    - unet: the model that will make the prediction.
    - shape: the dimensions image must be resized
    """
    destination = os.path.join(folder, sample_type)
    samples = os.listdir(destination)

    for sample in samples:
        if re.search("mask", sample):
            continue

        sample_number = re.search("\(\d+\)", sample).group()
        image = plt.imread(os.path.join(destination, sample))
        image_resized = cv2.resize(image, (shape, shape))

        prediction = unet.predict(np.array([image_resized]))

        plt.imsave(
            f"{destination}/{sample_type} {sample_number}_segmentation.png",
            prediction[0][:, :, 0],
        )


if __name__ == "__main__":
    # Train a model if it does not already exists
    if not os.path.isfile(os.path.join(os.getcwd(), "SonogramSegmentationModel.h5")):
        # Split the data into training and testing datasets
        split_sonogram_data()

        X_train, y_train, X_test, y_test = get_sonogram_data(shape=256)

        # Define the input layer
        inputs = tf.keras.layers.Input((256, 256, 3))

        # Create the U-Net model with a specified dropout rate
        unet_model = create_unet_model(inputs, dropout_rate=0.07)

        # Compile the model with the Adam optimizer and binary cross-entropy loss
        unet_model.compile(
            optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Fit the model on the input data
        fit_result = unet_model.fit(
            X_train,
            y_train,
            batch_size=10,
            epochs=100,
            verbose=2,
            validation_data=(X_test, y_test),
        )

        unet_model.save("SonogramSegmentationModel.h5")

    """
    Save the predictions in segmentation folders.
    The predictions will be used to train a classification model.
    """
    # Load the saved detection segmentation model
    unet_model = tf.keras.models.load_model("SonogramSegmentationModel.h5")

    # Save training data
    save_predictions(TRAINIG_PATH, "normal", unet_model, shape=256)
    save_predictions(TRAINIG_PATH, "benign", unet_model, shape=256)
    save_predictions(TRAINIG_PATH, "malignant", unet_model, shape=256)

    # Save testing data
    save_predictions(TESTING_PATH, "normal", unet_model, shape=256)
    save_predictions(TESTING_PATH, "benign", unet_model, shape=256)
    save_predictions(TESTING_PATH, "malignant", unet_model, shape=256)

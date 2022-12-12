# 473Project

In the root directory, you can run:

### In macOS

### `python3 -m venv venv`

### In Windows

### `python -m venv venv`

Creates a virtual environment directory in the project root directory.

### In macOS

### `source ./venv/bin/activate`

### In Windows

### `. venv/scripts/activate`

Activates the virtual environment in the project root directory.

### `pip install -r requirements.txt`

Installs all required python modules into the activated virtual environment.

## Project Description

The code is split into two main parts: an FNA folder containing the classifiers for the FNA-type biopsy and an R-CNN folder containing the models for ultrasound image segmentation and classification of the generated segmentation images. Each Python script can be run individually. In addition, a notebook entitled **Project Demo.ipynb** illustrates how the FNA classification works.

### Sonogram Segmentation

**SonogramSegmentation.py** script begins by splitting the data from the _Dataset_BUSI_with_GT_ folder into an 80% and 20% training split that will respectively be placed in the _Dataset_BUSI_with_GT_Train_ and the _Dataset_BUSI_with_GT_Test_ folders. After the data is split, it will begin training a model dedicated to generating the segmentation of the ultrasound images. Once the model is trained, it is saved as **SonogramSegmentationModel.h5** in the _models_ sub-folder and then used to generate and save the sonograms for both training and testing folders. We tested with different hyperparameters and achieved an accuracy of 94%.

### Classification Model

**ClassificationModel.py** script contains an _get_image_classification_model()_ function that returns a model structured to classify images. In our case, the images used will be the generated segmentation images.

### Diagnosis Classification

**DiagnosisClassification.py** script takes the generated images from the _Dataset_BUSI_with_GT_Train_ folder and the in the _normal_, _benign_, and _malignant_ sub-folders to train a model to classify the segmentation images as either normal, benign or malignant. Finally, the model is tested with the images from the _Dataset_BUSI_with_GT_Test_ folder and only achieves an accuracy of 64%.

### Normality Classification

**NormalityClassification.py** script takes the generated images from the _Dataset_BUSI_with_GT_Train_ folder and the in the _normal_, _benign_, and _malignant_ sub-folders to train a model to classify the segmentation images normal or abnormal (benign or malignant). The model is tested with the images from the _Dataset_BUSI_with_GT_Test_ folder and achieves an accuracy of about 93%. This alternative model can be used to determine whether a biopsy is required.

### Save Examples

**SaveExamples.py** script saves the ultrasound image, the mask, and the generated segmentation as a single image of specified samples found in the _Dataset_BUSI_with_GT_Test_ that are interesting for the report into the _data/examples_ folder. These samples illustrate what good and bad segmentations look like while also showing how the segmentation model can have difficulty discerning the tumor from the noise of the sonogram.

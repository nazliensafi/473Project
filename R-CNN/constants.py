import os

# paths
CWD = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(CWD, "data")
DATASET_PATH = os.path.join(DATA_PATH, "Dataset_BUSI_with_GT")
TRAINIG_PATH = DATASET_PATH + "_Train"
TESTING_PATH = DATASET_PATH + "_Test"
MODELS_PATH = os.path.join(CWD, "R-CNN", "models")

SHAPE = 256

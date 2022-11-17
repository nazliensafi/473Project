import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split


data_filepath = os.path.join(os.getcwd(), "data.csv")
df = pd.read_csv(data_filepath)

df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)


def categorical_to_numeric_diagnosis(x):
    if x == "M":
        return 1
    if x == "B":
        return 0


df["diagnosis"] = df["diagnosis"].apply(categorical_to_numeric_diagnosis)

features = list(df.columns[1:31])

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["diagnosis"].values, test_size=0.30, random_state=42
)

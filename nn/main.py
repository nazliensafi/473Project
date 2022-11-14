# imports
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix

data_filepath = os.path.join(os.getcwd(), "..", "data.csv")
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

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(
    Dense(units=32, kernel_initializer="uniform", activation="relu", input_dim=30)
)
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(Dense(units=16, kernel_initializer="uniform", activation="relu"))
classifier.add(Dropout(rate=0.1))

# Adding the third hidden layer
classifier.add(Dense(units=8, kernel_initializer="uniform", activation="relu"))
classifier.add(Dropout(rate=0.1))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.summary()

scaler = StandardScaler()

classifier.fit(
    scaler.fit_transform(X_train.values),
    np.array(y_train),
    batch_size=5,
    epochs=100,
)

classifier.save("breast_cancer_model.h5")  # Save trained ANN

y_prediction = classifier.predict(scaler.transform(X_test.values))

for sample_pred in y_prediction:
    diff = 1 - sample_pred[0]
    if sample_pred[0] > diff:
        sample_pred[0] = 1
    else:
        sample_pred[0] = 0


cm = confusion_matrix(y_test, y_prediction)
tn, fn, fp, tp = confusion_matrix(y_prediction, y_test).ravel()

print(cm)

accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print("Accuracy: " + str(accuracy * 100) + "%")

import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from NeuralNetwork import NeuralNetwork


def numerical_diagnosis(diagnosis):
    if diagnosis == "M":
        return 1
    if diagnosis == "B":
        return 0


if __name__ == "__main__":
    data_filepath = os.path.join(os.getcwd(), "..", "data.csv")
    df = pd.read_csv(data_filepath)

    df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

    # Transform diagnosis column to 1s and 0s
    df["diagnosis"] = df["diagnosis"].apply(numerical_diagnosis)

    features = list(df.columns[1:11])

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["diagnosis"].values, test_size=0.3
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.values)
    X_test = scaler.fit_transform(X_test.values)

    # Define layer sizes
    input_nodes = X_train.shape[1]
    hidden_nodes = 32
    output_nodes = 2

    # Create neural network
    nn = NeuralNetwork([input_nodes, hidden_nodes, output_nodes])

    nn.train(
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size=10,
        epochs=100,
        l_rate=0.2,
    )

    # Get model performance
    y_prediction = nn.predict(X_test)
    y_prediction = np.reshape(y_prediction, (len(y_prediction), 1))

    cm = confusion_matrix(y_test, y_prediction)
    tn, fn, fp, tp = confusion_matrix(y_prediction, y_test).ravel()

    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print(f"\nAccuracy: {str((accuracy * 100).round(4))}%")

    nn.show_accuracies()

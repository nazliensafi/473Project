import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, sizes):
        self.input_nodes = sizes[0]
        self.hidden_nodes = sizes[1]
        self.output_nodes = sizes[2]
        self.init_params()

        self.train_log = []
        self.test_log = []

    def init_params(self):
        self.w1 = np.random.randn(self.input_nodes, self.hidden_nodes) * 0.01
        self.b1 = np.zeros((1, self.hidden_nodes))
        self.w2 = np.random.randn(self.hidden_nodes, self.output_nodes) * 0.01
        self.b2 = np.zeros((1, self.output_nodes))

    def forward_prop(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self.ReLU(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.softmax(z2)

        return a1, a2

    def ReLU(self, input):
        output = np.maximum(0, input)
        return output

    def softmax(self, input):
        exp_x = np.exp(input - np.max(input, axis=1, keepdims=True))
        output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return output

    def backward_prop(self, a1, a2, X, y):
        dz2 = self.cross_entropy_derivative(a2, y)
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        dz1 = np.dot(dz2, self.w2.T)
        dz1 = self.ReLU_deriv(dz1, a1)
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)

        return dw1, db1, dw2, db2

    def ReLU_deriv(self, Z, out):
        Zc = Z.copy()
        Zc[out <= 0] = 0

        return Zc

    def optimize(self, l_rate, dw1, db1, dw2, db2):
        # Optimize the values of params from grads
        self.w1 -= l_rate * dw1
        self.b1 -= l_rate * db1
        self.w2 -= l_rate * dw2
        self.b2 -= l_rate * db2

    def cross_entropy(self, a2, y):
        m = y.shape[0]
        log_likelihood = -np.log(a2[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def cross_entropy_derivative(self, a2, y):
        m = y.shape[0]
        grad = a2.copy()
        grad[range(m), y] -= 1
        grad = grad / m
        return grad

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return out

    def predict(self, X):
        _, a2 = self.forward_prop(X)
        return np.argmax(a2, axis=1)

    def train(self, X_train, y_train, X_test, y_test, batch_size, epochs, l_rate):
        # Separate the date in batches
        batches = self.get_batches(X_train, y_train, batch_size)
        test_batches = self.get_batches(X_test, y_test, batch_size)

        for i in range(epochs):
            loss_batch = []
            test_loss_batch = []

            # Iterate over batches
            for X_batch, y_batch in batches:
                a1, a2 = self.forward_prop(X_batch)

                # Get loss
                loss = self.cross_entropy(a2, y_batch)
                loss_batch.append(loss)

                # Backpropagate
                dw1, db1, dw2, db2 = self.backward_prop(a1, a2, X_batch, y_batch)

                # Optimize
                self.optimize(l_rate, dw1, db1, dw2, db2)

            # return

            for X_batch_test, y_batch_test in test_batches:
                _, a2_test = self.forward_prop(X_batch_test)

                # # Get loss
                test_loss = self.cross_entropy(a2_test, y_batch_test)
                test_loss_batch.append(test_loss)

            # Measure mean losses
            mean_train_loss = np.mean(loss_batch).round(6)
            mean_test_loss = np.mean(test_loss_batch).round(6)

            # Measure performance of model
            # Measure accuracies
            train_acc = self.get_accuracy(X_train, y_train, batch_size)
            test_acc = self.get_accuracy(X_test, y_test, batch_size)

            # Add accuracies to log of accuracies
            self.train_log.append(train_acc)
            self.test_log.append(test_acc)

            print(
                f"Epoch {i+1}/{epochs} => Train Accuracy: {train_acc} - Train Loss: {mean_train_loss} - Test Accuracy: {test_acc} - Test Loss: {mean_test_loss}"
            )

    def get_batches(self, X, y, batch_size):
        n = X.shape[0]
        batches = []
        permutation = np.random.permutation(X.shape[0])
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i : i + batch_size, :]
            y_batch = y_shuffled[
                i : i + batch_size,
            ]

            batches.append((X_batch, y_batch))

        return batches

    def get_accuracy(self, X, y, batch_size):
        n = X.shape[0]
        y_pred = np.array([], dtype="int64")
        y_real = np.array([], dtype="int64")

        for i in range(0, n, batch_size):
            X_batch = X[i : i + batch_size, :]
            y_batch = y[
                i : i + batch_size,
            ]

            y_real = np.append(y_real, y_batch)
            y_pred = np.append(y_pred, self.predict(X_batch))

        accuracy = np.mean(y_real == y_pred)
        return accuracy.round(4)

    def show_accuracies(self):
        plt.plot(self.train_log, label="train accuracy")
        plt.plot(self.test_log, label="test accuracy")
        plt.legend(loc="best")
        plt.grid()
        plt.show()

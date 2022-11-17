import numpy as np
from data import X_train, y_train, X_test, y_test
from metrics import *


class NaiveBayes:

    def _init_(self, x, y):

        n, m = x.shape
        # unique classes in the data_set
        self.classes = np.unique(y)
        number_of_classes = len(self.classes)

        # stats of each feature
        self.Mean = np.zeros((number_of_classes, m), dtype=np.float64)
        self.Variance = np.zeros((number_of_classes, m), dtype=np.float64)
        self.Prior = np.zeros(number_of_classes, dtype=np.float64)

        for i, c in enumerate(self.classes):  # for each class
            xc = x[c == y]
            self.Mean[i, :] = xc.mean(axis=0)
            self.Variance[i, :] = xc.var(axis=0)
            self.Prior[i] = xc.shape[0] / float(n)

    def predict_a_sample (self, xi):

        posteriors = []
        for i, c in enumerate(self.classes):
            # get logs to avoid underflow
            pr = np.log(self.Prior[i])
            class_conditional = np.sum(np.log(self.probability_density(i, xi)))
            pst = pr + class_conditional
            posteriors.append(pst)
        return self.classes[np.argmax(posteriors)]

    def predict(self, x):

        y_predict = [self.predict_a_sample(xi) for xi in x]
        return np.array(y_predict)

    def probability_density(self, classind, xi):

        mean = self.Mean[classind]
        var = self.Variance[classind]
        numerator =np.exp(-((xi - mean)**2) / (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        p_x_given_y = numerator / denominator
        return p_x_given_y


if __name__ == '__main__':
    
    nb = NaiveBayes()
    nb._init_(X_train, y_train)
    pred = nb.predict(X_test.values)

    print("Naive Bayes accuracy :  ", accuracy(y_test, pred))
    print("Naive Bayes precision :  ", precision(y_test, pred))
    print("Naive Bayes recall :  ", recall(y_test, pred))
    print("Naive Bayes F1 :  ", f1(y_test, pred))

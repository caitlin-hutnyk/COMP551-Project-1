import numpy as np
import pandas as pd
from numpy import ma

from Importv2 import read_data
import LogRegression


def evaluate_acc(y, y_hat):
    error = 0
    if np.shape(y) != np.shape(y_hat):
        print("error: y != y_h")
        # raise SizeError('Size y != size yh')
    for i in range(np.shape(y)[0]):
        if y[i] != y_hat[i]:
            error += 1
    return error


def k_fold(X, k):
    # split into training and validation set by 1/k
    k_mask = np.random.rand(X.shape[0]) < (1 / k)
    train = X[~k_mask]
    validation = X[k_mask]
    return train, validation

def main():
    y_label, test_categorical, test_continuous, \
    train_validate_categorical, train_validate_continuous = read_data()
    X = np.concatenate((train_validate_categorical, train_validate_continuous), axis=1)
    print("in main")
    print(X.shape)
    print(train_validate_categorical.shape)
    print(train_validate_continuous.shape)

    X_train, X_validation = k_fold(X, 5)

    print(X_train.shape)
    print(X_validation.shape)

    # instantiate with learning rate and epsilon
    log_r = LogRegression.Log_Regression(0.01, 0.005)
    log_r.fit(X_train, y_label)
    y_h = log_r.predict()
    print(y_label)
    print(y_h)


if __name__ == "__main__":
    main()

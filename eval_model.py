import numpy as np
import pandas as pd
from numpy import ma, math

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
    # create a list of k pairs of train and validate sets
    split_list = []
    size = math.floor(X.shape[0] / k)
    # select blocks of size-sized instances to validate, and allocate
    # the rest to train
    for i in range(k - 2):
        v_data = X[i * size:(i + 1) * size]
        v_y, v_X = v_data[:, -1, np.newaxis], v_data[:, :-1]
        validate = (v_X, v_y)

        t_data = np.delete(X, np.s_[i * size: (i + 1) * size], 0)
        t_y, t_X = t_data[:, -1, np.newaxis], t_data[:, :-1]
        train = (t_X, t_y)

        assert (v_y.shape[1] + v_X.shape[1] == t_y.shape[1] + t_X.shape[1] ==
                v_data.shape[1] == t_data.shape[1])
        split_list.append((train, validate))

    # append the final block
    v_data = X[(k - 1) * size:]
    v_y, v_X = v_data[:, -1, np.newaxis], v_data[:, :-1]
    assert (v_y.shape[1] + v_X.shape[1] == v_data.shape[1])
    validate = (v_X, v_y)

    t_data = X[:(k - 1) * size]
    t_y, t_X = t_data[:, -1, np.newaxis], t_data[:, :-1]
    train = (t_X, t_y)

    split_list.append((train, validate))

    return split_list


def main():
    test_categorical, test_continuous, test_y, X = read_data()
    # X = np.concatenate((train_validate_categorical, train_validate_continuous), axis=1)
    print("in main --- x shape: {} ".format(X.shape))

    # instantiate with learning rate and epsilon
    log_r = LogRegression.Log_Regression(0.01, 0.005)

    # do k-fold split on the training data to get k folds of train and validate
    train_validate_list = k_fold(X, 5)

    # for each set, train the model and validate
    training_set_count = 1

    for train_validate_set in train_validate_list:
        train, validate = train_validate_set
        train_X, train_y = train
        validate_X, validate_y = validate

        # add first col of ones
        train_X = np.append(np.ones((train_X.shape[0], 1)), train_X, axis=1)
        validate_X = np.append(np.ones((validate_X.shape[0], 1)), validate_X, axis=1)

        print("train shape {} type {}" .format(train_X.shape, train_X.dtype))
        print("v shape {} type {}".format(validate_X.shape, validate_X.dtype))

        log_r.fit(train_X, train_y)
        y_h = log_r.predict(train_X)

        print("training set {} ".format(training_set_count))
        training_set_count = training_set_count + 1
        print("y label: {}" .format(train_y.dtype))
        print("y predicted: {}" .format(y_h.dtype))

        count = 0
        for i in range (y_h.shape[0] - 1):
            if y_h[i] == train_y[i]:
                count = count + 1
        print("Correctly assigned: {} out of {}".format(count, train_X.shape[0]))


if __name__ == "__main__":
    main()

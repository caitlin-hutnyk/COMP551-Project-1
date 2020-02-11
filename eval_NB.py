import numpy as np
from testImport import read_data
import NBv2 as nb
import constant


# returns performance rate
def evaluate_acc(y, y_hat):
    success = 0
    if np.shape(y) != np.shape(y_hat):
        print("error: y != y_h")
        # raise SizeError('Size y != size yh')
    for i in range(np.shape(y)[0]):
        if y[i] == y_hat[i]:
            success += 1
    return success / (np.shape(y)[0])


def main():
    dataset = constant.CENSUS
    X_con, X_cat, Y, test_con, test_cat, test_y = read_data(dataset, 0)
    normalise(X_con)

    train_validation_sets = k_fold(X_cat, X_con, Y, 5)

    for set in train_validation_sets:
        # extract training and validation sets
        train, validate = set
        t_cat, t_con, t_y = train
        v_cat, v_con, v_y = validate
        v_x = np.append(v_cat, v_con, axis=1)

        # calculate prior and likelihoods with training data
        priors = nb.computePrior(t_y, constant.CENSUS)
        w_cat = nb.computeLikelihoodBernoulli(t_cat, t_y)
        w_cont = nb.computeLikelihoodGaussian(t_con, t_y)
        w = np.append(w_cat, w_cont, axis=0)

        # calculate posterior probabilities of validation set and evaluate performance
        post = nb.posterior(priors, w, v_x)
        y_hat = nb.predict(post)
        rate = evaluate_acc(v_y, y_hat)
        print("success rate: " + str(rate))


def k_fold(X_cat, X_con, Y, k):
    # create a list of k pairs of train and validate sets
    split_list = []
    size = np.math.floor(X_cat.shape[0] / k)

    # select blocks of size-sized instances to validate, and allocate
    # the rest to train
    for i in range(k - 2):
        v_data_cat = X_cat[i * size:(i + 1) * size]
        v_data_con = X_con[i * size:(i + 1) * size]
        v_y = Y[i * size:(i + 1) * size]
        validate = (v_data_cat, v_data_con, v_y)

        t_data_cat = np.delete(X_cat, np.s_[i * size: (i + 1) * size], 0)
        t_data_con = np.delete(X_con, np.s_[i * size: (i + 1) * size], 0)
        t_y = np.delete(Y, np.s_[i * size: (i + 1) * size], 0)
        train = (t_data_cat, t_data_con, t_y)

        assert(v_data_cat.shape[0] == v_data_con.shape[0] == v_y.shape[0])
        split_list.append((train, validate))

    # append the final block
    v_data_cat = X_cat[(k - 1) * size:]
    v_data_con = X_con[(k - 1) * size:]
    v_y = Y[(k - 1) * size:]
    validate = (v_data_cat, v_data_con, v_y)

    t_data_cat = X_cat[:(k - 1) * size]
    t_data_con = X_con[:(k - 1) * size]
    t_y = Y[:(k - 1) * size:]
    train = (t_data_cat, t_data_con, t_y)
    split_list.append((train, validate))

    return split_list

def normalise(X_con):
    column_max_vals = np.max(X_con, axis=0)
    # for each continuous column and each instance,
    for col in range(X_con.shape[1]):
        for n in range(X_con.shape[0]):
            # divide by the max to normalise, and ensure we don't divide by zero
            if column_max_vals[col] > 0.001:
                X_con[n][col] /= column_max_vals[col]


if __name__ == "__main__":
    main()


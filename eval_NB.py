import numpy as np
from testImport import read_data
from test_models import k_fold_split
import NBv2 as nb
import constant


# returns performance rate
def evaluate_acc(y, y_hat):
    C = y.shape[1]
    N = y.shape[0]
    success = 0

    if np.shape(y) != np.shape(y_hat):
        print("error: y != y_h")
    for n in range(N):
        y_row = y[n, :]
        y_idx = np.nonzero(y_row)
        y_hat_row = y_hat[n, :]
        y_hat_idx = np.nonzero(y_hat_row)
        if (y_idx == y_hat_idx):
                success += 1
    return success / N


def main():
    dataset = constant.CREDIT
    X_con, X_cat, Y, test_con, test_cat, test_y = read_data(dataset, 0)

    train_validation_sets = k_fold_split(X_cat, X_con, Y, 5)

    for set in train_validation_sets:
        # extract training and validation sets
        train, validate = set
        t_cat, t_con, t_y = train
        v_cat, v_con, v_y = validate

        if t_y.shape[1] == 1:
            t_y, v_y = nb.convertY(t_y, v_y)

        # calculate prior, likelihoods and posterior probability depending
        # on what kind of data is in the dataset
        priors = nb.computePrior(t_y, dataset)

        if t_cat is not None and t_con is None:
            w_cat = nb.computeLikelihoodBernoulli(t_cat, t_y, priors)
            post = nb.posterior(priors, w_cat, None, v_cat, None)

        elif t_con is not None and t_cat is None:
            w_gauss = nb.computeGaussian(t_con, t_y)
            post = nb.posterior(priors, None, w_gauss, None, v_con)

        else:
            w_cat = nb.computeLikelihoodBernoulli(t_cat, t_y, priors)
            w_gauss = nb.computeGaussian(t_con, t_y)
            post = nb.posterior(priors, w_cat, w_gauss, v_cat, v_con)

        y_hat = nb.predict(post)
        rate = evaluate_acc(v_y, y_hat)
        print("success rate: " + str(rate))

if __name__ == "__main__":
    main()


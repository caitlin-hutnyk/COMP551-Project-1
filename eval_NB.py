import numpy as np
from testImport import read_data
from eval_model import k_fold_split
import NBv2 as nb
import constant


# returns performance rate
def evaluate_acc(y, y_hat):
    success = 0
    if np.shape(y) != np.shape(y_hat):
        print("error: y != y_h")
        # raise SizeError('Size y != size yh')
    for i in range(y.shape[0]):
        if y[i] == y_hat[i]:
            success += 1
    return success / (y.shape[0])


def main():
    dataset = constant.CENSUS
    X_con, X_cat, Y, test_con, test_cat, test_y = read_data(dataset, 0)
    # normalise(X_con)

    train_validation_sets = k_fold_split(X_cat, X_con, Y, 5)

    for set in train_validation_sets:
        # extract training and validation sets
        train, validate = set
        t_cat, t_con, t_y = train
        v_cat, v_con, v_y = validate

        # calculate prior and likelihoods with training data
        priors = nb.computePrior(t_y, dataset)

        if t_cat is not None and t_con is None:
            w_cat = nb.computeLikelihoodBernoulli(t_cat, t_y)
            post = nb.posterior(priors, w_cat, None, v_cat, None)

        elif t_con is not None and t_cat is None:
            w_gauss = nb.computeGaussian(t_con, t_y)
            post = nb.posterior(priors, None, w_gauss, None, v_con)

        else:
            w_cat = nb.computeLikelihoodBernoulli(t_cat, t_y)
            w_gauss = nb.computeGaussian(t_con, t_y)
            # w = np.append(w_cat, w_cont, axis=0)
            post = nb.posterior(priors, w_cat, w_gauss, v_cat, v_con)

        # calculate posterior probabilities of validation set and evaluate performance

        y_hat = nb.predict(post)
        rate = evaluate_acc(v_y, y_hat)
        print("success rate: " + str(rate))

# def normalise(X_con):
#     column_max_vals = np.max(X_con, axis=0)
#     # for each continuous column and each instance,
#     for col in range(X_con.shape[1]):
#         for n in range(X_con.shape[0]):
#             # divide by the max to normalise, and ensure we don't divide by zero
#             if column_max_vals[col] > 0.00001:
#                 X_con[n][col] /= column_max_vals[col]

if __name__ == "__main__":
    main()


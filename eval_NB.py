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
    for i in range(np.shape(y)[0]):
        if y[i] == y_hat[i]:
            success += 1
    return success / (np.shape(y)[0])


def main():
    dataset = constant.CREDIT
    X_con, X_cat, Y, test_con, test_cat, test_y = read_data(dataset, 0)
    normalise(X_con)

    train_validation_sets = k_fold_split(X_cat, X_con, Y, 5)

    for set in train_validation_sets:
        # extract training and validation sets
        train, validate = set
        t_cat, t_con, t_y = train
        v_cat, v_con, v_y = validate

        if v_cat is None and v_con is not None:
            v_x = v_con
        elif v_con is None and v_cat is None:
            v_x = v_cat
        else:
            v_x = np.append(v_cat, v_con, axis=1)

        # calculate prior and likelihoods with training data
        priors = nb.computePrior(t_y, dataset)
        w_cat, w_cont = None, None
        if t_cat is not None and t_con is None:
            w_cat = nb.computeLikelihoodBernoulli(t_cat, t_y)
            w = w_cat
        elif t_con is not None and t_cat is None:
            w_cont = nb.computeLikelihoodGaussian(t_con, t_y)
            w = w_cont
        else:
            w_cat = nb.computeLikelihoodBernoulli(t_cat, t_y)
            w_cont = nb.computeLikelihoodGaussian(t_con, t_y)
            w = np.append(w_cat, w_cont, axis=0)

        # calculate posterior probabilities of validation set and evaluate performance
        post = nb.posterior(priors, w, v_x)
        y_hat = nb.predict(post)
        rate = evaluate_acc(v_y, y_hat)
        print("success rate: " + str(rate))

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


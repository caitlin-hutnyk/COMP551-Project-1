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
    return success/(np.shape(y)[0])

def main():
    X_con, X_cat, Y, test_con, test_cat, test_y = read_data(2,0)
    # print("Y, x_con, x_cat")
    # print(Y.shape)
    # print(X_con.shape)
    # print(X_cat.shape)
    X = np.append(X_cat, X_con, axis=1)

    prior = nb.computePrior(Y, constant.CENSUS)
    w_cat = nb.computeLikelihoodBernoulli(X_cat, Y)
    w_cont = nb.computeLikelihoodGaussian(X_con, Y)
    w = np.append(w_cat, w_cont, axis=0)

    posterior = np.ones((X.shape[0], 2))
    for i in range(2):
        for n in range(X.shape[0]):
            Xn = X[n, :, np.newaxis]
            w_i = w[:, i, np.newaxis]
            prior_i = prior[i]
            posterior[n][1] = nb.posterior(prior_i, w_i, Xn)

    '''
    for i in range (100):
        print("posterior of x given 0: {}, of x given 1: {}".format(posterior[i][0], posterior[i][1]))
    '''
    y_hat = nb.predict(posterior)
    rate = evaluate_acc(y, y_hat)
    print("success rate: " + str(rate))





if __name__ == "__main__":
    main()

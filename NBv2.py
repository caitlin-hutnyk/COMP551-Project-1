import numpy as np
import constant


def computePrior(Y, dataset):
    # Y labels, of dimension Nx1
    # dataset 1: ionosphere - bernoulli
    # dataset 2: census - bernoulli
    # dataset 3: poker - multiclass, 10 classes
    # dataset 4: congressional - bernoulli

    # calculate a bernoulli prior
    if dataset != constant.POKER:
        c1 = 0
        for y in Y:
            if y == 1:
                c1 = c1 + 1
        c0 = Y.shape[0] - c1
        c0, c1 = c0/Y.shape[0], c1/Y.shape[0]
        print("in computePrior for dataset {}".format(dataset))
        print("instances: {}, class 0 prior: {}, class 1 prior: {}" .format(Y.shape[0], c0, c1))
        return c0, c1

def computeLikelihoodBernoulli(X, Y):
    print("in computeLikelihoodBernoulli")
    # X is a NxD design matrix
    # Y is a Nx1 label vector
    N, D = X.shape
    w = np.zeros((D, 2))
    total_y0, total_y1 = 0, 0

    # compute denominator
    for y in Y:
        if y == 1:
            total_y1 = total_y1 + 1
        else:
            total_y0 = total_y0 + 1
    print("y = 0:{}, y = 1:{} of total {}".format(total_y0, total_y1, N))

    for d in range(D):
        # set our counters to 0
        y0, y1 = 0, 0

        # for each instance n...
        for n in range(N):
            # identify the feature value as 1 or 0 to count
            if X[n][d] == 1:
                # and increment label count accordingly
                if Y[n] == 1:
                    y1 = y1 + 1
                else:
                    y0 = y0 + 1

        # loop over all instances N is complete, can compute likelihoods of
        # c = 0 and c = 1 for feature d
        w[d][0] = y0/total_y0
        w[d][1] = y1/total_y1

    # normalise
    for d in range(D):
        maxval = max(w[d][0], w[d][1])
        if maxval > 0.001:
            w[d][0] = w[d][0]/maxval
            w[d][1] = w[d][1]/maxval

    for i in range(5):
        print("The likelihoods of feature {} having class 0: {} class 1: {}".format(i, w[i][0], w[i][1]))

    return w

def computeLikelihoodGaussian(X, Y):
    print("in computeLikelihoodGaussian")
    N, D = X.shape
    mean, stdev = np.zeros((D, 2)), np.zeros((D, 2))
    w = np.zeros((D, 2))

    for i in range(2):
        # get all indexes where c = 0, or c = 1
        if i ==  0:
            c_index = np.where(Y==0)[0]
        else:
            c_index = np.nonzero(Y)[0]

        # calculate the mean and stdev of per feature for all instances, depending on
        # their class placement
        mean[:, i] = np.mean(X[c_index, :], 0)
        stdev[:, i] = np.std(X[c_index, :], 0)

    print("the mean and stdev of for each class-feature pair: ")
    # so we don't go out of bounds...
    m = X.shape[0]
    if (m > 6):
        m = 6
    for i in range(m):
        print("class 0, feature {}, mean: {}, stdev: {}".format(i, mean[i][0], stdev[i][0]))
        print("class 1, feature {}, mean: {}, stdev: {}".format(i, mean[i][1], stdev[i][1]))

    # calculate log likelihood, with var = 1
    for i in range(2):
        for d in range(D):
            likelihood = 0
            for n in range (N):
                likelihood = np.log(1/np.sqrt(np.pi*2)) + (-0.5*(X[n][d] - mean[d][i])**2)
            w[d][i] = -1 * likelihood

    # normalise
    for d in range(D):
        m = max(w[d][0], w[d][1])
        if m > 0.001:
            w[d][0] = w[d][0]/m
            w[d][1] = w[d][1]/m
        print("The gaussian likelihoods of feature {} having class 0: {} class 1: {}".format(d, w[d][0], w[d][1]))

    return w

def posterior(prior, w, x):
    D = x.shape[1]
    N = x.shape[0]
    s = 0
    for d in range(D):
        s = s + w[d] * x[d]
    return s

def predict(posterior):
    y_hat = np.ones((posterior.shape[0], 1))
    for i in range(posterior.shape[0]):
        if posterior[i][0] > posterior[i][1]:
            y_hat[i] = 0
        else:
            y_hat[i] = 1
    return y_hat

def assess(Y, y_hat):
    correct = 0;
    for i in range(Y.shape[0]):
        if Y[i] == y_hat[i]:
            correct += 1
    return correct

class NaiveBayes:
    pass




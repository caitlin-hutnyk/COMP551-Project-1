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
                c1 += 1
        c0 = Y.shape[0] - c1
        c0, c1 = c0/Y.shape[0], c1/Y.shape[0]
        # print("in computePrior for dataset {}".format(dataset))
        # print("instances: {}, class 0 prior: {}, class 1 prior: {}" .format(Y.shape[0], c0, c1))
        return c0, c1

def computeLikelihoodBernoulli(X, Y):
    # X is a NxD design matrix
    # Y is a Nx1 label vector
    N, D = X.shape
    w = np.zeros((D, 2))
    total_y0, total_y1 = 0, 0

    # compute denominator
    for y in Y:
        if y == 1:
            total_y1 += 1
    total_y0 = N - total_y1

    for i in range(2):
        # get all indexes where c = 0, or c = 1
        if i == 0:
            c_index = np.where(Y == 0)[0]
            total = total_y0
        else:
            c_index = np.nonzero(Y)[0]
            total = total_y1

        for d in range(D):
            # instances satisfying the condition xd = 1
            b_condition = 0
            # for each instance n...
            for c in c_index:
                if X[c][d] == 1:
                    b_condition += 1
            w[d][i] = b_condition / total
            # print("feature d: {}, satis b condition: {} of total: {}".format(d, b_condition, total))
            # print("bernoulli likelihood for feature {} is {} ".format(d, w[d][i]))

    # normalise
    # for d in range(D):
    #     maxval = max(w[d][0], w[d][1])
    #     if maxval > 0.001:
    #         w[d][0] = w[d][0]/maxval
    #         w[d][1] = w[d][1]/maxval

    # for i in range(5):
    #     print("The likelihoods of feature {} having class 0: {} class 1: {}".format(i, w[i][0], w[i][1]))

    return w

def computeGaussian(X, Y):
    N, D = X.shape
    mean, stdev = np.zeros((D, 2)), np.zeros((D, 2))
    w = np.zeros((D, 2))

    for i in range(2):
        # get all indexes where c = 0, or c = 1
        if i == 0:
            c_index = np.where(Y == 0)[0]
        else:
            c_index = np.nonzero(Y)[0]

        # calculate the mean and stdev of per feature for all instances, depending on
        # their class placement
        mean[:, i] = np.mean(X[c_index, :], 0)
        stdev[:, i] = np.std(X[c_index, :], 0)

    return mean, stdev

    # print("the mean and stdev of for each class-feature pair: ")
    # # so we don't go out of bounds when printing...
    # m = X.shape[0]
    # if (m > 6):
    #     m = 6
    # for i in range(m):
    #   print("class 0, feature {}, mean: {}, stdev: {}".format(i, mean[i][0], stdev[i][0]))
    #    print("class 1, feature {}, mean: {}, stdev: {}".format(i, mean[i][1], stdev[i][1]))

    # calculate log likelihood, with var = 1


    # normalise
    # for d in range(D):
    #     m = max(w[d][0], w[d][1])
    #     if m > 0.001:
    #         w[d][0] = w[d][0]/m
    #         w[d][1] = w[d][1]/m
    return w

def posterior(priors, w, w_gauss, x_cat, x_con):
    post_cat, post_con = None, None

    # calculate bernoulli posterior probabilities
    if w is not None:
        N = x_cat.shape[0]
        D = x_cat.shape[1]
        post_cat = np.zeros((N, 2))

        for i in range(len(priors)):
            for idx, row in enumerate(x_cat):
                for d in range(D):
                    post_cat[idx][i] += w[d][i] * row[d]
                post_cat[idx][i] *= priors[i]
            max_post_val = np.max(post_cat[:, i])
            post_cat[:, i] /= max_post_val

    # calculate gaussian posterior probabilities
    if w_gauss is not None:
        mu, stdv = w_gauss
        N = x_con.shape[0]
        D = x_con.shape[1]
        post_con = np.zeros((N, 2))

        for i in range(len(priors)):
            for idx, row in enumerate(x_con):
                likelihood = 0
                for d in range(D):
                    likelihood += 0.5 * ((row[d] - mu[d][i]) ** 2)
                print(likelihood)
                post_con[idx][i] = np.exp(np.log(priors[i]) + (-1 * likelihood))

    # return valid posterior probabilities
    if post_cat is not None:
        if post_con is not None:
            post = np.append(post_cat, post_con, axis=1)
        else:
            post = post_cat
    else:
        post = post_con

    return post

def predict(posterior):
    # posterior is a N x 2 matrix of posterior probability of each class-feature pair
    y_hat = np.ones((posterior.shape[0], 1))
    for i in range(posterior.shape[0]):
        if posterior[i][0] > posterior[i][1]:
            y_hat[i] = 0
    return y_hat

class NaiveBayes:
    pass




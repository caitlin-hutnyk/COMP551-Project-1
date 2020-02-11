import numpy as np
import constant

# returns priors Cx1
def computePrior(Y, dataset):
    # Y labels, of dimension Nx1
    # dataset 1: ionosphere - bernoulli
    # dataset 2: census - bernoulli
    # dataset 3: poker - multiclass, 10 classes
    # dataset 4: congressional - bernoulli
    N = Y.shape[0]
    C = Y.shape[1]

    priors = np.zeros(C)

    for c in range(C):
        count = 0
        for n in range(N):
            if Y[n][c] == 1:
                count += 1
        priors[c] = count / N

    return priors


def computeLikelihoodBernoulli(X, Y, priors):
    # X is a NxD design matrix
    # Y is a Nx2 label vector for datasets 1, 2 and 4
    # Y is a NxC matrix for datasets 3

    N, D = X.shape
    C = Y.shape[1]
    w = np.zeros((D, C))
    '''
    bottom = priors[c] * N
    top = 
    '''

    for i in range(C):
        # get all indexes where c = 0, or c = 1
        c_index = np.nonzero(Y.T[i])[0] 

        for d in range(D):
            # instances satisfying the condition xd = 1
            b_condition = 0
            # for each instance n...
            for c in c_index:
                if X[c][d] == 1:
                    b_condition += 1
            w[d][i] = b_condition / (priors[i] * N)

    return w


def computeGaussian(X, Y):
    N, D = X.shape
    C = Y.shape[1]
    mean, stdev = np.zeros((D, C)), np.zeros((D, C))
    w = np.zeros((D, 2))

    for i in range(C):
        # get all indexes where c = 0, or c = 1
        c_index = np.nonzero(Y.T[i])[0]

        mean[:, i] = np.mean(X[c_index, :], 0)
        stdev[:, i] = np.std(X[c_index, :], 0)

    return mean, stdev


def posterior(priors, w, w_gauss, x_cat, x_con):
    post_cat, post_con = None, None
    C = priors.shape[0]
    post = None

    # calculate bernoulli posterior probabilities
    if w is not None:
        N = x_cat.shape[0]
        D = x_cat.shape[1]
        post_cat = np.zeros((N, C))

        for i in range(C):
            for n in range(N):
                for d in range(D):
                    post_cat[n][i] += (w[d][i] * x_cat[n][d])
                post_cat[n][i] *= priors[i]
            max_post_val = np.max(post_cat[:, i])
            post_cat[:, i] /= max_post_val

    # calculate gaussian posterior probabilities
    if w_gauss is not None:
        mu, stdv = w_gauss
        N = x_con.shape[0]
        D = x_con.shape[1]
        post_con = np.zeros((N, C))

        for i in range(C):
            for n in range(N):
                likelihood = 0
                for d in range(D):
                    likelihood += 0.5 * ((x_con[n][d] - mu[d][i]) ** 2)
                post_con[n][i] = np.exp(np.log1p(priors[i]) + (-1 * likelihood))
            max_post_val = np.max(post_con[:, i])
            post_con[:, i] /= max_post_val

    # return valid posterior probabilities
    if post_cat is not None:
        N = post_cat.shape[0]
        post = np.zeros((N, C))
        if post_con is not None:
            for c in range(C):
                for n in range(N):
                    post[n][c] = post_cat[n][c] * post_con[n][c]
        else:
            post = post_cat
    else:
        post = post_con

    return post


def predict(posterior):
    # posterior is a N x 2 matrix of posterior probability of each class-feature pair
    # and N x C for poker hands
    N = posterior.shape[0]
    C = posterior.shape[1]
    y_hat = np.zeros((N, C))

    for n in range(N):
        class_max = -np.inf
        class_index = -1
        for c in range(C):
            if posterior[n][c] > class_max:
                class_max, class_index = posterior[n][c], c

        y_hat[n][class_index] = 1

    return y_hat


def convertY(t_y, v_y):
    N_ty = t_y.shape[0]
    N_vy = v_y.shape[0]
    t_y = np.append(np.zeros((N_ty, 1)), t_y, axis=1)
    v_y = np.append(np.zeros((N_vy, 1)), v_y, axis=1)

    for n in range(N_ty):
        if t_y[n][1] == 0:
            t_y[n][0] = 1

    for n in range(N_vy):
        if v_y[n][1] == 0:
            v_y[n][0] = 1

    return t_y, v_y
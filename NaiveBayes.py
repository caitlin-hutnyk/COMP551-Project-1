import numpy as np

# require X to be split up into X_cat and X_con
class Naive_Bayes:

	def __init__(self):
		self.priors = -1
		self.posterior = -1
		self.w = -1
		self.mean = -1
		self.stdev = -1
		self.datatype = -1


	def fit(self, x_con, x_cat, y):
		if y.shape[1] == 1:
			y = self.convertY(y)
		self.datatype = self.datatype(x_cat, x_con)

		self.priors = self.computePrior(y)
		if self.datatype is "categorical" or "mixed":
			self.w = self.computeLikelihoodBernoulli(x_cat, y)

		if self.datatype is "gaussian" or "mixed":
			self.mean, self.stdev = self.computeGaussian(x_con, y)

	def predict(self, x_con, x_cat):
		self.calculatePosterior(self, x_con, x_cat)
		y_hat = self.predict(self.posterior)
		return y_hat


	# returns priors Cx1
	def computePrior(self, Y):
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

	def computeGaussian(self, X, Y):
		N, D = X.shape
		C = Y.shape[1]
		mean, stdev = np.zeros((D, C)), np.zeros((D, C))

		for i in range(C):
			# get all indexes where c = 0, or c = 1
			c_index = np.nonzero(Y.T[i])[0]
			mean[:, i] = np.mean(X[c_index, :], 0)
			stdev[:, i] = np.std(X[c_index, :], 0)
		return mean, stdev

	def convertY(self, y):
		N = y.shape[0]
		y = np.append(np.zeros((N, 1)), y, axis=1)
		for n in range(N):
			if y[n][1] == 0:
				y[n][0] = 1
		return y

	def datatype(self, x_cat, x_con):
		if x_cat is not None and x_con is None:
			return "categorical"
		elif x_con is not None and x_cat is None:
			return "gaussian"
		else:
			return "mixed"

	def computeLikelihoodBernoulli(self, X, Y):
		# X is a NxD design matrix
		# Y is a Nx2 label vector for datasets 1, 2 and 4
		# Y is a NxC matrix for datasets 3

		N, D = X.shape
		C = Y.shape[1]
		w = np.zeros((D, C))

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
				w[d][i] = b_condition / (self.priors[i] * N)

		return w

	def calculatePosterior(self, w, mean, x_cat, x_con):
		post_cat, post_con = None, None
		C = self.priors.shape[0]

		# calculate bernoulli posterior probabilities
		if self.datatype is "categorical" or "mixed":
			N = x_cat.shape[0]
			D = x_cat.shape[1]
			post_cat = np.zeros((N, C))

			for i in range(C):
				for n in range(N):
					for d in range(D):
						post_cat[n][i] += (w[d][i] * x_cat[n][d])
					post_cat[n][i] *= self.priors[i]
				max_post_val = np.max(post_cat[:, i])
				post_cat[:, i] /= max_post_val

		# calculate gaussian posterior probabilities
		if self.datatype is "gaussian" or "mixed":
			N = x_con.shape[0]
			D = x_con.shape[1]
			post_con = np.zeros((N, C))

			for i in range(C):
				for n in range(N):
					likelihood = 0
					for d in range(D):
						likelihood += 0.5 * ((x_con[n][d] - mean[d][i]) ** 2)
					post_con[n][i] = np.exp(np.log1p(self.priors[i]) + (-1 * likelihood))
				max_post_val = np.max(post_con[:, i])
				post_con[:, i] /= max_post_val

		# return valid posterior probabilities
		if self.datatype is "categorical":
			self.posterior = post_cat

		elif self.datatype is "gaussian":
			self.posterior = post_con

		else:
			N = post_cat.shape[0]
			post = np.zeros((N, C))
			for c in range(C):
				for n in range(N):
					post[n][c] = post_cat[n][c] * post_con[n][c]
			self.posterior = post


	def predict(self, posterior):
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
import numpy as np

# require X to be split up into X_cat and X_con
class Naive_Bayes:

	self.u = -1
	self.v = -1
	self.mu = -1
	self.s = -1

	def __init__(self):
		pass

	def cost(u, v, X, y):
		pass

	def fit(X_cat, X_con, y):
		n,d1 = X_cat.shape
		d2 = X_con.shape[1]
		c = y.shape[1]

		# prior
		u = np.log(np.mean(y,0))[:,None]

		# likelihoods
		v = np.zeros((c,d1)) 						# for multiclass
		mu, s = np.zeros((c,d2)), np.zeros((c,d2)) 	# for continuous

		# compute v
		for c_ in range(c):
			inds = np.nonzero(y[:,c])[0]
			

		# compute mu,s
		for c_ in range(c):
			inds = np.nonzero(y[:,c])[0]
			mu[c,:] = np.mean(X_con[inds,:], 0)
			s[c,:] = np.std(X_con[inds,:],0)

		self.u = u
		self.v = v
		self.mu = mu
		self.s = s

	def predict(X_cat, X_con):
		# for purely continuous
		return self.u - np.sum( np.log(s[:,None,:]) + 0.5*(((X[None,:,:] - mu[:,None,:])/s[:,None,:])**2, 2)
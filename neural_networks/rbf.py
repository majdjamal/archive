
"""rbf.py: Radial Basis Function with Competitive Learning."""

__author__ = "Majd Jamal"

import numpy as np 

class CL:
	"""
	Competitive Learning, used to find the optimal centers.
	"""
	def __init__(self):
		self.C = None #Centers 

	def getCenters(self):
		""" Extract the solution
		:return: centers
		"""
		return self.C

	def optimize(self, X, units,  epochs = 100, learning_rate = 0.01):
		""" Trains the model to find the optimal centers
		:param X: Data matrix
		:param units: number of centers to optimize
		:param epochs: number of iterations in the training
		:param learning_rate: step size
		"""
		self.C = np.random.choice(X, units, replace=False)

		for i in range(epochs):
			for point in X:

				differences = np.abs(self.C - point)
				minimal = np.argmin(differences)
				self.C[minimal] += learning_rate * (point - self.C[minimal])

class RBF(CL):

	def __init__(self, hidden_units = 10, variance = 0.5):
		self.W = None #Weights
		self.centers = None 
		self.hidden_units = hidden_units
		self.variance = variance

	def gaussian(self, X, mu):
		""" Gaussian node
		:param X: Data points
		:param mu: Mean values, which is also refered to as centers.
		:return: Gaussian function values
		"""
		func = np.subtract(X, mu)
		func = - np.square(func)
		func /= (2*self.variance)
		return np.exp(func)

	def getWeights(self):
		""" Extract weights
		:return: Weights that are used for the approximation
		"""
		return self.W

	def O(self, X):
		""" Computes the PHI matrix
		:param X: Data points
		:return PHI: PHI matrix 
		"""
		Npts = X.shape[0]
		PHI = np.zeros((Npts, self.hidden_units))

		for i in range(len(X)):
			for j in range(len(self.centers)):

				PHI[i][j] = self.gaussian(X[i], self.centers[j])

		return PHI

	def fit(self, X, y):
		""" Trains the model
		:param X: Data points
		:param y: Targets
		"""
		self.optimize(X, self.hidden_units)
		self.centers = self.getCenters()

		PHI = self.O(X)

		self.W, _, _, _ = np.linalg.lstsq(PHI, y)

	def predict(self, X):
		""" Makes predictions
		:param X: Data points 
		:return: pred: Predicted function values
		"""
		PHI = self.O(X)

		return PHI @ self.W

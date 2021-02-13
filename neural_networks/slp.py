
"""slp.py: Single Layer Perceptron with Delta Rule"""

__author__ = "Majd Jamal"

import numpy as np

class SLP:
	"""
	Single-Layer Perceptron
	"""

	def __init__(self):
		self.W = None #Weights

	def activation(self, pred):
		""" A node that determines the output of the network.
		:param pred: numerical predictions 
		:return: classifications
		"""
		return np.where(pred > 0, 1, -1)

	def getWeights(self):
		""" Extract weights
		:return: Weights that are used to transform data
		"""
		return self.W

	def fit(self, X, y, learning_rate = 0.01, epochs = 20):
		""" Trains the model
		:param X: Patterns. Data points are represented by columns and attributres by rows.
		:param y: Targets, also known as labels. 
		:param learning_rate: Learning rate, wich resembles a step size in the delta rule.
		:param epochs: Number of iterations
		"""
		Ndim, Npts = X.shape
		self.W = np.random.normal(0, 0.5, Ndim)	#Initialize weights from a normal distribution and zero mean.

		for epoch in range(epochs):

			delta = self.W @ X - y
			delta *= learning_rate
			delta = delta @ X.T
			self.W += delta

	def predict(self, X):
		""" Makes predictions
		:param X: Patterns. Data points are represented by columns and attributres by rows.
		:return: pred: classificaiton of data points.
		"""
		pred = self.W @ X
		pred = self.activation(pred)
		return pred


"""mlp.py: Multilayer Perceptron"""

__author__ = "Majd Jamal"

import matplotlib.pyplot as plt
import numpy as np

class MLP:
	"""
	Multilayer Perceptron
	"""
	def __init__(self, hidden_units, learning_rate = 0.01):

		self.hidden_units = hidden_units
		self.learning_rate = learning_rate

		self.V = None	#Weights V
		self.W = None	#Weights W

		self.Npts = 0

		self.patterns = None
		self.targets = None

		self.h = None	#Outputs from hidden layer
		self.o = None	#Outputs from the network

		self.delta_o = None	#Derivative computations for output
		self.delta_h = None	#Derivative computations for hidden layer


	def sigmoid(self, X, prime = False):
		"""Computes the Sigmoid function,
		   and it's derivative.
		:param X: Data matrix
		:param prime: Boolean, false return regular sigmoid,
		              true returns the derivative.
		:return: sigmoid function values
		"""
		if prime:
			return np.multiply( (1 + X), (1 - X)) / 2

		return 2/(1 + np.exp(-X)) - 1

	def forward(self):
		""" Forward pass,
			this function takes an input
			and pass it through the network
			to produce an output.
		"""
		self.h = self.sigmoid(self.W @ self.patterns)

		ones = np.ones(self.Npts)
		self.h = np.vstack((self.h, ones))

		self.o = self.sigmoid(self.V @ self.h)

	def backward(self):
		""" Backward propagation.
			This function evaluates
			the derivatives that are
			used to update the weights.
		"""
		activation_o = self.sigmoid(self.o)

		self.delta_o = np.multiply(
			np.subtract(self.o, self.targets)
			, self.sigmoid(self.o, prime=True)
			)

		self.delta_h = np.multiply(
			(self.V.T @ self.delta_o)
			, self.sigmoid(self.h, prime=True))


		self.delta_h = self.delta_h[:-1]


	def update(self):
		""" Updates weights.
		"""
		self.W -= self.learning_rate * self.delta_h @ self.patterns.T
		self.V -= self.learning_rate * self.delta_o @ self.h.T


	def fit(self, X, y, epochs=1000):
		""" Trains the model.
		:param X: Patterns. Shape = (Ndim, Npts)
		:param y: y. Sigals Shape = (Npts,)
		:param epochs: Number of training iterations
		"""
		visible_units, self.Npts = X.shape

		self.W = np.random.normal(size = (self.hidden_units, visible_units))
		self.V = np.random.normal(size = (1, self.hidden_units + 1))

		self.patterns = X
		self.targets = y.reshape(1, -1)

		for epoch in range(epochs):

			self.forward()
			self.backward()
			self.update()

	def predict(self, X):
		""" Classifies points
		:param X: Patterns. Shape = (Ndim, Npts)
		:return: Classification-s
		"""
		Ndim, Npts = X.shape

		hidden = self.sigmoid(self.W @ X)

		ones = np.ones(Npts)
		hidden = np.vstack((hidden, ones))

		output = self.sigmoid(self.V @ hidden)

		return np.where(output > 0, 1, -1)

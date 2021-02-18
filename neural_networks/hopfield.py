
"""hopfield.py: Hopfield Network with two prediction methods, namely Little Model and Sequential Update."""

__author__ = "Majd Jamal"

import numpy as np
import matplotlib.pyplot as plt

class Hopfield:
	"""
	Hopfield Network with Little Model and Sequential Update.
	"""

	def __init__(self):
		self.W = None	#Weights, which is the memory.

	def sign(self, X):
		""" Activation node
		:param X: Predictions
		:return: Output: The sign of each prediction.
		"""
		return np.where(X >= 0, 1, -1)

	def fit(self, X):		
		""" Train the memory
		:param X: Patterns. Data points as numpy arrays.
		"""
		self.W = X.T @ X
	
	def predict(self, X):
		""" Predict with Batch mode
		:param X: data point.
		:return: Re-created data point.
		"""
		return self.sign(self.W@X.T).T
    
	def sequential_predict(self, X, steps):
		""" Predict with Sequential Update
		:param X: data point.
		:param steps: number of iterations
		:return updatedX: Re-created data point.
		"""
		updatedX = X
		Ndim = len(X)

		for step in range(steps):
			randomInt = np.random.randint(0, Ndim)
			var = self.W[randomInt] @ updatedX
			var = self.sign(var)
			updatedX[randomInt] = var

		return updatedX



"""knn.py: k-Nearest Neighbor Classifier"""

__author__ = "Majd Jamal"

import numpy as np
from scipy.spatial import distance_matrix

class kNN:
	"""
	k-Nearest Neighbor
	"""
	def __init__(self, k_neigh = 10):

		self.k_neigh = k_neigh
		self.patterns = None
		self.targets = None

	def score(self, true, pred):
		""" Computes the ratio of misclassifications.
		:param true: true value-s
		:param pred: predicted value-s
		:return score: error rate
		"""
		N = pred.size
		score = 1 - np.sum(np.where(true == pred, 0, 1)) / N
		return score

	def fit(self, X, y):
		""" Adjusts patterns and targets.
		:param X: Training data (Shape = (Npts, Ndim))
		:param y: Training labels
		"""
		self.patterns = X
		self.targets = y

	def predict(self, X):
		""" Predicts labels of data points.
		:param X: Data points (Shape = (Npts, Ndim))
		:return predictions: predicted labels
		"""
		N, D = X.shape

		predictions = np.zeros(N)

		D = distance_matrix(X, self.patterns)

		for i in range(N):

			neighbors = D[i].argsort()[:self.k_neigh]
			neighbor_labels = self.targets[neighbors]

			values, counts = np.unique(neighbor_labels, return_counts=True)

			ind = np.argmax(counts)
			label = values[ind]
			predictions[i] = label

		return predictions


"""rbm.py: Restricted Boltzmann Machine with Contrastive Divergence."""

__author__ = "Majd Jamal"

import numpy as np 

class RBM:
	"""
	Restricted Boltzmann Machine
	"""
	def __init__(self, ndim_visible, ndim_hidden):

		self.ndim_visible = ndim_visible	

		self.ndim_hidden = ndim_hidden

		self.W = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

		self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

		self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

		self.learning_rate = 0.1

	def sigmoid(self, X):
		""" Sigmoid function
		:param X: Data matrix
		:return: Function values
		"""
		return 1./(1.+np.exp(-X))

	def sample_binary(self, probabilities):
		""" Sigmoid function
		:param X: Data matrix
		:return: Function values
		"""
		activations = 1. * (probabilities >= np.random.random_sample(size=probabilities.shape) )
		return activations

	def v_given_h(self, X_hidden):
		"""	Computes probabilities p(v|h) and activations v ~ p(v|h)
        :param X_hidden: data points from the hidden layer.  
		:return prob: probabilities of hidden layer
		:return states: activities of hidden layer
        """

		prob = self.sigmoid(X_hidden @ self.W.T + self.bias_v)
		states = self.sample_binary(prob)
		return prob, states

	def h_given_v(self, X_visible):
		"""	Computes probabilities p(h|v) and activations h ~ p(h|v)
        :param X_visible: Visible data points. 
		:return prob: probabilities of hidden layer
		:return states: activities of hidden layer
        """

		prob = self.sigmoid(X_visible @ self.W + self.bias_h)
		states = self.sample_binary(prob)
		return prob, states

	def update(self, v_0, h_0, v_k, h_k):
		"""	Update the weight and bias parameters.
        :param v_0: activities or probabilities of visible layer
        :param h_0: activities or probabilities of hidden layer
        :param v_k: activities or probabilities of visible layer
        :param h_k: activities or probabilities of hidden layer
        """

		n_samples = v_0.shape[0]
		
		self.W += self.learning_rate * (v_0.T @ h_0 - v_k.T @ h_k) / n_samples
		self.bias_v += self.learning_rate * (v_0 - v_k).mean(axis=0)
		self.bias_h += self.learning_rate * (h_0 - h_k).mean(axis=0)

	def evaluate(self, X):
		"""	Evaluates the model with reconstruction loss 
			between true and predicted values.
		:param X: Data matrix
        :return: reconstruction loss
        """
		h_prob, h_states = self.h_given_v(X)
		v_prob, v_states = self.v_given_h(h_states)
		return np.linalg.norm(X - v_prob)


	def fit(self, X, epochs=100):
		"""	Train the Bolzmann Machine with Contrastive Divergence.
		:param X: Data matrix
        """
		Npts, Ndim = X.shape

		batches = np.split(X, Npts*(1/4))
    
		for i in range(epochs):

			for mini_batch in batches:
				h_prob, h_states = self.h_given_v(mini_batch)
				v_prob, v_states = self.v_given_h(h_states)
				self.update(mini_batch, h_states, v_prob, h_prob)

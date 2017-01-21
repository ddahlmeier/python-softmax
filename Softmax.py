"""Softmax multi-class logistic regression classifier in Python."""

# References

# MultiClass Logistic Classifier in Python
# https://www.codeproject.com/Articles/821347/MultiClass-Logistic-Classifier-in-Python
import numpy as np


class Softmax(object):
      """Softmax classfier layer"""

      def __init__(self):
      	  self.W = None
	  self.b = None

      def make_onehot(self, Y, n_dim):
            """Compute class indices to one-hot vectors
            
            Y is a vector of numeric class labels
            n_dim is a scalar 
            """
            y_onehot = np.zeros((Y.shape[0], n_dim))
            y_onehot[np.arange(Y.shape[0]), Y] = 1
            return y_onehot

      def fit(self, X, Y, iters=100, learn_rate=0.01, ref=0.1):
            """Train the model.
            X is the feature matrix (n_samples, n_features)
            Y is a vector of numeric zero-index class labels dimension (n_samples)
            W is the weight matrix of dimension (n_features, n_classes)
            b is the a bias vector of dimenison (n_classes)
            """

            # initialize weight matrix and bias vector
            self.W = np.random.normal(0, 0.1, (X.shape[1], np.max(Y)+1))
            self.b = np.zeros(np.max(Y)+1)

            # train model
            # TODO

      def softmax(self, Z):
            """Compute softmax activations

            Z is a matrix with dimension (n_batch, n_classes)
            """
            ex = np.exp(Z-np.max(Z, axis=1, keepdims=True))
            return ex / ex.sum(axis=1, keepdims=True)

      def predict(self, X):
            """return prediction of most likely class for each instance"""
            return np.argmax(self.predict_log_proba(X), axis=1)

      def predict_log_proba(self, X):
            """ return probability vector for each instance
            
            X is a feature matrix of dimenison (n_batch, n_features)
            returns matrix of dimension (n_batch, n_classes)
            """
            return self.softmax(np.dot(X, self.W) + self.b)

      def loss(self, X, Y, reg):
            """compute cross-entropy loss

            X is a feature matrix of dimension (n_batch, n_features)
            Y is a vector of the true class labels of dimension (n_batch)
            """
            # compute softmax activations
            Yh = self.predict_lob_proba(X)
            # loss is the negative log probability of correct class
            #loss = np.sum(-n[p.log(Yh[np.arange(X.shape[0]), Y]))
            # add regularization penalty
            #loss += reg * np.sum(np.square(self.W))
            #return loss / X.shape[0]

      def grad(self, X, Y, reg):
            """compute gradient of cost function with respect to inputs"""
            Yh = self.predict_lob_proba(X)

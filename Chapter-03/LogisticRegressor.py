
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class LogisticRegressor:
    def __init__(self, d, n, lr=0.001):

        # Place holders for input-output training data
        self.X = tf.placeholder(tf.float32, \
                                shape=[None, d], name='input')
        self.Y = tf.placeholder(tf.float32, \
                                name='output')
        # Variables for weight and bias
        self.b = tf.Variable(tf.zeros(n), dtype=tf.float32)
        self.W = tf.Variable(tf.random_normal([d, n]), \
                             dtype=tf.float32)

        # The Linear Regression Model
        h = tf.matmul(self.X, self.W) + self.b
        self.Ypred = tf.nn.sigmoid(h)

        # Loss function
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.Ypred), \
                                                  reduction_indices=1), name='cross-entropy-loss')

        # Gradient Descent with learning
        # rate of 0.05 to minimize loss
        optimizer = tf.train.GradientDescentOptimizer(lr)
        self.optimize = optimizer.minimize(self.loss)

        # Initializing Variables
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def fit(self, X, Y, epochs=500):
        total = []
        for i in range(epochs):
            _, l = self.sess.run([self.optimize, self.loss], \
                                 feed_dict={self.X: X, self.Y: Y})
            total.append(l)
            if i % 1000 == 0:
                print('Epoch {0}/{1}: Loss {2}'.format(i, epochs, l))
        return total

    def predict(self, X):
        return self.sess.run(self.Ypred, feed_dict={self.X: X})

    def get_weights(self):
        return self.sess.run([self.W, self.b])


import tensorflow as tf
from sklearn.utils import check_array
from sklearn.metrics import mean_absolute_error
import numpy as np
import math
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


class Fuzzyloopa():

    def __init__(self, n_inputs, n_rules, learning_rate = 1e-2):
        self.__numI = n_inputs
        self.__numR = n_rules
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.__numI))
        self.targets = tf.placeholder(tf.float32, shape=None)
        # numbers of ai varibles are numR * numI
        self.ai = tf.get_variable('ai', [self.__numR * self.__numI], initializer=tf.random_normal_initializer(0, 1))
        self.ci = tf.get_variable('ci', [self.__numR * self.__numI], initializer=tf.random_normal_initializer(0, 1))
        # this step is 2-3 layers
        self.rules = tf.reduce_prod(
            tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), self.ai) / (self.ci))),
                       (-1, n_rules, n_inputs)), axis=2)

        y = tf.get_variable('y', [1, n_rules], initializer=tf.random_normal_initializer(0,1))
        num = tf.reduce_sum(tf.multiply(self.rules, y), axis=1)
        den = tf.clip_by_value(tf.reduce_sum(self.rules, axis=1), 1e-12, 1e12)
        self.out = tf.divide(num, den)

        self.loss = tf.losses.huber_loss(self.targets, self.out)
        # self.loss = tf.losses.mean_squared_error(self.targets, self.out)
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.init_variables = tf.global_variables_initializer()

    def MAPE(self, out, targets):
        # y_true, y_pred = check_array(targets, out)
        return np.mean(np.abs((targets - out) / targets)) * 100


    def MAD(self, out, targets):
        return mean_absolute_error(targets, out)


    def SMAPE(self, out, targets):
        return 100 / len(targets) * np.sum(2 * np.abs(out - targets) / (np.abs(targets) + np.abs(out)))


    def make_prediction(self, sess, x, targets):
        return sess.run([self.out, self.loss], feed_dict={self.inputs: x, self.targets: targets})


    def plot_rules(self):
        a = tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, self.__numR)), self.ai) / (self.ci))),
                   (-1, self.__numR, self.__numI))
        return a


    def plot_rules(self, sess, num):
        ai_, ci_ = sess.run([self.ai, self.ci])
        ai_ = ai_[num::self.__numI + num]
        ci_ = ci_[num::self.__numI + num]
        x = [i for i in range(-50,50)]
        # for j in range(len(ai_)):
        res = [math.exp(-((x[i] - ci_[0])/ai_[0])**2) for i in range(100)]
        plt.plot(res)
        plt.show()
        # return ai_


    def train(self, sess, x , targets):
        out, l, _ = sess.run([self.out, self.loss, self.optimize], feed_dict={self.inputs: x, self.targets: targets})
        # print(graph)
        return l, out


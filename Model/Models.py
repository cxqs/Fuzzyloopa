import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd
import gviz_api


class CFuzzyloopa():

    def __init__(self, n_inputs, n_rules, n_output, learning_rate = 1e-2, ai=None, ci=None):
        self.__numI = n_inputs
        self.__numR = n_rules
        self.__numO = n_output
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.__numI))
        self.targets = tf.placeholder(tf.float32, shape=None)
        # numbers of ai varibles are numR * numI
        if (ai == None and ci == None):
            self.ai = tf.get_variable('ai', [self.__numR * self.__numI], initializer=tf.random_normal_initializer(0, 1))
            self.ci = tf.get_variable('ci', [self.__numR * self.__numI], initializer=tf.random_normal_initializer(0, 1))
        else:
            # tf.convert_to_tensor(ai, dtype=tf.float32)
            ai_init = tf.constant(ai)
            ci_init = tf.constant(ci)
            self.ai = tf.get_variable('ai', initializer=ai_init)
            self.ci = tf.get_variable('ci', initializer=ci_init)
        # this step is 2-3 layers
        self.rules = tf.reduce_prod(
            tf.reshape(
                tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, self.__numR)), self.ai) / (self.ci))),
                (-1, self.__numR, self.__numI)), axis=2)
        print(str(self.rules.shape[0]))
        self.y = tf.get_variable('y', [self.__numO, self.__numR], initializer=tf.random_normal_initializer(0, 1))
        self.y = tf.tile(self.y, [1, tf.shape(self.rules)[0]])
        self.helper = tf.reshape(tf.tile(tf.reshape(self.rules, [tf.shape(self.rules)[0] * tf.shape(self.rules)[1]]),
                                         [self.__numO]), [self.__numO, tf.shape(self.rules)[0] * tf.shape(self.rules)[1]])

        self.mul = tf.multiply(self.helper, self.y)
        self.back_to_matrix = tf.reduce_sum(tf.reshape(self.mul, [self.__numO, tf.shape(self.rules)[0], self.__numR]), axis=2)
        # self.num = tf.reduce_sum(tf.multiply(self.rules, self.y), axis=1)
        self.den = tf.reshape(tf.clip_by_value(tf.reduce_sum(self.rules, axis=1), 1e-12, 1e12), [1,tf.shape(self.rules)[0]])
        self.out = tf.reshape(self.back_to_matrix / self.den, [tf.shape(self.rules)[0], self.__numO])
        # self.out = tf.divide(self.back_to_matrix, den)
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

    def train(self, sess, x , targets):
        # y, y1 = sess.run([self.y, self.y1])
        rule, helper, mul, back, den = sess.run([self.rules, self.helper, self.mul, self.back_to_matrix, self.den], feed_dict={self.inputs: x, self.targets: targets})
        ai_, ci_,out, l, _ = sess.run([self.ai, self.ci, self.out, self.loss, self.optimize], feed_dict={self.inputs: x, self.targets: targets})
        # input, output = sess.run([self.inputs,self.targets], feed_dict={self.inputs: x, self.targets: targets})
        # print(graph)
        return l, out, ai_, ci_


class Fuzzyloopa():

    def __init__(self, n_inputs, n_rules, learning_rate = 1e-2, ai=None, ci=None):
        self.__numI = n_inputs
        self.__numR = n_rules
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.__numI))
        self.targets = tf.placeholder(tf.float32, shape=None)
        # numbers of ai varibles are numR * numI
        if (ai == None and ci == None):
            self.ai = tf.get_variable('ai', [self.__numR * self.__numI], initializer=tf.random_normal_initializer(0, 1))
            self.ci = tf.get_variable('ci', [self.__numR * self.__numI], initializer=tf.random_normal_initializer(0, 1))
        else:
            # tf.convert_to_tensor(ai, dtype=tf.float32)
            ai_init = tf.constant(ai)
            ci_init = tf.constant(ci)
            self.ai = tf.get_variable('ai', initializer=ai_init)
            self.ci = tf.get_variable('ci', initializer=ci_init)
        # this step is 2-3 layers
        self.rules = tf.reduce_prod(
            tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), self.ai) / (self.ci))),
                       (-1, n_rules, n_inputs)), axis=2)

        y = tf.get_variable('y', [1, n_rules], initializer=tf.random_normal_initializer(0,1))
        self.num = tf.reduce_sum(tf.multiply(self.rules, y), axis=1)
        den = tf.clip_by_value(tf.reduce_sum(self.rules, axis=1), 1e-12, 1e12)
        self.out = tf.divide(self.num, den)

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
        ai_, ci_,out, l, num, rule, _ = sess.run([self.ai, self.ci, self.out, self.loss, self.num, self.rules, self.optimize], feed_dict={self.inputs: x, self.targets: targets})
        # input, output = sess.run([self.inputs,self.targets], feed_dict={self.inputs: x, self.targets: targets})
        return l, out, ai_, ci_


# data=[[1.0,2.0,4.0,5.0],[0.0,6.0,7.0,8.0],[8.0,1.0,1.0,1.0]]
# multiply = tf.constant([2,1])
# data2 = tf.tile(data, multiply)
# data1=[[8.0,1.0,1.0,1.0], [8.0,1.0,1.0,1.0]]
# data3 = tf.reshape(data2, [-1, 3, 4])
# # X=tf.constant(data3)
# X = data3
# Y=tf.constant(data1)
# matResult=tf.matmul(X, X, transpose_b=True)
#
# multiplyResult=tf.reduce_sum(tf.multiply(X,X),axis=1)
# multi = tf.multiply(X,Y)
# with tf.Session() as sess:
#    print('matResult')
#    print(sess.run(data3))
#    print()
#    print(multi)
#    print(sess.run([multi]))
#    print()

data=[[[1.0, 2.0, 4.0, 5.0],[0.0, 6.0, 7.0, 8.0],[8.0, 1.0, 1.0, 1.0]]]
data1=[[1.0, 2.0, 4.0, 5.0]]
x = tf.constant(data)
y = tf.constant(data1)
# y = tf.tile(y, [1, x.shape[0]])
l = tf.divide(x,y)
# x = tf.reshape(tf.tile(tf.reshape(x, [x.shape[0]*x.shape[1]]), [2]),[2, x.shape[0]*x.shape[1]])
# p = tf.reshape(x, [2,3,4])
# w = tf.constant([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
# xw = tf.multiply(x, y)
# p = tf.reshape(xw, [2,3,4])
# max_in_rows = tf.reduce_max(xw, 1)

sess = tf.Session()
print(sess.run(x))
print(sess.run(l))
print("gogogo")
print(sess.run(y))
# ==> [[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
#      [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]]

# print(sess.run(max_in_rows))
# ==> [25.0, 25.0, 25.0, 25.0, 25.0]
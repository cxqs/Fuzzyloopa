import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import numpy as np
import math
import matplotlib.pyplot as plt
import json

class CFuzzyloopa():

    def __init__(self, n_inputs, n_rules, n_output, learning_rate = 1e-2, ai=None, ci=None, y=None):
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
        if y == None:
            self.y = tf.get_variable('y', [self.__numO, self.__numR], initializer=tf.random_normal_initializer(0, 1))
        else:
            y_init = tf.constant(y)
            self.y = tf.get_variable('y', initializer=y_init)
        self.y_bigger = tf.tile(self.y, [1, tf.shape(self.rules)[0]])
        self.helper = tf.reshape(tf.tile(tf.reshape(self.rules, [tf.shape(self.rules)[0] * tf.shape(self.rules)[1]]),
                                         [self.__numO]), [self.__numO, tf.shape(self.rules)[0] * tf.shape(self.rules)[1]])

        self.mul = tf.multiply(self.helper, self.y_bigger)
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


    def make_prediction(self, sess, x, targets=None):
        if targets is None:
            return sess.run(self.out, feed_dict={self.inputs: x})
        else:
            return sess.run([self.out, self.loss], feed_dict={self.inputs: x, self.targets: targets})


    def train(self, sess, x , targets):
        # y, y1 = sess.run([self.y, self.y1])
        rule, helper, mul, back, den = sess.run([self.rules, self.helper, self.mul, self.back_to_matrix, self.den], feed_dict={self.inputs: x, self.targets: targets})
        y, ai_, ci_,out, l, _ = sess.run([self.y, self.ai, self.ci, self.out, self.loss, self.optimize], feed_dict={self.inputs: x, self.targets: targets})
        return l, out, ai_, ci_, y


    def save_weights(self, path, sess):
        y_, ai_, ci_ = sess.run([self.y, self.ai, self.ci])
        json_dict = {}
        json_dict['ai'] = ai_.tolist()
        json_dict['ci'] = ci_.tolist()
        json_dict['y'] = y_.tolist()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(json_dict))


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
import tensorflow as tf
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class Fuzzyloopa():

    def __init__(self, n_inputs, n_rules, learning_rate = 1e-2):
        self.__numI = n_inputs
        self.__numR = n_rules
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.__numI))
        self.targets = tf.placeholder(tf.float32, shape=None)
        # numbers of ai varibles are numR * numI
        self.ai = tf.get_variable('ai', [self.__numR * self.__numI], initializer=tf.random_normal_initializer(0,1))
        self.ci = tf.get_variable('ci', [self.__numR * self.__numI], initializer=tf.random_normal_initializer(0, 1))
        # this step is 2-3 layers
        self.rules = tf.reduce_prod(
            tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs, (1, n_rules)), self.ai) / (self.ci))),
                       (-1, n_rules, n_inputs)), axis=2)
        # self.rules = self.rules / tf.reduce_sum(self.rules)
        # __b = tf.Variable(np.random.randn(), name='b')
        # __w = tf.Variable(np.random.randn(), name='w')
        # self.y_pred = tf.reduce_sum(tf.add(tf.multiply(self.rules, __w), __b), axis=1)

        y = tf.get_variable('y', [1, n_rules], initializer=tf.random_normal_initializer(0,1))
        num = tf.reduce_sum(tf.multiply(self.rules, y), axis=1)
        den = tf.clip_by_value(tf.reduce_sum(self.rules, axis=1), 1e-12, 1e12)
        self.out = tf.divide(num, den)

        self.loss = tf.losses.huber_loss(self.targets, self.out)
        self.optimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        self.init_variables = tf.global_variables_initializer()


    def make_prediction(self, sess, x):
        return sess.run(self.out, feed_dict={self.inputs:x})


    def train(self, sess, x , targets):
        out, l, _ = sess.run([self.out, self.loss, self.optimize], feed_dict={self.inputs: x, self.targets: targets})
        return l, out
# rules = tf.constant([[1, 1, 1, 1], [1,1,1,1], [1,1,1,1]])
# input = tf.constant([1,2,1,1],[1,1,1,1])
# n_rules = 2
#inputs = tf.placeholder(tf.float32, shape=(None, 4))
# matrix =  tf.reduce_prod(
#             tf.reshape(tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(rules, (1, n_rules)), input)) / tf.square(input)),
#                        (-1, n_rules, 2)), axis=2)
# ai = tf.get_variable('ai', [3 * 5], initializer=tf.random_normal_initializer(0,1))
if __name__ == "__main__":
    with tf.Session() as sess:
        a1 = tf.get_variable('ai', [3 * 5], initializer=tf.random_normal_initializer(0,1))
        a1.initializer.run()
        print(a1.eval())


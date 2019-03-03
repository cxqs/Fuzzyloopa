import tensorflow as tf
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class Fuzzyloopa():

    def __init__(self, n_inputs, n_rules):
        self.numI = n_inputs
        self.numR = n_rules
        self.inputs = tf.placeholder(tf.float32, shape=(None, n_inputs))
        self.targets = tf.placeholder(tf.float32, shape=None)
        self.ai = tf.get_variable('ai', [self.numR, self.numI], initializer=tf.random_normal_initializer(0,1))
        self.ci = tf.get_variable('ci', [self.numR, self.numI], initializer=tf.random_normal_initializer(0, 1))
        y = tf.get_variable('y', [1, n_rules], initializer=tf.random_normal_initializer(0,1))
        self.rules = tf.exp(-tf.square((tf.tile(self.inputs, (1, n_rules)) - self.ai) / self.ci))
        self.rules = tf.reshape(self.rules,(-1, n_rules, n_inputs))

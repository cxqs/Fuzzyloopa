from Model import Fuzzyloopa
from pre.Preprocessing import Prepocessing as prep

import time
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mackey(n_iters):
    x = np.zeros((n_iters,))
    x[0:30] = 0.23 * np.ones((30,))
    t_s = 30
    for i in range(30, n_iters - 1):
        a = x[i]
        b = x[i - t_s]
        y = ((0.2 * b) / (1 + b ** 10)) + 0.9 * a
        x[i + 1] = y
    return x


D = 4  # number of regressors
T = 1  # delay
N = 1000  # Number of points to generate
mg_series = mackey(N)[499:]  # Use last 1500 points
# frame = pd.read_csv('TimeSeries/Algn.csv')
# series = frame['Adj Close'].values[:1000]
#
# series= prep('TimeSeries/Algn.csv')
# series.creat_features()
# series.creat_target()
#
# data = np.array(series.features)
# lbls = np.array(series.targets)

data = np.zeros((N - 500 - T - (D - 1) * T, D))
lbls = np.zeros((N - 500 - T - (D - 1) * T,))



for t in range((D - 1) * T, N - 500 - T):
    data[t - (D - 1) * T, :] = [mg_series[t - 3 * T], mg_series[t - 2 * T], mg_series[t - T], mg_series[t]]
    lbls[t - (D - 1) * T] = mg_series[t + T]
trnData = data[:lbls.size - round(len(lbls) * 0.3), :]
trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]
chkData = data[lbls.size - round(lbls.size * 0.3):, :]
chkLbls = lbls[lbls.size - round(lbls.size * 0.3):]


m = 5  # number of rules
alpha = 0.01  # learning rate

fis = Fuzzyloopa(n_inputs=D, n_rules=m, learning_rate=alpha)

# Training
num_epochs = 100

# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    time_start = time.time()
    for epoch in range(100):
        trn_loss, trn_pred = fis.train(sess, trnData, trnLbls)
        val_pred, val_loss = fis.make_prediction(sess, chkData, chkLbls)
        if epoch % 10 == 0:
            print("Train cost after epoch %i: %f" % (epoch, trn_loss))
        if epoch == num_epochs - 1:
            time_end = time.time()
            print("Elapsed time: %f" % (time_end - time_start))
            print("Validation loss: %f" % val_loss)
            pred = np.vstack((np.expand_dims(trn_pred, 1), np.expand_dims(val_pred, 1)))
            plt.figure(1)
            # plt.plot(series.targets)
            plt.plot(mg_series)
            plt.plot(pred)
        trn_costs.append(trn_loss)
        val_costs.append(val_loss)
    plt.show()
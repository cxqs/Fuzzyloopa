from Models import Fuzzyloopa, CFuzzyloopa
from Preprocessing.Preprocessing import Prepocessing as prep

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


# data = np.zeros((N - T - (D - 1) * T, D))
# lbls = np.zeros((N - T - (D - 1) * T,))



# for t in range((D - 1) * T, N - T):
#     data[t - (D - 1) * T, :] = [mg_series[t - 3 * T], mg_series[t - 2 * T], mg_series[t - T], mg_series[t]]
#     lbls[t - (D - 1) * T] = mg_series[t + T]
# trnData = data[:lbls.size - round(len(lbls) * 0.3), :]
# trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]
# chkData = data[lbls.size - round(lbls.size * 0.3):, :]
# chkLbls = lbls[lbls.size - round(lbls.size * 0.3):]

D = 15  # number of regressors
T = 1  # delay
# N = 1000  # Number of points to generate
# frame = pd.read_csv('TimeSeries/zuerich.csv')
# mg_series = mackey(N)[499:]  # Use last 1500 points
# mg_series = frame["Zuerich"].values
# frame = pd.read_csv('TimeSeries/Algn.csv')
# series = frame['Adj Close'].values[:1000]

series= prep('TimeSeries/zuerich.csv', D, 1, "Zuerich")
series.print_info()
series.creat_features()
series.normalize_features()
series.creat_target()

data_new = np.array(series.features)[:2800]
lbls_new = np.array(series.targets)[:2800]

trnData_new = data_new[:lbls_new.size - round(len(lbls_new) * 0.3), :]
trnLbls_new = lbls_new[:lbls_new.size - round(lbls_new.size * 0.3)]
chkData_new = data_new[lbls_new.size - round(lbls_new.size * 0.3):, :]
chkLbls_new = lbls_new[lbls_new.size - round(lbls_new.size * 0.3):]


# m = 22  # number of rules best for 1
m = 22
alpha = 0.001  # learning rate
# batch_size = 700 best for
batch_size = 700

# cfis = Fuzzyloopa(n_inputs=D, n_rules=m, learning_rate=alpha)
cfis = CFuzzyloopa(n_inputs=D, n_rules=m, n_output=1, learning_rate=alpha)
# Training
num_epochs = 20000
# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    sess.run(cfis.init_variables)
    trn_costs = []
    val_costs = []
    before = 0
    ai_ = []
    ci_ = []
    trn_loss, trn_pred = None, None
    for epoch in range(num_epochs):
        for i in range(0,len(trnData_new),batch_size):
            trn_loss, trn_pred, ai_, ci_ = cfis.train(sess, trnData_new[i:i+batch_size], trnLbls_new[i:i+batch_size])
        trn_pred, trn_loss = cfis.make_prediction(sess, trnData_new, trnLbls_new)
        if epoch % 10 == 0:
            print("Train cost after epoch %i: \nhuber: %f" % (epoch, trn_loss))
            mad = cfis.MAD(trn_pred, trnLbls_new)
            mape = cfis.MAPE(trn_pred, trnLbls_new)
            smape = cfis.SMAPE(trn_pred, trnLbls_new)
            print('mape: {}\nmad: {}\nsmape: {}\n'.format(mape, mad, smape))
            # ctrn_loss, ctrn_pred, cai_, cci_ = cfis.train(sess, trnData_new, trnLbls_new)
        if epoch == num_epochs - 1:
            # cfis.plot_rules(sess, 2)
            val_pred, val_loss = cfis.make_prediction(sess, chkData_new, chkLbls_new)
            val_pred_ex, val_loss_ex = cfis.make_prediction(sess, chkData_new[0:3], chkLbls_new[0:3])
            # pred = np.vstack((np.expand_dims(trn_pred, 1), np.expand_dims(val_pred, 1)))
            print(val_loss)
            plt.plot([x for x in range(0, lbls_new.size)], lbls_new, color='blue')
            plt.plot([x for x in range(lbls_new.size - round(len(lbls_new) * 0.3), lbls_new.size)], val_pred, color='yellow')
            plt.plot([x for x in range(0,lbls_new.size - round(len(lbls_new) * 0.3))], trn_pred, color='green')
    plt.show()
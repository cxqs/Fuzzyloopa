from Models import Fuzzyloopa
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
frame = pd.read_csv('TimeSeries/zuerich.csv')
# mg_series = mackey(N)[499:]  # Use last 1500 points
# mg_series = frame["Zuerich"].values
# frame = pd.read_csv('TimeSeries/Algn.csv')
# series = frame['Adj Close'].values[:1000]
#
series= prep('TimeSeries/Algn.csv', D, "Adj Close")
series.print_info()
series.creat_features()
series.creat_target()
#
data_new = np.array(series.features)[500:1000]
lbls_new = np.array(series.targets)[500:1000]

# data = np.zeros((N - T - (D - 1) * T, D))
# lbls = np.zeros((N - T - (D - 1) * T,))



# for t in range((D - 1) * T, N - T):
#     data[t - (D - 1) * T, :] = [mg_series[t - 3 * T], mg_series[t - 2 * T], mg_series[t - T], mg_series[t]]
#     lbls[t - (D - 1) * T] = mg_series[t + T]
# trnData = data[:lbls.size - round(len(lbls) * 0.3), :]
# trnLbls = lbls[:lbls.size - round(lbls.size * 0.3)]
# chkData = data[lbls.size - round(lbls.size * 0.3):, :]
# chkLbls = lbls[lbls.size - round(lbls.size * 0.3):]

trnData_new = data_new[:lbls_new.size - round(len(lbls_new) * 0.3), :]
trnLbls_new = lbls_new[:lbls_new.size - round(lbls_new.size * 0.3)]
chkData_new = data_new[lbls_new.size - round(lbls_new.size * 0.3):, :]
chkLbls_new = lbls_new[lbls_new.size - round(lbls_new.size * 0.3):]


m = 20  # number of rules
# m = 10
alpha = 0.01  # learning rate

ai = [ 3.3417804,   2.4792037,   3.3038974,   3.2569113,  -0.3606914,   1.2810735,
 -0.47960976,  0.21745922,  0.30546865,  1.2757643,   0.20249526, -0.62234116,
  0.29753733, -1.3188442,   1.2533172,   0.8017225,   0.09012651,  0.5127314,
 -1.6711113,   0.56070346, -1.0299779,   0.386763,    0.29509157,  0.57632816,
  0.92725724,  0.7118579,   0.7328319,   1.0096419,  -1.1960636,  -0.89693844,
 -0.9116366,   0.8018516,  -0.8771843,   1.4065806,  -0.31195757,  1.3654612,
 -0.24989913,  0.48568895,  1.2807167,  -0.19832258,  0.29606855,  0.2446179,
 -0.56884235,  0.2768941,   0.10673622, -0.1705654,   0.5099891,  -0.72171605,
  5.164786,    2.6424053,  -0.24313544,  7.84002,     1.5434158,   0.85288656,
 -0.02034415, -0.3153839,   0.8430154,  -0.75862825, -1.3017598,  -0.09646904,
  1.8866267,  -1.0692087,   1.5381913,  -3.2453372,  -1.0433218,  -0.57226723,
 -0.66502047,  0.11316385, -0.9902626,  -0.7993953,  -0.6382371,  -1.6664555,
 -0.10946628,  0.9446097,   2.379941,    1.7645154,   0.5101593,  -0.16252726,
  0.33576387,  0.6866665 ]

ci = [ 5.584588,    4.793802,    4.02952,   -4.06011,     0.1081356,  -0.925067,
 -0.4959379,   0.8363054,   1.2502332,  -1.7618656,  -0.65886766,  0.386207,
  0.5149843,   1.2574258,   0.8933777,   1.2945822,  -0.74716985, -0.4265025,
 -1.3401937,  -0.82246196,  0.0126244,   0.5336313,   0.5651808,  -0.08543375,
 -1.3947186,   0.15033567,  1.0751579,  -0.93908316,  0.84019774, -0.54418176,
 -0.3952142,  -0.5004632,   0.5125287,  -0.9740452,   0.29803014,  1.2412325,
  0.2019312,  0.02825334, -1.3929935,  -1.0680609,   1.1559566,  -0.21055202,
  0.86910886,  0.7874488,  -0.6511206,   1.2921422,   0.4117043,  -0.61076087,
 -4.8468957,   4.590745,   -4.9319706,   6.562854,    1.7311023,   0.41180715,
  1.0193431,  -0.29825374, -0.60405266, -1.1156625,   1.4747849,  -1.6098958,
  1.4741768,  -0.47115132,  0.22690246,  1.237101,    0.30943826,  1.076106,
  0.04192426,  1.4916527,  -1.3864969,   0.28438193, -1.007004,   -2.4688423,
 -0.22718635,  0.28712523, -0.36871764, -0.3263881,  -0.8420689,   0.42085102,
 -1.2117531,   0.45000574]
fis = Fuzzyloopa(n_inputs=D, n_rules=m, learning_rate=alpha, ai=ai, ci=ci)
# Training
num_epochs = 5000
# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    sess.run(fis.init_variables)
    trn_costs = []
    val_costs = []
    before = 0
    time_start = time.time()
    ai_ = []
    ci_ = []
    for epoch in range(num_epochs):
        trn_loss, trn_pred, ai_, ci_ = fis.train(sess, trnData_new, trnLbls_new)
        if epoch % 10 == 0:
            print("Train cost after epoch %i: \nhuber: %f" % (epoch, trn_loss))
            mad = fis.MAD(trn_pred, trnLbls_new)
            mape = fis.MAPE(trn_pred, trnLbls_new)
            smape = fis.SMAPE(trn_pred, trnLbls_new)
            print('mape: {}\nmad: {}\nsmape: {}\n'.format(mape, mad, smape))
        if epoch == num_epochs - 1:
            # print(ai_)
            # print(ci_)
            fis.plot_rules(sess, 0)
            val_pred, val_loss = fis.make_prediction(sess, chkData_new, chkLbls_new)
            pred = np.vstack((np.expand_dims(trn_pred, 1), np.expand_dims(val_pred, 1)))
            print(val_loss)
            # plt.plot([x for x in range(700,1000)],val_pred)
            plt.plot([x for x in range(0, lbls_new.size)], lbls_new, color='blue')
            # plt.plot([x for x in range(lbls_new.size - round(len(lbls_new) * 0.3), lbls_new.size)], val_pred, color='yellow')
            plt.plot([x for x in range(0,lbls_new.size - round(len(lbls_new) * 0.3))], trn_pred, color='green')
    plt.show()
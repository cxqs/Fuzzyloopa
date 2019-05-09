from Model.Models import Fuzzyloopa, CFuzzyloopa
from Preprocessing.Preprocessing import Prepocessing as prep
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json


def read_weights(path):
    with open(path) as json_file:
        data = json.load(json_file)
        ai_algn = data['ai']
        ci_algn = data['ci']
        y_algn = data['y']
    return ai_algn, ci_algn, y_algn


D = 7  # number of regressors
T = 1  # delay

PATH_DATA = 'TimeSeries/zuerich.csv'
NAME_COL = 'Zuerich'
PATH_WEIGHT = 'Weights/Zuerich_22_5_7_2400_2600.txt'

series = prep(PATH_DATA, D, T, NAME_COL)
series.print_info()
series.creat_features()
series.normalize_features()
series.creat_target()

data_new = np.array(series.features)[:]
lbls_new = np.array(series.targets)[:]

trnData_new = data_new[:len(lbls_new) - round(len(lbls_new) * 0.3), :]
trnLbls_new = lbls_new[:len(lbls_new) - round(len(lbls_new) * 0.3)]
chkData_new = data_new[len(lbls_new) - round(len(lbls_new) * 0.3):, :]
chkLbls_new = lbls_new[len(lbls_new) - round(len(lbls_new) * 0.3):]


# m = 22  # number of rules best for 1
m = 22
alpha = 0.001  # learning rate
# batch_size = 700 best for
batch_size = 10


trnData_new = data_new[0:1000, :]
trnLbls_new = lbls_new[0:1000,:]

chkData_new = data_new[2500:2501, :]
chkLbls_new = lbls_new[2500:2501, :]

# ai, ci ,y = read_weights('Weights/Algn_22_1_7.txt')
cfis = CFuzzyloopa(n_inputs=D, n_rules=m, n_output=T, learning_rate=alpha)
# Training
num_epochs = 10000
# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    sess.run(cfis.init_variables)
    for epoch in range(num_epochs):
        for i in range(0,len(trnData_new),batch_size):
            trn_loss, trn_pred, ai_, ci_, y_ = cfis.train(sess, trnData_new[i:i+batch_size], trnLbls_new[i:i+batch_size])
        trn_pred, trn_loss = cfis.make_prediction(sess, trnData_new, trnLbls_new)
        if epoch % 10 == 0:
            print("Train cost after epoch %i: \nhuber: %f" % (epoch, trn_loss))
            mad = cfis.MAD(trn_pred, trnLbls_new)
            mape = cfis.MAPE(trn_pred, trnLbls_new)
            smape = cfis.SMAPE(trn_pred, trnLbls_new)
            print('mape: {}\nmad: {}\nsmape: {}\n'.format(mape, mad, smape))
        if epoch == num_epochs - 1:
            cfis.save_weights(PATH_WEIGHT, sess)
            lbls_new1 = series.creat_target(t=1)
            val_pred = cfis.make_prediction(sess, chkData_new)
            plt.plot([x for x in range(2300, len(lbls_new1[:2505]))], lbls_new[2300:2505], color='blue')
            # plt.plot([x for x in range(len(lbls_new) - round(len(lbls_new) * 0.3), len(lbls_new) - round(len(lbls_new) * 0.3) + T)], val_pred[0], color= 'yellow')
            # plt.plot([x for x in range(len(lbls_new) - round(len(lbls_new) * 0.3), len(lbls_new))], val_pred, color='yellow')
            plt.plot([x for x in range(2500 ,2505)], val_pred[0], color='red')
            # plt.plot([x for x in range(0,lbls_new.size - round(len(lbls_new) * 0.3))], trn_pred, color='green')
    plt.show()
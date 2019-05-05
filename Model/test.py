from Model.Models import Fuzzyloopa, CFuzzyloopa
from Preprocessing.Preprocessing import Prepocessing as prep

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

D = 15  # number of regressors
T = 1  # delay

series= prep('TimeSeries/zuerich.csv', D, T, 'Zuerich')
series.print_info()
series.creat_features()
series.normalize_features()
series.creat_target()

data_new = np.array(series.features)[:2800]
lbls_new = np.array(series.targets)[:2800]

trnData_new = data_new[:len(lbls_new) - round(len(lbls_new) * 0.3), :]
trnLbls_new = lbls_new[:len(lbls_new) - round(len(lbls_new) * 0.3)]
chkData_new = data_new[len(lbls_new) - round(len(lbls_new) * 0.3):, :]
chkLbls_new = lbls_new[len(lbls_new) - round(len(lbls_new) * 0.3):]


# m = 22  # number of rules best for 1
m = 22
alpha = 0.001  # learning rate
# batch_size = 700 best for
batch_size = 700

# cfis = Fuzzyloopa(n_inputs=D, n_rules=m, learning_rate=alpha)
cfis = CFuzzyloopa(n_inputs=D, n_rules=m, n_output=T, learning_rate=alpha)
# Training
num_epochs = 40000
# Initialize session to make computations on the Tensorflow graph
with tf.Session() as sess:
    sess.run(cfis.init_variables)
    trn_costs = []
    val_costs = []
    before = 0
    ai_ = []
    ci_ = []
    y_ = []
    trn_loss, trn_pred = None, None
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
            # ctrn_loss, ctrn_pred, cai_, cci_ = cfis.train(sess, trnData_new, trnLbls_new)
        if epoch == num_epochs - 1:
            json_dict = {}
            json_dict['ai'] = ai_.tolist()
            json_dict['ci'] = ci_.tolist()
            json_dict['y'] = y_.tolist()
            with open('Weights/Zuerich_22_1_15.txt', 'w', encoding='utf-8') as f:
                f.write(json.dumps(json_dict))
            val_pred, val_loss = cfis.make_prediction(sess, chkData_new, chkLbls_new)
            val_pred_ex, val_loss_ex = cfis.make_prediction(sess, chkData_new[0:3], chkLbls_new[0:3])
            print(val_loss)
            # res = np.reshape(trn_pred,[1,-1])
            # res1 = np.reshape(trnData_new, [1, -1])
            # plt.plot(list(res[0]))
            # plt.plot(list(res1[0]))
            plt.plot([x for x in range(0, len(lbls_new))], lbls_new, color='blue')
            plt.plot([x for x in range(len(lbls_new) - round(len(lbls_new) * 0.3), len(lbls_new))], val_pred, color='yellow')
            plt.plot([x for x in range(0,lbls_new.size - round(len(lbls_new) * 0.3))], trn_pred, color='green')
    plt.show()
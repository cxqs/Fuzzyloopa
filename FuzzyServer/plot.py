import json
import tensorflow as tf
import numpy as np

from flask import Flask, render_template, request

from Model.Models import CFuzzyloopa
from Preprocessing.Preprocessing import Prepocessing as prep

app = Flask(__name__)

D = 15
T = 1
ALPHA = 0.01  # learning rate
m = 22
ai_algn = 0
ci_algn = 0
y_algn = 0

with open('../Weights/Algn_22_1.txt') as json_file:
    data = json.load(json_file)
    ai_algn = data['ai']
    ci_algn = data['ci']
    y_algn = data['y']


@app.route('/')
def line(line = 'Bitcoin Monthly Price in USD'):
    series = prep('../TimeSeries/Algn.csv', D, T, "Adj Close")
    series.creat_features()
    series.normalize_features()
    series.creat_target()

    data = np.array(series.features)[:2800]
    lbls = np.array(series.targets)[:2800]

    cfis = CFuzzyloopa(n_inputs=D, n_rules=m, n_output=T, learning_rate=ALPHA, ai=ai_algn, ci=ci_algn, y=y_algn)

    with tf.Session() as sess:
        sess.run(cfis.init_variables)

        chkData_new = data[lbls.size - round(lbls.size * 0.3):, :]
        chkLbls_new = lbls[lbls.size - round(lbls.size * 0.3):]

        all_data = []
        for ch in lbls:
            all_data.append(ch[0])

        val_pred, val_loss = cfis.make_prediction(sess, chkData_new, chkLbls_new)
        prediction = [-1 for i in range(len(all_data) - len(val_pred))]
        for ch in val_pred:
            prediction.append(ch[0])

        labels = [i for i in range(2800)]
        line_labels = labels
        line_values = all_data

    return render_template('base.html', labels=line_labels, values=line_values, predic=prediction)


@app.route('/predict_align/', methods=['POST'])
def do_align():
    series = prep('../TimeSeries/Algn.csv', D, T, "Adj Close")
    series.creat_features()
    series.normalize_features()
    series.creat_target()

    data = np.array(series.features)[:2800]
    lbls = np.array(series.targets)[:2800]

    cfis = CFuzzyloopa(n_inputs=D, n_rules=m, n_output=T, learning_rate=ALPHA, ai=ai_algn, ci=ci_algn, y=y_algn)

    with tf.Session() as sess:
        sess.run(cfis.init_variables)

        chkData_new = data[lbls.size - round(lbls.size * 0.3):, :]
        chkLbls_new = lbls[lbls.size - round(lbls.size * 0.3):]

        all_data = []
        for ch in lbls:
            all_data.append(ch[0])

        val_pred, val_loss = cfis.make_prediction(sess, chkData_new, chkLbls_new)
        prediction = [-1 for i in range(len(all_data) - len(val_pred))]
        for ch in val_pred:
            prediction.append(ch[0])

        labels = [i for i in range(2800)]
        line_labels = labels
        line_values = all_data

    return render_template('base.html', labels=line_labels, values=line_values, predic=prediction)

@app.route('/predict_Zuerich/', methods=['POST'])
def doit():

    with open('../Weights/Zuerich_22_1_15.txt') as json_file:
        data = json.load(json_file)
        ai_Zuerich = data['ai']
        ci_Zuerich = data['ci']
        y_Zuerich = data['y']

    series = prep('../TimeSeries/zuerich.csv', D, T, "Zuerich")
    series.creat_features()
    series.normalize_features()
    series.creat_target()

    data = np.array(series.features)[:2800]
    lbls = np.array(series.targets)[:2800]

    cfis = CFuzzyloopa(n_inputs=D, n_rules=m, n_output=T, learning_rate=ALPHA, ai=ai_Zuerich, ci=ci_Zuerich, y=y_Zuerich)

    with tf.Session() as sess:
        sess.run(cfis.init_variables)

        chkData_new = data[lbls.size - round(lbls.size * 0.3):, :]
        chkLbls_new = lbls[lbls.size - round(lbls.size * 0.3):]

        all_data = []
        for ch in lbls:
            all_data.append(ch[0])

        val_pred, val_loss = cfis.make_prediction(sess, chkData_new, chkLbls_new)
        prediction = [-1 for i in range(len(all_data) - len(val_pred))]
        for ch in val_pred:
            prediction.append(ch[0])

        labels = [i for i in range(2800)]
        line_labels = labels
        line_values = all_data

    return render_template('base.html', labels=line_labels, values=line_values, predic=prediction)

if __name__ == '__main__':
    app.run()
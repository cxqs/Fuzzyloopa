import json
import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, abort, redirect, url_for
from Model.Models import CFuzzyloopa
from Preprocessing.Preprocessing import Prepocessing as prep


D = 7
T = 1
ALPHA = 0.01  # learning rate
M = 22


def read_data(path, name, d, t, start, finish, normalize=True):
    series = prep(path, d, t, name)
    series.creat_features()
    if normalize:
        series.normalize_features()
    series.creat_target()

    data = np.array(series.features)[:finish]
    lbls = np.array(series.targets)[:finish]

    chkData_new = data[start:finish, :]
    chkLbls_new = lbls[start:finish, :]
    return lbls, chkData_new, chkLbls_new


def read_weights(path):
    with open(path) as json_file:
        data = json.load(json_file)
        ai_algn = data['ai']
        ci_algn = data['ci']
        y_algn = data['y']
    return ai_algn, ci_algn, y_algn


app = Flask(__name__)

@app.route('/<somevar>')
@app.route('/')
def main(somevar=None):
    a = somevar
    if somevar == None:
        return render_template('home.html')
    else:
        return render_template('home.html', messegeid="Подходящих весов для данного временного ряда пока нет")


@app.route('/predict_align/', methods=['POST'])
def go_align():

    start = 2400
    finish = 2600
    ai_algn, ci_algn, y_algn = read_weights('../Weights/Algn_22_1_7_2400_2600.txt')
    lbls, chkData_new, chkLbls_new = read_data('../TimeSeries/Algn.csv', 'Adj Close', D, T, start, finish)
    cfis = CFuzzyloopa(n_inputs=D, n_rules=M, n_output=T, learning_rate=ALPHA, ai=ai_algn, ci=ci_algn, y=y_algn)

    with tf.Session() as sess:
        sess.run(cfis.init_variables)

        all_data = []
        for ch in lbls:
            all_data.append(ch[0])

        val_pred, val_loss = cfis.make_prediction(sess, chkData_new, chkLbls_new)
        mad = cfis.MAD(val_pred, chkLbls_new)
        mape = cfis.MAPE(val_pred, chkLbls_new)
        smape = cfis.SMAPE(val_pred, chkLbls_new)
        prediction = [-1 for i in range(len(all_data) - len(val_pred))]
        for ch in val_pred:
            prediction.append(ch[0])

        labels = [i for i in range(2800)]
        line_labels = labels
        line_values = all_data


    return render_template('plot.html', labels=line_labels, values=line_values, predic=prediction,label_name='Align', huber=val_loss, mad=mad, mape=mape, smape=smape)

@app.route('/predict_Temps/', methods=['POST'])
def go_temps():

    start = 2000
    finish = 2001
    ai_Temps, ci_Temps, y_Temps = read_weights('../Weights/owndata/temps.txt')
    cfis = CFuzzyloopa(n_inputs=D, n_rules=M, n_output=T, learning_rate=ALPHA, ai=ai_Temps, ci=ci_Temps, y=y_Temps)
    lbls, chkData_new, chkLbls_new = read_data('../TimeSeries/temps.csv', 'temps', D, T, start, finish, normalize=False)

    with tf.Session() as sess:
        sess.run(cfis.init_variables)

        all_data = []
        for ch in lbls:
            all_data.append(ch[0])

        val_pred, val_loss = cfis.make_prediction(sess, chkData_new, chkLbls_new)
        mad = cfis.MAD(val_pred, chkLbls_new)
        mape = cfis.MAPE(val_pred, chkLbls_new)
        smape = cfis.SMAPE(val_pred, chkLbls_new)
        prediction = [-1 for i in range(0,start)]
        prediction.append(val_pred[0][0])


        labels = [i for i in range(0,finish)]
        line_labels = labels
        line_values = all_data[0:finish]

    return render_template('plot.html', labels=line_labels, values=line_values, predic=prediction,label_name='Temps', huber=val_loss, mad=mad, mape=mape, smape=smape)


@app.route('/predict_Mackey/', methods=['POST'])
def go_mackey():

    start = 2500
    finish = 2505
    ai_Mackey, ci_Mackey, y_Mackey = read_weights('../Weights/Mackey_22_5_7_2500_2505.txt')
    cfis = CFuzzyloopa(n_inputs=D, n_rules=M, n_output=5, learning_rate=ALPHA, ai=ai_Mackey, ci=ci_Mackey, y=y_Mackey)
    lbls, chkData_new, chkLbls_new = read_data('../TimeSeries/Mackey.csv', 'mackey', D, 5, start, finish)

    with tf.Session() as sess:
        sess.run(cfis.init_variables)

        all_data = []
        for ch in lbls:
            all_data.append(ch[0])

        val_pred, val_loss = cfis.make_prediction(sess, chkData_new, chkLbls_new)
        mad = cfis.MAD(val_pred, chkLbls_new)
        mape = cfis.MAPE(val_pred, chkLbls_new)
        smape = cfis.SMAPE(val_pred, chkLbls_new)
        prediction = [-1 for i in range(2300,start)]
        prediction.extend(val_pred[0][:])


        labels = [i for i in range(2300,finish)]
        line_labels = labels
        line_values = all_data[2300:finish]

    return render_template('plot.html', labels=line_labels, values=line_values, predic=prediction,label_name='Zuerich', huber=val_loss, mad=mad, mape=mape, smape=smape)


@app.route('/uploads/', methods=['GET', 'POST'])
def go_upload():

    D = 7
    T = 1
    ALPHA = 0.01  # learning rate
    M = 22
    try:
        if request.method == 'POST':
            if not 'file' in request.files:
                abort(400, 'Some problem with download')
                return render_template('/')
            file = request.files['file']
            df = pd.read_csv(file.stream)
            name = file.filename
            name_of_col = name.split('.')[0]
            ai_, ci_, y_ = read_weights('../Weights/owndata/{0}.txt'.format(name_of_col.lower()))
            if name == 'manning.csv':
                T = 4
            cfis = CFuzzyloopa(n_inputs=D, n_rules=M, n_output=T, learning_rate=ALPHA, ai=ai_, ci=ci_, y=y_)
            with tf.Session() as sess:
                sess.run(cfis.init_variables)
                all_data = df[name_of_col.lower()].values
                data_plot = list(all_data)[:]
                if name == 'manning.csv':
                    mean = df[name_of_col.lower()].mean()
                    std = df[name_of_col.lower()].std()
                    all_data = (all_data - mean) / std
                data_for_pred = list(np.reshape(all_data[-D-T:-T],[1,-1]))
                predicted = all_data[-1]
                labels = [i for i in range(0,2800)]
                line_labels = labels
                line_values = data_plot

                val_pred = cfis.make_prediction(sess, data_for_pred)
                prediction = [-1 for i in range(0, 2800-T)]
                prediction.extend(val_pred[0,:])
    except:
        return redirect(url_for('main', somevar='error'))
    return render_template('fromowndata.html', labels=line_labels, values=line_values, predic=prediction,
                               label_name='real data')

if __name__ == '__main__':
    app.run()
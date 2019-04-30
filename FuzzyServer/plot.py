from flask import Flask, Markup, render_template, request, redirect, url_for
from Preprocessing.Preprocessing import Prepocessing as prep
import numpy as np
import pandas as pd

app = Flask(__name__)

D = 15  # number of regressors

series= prep('../TimeSeries/Algn.csv', D, 1, "Adj Close")
series.print_info()
series.creat_features()
series.normalize_features()
series.creat_target()

# data_new = np.array(series.features)[:50]
lbl = []
for i in series.targets[:1000]:
    lbl.append(i[0])
lbls_new = lbl

# labels = [
#     'JAN', 'FEB', 'MAR', 'APR',
#     'MAY', 'JUN', 'JUL', 'AUG',
#     'SEP', 'OCT', 'NOV', 'DEC'
# ]
#

labels = [i for i in range(len(lbls_new[:1000]))]
values = lbls_new[:1000]

@app.route('/')
def line(line = 'Bitcoin Monthly Price in USD'):
    line_labels = labels
    line_values = values
    # return render_template('base.html', title=line, max=17000, labels=line_labels,
    #                        values=line_values)
    return render_template('base.html', title='Ploting...', max=17000, labels=line_labels, values=line_values)


@app.route('/predict/', methods=['POST'])
def doit():
    index = request.form['index']
    print(index)
    return redirect("http://127.0.0.1:5000")

app.run()
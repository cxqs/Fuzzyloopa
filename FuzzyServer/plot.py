from flask import Flask, Markup, render_template, request, redirect, url_for

app = Flask(__name__)

labels = [
    'JAN', 'FEB', 'MAR', 'APR',
    'MAY', 'JUN', 'JUL', 'AUG',
    'SEP', 'OCT', 'NOV', 'DEC'
]

values = [
    967.67, 1190.89, 1079.75, 1349.19,
    2328.91, 2504.28, 2873.83, 4764.87,
    4349.29, 6458.30, 9907, 16297
]


@app.route('/')
def line(line = 'Bitcoin Monthly Price in USD'):
    line_labels = labels
    line_values = values
    return render_template('base.html', title=line, max=17000, labels=line_labels,
                           values=line_values)


@app.route('/predict/', methods=['POST'])
def doit():
    index = request.form['index']
    print(index)
    return redirect("http://127.0.0.1:5000")

app.run()
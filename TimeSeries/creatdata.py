import numpy as np
import pandas as pd

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

data = mackey(2800)
df = pd.DataFrame(data=data,columns=['mackey'],index=[i for i in range(2800)])
df.to_csv('../TimeSeries/Mackey.csv', sep=',')
print(data)
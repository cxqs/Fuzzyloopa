import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def check_path(path:str):
    path = os.path.abspath('../' + path)
    if not path.endswith('.csv'):
        return False
    if not os.path.isfile(path):
        return False
    return True


class Prepocessing():

    def __init__(self, path:'str'):
        if check_path(path):
            self.frame = pd.read_csv(os.path.abspath('../' + path))


    def print_info(self):
        print('---------------describe--------------------')
        print(self.frame['Adj Close'].describe())
        print('--------------null----------------------')
        print(self.frame.isnull().sum())


    def creat_target(self):
        return np.array(self.frame['Adj Close'].values[30:])


    def creat_features(self):
        features = []
        for i in range(len(self.frame['Adj Close'].values)-30):
            features.append(self.frame['Adj Close'].values[i:i+30])
        return np.array(features)


    def plot_graph(self):
        plt.plot(self.frame['Adj Close'])
        plt.show()

cls = Prepocessing('TimeSeries//Algn.csv')
y = cls.creat_target()
x = cls.creat_features()
print(len(y))
print(len(x))
print(len(x[4494]))
print(y[4494])

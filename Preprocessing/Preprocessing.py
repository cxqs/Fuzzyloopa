import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def check_path(path:str):
    path = os.path.abspath(path)
    if not path.endswith('.csv'):
        return False
    if not os.path.isfile(path):
        return False
    return True


class Prepocessing():

    def __init__(self, path:'str', num, column):
        self.column = column
        self.num = num
        if check_path(path):
            self.frame = pd.read_csv(path)
            self.data = self.frame[column]


    def print_info(self):
        print('---------------describe--------------------')
        print(self.frame[self.column].describe())
        print('--------------null----------------------')
        print(self.frame.isnull().sum())


    def creat_target(self):
        self.targets = np.array(self.frame[self.column].values[self.num:])


    def creat_features(self):
        features = []
        for i in range(len(self.frame[self.column].values)-self.num):
            features.append(self.frame[self.column].values[i:i+self.num])
        self.features = np.array(features)


    def split(self):
        train_start = 0
        train_end = int(np.floor(0.8 * len(self.features)))
        test_start = train_end
        test_end = len(self.features)
        return self.features[np.arange(train_start, train_end), :]

    def plot_graph(self):
        plt.plot(self.frame[self.column])
        plt.show()

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

    def __init__(self, path:'str', num, output, column):
        self.output = output
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


    def normalize_features(self):
        mean = self.features.mean()
        std = self.features.std()
        self.features = (self.features - mean) / std


    def creat_target(self):
        targets = []
        for i in range(self.num, len(self.frame[self.column].values)-self.output):
            targets.append(self.frame[self.column].values[i:i+self.output])
            # features.append(self.frame[self.column].values[i:i+self.num])
        self.targets = np.array(targets)
        # self.targets = np.array(self.frame[self.column].values[self.num:])


    def creat_features(self):
        features = []
        for i in range(len(self.frame[self.column].values)-self.num - self.output):
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

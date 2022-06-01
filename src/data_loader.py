import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self):
        self.columns = None
        self.classes = None
        self.X = None
        self.y = None

    def load(self, filename):
        """
        Loads .csv file from given location.
        :return:
        """
        df = pd.read_csv(filename)

        self.columns = df.drop('label', axis=1).columns
        self.classes = np.unique(df['label'])

        # create target and data with standardized values
        self.y = LabelEncoder().fit_transform(y=df['label'].to_numpy())
        self.X = StandardScaler().fit_transform(df.drop('label', axis=1).to_numpy())

    def train_test_data(self, test_size=None, random_state=None):
        """
        Split loaded datasets with sklearn 'train_test_split' method.
        :return: X_train, X_test, y_train, y_test datasets
        """
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

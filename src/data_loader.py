#--------------------------------------------------
# Created By  : Krzysztof Palmi, Łukasz Sawicki 
# Created Date: 04.06.2022
# Class:        Machine Learning, 2022 summer
#--------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import configparser


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


def load_config(filename: str) -> dict:
    config_dict = dict()

    config = configparser.ConfigParser()
    config.read(filename)

    do_test = config.getboolean("PARAMETERS", "do_test")
    config_dict['do_test'] = do_test
    section = "TEST" if do_test else "PARAMETERS"

    for row in config.items(section):
        key = row[0]
        if key == 'do_test':
            continue

        values = row[1].replace(' ', '').split(',')

        # replace string with int or float in pythonic way :D
        try:
            # replace int
            values = [int(value) for value in values]
        except ValueError:
            try:
                # replace floats
                values = [float(value) for value in values]
            except ValueError:
                pass

        config_dict[key] = values

    # add info about inout and output files
    config_dict['dataset'] = config.get("FILES", "dataset")
    config_dict['output'] = config.get("FILES", "output")

    return config_dict

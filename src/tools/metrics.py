# --------------------------------------------------
# Created By  : Krzysztof Palmi, Łukasz Sawicki 
# Created Date: 04.06.2022
# Class:        Machine Learning, 2022 summer
# --------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join


class Metrics:
    def __init__(self, y_true, y_pred, train_time, classes, training_params):
        self._y_true = y_true
        self._y_pred = y_pred
        self._train_time = train_time
        self._classes = classes
        self._training_params = training_params

        self._matrix = self._create_matrix()

    def accuracy(self):
        return accuracy_score(self._y_true, self._y_pred)

    def f1(self):
        if len(self._classes) == 2:
            return f1_score(self._y_true, self._y_pred, average='binary')
        else:
            return f1_score(self._y_true, self._y_pred, average='macro')

    def auc(self):
        return roc_auc_score(self._y_true, self._y_pred, average='macro')

    def _create_matrix(self):
        cm = confusion_matrix(self._y_true, self._y_pred)
        cm = pd.DataFrame(cm, index=self._classes, columns=self._classes)

        return cm

    def confusion_matrix(self):
        return self._matrix

    def calc_results(self):
        p = self._training_params
        return [p['n_estimators'], p['max_depth'], p['min_samples_split'], p['min_samples_leaf'], p['criterion'],
                p['threshold'], p['split_method'], f"{self.accuracy():.2f}", f"{self.f1():.2f}",
                f"{self._train_time:.2f}"]

    def save_metrics(self, filename):
        plt.figure(figsize=(7, 6))
        sns.heatmap(self._matrix, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')

        # save confusion_matrix
        plt.savefig(join('../results/', filename))
        plt.clf()

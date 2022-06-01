import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join


class Metrics:
    def __init__(self, y_true, y_pred, classes):
        self._y_true = y_true
        self._y_pred = y_pred
        self._classes = classes

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

    def save_metrics(self, path_to_folder='./', n_estimator=0, max_depth=0, criterion=0, split_method=0):
        plt.figure(figsize=(7, 6))
        sns.heatmap(self._matrix, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')

        # save confusion_matrix
        filename = f"e{n_estimator}_d{max_depth}_{criterion}_{split_method}" + '.jpg'
        plt.savefig(join(path_to_folder, filename))
        plt.clf()

        # TODO AUC się nie liczy
        # save metrics to .csv file
        with open(join(path_to_folder, 'results.csv'), 'a') as f:
            f.write(f"{n_estimator},{max_depth},{criterion},{split_method},{self.accuracy()},{self.f1()},{0}\n")

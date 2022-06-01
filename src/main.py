import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from classifiers import RandomForestClassifier
from metrics import Metrics
from data_loader import DataLoader
import time

# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html
# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/turniejRuletka.pdf


def main():
    # data = datasets.load_breast_cancer()
    # X, y = data.data, data.target
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1
    # )
    loader = DataLoader()
    loader.load('../data/wine.csv')
    X_train, X_test, y_train, y_test = loader.train_test_data()

    clf = RandomForestClassifier(n_estimators=5, max_depth=10, min_samples_split=2, min_samples_leaf=3,
                                 criterion='gini', threshold=0.5, split_method='roulette')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = Metrics(y_test, y_pred, classes=loader.classes)
    metrics.save_metrics(path_to_folder='../results/', n_estimator=5, max_depth=10, criterion='gini',
                         split_method='roulette')

    # TODO wynmiki są za dobre...
    # TODO lepsze zapisywanie wyników zrobić
    # TODO trzeba przerobić kod, zeby nie było widać, ze z neta


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("---------- {:.10f}s ----------".format(end_time - start_time))

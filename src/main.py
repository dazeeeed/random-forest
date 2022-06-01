import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from classifiers import DecisionTreeClassifier, RandomForestClassifier
import time
import pandas as pd

# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html
# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/turniejRuletka.pdf


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def metrics(y_true, y_pred):
    df = pd.DataFrame(data=[[i, j] for i, j in zip(y_true, y_pred)], columns=['label', 'pred'])
    confusion_matrix = pd.crosstab(df['label'], df['pred'], rownames=['Actual'], colnames=['Predicted'])

    return confusion_matrix


def main():
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = RandomForestClassifier(n_estimators=5, max_depth=5, min_samples_split=2, min_samples_leaf=3,
                                 criterion='gini', threshold=0.6, split_method='roulette')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)

    acc = accuracy(y_test, y_pred)
    print("Accuracy: ", acc)

    cm = metrics(y_test, y_pred)
    print(cm)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("---------- {:.10f}s ----------".format(end_time - start_time))
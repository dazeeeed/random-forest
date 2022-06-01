import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from classifiers import DecisionTreeClassifier, RandomForestClassifier
import time

# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html
# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/turniejRuletka.pdf


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def main():
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = RandomForestClassifier(n_estimators=3, max_depth=4, min_samples_split=2, min_samples_leaf=1,
                                 criterion='gini', threshold=0.4, split_method='classic')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)

    acc = accuracy(y_test, y_pred)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("---------- {:.10f}s ----------".format(end_time - start_time))
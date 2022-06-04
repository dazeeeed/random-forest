#--------------------------------------------------
# Created By  : Krzysztof Palmi, Łukasz Sawicki 
# Created Date: 04.06.2022
# Class:        Machine Learning, 2022 summer
#--------------------------------------------------

import pandas as pd
from classifiers import RandomForestClassifier
from metrics import Metrics
from data_loader import DataLoader, load_config
from itertools import product
import time

# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html
# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/turniejRuletka.pdf


def create_train_params(config_dict: dict):
    """Creates all possible training parameters"""
    config_dict = config_dict.copy()
    config_dict.pop('do_test', None)
    config_dict.pop('dataset', None)
    config_dict.pop('output', None)

    return [dict(zip(config_dict, v)) for v in product(*config_dict.values())]


def main():
    config_dict = load_config('config.ini')

    # load dataset
    loader = DataLoader()
    loader.load('../data/' + config_dict['dataset'])
    X_train, X_test, y_train, y_test = loader.train_test_data()

    # create list of training parameters, if flag 'de_test' in config file is not set, then only one
    # training parameters set will be created
    training_parameters = create_train_params(config_dict)

    # keep training results with params, accuracy, f1 score and train time
    training_results = list()

    for i, tp in enumerate(training_parameters):
        print(f"Training with set {i + 1} / {len(training_parameters)}: {tp}")
        clf = RandomForestClassifier(n_estimators=tp['n_estimators'], max_depth=tp['max_depth'],
                                     min_samples_split=tp['min_samples_split'], min_samples_leaf=tp['min_samples_leaf'],
                                     criterion=tp['criterion'], threshold=tp['threshold'],
                                     split_method=tp['split_method'])

        # train model and measure training time
        start_time = time.time()
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time

        # predict labels from training set
        y_pred = clf.predict(X_test)

        metrics = Metrics(y_test, y_pred, train_time=training_time, classes=loader.classes, training_params=tp)
        training_results.append(metrics.calc_results())

        # metrics.save_metrics(path_to_folder='../results/', n_estimator=5, max_depth=10, criterion='gini',
        #                      split_method='roulette')

    # create DF with results and save it
    columns = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion', 'threshold',
               'split_method', 'accuracy', 'f1', 'train_time']
    df_res = pd.DataFrame(data=training_results, columns=columns)
    df_res['dataset'] = config_dict['dataset']
    df_res.to_csv('../data/' + config_dict['output'] + '.csv', index=False)

    # TODO wynmiki są za dobre...
    # TODO lepsze zapisywanie wyników zrobić
    # TODO trzeba przerobić kod, zeby nie było widać, ze z neta


if __name__ == "__main__":
    main()

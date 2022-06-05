# --------------------------------------------------
# Created By  : Krzysztof Palmi, Łukasz Sawicki 
# Created Date: 04.06.2022
# Class:        Machine Learning, 2022 summer
# --------------------------------------------------

from src.tools.classifiers import RandomForestClassifier
from src.tools.metrics import Metrics
from src.tools.data_loader import DataLoader, load_config
from src.tools.utilities import create_train_params, save_results
import time

# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/index.html
# http://staff.elka.pw.edu.pl/~rbiedrzy/UMA/turniejRuletka.pdf


def main():
    config_dict = load_config('config.ini')

    # load dataset
    loader = DataLoader()
    loader.load('../data/' + config_dict['dataset'])
    X_train, X_test, y_train, y_test = loader.train_test_data()

    # create list of training parameters, if flag 'do_test' in config file is not set, then only one
    # training parameters set will be created
    training_parameters = create_train_params(config_dict)

    # keep training results with params, accuracy, f1 score and train time
    training_results = list()

    # store best model (based on accuracy) for given set of parameters
    best_model = {'model': None, 'acc': -1.0, 'train_params': None, 'metrics': None}

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

        new_score = metrics.accuracy()
        if new_score > best_model['acc']:
            best_model['model'] = clf
            best_model['acc'] = new_score
            best_model['train_params'] = tp
            best_model['metrics'] = metrics

    # save results
    save_results(config_dict=config_dict, training_results=training_results, best_model=best_model)


if __name__ == "__main__":
    main()

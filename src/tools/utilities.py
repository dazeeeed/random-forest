# --------------------------------------------------
# Created By  : Krzysztof Palmi, Łukasz Sawicki
# Created Date: 04.06.2022
# Class:        Machine Learning, 2022 summer
# --------------------------------------------------

import pandas as pd
from itertools import product


def save_results(config_dict: dict, training_results: list, best_model: dict):
    # create DF with results and save it
    columns = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion', 'threshold',
               'split_method', 'accuracy', 'f1', 'train_time']

    df_res = pd.DataFrame(data=training_results, columns=columns)
    df_res['dataset'] = config_dict['dataset']
    df_res.to_csv('../results/' + config_dict['output'] + '.csv', index=False)

    # save info abut best model
    with open(f"../results/{config_dict['output']}.txt", 'w') as f:
        f.write(f"Best model: {best_model['train_params']}\n"
                f"Accuracy: {best_model['acc']}")

    # save best model's confusion matrix
    best_metric = best_model['metrics']
    best_metric.save_metrics(filename=f"../results/best_{config_dict['output']}.jpg")


def create_train_params(config_dict: dict):
    """Creates all possible training parameters"""
    config_dict = config_dict.copy()
    config_dict.pop('do_test', None)
    config_dict.pop('dataset', None)
    config_dict.pop('output', None)

    return [dict(zip(config_dict, v)) for v in product(*config_dict.values())]

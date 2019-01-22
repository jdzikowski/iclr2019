import os
import os.path as osp
import argparse
import json
import hashlib
from collections import namedtuple
from cross_validation import cross_validation
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True,
                            help='Path to config JSON file')
parser.add_argument('--results_path', type=str, required=True,
                            help='Path to directory where results should be stored')
args = parser.parse_args()

def get_all_param_sets(hyper_params):
    param_list = [(key, hyper_params[key]) for key in hyper_params]
    param_sets = [{}]
    for param in hyper_params:
        new_param_sets = []
        options = hyper_params[param]
        for option in options:
            for param_set in param_sets:
                new_param_set = param_set.copy()
                new_param_set[param] = option
                new_param_sets.append(new_param_set)
        param_sets = new_param_sets

    return param_sets

def analyze_data(data_path):
    data = torch.load(data_path)
    train_histories = data['train_history']
    test_histories = data['test_history']
    #results = data['results']
    train_histories = np.array(train_histories)
    test_histories = np.array(test_histories)
    epoch_test_mean = test_histories.mean(axis=0)
    epoch_test_std = test_histories.std(axis=0)
    best_epoch = np.argmax(epoch_test_mean)
    print('Best epoch: %d, mean acc: %f, std: %f' 
        % (best_epoch + 1, epoch_test_mean[best_epoch], epoch_test_std[best_epoch]),'\n')


configs = []
with open(args.config_path) as json_file:
    config_data = json.load(json_file)
    models = config_data['models']
    datasets = config_data['datasets']
    base_config = {}
    for config_key in config_data:
        if config_key == 'models' or config_key == 'datasets':
            continue
        base_config[config_key] = config_data[config_key]

    for dataset in datasets:
        for model in models:    
            hyper_params = dataset['hypertuned_params']
            all_param_sets = get_all_param_sets(hyper_params)
            for param_set in all_param_sets:
                config = dataset.copy()
                config.pop('hypertuned_params', None)
                config = {**config, **base_config, **param_set, **model}
                configs.append(config)

if not osp.exists(args.results_path):
    os.mkdir(args.results_path)

for config in configs:
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    config_results_path = osp.join(args.results_path, model_name, dataset_name, config_hash)
    config['results_path'] = config_results_path
    os.makedirs(config_results_path, exist_ok=True)
    if not osp.exists(osp.join(config_results_path, 'done')):
        #print('Model %s - dataset %s: There are no results for config %s, skipping\n' 
        #    % (config['model_name'], config['dataset_name'], config_hash))
        continue
    print(config)
    data_path = osp.join(config_results_path, 'data.pth')
    analyze_data(data_path)
    

    
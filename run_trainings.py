import os
import os.path as osp
import argparse
import json
import hashlib
from collections import namedtuple
from cross_validation import cross_validation
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True,
                            help='Path to config JSON file')
parser.add_argument('--data_path', type=str, required=True,
                            help='Path to directory with datasets')
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

def save_results(results, config, path):
    avg, std, details = results
    print('Saving results to', path)
    with open(osp.join(path, 'config.json'), 'w') as config_file:
        config_file.write(json.dumps(config, sort_keys=True))
    with open(osp.join(path, 'results'), 'w') as results_file:
        print('Avg: %f, Std: %f' % (avg, std), file=results_file)
    data_path = osp.join(path, 'data.pth')
    torch.save(details, data_path)
    os.mknod(osp.join(path, 'done'))
    os.remove(osp.join(path, 'checkpoint'))

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

    for model in models:
        for dataset in datasets:
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
    print(config)
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    config_results_path = osp.join(args.results_path, model_name, dataset_name, config_hash)
    config['data_path'] = args.data_path
    config['results_path'] = config_results_path
    os.makedirs(config_results_path, exist_ok=True)
    if osp.exists(osp.join(config_results_path, 'done')):
        print('Model %s - dataset %s: There is already a result for config %s, skipping\n' 
            % (config['model_name'], config['dataset_name'], config_hash))
        continue

    checkpoint = None
    checkpoint_path = osp.join(config_results_path, 'checkpoint')
    if osp.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    results = cross_validation(config, checkpoint=checkpoint)
    avg, std, details = results
    print('Model %s - dataset %s: %f +- %f' % (config['model_name'], config['dataset_name'], avg, std))
    save_results(results, config, config_results_path)
    print('\n')

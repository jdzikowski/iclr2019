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

"""
parser.add_argument('--caption_root', type=str, required=True,
                            help='root directory that contains captions')
parser.add_argument('--trainclasses_file', type=str, required=True,
                            help='text file that contains training classes')
parser.add_argument('--save_filename_G', type=str, required=True,
                            help='checkpoint file of generator')
parser.add_argument('--save_filename_D', type=str, required=True,
                            help='checkpoint file of discriminator')
parser.add_argument('--log_interval', type=int, default=10,
                            help='the number of iterations (default: 10)')
parser.add_argument('--num_threads', type=int, default=8,
                            help='number of threads for fetching data (default: 8)')
parser.add_argument('--num_epochs', type=int, default=600,
                            help='number of threads for fetching data (default: 600)')
parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                            help='learning rate (dafault: 0.0002)')
parser.add_argument('--lr_decay', type=float, default=0.5,
                            help='learning rate decay (dafault: 0.5)')
parser.add_argument('--momentum', type=float, default=0.5,
                            help='beta1 for Adam optimizer (dafault: 0.5)')
parser.add_argument('--lambda_cond_loss', type=float, default=10,
                            help='lambda of conditional loss (default: 10)')
parser.add_argument('--lambda_recon_loss', type=float, default=0.2,
                            help='lambda of reconstruction loss (default: 0.2)')
parser.add_argument('--no_cuda', action='store_true',
                            help='do not use cuda')
"""
args = parser.parse_args()

def get_all_param_sets(hypere_params):
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

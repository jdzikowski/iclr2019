import os
import argparse
import json
from collections import namedtuple
from cross_validation import cross_validation

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True,
                            help='Path to config JSON file')
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


with open(args.config_path) as json_file:
    #configs = json.load(json_file, object_hook=lambda d: namedtuple('Config', d.keys())(*d.values()))
    config_data = json.load(json_file)

    #print(config_data)
    models = config_data['models']
    datasets = config_data['datasets']
    base_config = {}
    for config_key in config_data:
        if config_key == 'models' or config_key == 'datasets':
            continue
        base_config[config_key] = config_data[config_key]
    
    configs = []
    #print('MODELS:')
    for model in models:
        #print(model)
        #print('DATASETS:')
        for dataset in datasets:
            #print(dataset)
            hyper_params = dataset['hypertuned_params']
            all_param_sets = get_all_param_sets(hyper_params)
            for param_set in all_param_sets:
                config = dataset.copy()
                config.pop('hypertuned_params', None)
                config = {**config, **base_config, **param_set, **model}
                configs.append(config)

    for config in configs:
        print(config, '\n')
        avg, std, details = cross_validation(config)
        print('Model %s - dataset %s: %f +- %f' % (config['model_name'], config['dataset_name'], avg, std))

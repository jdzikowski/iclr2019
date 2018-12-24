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

with open(args.config_path) as json_file:
    #configs = json.load(json_file, object_hook=lambda d: namedtuple('Config', d.keys())(*d.values()))
    configs = json.load(json_file)
    for config in configs:
        print(config)
        avg, std = cross_validation(config)
        print('Result on %s: %f +- %f' % (config['dataset_name'], avg, std))


import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add
from data import NodeDegreeFeatureDataLoader, SameFeatureDataLoader
from train import train

def prepare_config_for_dataset(config, dataset):
    if config['node_features'] == 'node_degree':
        dataset_loader = DataLoader(dataset, batch_size=config['batch_size'])
        max_node_degree = 0
        for data in dataset_loader:
            edge_index, _ = remove_self_loops(data.edge_index)
            row, col = edge_index
            out = scatter_add(torch.ones(len(data.batch))[col], row, dim=0, dim_size=len(data.batch))
            max_node_degree = max(max_node_degree, out.max())
        config['max_node_degree'] = max_node_degree
        config['num_features'] = max_node_degree + 1
    elif config['node_features'] == 'categorical':
        dataset_loader = DataLoader(dataset, batch_size=config['batch_size'])
        data = next(iter(dataset_loader))
        config['num_features'] = data.x.shape[1]
    elif config['node_features'] == 'same':
        config['num_features'] = 1
    
    if dataset.num_classes is not None and dataset.num_classes != config['num_classes']:
        print('Config num classes %d doesn\'t match dataset\'s num classes %d!' % (config['num_classes'], dataset.num_classes))
        config['num_classes'] = dataset.num_classes

def cross_validation(config, checkpoint=None):
    dataset_name = config['dataset_name']
    dataset_path = osp.join(config['data_path'], dataset_name)
    dataset = TUDataset(dataset_path, name=dataset_name)
    if checkpoint is None:
        dataset_permutation = torch.randperm(len(dataset))
    else:
        dataset_permutation = checkpoint['dataset_permutation']
    dataset = dataset[dataset_permutation]

    prepare_config_for_dataset(config, dataset)

    cross_validation_batches = config['cross_validation_batches']
    cross_validation_batch_size = len(dataset) // cross_validation_batches
    results = [] if checkpoint is None else checkpoint['results']
    train_histories = [] if checkpoint is None else checkpoint['train_histories']
    test_histories = [] if checkpoint is None else checkpoint['test_histories']
    start_cross_validation_batch = 0 if checkpoint is None else checkpoint['current_cross_validation_batch']
    for i in range(start_cross_validation_batch, cross_validation_batches):
        #save_cross_validation_progress(config, i, results, train_histories, test_histories)
        print('%s cross validation batch %d/%d' % ('Starting' if checkpoint is None else 'Continuing', i+1, cross_validation_batches))
        start_index = i * cross_validation_batch_size
        end_index = (i + 1) * cross_validation_batch_size if i + 1 < cross_validation_batches else len(dataset)
        test_dataset = dataset[start_index:end_index]     
        if start_index == 0:
            train_dataset = dataset[end_index:]
        elif end_index == len(dataset):
            train_dataset = dataset[:start_index] 
        else:
            train_dataset = dataset[:start_index] + dataset[end_index:]

        if config['node_features'] == 'categorical':
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
        elif config['node_features'] == 'node_degree':
            test_loader = NodeDegreeFeatureDataLoader(test_dataset, config['max_node_degree'], batch_size=config['batch_size'])
            train_loader = NodeDegreeFeatureDataLoader(train_dataset, config['max_node_degree'], batch_size=config['batch_size'])
        elif config['node_features'] == 'same':
            test_loader = SameFeatureDataLoader(test_dataset, batch_size=config['batch_size'])
            train_loader = SameFeatureDataLoader(train_dataset, batch_size=config['batch_size'])

        save_data = {
            'dataset_permutation' : dataset_permutation,
            'results' : results,
            'train_histories' : train_histories,
            'test_histories' : test_histories,
            'current_cross_validation_batch' : i
        }
        if i > start_cross_validation_batch:
            checkpoint = None

        train_history, test_history = train(config, train_loader, test_loader, save_data, checkpoint=checkpoint) 
        train_histories.append(train_history)
        test_histories.append(test_history)
        best_epoch = np.argmax(test_history) + 1
        results.append(test_history[best_epoch - 1])
        print('Cross validation batch %d/%d: %f in epoch %d' % (i+1, cross_validation_batches, results[-1], best_epoch))

    avg = np.mean(results)
    std = np.std(results)
    details = {
        'results' : results,
        'train_history' : train_histories,
        'test_history' : test_histories
    }
    return avg, std, details

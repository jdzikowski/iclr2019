import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops

class NodeDegreeFeatureDataLoader(DataLoader):
    def __init__(self, dataset, max_node_degree, batch_size=1, shuffle=True, **kwargs):
        super().__init__(dataset, batch_size, shuffle, **kwargs)
        self.max_node_degree = max_node_degree
 
    def __iter__(self):
        self.base_iterator = super().__iter__()
        return self
    
    def __next__(self):
        data = next(self.base_iterator)
        #print(data)
        if data.x is None:
            data.x = torch.zeros((len(data.batch), self.max_node_degree + 1))
            node_degrees = self.get_node_degrees(data)
            data.x[:, node_degrees] = 1
        return data
                                         
    def get_node_degrees(self, data):
        edge_index, _ = remove_self_loops(data.edge_index)
        row, col = edge_index
        return scatter_add(torch.ones(len(data.batch))[col], row, dim=0, dim_size=len(data.batch)).long()

class SameFeatureDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super().__init__(dataset, batch_size, shuffle, **kwargs)
 
    def __iter__(self):
        self.base_iterator = super().__iter__()
        return self
    
    def __next__(self):
        data = next(self.base_iterator)
        #print(data)
        if data.x is None:
            data.x = torch.ones(len(data.batch))
        return data

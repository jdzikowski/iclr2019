import torch
import torch.nn.functional as F
from torch.nn import Sequential, ModuleList, Linear, ReLU, BatchNorm1d, Dropout, LogSoftmax
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool
from torch_scatter import scatter_add, scatter_max, scatter_mean
from torch_geometric.utils import add_self_loops, remove_self_loops

class GNNConv_Variant(GINConv):
    def __init__(self, nn, aggregate_func, eps=0, train_eps=False):
        super().__init__(nn, eps, train_eps)
        self.aggregate_func = aggregate_func
        self.train_eps = train_eps
    
    # This is almost identical to GINConv's implementation
    # GIN (Sum) is (1 + eps) * x + out
    # GNN Max is max(x, out), therefore we can include x in scatter_max
    # GNN Mean is analogous to above - we include x in scatter_mean
    def forward(self, x, edge_index):
        # The below line is used to turn 1D features into 2D.
        # Each node now has a vector of size 1 instead of just 1 number
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # Ensuring every node has exactly one self-loop
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        # Sum, Mean or Max aggregation
        out = self.aggregate_func(x[col], row, dim=0, dim_size=x.size(0))
        # scatter_max returns a tuple instead of a single value
        if isinstance(out, tuple):
            out = out[0]
        # For sum aggregation we have to add epsilon times node's feature vector
        # Assuming eps is 0 in case of Max and Mean aggregation
        if self.eps != 0 or self.train_eps:
            out = self.eps * x + out
        # Feeding aggregated node features through MLP
        out = self.nn(out)
        return out


class GNN_Variant(torch.nn.Module):
    def __init__(self, aggregation_op, readout_op, num_aggregation_layers, mlp_num_layers, 
                 num_features, num_classes, dim=32, eps=0, train_eps=False, dropout_rate=0.5):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.num_aggregation_layers = num_aggregation_layers
        self.aggregators = ModuleList()
        
        if aggregation_op == 'sum':
            aggregate_func = scatter_add    
        elif aggregation_op == 'mean':
            aggregate_func = scatter_mean
        elif aggregation_op == 'max':
            aggregate_func = scatter_max
        else:
            raise Exception('Invalid aggregation op %s' % aggregation_op)

        for k in range(num_aggregation_layers):
            mlp_layer = []
            for i in range(mlp_num_layers):
                input_dim = num_features if k == 0 and i == 0 else dim
                output_dim = dim
                mlp_layer.extend([Linear(input_dim, output_dim),
                                  Dropout(self.dropout_rate), 
                                  ReLU(), 
                                  BatchNorm1d(output_dim)])
            mlp = Sequential(*mlp_layer)           
            self.aggregators.append(GNNConv_Variant(mlp, aggregate_func, eps=eps, train_eps=train_eps))

        if readout_op == 'sum':
            self.readout = global_add_pool
        elif readout_op == 'mean':
            self.readout = global_mean_pool
        elif readout_op == 'max':
            self.readout = global_max_pool
        else:
            raise Exception('Invalid readout op %s' % readout_op)

        self.classifier = Sequential(
            Linear(num_features + num_aggregation_layers * dim, num_features + num_aggregation_layers * dim),
            ReLU(),
            Dropout(self.dropout_rate),
            Linear(num_features + num_aggregation_layers * dim, num_classes),
            LogSoftmax(dim=-1)
        )

    def forward(self, x, edge_index, batch):
        layer_readouts = [self.readout(x, batch)]
        for k in range(self.num_aggregation_layers):
            x = self.aggregators[k](x, edge_index)
            layer_readouts.append(self.readout(x, batch))
        x = torch.cat(layer_readouts, dim=1)
        return self.classifier(x)
import torch
import torch.nn.functional as F
from model import GIN_MLP, GNN_Variant

def train(config, training_set, test_set):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device used is:', device)
    #model = GIN_MLP(config['num_features'], config['num_classes'], config['hidden_layer_dim']).to(device)
    model = GNN_Variant(config['aggregation_op'], 
                        config['readout_op'], 
                        config['num_aggregation_layers'], 
                        config['mlp_num_layers'], 
                        config['num_features'], 
                        config['num_classes'], 
                        dim=config['hidden_layer_dim'], 
                        eps=config['epsilon'], 
                        train_eps=config['train_epsilon'],
                        dropout_rate=config['dropout_rate']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_history = []
    test_history = []
    num_epochs = config['num_epochs']
    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(model, epoch, training_set, optimizer, device)
        train_acc = eval(model, training_set, device)
        test_acc = eval(model, test_set, device)
        train_history.append(train_acc)
        test_history.append(test_acc)
        if epoch % 50 == 0:
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss, train_acc, test_acc))
    return train_history, test_history


def training_epoch(model, epoch, dataset_loader, optimizer, device):
    model.train()

    if epoch % 50 == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in dataset_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(dataset_loader.dataset)


def eval(model, dataset_loader, device):
    model.eval()
    correct = 0
    for data in dataset_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(dataset_loader.dataset)



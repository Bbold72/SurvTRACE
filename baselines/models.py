import torch 


def simple_dln(config):
    hidden_size = config.hidden_size
    dropout = config.dropout

    net = torch.nn.Sequential(
        torch.nn.Linear(config.num_feature, hidden_size),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(hidden_size),
        torch.nn.Dropout(dropout),
        
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(hidden_size),
        torch.nn.Dropout(dropout),
        
        torch.nn.Linear(hidden_size, config.out_feature)
    )
    return net
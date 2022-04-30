# deep neural networks that are passed to some models from pycox

import torch 
import torchtuples as tt # Some useful functions


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


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, config, batch_norm=True):
        dropout = config.dropout
        num_nodes_shared = config.hidden_size_shared

        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            config.num_feature, 
            num_nodes_shared, 
            num_nodes_shared,
            batch_norm, 
            dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(config.num_event):
            net = tt.practical.MLPVanilla(
                num_nodes_shared, 
                config.hidden_size_indiv, 
                config.out_feature,
                batch_norm, 
                dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out
import torch
import torch.nn as nn
class FCNN(nn.Module):
    def __init__(self, config, input_size, device, output_size=2):
        super().__init__()
        input_sizes = [input_size] + config.hidden_layers
        output_sizes = config.hidden_layers + [output_size]
        m = []
        if hasattr(config, 'input_dropout'):
            m.append(nn.Dropout(config.input_dropout))
        for i in range(len(input_sizes)):
            m.append(nn.Linear(input_sizes[i], output_sizes[i]))
            if i != len(config.hidden_layers):
                m.append(nn.ReLU())
                if hasattr(config, 'dropout'):
                    m.append(nn.Dropout(config.dropout))

        self.linear_relu_stack = nn.Sequential(*m)

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
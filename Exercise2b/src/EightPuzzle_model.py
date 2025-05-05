import torch
import torch.nn as nn

class EightPuzzleMLP(nn.Module):
    def __init__(self, config):
        super(EightPuzzleMLP, self).__init__()
        model_params = config['model_params']
        input_size = model_params['input_size']
        hidden_layers = model_params['hidden_layers']
        output_size = 9 * 9  

        activation = model_params['activation']
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(0.5))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.flatten = nn.Flatten()
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        out = self.network(x)
        out = out.view(-1, 9, 9)  
        return out

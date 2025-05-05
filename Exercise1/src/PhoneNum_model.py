import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class PhoneNumberMLP(nn.Module):
    """
    Multi-Layer Perceptron for phone number digit recognition
    """
    def __init__(self, config):
        super(PhoneNumberMLP, self).__init__()
        
        
        self.input_size = config['model']['input_size']  
        self.hidden_sizes = config['model']['hidden_sizes']
        self.num_digits = config['model']['num_digits']  
        self.num_classes = config['model']['num_classes']  
        self.dropout_rate = config['model']['dropout_rate']
        
        
        self.flatten = nn.Flatten()
        
       
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        
        conv_output_size = 128 * 7 * 70
        
        
        mlp_layers = []
        prev_size = conv_output_size
        
        for hidden_size in self.hidden_sizes:
            mlp_layers.append(nn.Linear(prev_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.BatchNorm1d(hidden_size))  
            if self.dropout_rate > 0:
                mlp_layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size
        
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        
        self.digit_classifiers = nn.ModuleList([
            nn.Linear(self.hidden_sizes[-1], self.num_classes)
            for _ in range(self.num_digits)
        ])
    
    def forward(self, x):
        
        
        x = self.conv_layers(x)
        x = self.flatten(x)
        
        
        features = self.mlp_layers(x)
        
       
        digit_outputs = [classifier(features) for classifier in self.digit_classifiers]
        
        return digit_outputs


def load_model_from_config(config_path):
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    
    model = PhoneNumberMLP(config)
    
    return model, config


def save_model(model, save_path, config=None):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    
    torch.save(model.state_dict(), save_path)
    
    
    if config:
        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
    
    print(f"Model saved to {save_path}")


def load_model(model_path, config_path=None):
    """
    Load a saved model from path
    """
    
    if not config_path:
        config_path = model_path.replace('.pth', '_config.json')
    
    
    model, config = load_model_from_config(config_path)
    
    
    model.load_state_dict(torch.load(model_path))
    
    return model, config


if __name__ == "__main__":
   
    example_config = {
        "model": {
            "input_size": 1 * 28 * 280,  
            "hidden_sizes": [512, 256],
            "num_digits": 10,
            "num_classes": 10,  
            "dropout_rate": 0.3
        }
    }
    
   
    os.makedirs('../src/config', exist_ok=True)
    with open('../src/config/improved_config.json', 'w') as f:
        json.dump(example_config, f, indent=4)
    
    
    model = PhoneNumberMLP(example_config)
    print(model)
    
    
    example_input = torch.randn(32, 1, 28, 280)  
    outputs = model(example_input)
    
    print(f"Number of output tensors: {len(outputs)}")
    print(f"Shape of first output tensor: {outputs[0].shape}")  
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """
    A simple MLP for synthetic data or flattened image data.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=2):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # Ensure input is flattened
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)
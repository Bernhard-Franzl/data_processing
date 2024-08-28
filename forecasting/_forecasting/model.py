import torch
import torch.nn as nn

class OccupancyDenseNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        
        x = x.view(-1, self.input_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class Model(nn.Module):
    def __init__(self, action_dim, hidden_dim=256):
        super(Model, self).__init__()

        # CNN layers with a third layer added
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)


        # Pooling layer for additional downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(8 * 8 * 128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(self.weights_init)
    

    def forward(self, x):
        # X represents the observation. P represents player. 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # Pooling after third conv layer
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers with optional dropout
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        output = self.output(x)

        return output

    def save_the_model(self, filename='models/latest.pt'):
        torch.save(self.state_dict(), filename)

    def load_the_model(self, filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(filename))
            print(f"Loaded weights from {filename}")
        except FileNotFoundError:
            print(f"No weights file found at {filename}")

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# Helper Functions
def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

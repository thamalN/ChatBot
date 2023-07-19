import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, hidden_layers)
        self.linear3 = nn.Linear(hidden_layers, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out

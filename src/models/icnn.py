import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class I_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(I_CNN, self).__init__()
        # Define ReLU activation
        self.relu = nn.ReLU()

        # Use nn.ModuleList for layers to ensure they are registered as model parameters
        self.modules_x = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=True) for _ in range(num_layers)])
        self.modules_z = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Initialize zl as a zero tensor with the same device as x
        zl = torch.zeros(x.shape[0], self.modules_x[0].out_features, device=x.device)
        for i in range(len(self.modules_x)):
            self.modules_z[i].weight.data.clamp_(min=0)
            zl = self.relu(self.modules_z[i](zl) + self.modules_x[i](x))  # Apply ReLU activation
        return zl

    def _initialize_weights(self):
        # Loop through all layers and apply Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  # Apply Xavier uniform initialization
                if module.bias is not None:
                    init.zeros_(module.bias)  # Initialize biases to 0
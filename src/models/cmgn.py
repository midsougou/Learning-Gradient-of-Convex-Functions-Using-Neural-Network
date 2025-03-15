import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class C_MGN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation='sigmoid'):
        super(C_MGN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # the matrix params is shared across layers
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim))
        
        # biases b0 to bL
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_layers)])
        self.bL = nn.Parameter(torch.randn(output_dim))
        
        self.V = nn.Parameter(torch.randn(input_dim, output_dim))
        
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'softplus':
            self.activation = F.softplus
        else:
            raise ValueError("Activation function not supported.")
    
    def forward(self, x):
        # first layer
        z_prev = torch.matmul(x, self.W) + self.biases[0]
        
        for l in range(1, self.num_layers):
            z_l = torch.matmul(x, self.W) + self.activation(z_prev) + self.biases[l]
            z_prev = z_l
        inter_1 = torch.matmul(self.activation(z_prev), self.W.t()) # (batch_size, hidden_dim) * (hidden_dim, input_dim)
        # x@V (b, i) * (i, o) => (b, o)
        # x@V@V.T (b, o) * (o, i) = > (b, i)
        inter_2 = torch.matmul(torch.matmul(x, self.V), self.V.t()) 
        output = inter_1 + inter_2 + self.bL  # (batch_size, input_dim) +  (batch_size, input_dim) + (batch_size, input_dim)
        
        return output


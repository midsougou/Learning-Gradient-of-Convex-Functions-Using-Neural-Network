import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class M_MGN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(M_MGN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Define modules (W_k, b_k, and activation functions)
        self.W_k = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=True) for _ in range(num_layers)
        ])

        # Each module has its own activation (e.g., tanh, softplus)
        self.activations = nn.ModuleList([nn.Tanh() for _ in range(num_layers)])

        # V^T V term (PSD by construction)
        self.V = nn.Parameter(torch.randn(hidden_dim, input_dim))  # Shape: [hideen_dim, input_dim]

        # Bias term (a)
        self.a = nn.Parameter(torch.randn(output_dim))  # Learned bias


    def forward(self, x):
        batch_size = x.shape[0]

        # Initialize output with bias term (broadcasted to batch)
        out = self.a.unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, input_dim]

        # Add V^T V x term (ensures PSD Jacobian)
        V_sq = self.V.t() @ self.V  # Shape: [input_dim, input_dim]
        out = out + x@ V_sq  # Shape: [batch_size, input_dim]

        # Loop over modules and compute terms
        for k in range(self.num_layers):
            # Compute z_k = W_k x + b_k
            z_k = self.W_k[k](x)  # Shape: [batch_size, hidden_dim]
            # Compute s_k(z_k) = sum_i log(cosh(z_k_i)) (scalar per sample)
            s_k = torch.sum(torch.log(torch.cosh(z_k)), dim=1)  # Shape: [batch_size]

            # Compute activation σ_k(z_k)
            sigma_k = self.activations[k](z_k)  # Shape: [batch_size, hidden_dim]

            # Compute s_k(z_k) * W_k^T σ_k(z_k)
            W_k_T = self.W_k[k].weight.t()  # Shape: [input_dim, hidden_dim]
            term = (W_k_T @ sigma_k.t()).t()  # Shape: [batch_size, input_dim]
            term = s_k.unsqueeze(-1) * term  # Broadcast s_k and multiply

            out += term

        return out  # Shape: [batch_size, input_dim]

    def logcosh(self, x):
        return torch.log(torch.cosh(x))


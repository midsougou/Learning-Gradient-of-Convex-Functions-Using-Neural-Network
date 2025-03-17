import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Callable


class Trainer:
    """General-purpose trainer for M-MGN supporting multiple tasks."""
    def __init__(
        self,
        task: str = 'gradient',  # 'gradient' or 'optimal_transport'
        dataset: str = '2D_distribution',
        n_epochs: int = 50,
        lr: float = 0.01,
        criterion: str = 'L1loss',
        optimizer: str = 'Adam',
        weight_decay: float = 0,
        betas: Tuple[float, float] = (0.9, 0.999),
        model: nn.Module = None,
        model_name: Optional[str] = None,
        true_fx: Optional[Callable] = None,
        batch_size: int = 32,
        device: str = 'cpu',
        verbose: bool = True
    ):
        self.task = task
        self.dataset = dataset
        self.device = device
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.lr = lr
        self.batch_size = batch_size
        self.metrics = {
            'train_loss': [], 'val_loss': [], 'train_cost': [], 'val_cost': []
        }

        # Initialize model
        self.model = model.to(device)
        self.model_name = model_name

        # Initialize the function to approximate (if applicable)
        self.true_fx = true_fx

        # Initialize optimizer and loss
        self.optimizer = self._get_optimizer(optimizer=optimizer, weight_decay=weight_decay, betas=betas)
        self.criterion = self._get_criterion(criterion=criterion)

    def _get_optimizer(self, optimizer: str, weight_decay: float, betas: Tuple[float, float]):
        """Initialize the optimizer."""
        if optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.")

    def _get_criterion(self, criterion: str):
        """Initialize the criterion."""
        if criterion.lower() == 'l1loss':
            return nn.L1Loss()
        elif criterion.lower() in ['kld', 'nll']:
            return lambda x: self._custom_loss(x)
        elif criterion.lower() == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Criterion '{criterion}' is not supported.")
    def _get_grad(self, x):
        
        if self.model_name.lower() == 'icnn':
            # Compute gradients
            outputs = self.model(x.squeeze(0))
            grad = torch.autograd.grad(
                outputs=outputs,
                inputs=x,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True
            )[0]  # Extract gradient tensor

            return grad
        return self.model
    
    def _custom_loss(self, x_batch):
        """Custom loss for optimal transport (NLL with Jacobian)."""
        x_batch.requires_grad_(True)
        g_x = self._get_grad(x_batch)  # Forward pass

        # Compute Jacobian for each sample in the batch
        batch_size = x_batch.shape[0]
        log_det = torch.zeros(batch_size, device=x_batch.device)

        for i in range(batch_size):
            J = torch.autograd.functional.jacobian(lambda x: self._get_grad(x), x_batch[i], create_graph=True)
            det = torch.det(J)
            log_det[i] = torch.log(det.abs() + 1e-6)  # Avoid log(0)

        # Gaussian log-likelihood
        log_p = -0.5 * (g_x ** 2).sum(dim=1) - torch.log(torch.tensor(2 * torch.pi))

        # Total loss
        return - (log_p + log_det).mean()

    def compute_transport_cost(self, x, g_x):
        """Compute Brenier's transport cost: E[||x - g(x)||^2]."""
        return torch.mean(torch.sum((x - g_x) ** 2, dim=1))

    def compute_jacobian(self, x):
        """Compute the Jacobian of g(x) w.r.t. x."""
        batch_size = x.shape[0]
        jacobian = torch.zeros(batch_size, x.shape[1], x.shape[1], device=x.device)
        for i in range(batch_size):
            jacobian[i] = torch.autograd.functional.jacobian(lambda x: self.model(x), x[i].unsqueeze(0))
        return jacobian

    def evaluate(self, loader: DataLoader, split: str = 'val') -> Tuple[float, float]:
        """Generic evaluation method."""
        self.model.eval()
        total_loss = total_cost = total_samples = 0

        with torch.no_grad():
            for batch in loader:
                x = batch[0]  # Input data
                if self.task == 'gradient':
                    true_grad = batch[1]  # Ground truth gradient
                else:
                    true_grad = None

                x.requires_grad = True
                # Forward pass
                g_x = self.model(x)

                # Compute loss
                if self.task == 'gradient':
                    loss = self.criterion(g_x, true_grad)
                else:  # Optimal transport
                    loss = self.criterion(x)

                # Compute transport cost
                cost = self.compute_transport_cost(x, g_x)

                # Accumulate metrics
                total_loss += loss.item() * x.shape[0]
                total_cost += cost.item() * x.shape[0]
                total_samples += x.shape[0]

        return total_loss / total_samples, total_cost / total_samples

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop."""
        best_val_loss = float('inf')
        for epoch in (pbar := tqdm(range(self.n_epochs), disable=not self.verbose)):
            self.model.train()
            train_loss = train_cost = 0

            for batch in train_loader:
                x = batch[0]  # Input data
                if self.task == 'gradient':
                    true_grad = batch[1]  # Ground truth gradient
                else:
                    true_grad = None

                # Forward pass
                g_x = self.model(x)
                
                # Compute loss
                if self.task == 'gradient':
                    loss = self.criterion(g_x, true_grad)
                else:  # Optimal transport
                    loss = self.criterion(x)

                # Compute transport cost
                cost = self.compute_transport_cost(x, g_x)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                train_loss += loss.item() * x.shape[0]
                train_cost += cost.item() * x.shape[0]

            # Evaluate on validation set (if provided)
            val_loss, val_cost = self.evaluate(val_loader) if val_loader else (0, 0)

            # Update metrics
            self.metrics['train_loss'].append(train_loss / len(train_loader.dataset))
            self.metrics['train_cost'].append(train_cost / len(train_loader.dataset))
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_cost'].append(val_cost)

            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1} | "
                                f"Train Loss: {train_loss / len(train_loader.dataset):.4f} | "
                                f"Train Cost: {train_cost / len(train_loader.dataset):.4f} | "
                                f"Val Loss: {val_loss:.4f} | "
                                f"Val Cost: {val_cost:.4f}")

    def plot_train_metrics(self, plot_cost=False):
        """Plot training and validation metrics using Plotly."""
        if not self.metrics['train_loss'] or not self.metrics['val_loss']:
            raise ValueError("Training metrics are empty. Ensure the model has been trained.")
        cols = 2 if plot_cost else 1
        # Create subplots
        fig = make_subplots(rows=1, cols=cols, subplot_titles=(
            "Train vs Validation Loss", "Train vs Validation Cost"
        ))

        # Add training and validation loss plot
        fig.add_trace(
            go.Scatter(x=list(range(len(self.metrics['train_loss']))), y=self.metrics['train_loss'], name="Train Loss", mode="lines"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(self.metrics['val_loss']))), y=self.metrics['val_loss'], name="Validation Loss", mode="lines"),
            row=1, col=1
        )

        # Add training and validation cost plot
        if plot_cost:
            fig.add_trace(
                go.Scatter(x=list(range(len(self.metrics['train_cost']))), y=self.metrics['train_cost'], name="Train Cost", mode="lines"),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(range(len(self.metrics['val_cost']))), y=self.metrics['val_cost'], name="Validation Cost", mode="lines"),
                row=1, col=2
            )

        # Update layout
        fig.update_layout(
            title=f"Training Metrics for {self.dataset}",
            showlegend=True,
            width=1200 if plot_cost else 600,
            height=500
        )

        # Show the plot
        fig.show()
        
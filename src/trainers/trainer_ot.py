import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, mean_source, cov_source, target_samples, epochs, lr, batch_size, plot_interval, optimizer, plot=False):
        self.model = model
        self.mean_source = mean_source
        self.cov_source = cov_source
        self.target_samples = target_samples
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.plot_interval = plot_interval
        self.optimizer = optimizer
        self.plot = plot
        self.source_samples = self.generate_source_data()

    def generate_source_data(self):
        L = torch.linalg.cholesky(self.cov_source)
        target_samples = torch.randn(self.target_samples.shape[0], 2)
        return target_samples @ L.T + self.mean_source


    def _custom_loss(self, x_batch):
        """Custom loss for optimal transport (NLL with Jacobian)."""
        x_batch.requires_grad_(True)

        # Compute gradients for the entire batch at once
        g_x = self._get_grad(x_batch)  # Forward pass
        
        # Compute Jacobian for the entire batch
        batch_size = x_batch.shape[0]

        # The jacobian function works on batches, so we can pass the entire batch at once
        jacobian = torch.autograd.functional.jacobian(self._get_grad, x_batch)  # Shape: [batch_size, batch_size, hidden_dim]

        # Compute the determinant for each sample in the batch
        log_det = torch.zeros(batch_size, device=x_batch.device)

        for i in range(batch_size):
            # For each sample, compute the SVD of the Jacobian matrix
            J = jacobian[i]  # Jacobian of the i-th sample (shape: [batch_size, hidden_dim])
            
            # Compute the singular values using SVD
            U, S, V = torch.svd(J)  # U, S, V are the SVD outputs
            
            # The log determinant can be approximated by the sum of the log of the singular values
            log_det[i] = torch.sum(torch.log(S.abs() + 1e-6))  # Avoid log(0)


        # Gaussian log-likelihood (NLL)
        log_p = -0.5 * (g_x ** 2).sum(dim=1) - torch.log(torch.tensor(2 * torch.pi))

        # Total loss (negative log-likelihood + log determinant term)
        return - (log_p + log_det).mean()

    def _get_grad(self, x_batch):
        """Calculate the gradient of the model output w.r.t. the input."""
        return self.model(x_batch)

    def wasserstein_distance(self, source, target):
        mu_s, mu_t = source.mean(dim=0), target.mean(dim=0)
        sigma_s, sigma_t = torch.cov(source.T), torch.cov(target.T)
        cost = torch.norm(mu_s - mu_t) ** 2 + torch.trace(sigma_s + sigma_t - 2 * (sigma_s @ sigma_t).sqrt())
        return cost.item()

    def plot_transport_evolution(self, num_samples_to_plot=1000):
        self.model.eval()
        source_samples = self.generate_source_data()
        transformed_samples_evol = self.model(source_samples)
        return transformed_samples_evol

    def train(self):
        costs_theorical = []
        transformed_samples_history = {}
        stored_costs = []

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            idx = torch.randint(0, self.source_samples.shape[0], (self.batch_size,))
            x_batch = self.source_samples[idx]

            # Using custom loss function
            loss = self._custom_loss(x_batch)
            
            # Compute the Wasserstein distance for the current transformation
            transformed_samples = self._get_grad(x_batch)  # Assuming the transformation is done in the model
            cost_theory = self.wasserstein_distance(transformed_samples, self.target_samples)
            costs_theorical.append(cost_theory)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Storing costs
            stored_costs.append(cost_theory)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Custom Loss: {loss.item():.4f}")
                transformed_samples_evol = self.plot_transport_evolution()
                transformed_samples_history[epoch] = transformed_samples_evol.detach().cpu().numpy()

            if epoch % self.plot_interval == 0 and self.plot:
                plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.scatter(transformed_samples_evol[:, 0].detach(), transformed_samples_evol[:, 1].detach(),
                            c=transformed_samples_evol[:, 0].detach(), cmap="rainbow", alpha=0.6, label=f"Epoch {epoch}")
                plt.xlim(-7, 7)
                plt.ylim(-7, 7)
                plt.title(f"Transformed Samples Over Time : cost {cost_theory}")
                plt.legend(loc="upper right", fontsize=8)

                plt.subplot(1, 2, 2)
                plt.scatter(self.target_samples[:, 0], self.target_samples[:, 1],
                            c=self.target_samples[:, 0], cmap="rainbow", alpha=0.6, label="Target Distribution")
                plt.xlim(-7, 7)
                plt.ylim(-7, 7)
                plt.title("Target Distribution")
                plt.legend(loc="upper right", fontsize=8)
                plt.show()

        return costs_theorical, stored_costs, transformed_samples_history
    
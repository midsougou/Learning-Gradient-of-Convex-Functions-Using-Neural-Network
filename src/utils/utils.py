import torch.distributions as dist
import torch

def kl_divergence_loss(model, source_samples):
    """
    returns back the Kullback-Leibler divergence and also the transformed samples returned by the model
    """
    transformed_samples = model(source_samples) # source one are N(µ, A.T@A)
    p_target = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))  # Standard normal
    q_transformed = dist.MultivariateNormal(transformed_samples.mean(dim=0), torch.cov(transformed_samples.T))
    
    return torch.distributions.kl_divergence(p_target, q_transformed).mean(), transformed_samples

def generate_source_data(mean_source, cov_source, num_samples):
    L = torch.linalg.cholesky(cov_source)
    target_samples = torch.randn(num_samples, 2)
    source_samples = target_samples @ L.T + mean_source
    return source_samples
def wasserstein_distance(source, target):
    mu_s, mu_t = source.mean(dim=0), target.mean(dim=0)
    sigma_s, sigma_t = torch.cov(source.T), torch.cov(target.T)
    cost = torch.norm(mu_s - mu_t) ** 2 + torch.trace(sigma_s + sigma_t - 2 * (sigma_s @ sigma_t).sqrt())
    return cost.item()


def plot_transport_evolution(model, num_samples_to_plot=1000):
    model.eval()
    source_samples = generate_source_data(mean_source, cov_source, num_samples=num_samples_to_plot)
    # target_samples_evol = torch.randn(num_samples_to_plot, 2) # N(0, 1)
    transformed_samples_evol = model(source_samples)
    return transformed_samples_evol


def train_optimal_coupling(model, epochs, lr, batch_size, plot_interval, input_dim, output_dim, optimizer, plot=False):
    costs_theorical = []
    transformed_samples_history = {}
    stored_costs = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # we sample batch from the source distribution (N(µ, A.T@A))
        idx = torch.randint(0, source_samples.shape[0], (batch_size,))
        x_batch = source_samples[idx]

        # Compute KL loss
        loss, transformed_samples = kl_divergence_loss(model, x_batch)
        cost_theory = wasserstein_distance(transformed_samples, target_samples)
        costs_theorical.append(cost_theory)
        # distance btw two samples : sqrt(sum(xi - f(xi)^2)) (here only two dimensions (the sum))
        # we do the average across all the points of the current batch
        cost = torch.mean(torch.sqrt(torch.sum((transformed_samples - x_batch) ** 2, axis=1)))
        

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, KL Loss: {loss.item():.4f}")
            transformed_samples_evol = plot_transport_evolution(model, num_samples_to_plot=1000)
            transformed_samples_history[epoch] = transformed_samples_evol.detach().cpu().numpy() 
            stored_costs.append(cost.item())

        if epoch % plot_interval == 0 and plot==True:   
            plt.figure(figsize=(16,8)) 
            plt.subplot(1,2,1)
            plt.scatter(transformed_samples_evol[:,0].detach(), transformed_samples_evol[:,1].detach(), 
                        c=transformed_samples_evol[:,0].detach(), cmap="rainbow", alpha=0.6, label=f"Epoch {epoch}")
            plt.xlim(-7, 7)
            plt.ylim(-7, 7)
            plt.title(f"Transformed Samples Over Time : cost {cost}")
            plt.legend(loc="upper right", fontsize=8)  # Keep track of epochs

            plt.subplot(1,2,2)
            plt.scatter(target_samples[:,0], target_samples[:,1], 
                    c=target_samples[:,0], cmap="rainbow", alpha=0.6, label="Target Distribution")
            plt.xlim(-7, 7)
            plt.ylim(-7, 7)
            plt.title("Target Distribution")
            plt.legend(loc="upper right", fontsize=8)
            plt.show()
    return costs_theorical, stored_costs, transformed_samples_history
import torch
import matplotlib.pyplot as plt
import numpy as np


def verify_psd(func, model, batch_size=10000):

    batch = func(torch.rand(size=(batch_size, 2)))

    input_dim = batch.shape[1]

    grad_f_x = model(batch)  

    # Calculate the Jacobian for each sample in the batch
    jacobians = [torch.autograd.functional.jacobian(model, batch[i].unsqueeze(0)) for i in range(batch.shape[0])]

    # Reshape each Jacobian to be a square matrix
    jacobians = [j.reshape(input_dim, input_dim) for j in jacobians]

    # Check PSD for each Jacobian in the batch
    for jacobian in jacobians:
        if not torch.all(torch.linalg.eigvalsh(jacobian) >= -1e-6): return 'Jacobian is not PSD' 
    
    return 'Jacobian is PSD'


def plot_gradient_field(gradient, model, model_name='ICNN', grid_size=20):
    # Generate a grid of points for visualization
    x1 = torch.linspace(0, 1, grid_size)
    x2 = torch.linspace(0, 1, grid_size)
    x1_grid, x2_grid = torch.meshgrid(x1, x2)
    x_grid = torch.stack([x1_grid.flatten(), x2_grid.flatten()], dim=1)

    # Compute true and predicted gradients on the grid
    true_grad_grid = gradient(x_grid)
    with torch.no_grad():
        pred_grad_grid = model(x_grid)

    # Reshape for visualization
    true_grad_grid = true_grad_grid.reshape(grid_size, grid_size, 2)
    pred_grad_grid = pred_grad_grid.reshape(grid_size, grid_size, 2)

    # Plot true gradient field
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.quiver(x1_grid, x2_grid, true_grad_grid[:, :, 0], true_grad_grid[:, :, 1], color='blue')
    plt.title("True Gradient Field")
    plt.xlabel("x1")
    plt.ylabel("x2")

    # Plot predicted gradient field
    plt.subplot(1, 2, 2)
    plt.quiver(x1_grid, x2_grid, pred_grad_grid[:, :, 0], pred_grad_grid[:, :, 1], color='red')
    plt.title(f"Predicted Gradient Field wit model {model_name}")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.show()

def plot_error_maps(gradients, target, labels, device, xi_res=100, yi_res=100):
    # Create meshgrid
    xi = torch.linspace(0, 1, xi_res)
    yi = torch.linspace(0, 1, yi_res)
    Xi, Yi = torch.meshgrid(xi, yi, indexing="ij")
    space = torch.cat([Xi.reshape(-1, 1), Yi.reshape(-1, 1)], dim=1)

    # Compute target (ensure device consistency)
    with torch.no_grad():
        targ = target(space)
    print("Target output shape:", targ.shape)

    # Initialize figure
    n_plots = len(gradients)
    plt.rcParams['figure.figsize'] = (6 * n_plots, 10)
    fig, ax = plt.subplots(1, n_plots, figsize=(6 * n_plots, 10))
    if n_plots == 1:
        ax = [ax]

    max_error = 0
    error_maps = []

    # Compute error maps
    for grad in gradients:
        output = grad(space) # Ensure grad handles device correctly
        # Ensure tensors are on CPU and convert to NumPy
        error_map = (targ - output).norm(dim=1).reshape(Xi.shape).T
        error_map = error_map.to(device).detach().numpy()  # Convert to NumPy
        error_maps.append(error_map)
        max_error = max(max_error, error_map.max())

    # Plotting
    xi_np = xi.numpy()
    yi_np = yi.numpy()
    print('xi_np')
    for i, (error_map, label) in enumerate(zip(error_maps, labels)):
        contour = ax[i].contourf(xi_np, yi_np, error_map, levels=10, cmap="RdYlGn_r", vmax=max_error)
        ax[i].set_title(label, fontsize=20)
        ax[i].tick_params(axis='both', which='major', labelsize=15)

    # Colorbar
    cbar = fig.colorbar(contour, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize=15)
    
'''
def plot_error_maps(gradients, target, labels, xi_res=100, yi_res=100):
    """
    Plots error maps for multiple gradients side by side.
    
    Args:
        gradients (list): List of gradient models or outputs to evaluate.
        space (torch.Tensor): Input space (e.g., 2D grid) to compute errors.
        target (torch.Tensor): Target values for comparison.
        labels (list): List of strings for the subplot titles.
        xi_res (int): Resolution of the xi axis for the meshgrid.
        yi_res (int): Resolution of the yi axis for the meshgrid.
    """
    # Create meshgrid
    xi = torch.linspace(0, 1, xi_res)
    yi = torch.linspace(0, 1, yi_res)
    Xi, Yi = torch.meshgrid(xi, yi, indexing="ij")
    space = torch.cat([torch.reshape(Xi, (-1, 1)), torch.reshape(Yi, (-1, 1))], dim=1)

    # Compute the target
    targ = target(space)
    print("Target output shape:", targ.shape)

    # Initialize the figure with subplots
    n_plots = len(gradients)
    plt.rcParams['figure.figsize'] = (6 * n_plots, 10)
    fig, ax = plt.subplots(1, n_plots, figsize=(6 * n_plots, 10))

    if n_plots == 1:  # Handle single plot case
        ax = [ax]

    max_error = 0  # Determine global max error for consistent color scale
    error_maps = []

    # Compute error maps for each gradient
    for grad in gradients:
        output = grad(space).detach()  # Get model output
        print("Model output shape:", output.shape)
        error_map = (targ - output).norm(dim=1).reshape(Xi.shape).T
        error_maps.append(error_map)
        max_error = max(max_error, error_map.max().item())

    # Plot each gradient's error map
    for i, (error_map, label) in enumerate(zip(error_maps, labels)):
        contour = ax[i].contourf(xi, yi, error_map, levels=50, cmap="RdYlGn_r", vmax=max_error)
        ax[i].set_title(label, fontsize=20)
        ax[i].tick_params(axis='both', which='major', labelsize=15)

    # Add a shared colorbar
    cbar = fig.colorbar(contour, ax=ax, orientation="vertical", shrink=0.9)
    cbar.ax.tick_params(labelsize=15)
    
    plt.show()


# Example usage
#xi = torch.linspace(0, 1, 100)
#yi = torch.linspace(0, 1, 100)
#Xi, Yi = torch.meshgrid(xi, yi)
#space = torch.cat([torch.reshape(Xi, (-1, 1)), torch.reshape(Yi, (-1, 1))], 1)

'''
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def get_appropriate_colormap(num_categories):
    """Get appropriate colormap based on number of categories."""
    if num_categories <= 10:
        return plt.get_cmap('tab10', num_categories)
    elif num_categories <= 20:
        return plt.get_cmap('tab20', num_categories)
    else:
        # For many categories, use a continuous colormap
        return plt.get_cmap('viridis', num_categories)

def get_category_names_from_dataloader(dataloader):
    """Extract category names from dataloader if available (e.g., QuickDraw dataset)."""
    try:
        dataset = dataloader.dataset
        # Handle CombinedDataset
        if hasattr(dataset, 'datasets'):
            # For combined dataset, check if QuickDraw is included
            if 'quickdraw' in dataset.datasets:
                quickdraw_dataset = dataset.datasets['quickdraw']
                if hasattr(quickdraw_dataset, 'categories'):
                    # Create a mapping for the combined dataset
                    category_names = {}
                    current_offset = 0
                    for name, sub_dataset in dataset.datasets.items():
                        if name == 'quickdraw':
                            for i, cat_name in enumerate(quickdraw_dataset.categories):
                                category_names[current_offset + i] = cat_name
                            current_offset += len(quickdraw_dataset.categories)
                        else:
                            # Standard datasets (MNIST, Fashion, CIFAR) have 10 classes with numeric labels
                            for i in range(10):
                                if name == 'mnist':
                                    category_names[current_offset + i] = f"MNIST_{i}"
                                elif name == 'fashion_mnist':
                                    category_names[current_offset + i] = f"Fashion_{i}"
                                elif name == 'cifar':
                                    category_names[current_offset + i] = f"CIFAR_{i}"
                                else:
                                    category_names[current_offset + i] = f"{name}_{i}"
                            current_offset += 10
                    return category_names
        # Handle QuickDraw dataset directly
        elif hasattr(dataset, 'categories'):
            return {i: name for i, name in enumerate(dataset.categories)}
        # Handle RotatedMNIST or other wrapped datasets
        elif hasattr(dataset, 'base_dataset') and hasattr(dataset.base_dataset, 'categories'):
            return {i: name for i, name in enumerate(dataset.base_dataset.categories)}
    except Exception as e:
        print(f"Could not extract category names: {e}")
    return None

# New function to compute, save and load latent embeddings for visualization
def compute_latent_embeddings(model, dataloader, device, output_dir, force_recompute=False):
    """
    Compute, save and potentially load latent embeddings and their 2D projections for visualization.
    
    Args:
        model: The VAE model
        dataloader: DataLoader containing the test dataset
        device: Device to run computation on
        output_dir: Directory to save/load embeddings
        force_recompute: Whether to force recomputation even if embeddings exist
        
    Returns:
        Dictionary containing:
            - latent_representations: Original latent vectors
            - latent_2d: 2D projections (original or via t-SNE)
            - plot_labels: Labels for coloring points
            - selected_experts: Expert indices for each sample
            - base_title: Plot title depending on embedding method
            - xlabel: X-axis label
            - ylabel: Y-axis label
    """
    # Create file paths for saved embeddings
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    latent_path = os.path.join(embeddings_dir, 'latent_vectors.npy')
    latent_2d_path = os.path.join(embeddings_dir, 'latent_2d.npy')
    labels_path = os.path.join(embeddings_dir, 'labels.npy')
    experts_path = os.path.join(embeddings_dir, 'selected_experts.npy')
    
    # Check if saved embeddings exist and we're not forcing recomputation
    if not force_recompute and os.path.exists(latent_path) and os.path.exists(latent_2d_path) and \
       os.path.exists(labels_path) and os.path.exists(experts_path):
        print("Loading precomputed embeddings...")
        latent_representations = np.load(latent_path)
        latent_2d = np.load(latent_2d_path)
        plot_labels = np.load(labels_path)
        selected_experts = np.load(experts_path)
    else:
        print("Computing latent embeddings...")
        model.eval()
        latent_representations = []
        plot_labels = []
        selected_experts = []

        with torch.no_grad():
            for data_batch, labels_batch in dataloader:
                data_batch = data_batch.to(device)
                
                # Get latent vectors
                mu, log_var = model.encoder(data_batch)
                z = model.reparameterize(mu, log_var)
                
                # Get logits from the gating network
                logits = model.gating_network(z)
                
                # Determine the selected expert for each sample using argmax
                top_expert_indices = torch.argmax(logits, dim=-1)
                
                # Store data
                latent_representations.append(z.cpu().numpy())
                plot_labels.append(labels_batch.cpu().numpy())
                selected_experts.append(top_expert_indices.cpu().numpy())

        # Concatenate all data
        latent_representations = np.concatenate(latent_representations, axis=0)
        plot_labels = np.concatenate(plot_labels, axis=0)
        selected_experts = np.concatenate(selected_experts, axis=0)
        
        # Subsample vectors for quicker t-SNE (max 10,000)
        max_samples = 10000
        if latent_representations.shape[0] > max_samples:
            indices = np.random.choice(latent_representations.shape[0], max_samples, replace=False)
            latent_representations = latent_representations[indices]
            plot_labels = plot_labels[indices]
            selected_experts = selected_experts[indices]
        
        # Calculate 2D embeddings based on latent dimension
        if model.latent_dim > 2:
            print("Performing t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, 
                       perplexity=min(30, len(latent_representations)-1), 
                       n_iter=300)
            latent_2d = tsne.fit_transform(latent_representations)
            base_title = 't-SNE Latent Space Visualization'
            xlabel = 't-SNE Dimension 1'
            ylabel = 't-SNE Dimension 2'
        elif model.latent_dim == 2:
            latent_2d = latent_representations
            base_title = '2D Latent Space Visualization'
            xlabel = 'Latent Dimension 1'
            ylabel = 'Latent Dimension 2'
        else:  # latent_dim == 1
            latent_2d = np.stack([
                latent_representations[:, 0], 
                np.random.randn(latent_representations.shape[0]) * 0.1
            ], axis=1)
            base_title = '1D Latent Space Visualization'
            xlabel = 'Latent Dimension 1'
            ylabel = ''
        
        # Save computed embeddings
        np.save(latent_path, latent_representations)
        np.save(latent_2d_path, latent_2d)
        np.save(labels_path, plot_labels)
        np.save(experts_path, selected_experts)
    
    return {
        'latent_representations': latent_representations,
        'latent_2d': latent_2d,
        'plot_labels': plot_labels,
        'selected_experts': selected_experts,
        'base_title': 't-SNE Latent Space Visualization' if model.latent_dim > 2 else 
                     ('2D Latent Space Visualization' if model.latent_dim == 2 else '1D Latent Space Visualization'),
        'xlabel': 't-SNE Dimension 1' if model.latent_dim > 2 else 'Latent Dimension 1',
        'ylabel': 't-SNE Dimension 2' if model.latent_dim > 2 else ('Latent Dimension 2' if model.latent_dim == 2 else '')
    }

# Function to visualize latent space (colors by true class label or rotation)
def visualize_latent_space(model, dataloader, device, save_path='tsne_latent_space_moe.png', output_dir=None, force_recompute=False):
    # Determine output directory for embeddings
    if output_dir is None:
        output_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    
    # Compute or load embeddings
    embeddings = compute_latent_embeddings(model, dataloader, device, output_dir, force_recompute)
    
    # Extract visualization data
    latent_2d = embeddings['latent_2d']
    plot_labels_class = embeddings['plot_labels']
    base_title = embeddings['base_title']
    xlabel = embeddings['xlabel']
    ylabel = embeddings['ylabel']
    
    # Get category names if available
    category_names = get_category_names_from_dataloader(dataloader)
    
    # Determine number of unique classes
    unique_classes = np.unique(plot_labels_class)
    num_classes = len(unique_classes)
    
    # Get appropriate colormap based on number of classes
    cmap = get_appropriate_colormap(num_classes)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    title = f'{base_title} (Colored by Class)'
    
    # Create scatter plot
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=plot_labels_class, 
                         cmap=cmap, s=5, alpha=0.7)
    
    # Create colorbar or legend based on number of classes and availability of category names
    if category_names and num_classes <= 20:
        # Use discrete legend with category names for small number of classes
        handles = []
        for class_idx in unique_classes:
            if class_idx in category_names:
                # Create a handle for each class
                class_mask = plot_labels_class == class_idx
                if np.any(class_mask):
                    color = cmap(class_idx / max(unique_classes) if len(unique_classes) > 1 else 0)
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8, 
                                            label=category_names[class_idx]))
        
        if handles:
            plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Use colorbar for many classes or when category names are not available
        cbar = plt.colorbar(scatter, label='Class')
        if category_names and num_classes <= 10:
            # Set tick labels to category names if available and few enough
            cbar.set_ticks(unique_classes)
            cbar.set_ticklabels([category_names.get(i, str(i)) for i in unique_classes])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Latent space visualization saved to {save_path}")
    plt.close()


# Function to visualize expert activation in latent space
def visualize_expert_activation_space(model, dataloader, device, save_path='expert_activation_space.png', output_dir=None, force_recompute=False):
    # Determine output directory for embeddings
    if output_dir is None:
        output_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    
    # Compute or load embeddings
    embeddings = compute_latent_embeddings(model, dataloader, device, output_dir, force_recompute)
    
    # Extract visualization data
    latent_2d = embeddings['latent_2d']
    selected_experts = embeddings['selected_experts']
    base_title = embeddings['base_title']
    xlabel = embeddings['xlabel']
    ylabel = embeddings['ylabel']
    
    # Create title specific to expert visualization
    title = f'{base_title} (Colored by Selected Expert)'

    # Create a scatter plot, coloring by selected expert index
    plt.figure(figsize=(10, 8))
    # Use a colormap suitable for categorical data that works with any number of experts
    if model.num_experts <= 10:
        cmap = plt.get_cmap('tab10', model.num_experts)
    elif model.num_experts <= 20:
        cmap = plt.get_cmap('tab20', model.num_experts)
    else:
        # For more than 20 experts, use a continuous colormap
        cmap = plt.get_cmap('viridis', model.num_experts)
    
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                          c=selected_experts, cmap=cmap, s=5, alpha=0.7)

    # Create a colorbar with discrete ticks for each expert
    cbar = plt.colorbar(scatter, ticks=np.arange(model.num_experts))
    cbar.set_label('Selected Expert Index')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"Expert activation visualization saved to {save_path}")
    plt.close()


# Function to visualize reconstructions
def visualize_reconstructions(model, dataloader, device, num_samples=10, save_path='reconstructions_moe.png'):
    model.eval()

    # Get a batch of data
    data, labels = next(iter(dataloader))
    data, labels = data[:num_samples].to(device), labels[:num_samples]

    # Get category names if available
    category_names = get_category_names_from_dataloader(dataloader)

    # Get reconstructions (model is in eval mode)
    with torch.no_grad():
        # Model returns: reconstructed_x, mu, log_var, logits, expert_selection_one_hot
        reconstructed_x, _, _, logits, _ = model(data)
        # Get the most likely expert for visualization purposes (optional)
        most_likely_expert = torch.argmax(logits, dim=-1)

    # Plot original and reconstructed images
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        # Get label text (category name if available, otherwise numeric label)
        label_text = category_names.get(labels[i].item(), str(labels[i].item())) if category_names else str(labels[i].item())
        
        # Original image
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(data[i].cpu().squeeze().numpy(), cmap='gray')
        plt.title(f"Orig: {label_text}")
        plt.axis('off')

        # Reconstructed image
        plt.subplot(2, num_samples, i + num_samples + 1)
        plt.imshow(reconstructed_x[i].cpu().squeeze().numpy(), cmap='gray')
        # Optionally show the dominant expert for this reconstruction
        plt.title(f"Recon (Exp: {most_likely_expert[i].item()})")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reconstructions visualization saved to {save_path}")
    plt.close()

def visualize_expert_correlation(model, dataloader, device, num_experts, save_path='expert_correlation.png'):
    """
    Visualize which experts are activated for which class labels.
    Creates a grid of histograms showing the distribution of class labels for each expert.
    Automatically adjusts the grid layout based on the number of experts.
    Also calculates and reports:
    - Mutual information between experts and class labels
    - Correlation between different experts
    """
    selected_experts = []
    class_labels = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.cpu().numpy()
            
            # Get the model output (model is in eval mode)
            # Returns: reconstructed_x, mu, log_var, logits, expert_selection_output
            _, _, _, logits, _ = model(data)
            
            # Get the selected expert for each sample
            top_expert_indices = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Store selected experts and corresponding class labels
            selected_experts.append(top_expert_indices)
            class_labels.append(labels)
            
    # Concatenate all data
    selected_experts = np.concatenate(selected_experts, axis=0)
    class_labels = np.concatenate(class_labels, axis=0)
    
    # Get category names if available
    category_names = get_category_names_from_dataloader(dataloader)
    
    # Calculate optimal grid dimensions based on number of experts
    if num_experts <= 3:
        n_cols = num_experts
        n_rows = 1
    else:
        n_cols = min(5, num_experts)  # Maximum 5 columns
        n_rows = (num_experts + n_cols - 1) // n_cols  # Ceiling division
    
    # Create a grid of histograms, one for each expert
    fig_width = min(3 * n_cols, 15)  # Scale width based on columns, max 15
    fig_height = min(3 * n_rows, 12)  # Scale height based on rows, max 12
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    
    # Handle the case where there's only one expert (axes won't be an array)
    if num_experts == 1:
        axes = np.array([axes])
    
    # Flatten axes for easy iteration
    axes = np.array(axes).flatten()
    
    # Set a common title for the entire figure
    fig.suptitle('Expert Assignment Counts per Class Label', fontsize=16)
    
    # Define colors for each expert (use a colormap that works well for any number of experts)
    colors = plt.cm.tab10(np.linspace(0, 1, num_experts)) if num_experts <= 10 else plt.cm.viridis(np.linspace(0, 1, num_experts))
    
    # Create a histogram for each expert
    for expert_idx in range(num_experts):
        ax = axes[expert_idx]
        
        # Get indices where this expert was selected
        expert_mask = (selected_experts == expert_idx)
        
        # Get the class labels for these samples
        expert_class_labels = class_labels[expert_mask]
        
        # Create histogram - adjust bins based on the range of labels
        # For combined dataset (MNIST+FashionMNIST+CIFAR10), we need 30 bins (0-29)
        max_label = np.max(class_labels) if len(class_labels) > 0 else 9
        num_bins = max_label + 2  # +2 for left and right edges
        
        if len(expert_class_labels) > 0:  # Only plot if there are samples
            counts, bins, _ = ax.hist(
                expert_class_labels, 
                bins=np.arange(num_bins) - 0.5,  # Create bins centered on integers
                alpha=0.7, 
                color=colors[expert_idx]
            )
            
            # Set title and labels
            ax.set_title(f'Expert {expert_idx}')
            ax.set_xlabel('Class Label')
            ax.set_ylabel('Count')
            
            # Set x-ticks based on the range of labels
            unique_labels = np.unique(class_labels)
            if max_label > 20:
                # Show fewer ticks for readability when there are many classes
                tick_indices = np.arange(0, max_label+1, 5)
                ax.set_xticks(tick_indices)
                if category_names:
                    # Show category names for selected ticks
                    tick_labels = [category_names.get(i, str(i)) for i in tick_indices]
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            else:
                ax.set_xticks(unique_labels)
                if category_names and len(unique_labels) <= 15:
                    # Show category names for all ticks if there aren't too many
                    tick_labels = [category_names.get(i, str(i)) for i in unique_labels]
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
                
            ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(num_experts, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for the suptitle
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Expert correlation visualization saved to {save_path}")
    plt.close()


def visualize_expert_frequencies(model, dataloader, device, num_experts, save_path='expert_histogram.png'):
    selected_experts = []
    lb_losses = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            # Get the model output (model is in eval mode)
            # Returns: reconstructed_x, mu, log_var, logits, expert_selection_output
            reconstructed_x, mu, log_var, logits, expert_selection_output = model(data)

            # Store selected experts for histogram (use argmax on logits, as model is in eval mode)
            top_expert_indices = torch.argmax(logits, dim=-1)
            top_expert_indices_np = top_expert_indices.cpu().numpy()
            selected_experts.append(top_expert_indices_np)
    selected_experts = np.concatenate(selected_experts, axis=0)
    # histogram of selected experts
    plt.figure(figsize=(10, 6))
    plt.hist(selected_experts, bins=np.arange(num_experts + 1) - 0.5, density=True, alpha=0.7, color='blue')
    plt.xticks(np.arange(num_experts))
    plt.xlabel('Expert Index')
    plt.ylabel('Probability')
    plt.title('Histogram of Selected Experts')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def visualize_pretrained_model(model_dir, output_dir=None, download_data=False):
    """
    Load a pretrained VAE mixture of experts model and perform all visualizations.
    
    Args:
        model_dir (str): Path to the directory containing the model and hyperparams.txt
        output_dir (str, optional): Directory to save visualizations. If None, uses model_dir.
        download_data (bool): Whether to download dataset files if they're not already available.
                             Set to False to avoid downloading large datasets.
    
    Returns:
        model: The loaded model
    """
    # If no output directory specified, use the model directory
    if output_dir is None:
        output_dir = model_dir
    
    # Check if directories exist
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse hyperparameters from hyperparams.txt
    hyperparams_path = os.path.join(model_dir, 'hyperparams.txt')
    if not os.path.exists(hyperparams_path):
        raise ValueError(f"Hyperparameters file {hyperparams_path} does not exist")
    
    # Load hyperparameters
    hyperparams = {}
    with open(hyperparams_path, 'r') as f:
        for line in f:
            if '//' in line:  # Skip comment lines
                continue
            if '=' in line:
                key, value = line.strip().split('=')
                if value.isdigit():
                    hyperparams[key] = int(value)
                elif value.replace('.', '').isdigit():
                    hyperparams[key] = float(value)
                else:
                    hyperparams[key] = value
    
    # Extract necessary parameters
    latent_dim = hyperparams.get('latent_dim', 32)
    num_experts = hyperparams.get('num_experts', 5)
    batch_size = hyperparams.get('batch_size', 128)
    dataset_name = hyperparams.get('dataset', 'mnist')
    model_size = hyperparams.get('model_size', 'lite')
    encoder_size = hyperparams.get('encoder_size', 'medium')
    
    print(f"Loaded hyperparameters: latent_dim={latent_dim}, num_experts={num_experts}, dataset={dataset_name}, model_size={model_size}")
    
    # Import necessary modules
    import sys
    sys.path.append("/home/stnikoli/semestral-project-awareness/vae")
    from vae_mixture_of_experts import VAE_MixtureOfExperts
    from datasets import QuickDrawImageDataset, CombinedDataset
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model 
    model = VAE_MixtureOfExperts(latent_dim, num_experts, model_size=model_size, encoder_size=encoder_size).to(device)
    
    # Find the final model or latest checkpoint
    model_path = os.path.join(model_dir, 'vae_moe_model_final.pth')
    if not os.path.exists(model_path):
        # Look for checkpoint with highest epoch number
        checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('vae_moe_model_epoch_') and f.endswith('.pth')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model_path = os.path.join(model_dir, checkpoint_files[-1])
        else:
            raise ValueError(f"No model checkpoint found in {model_dir}")
    
    # Load model state
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model state from {model_path}")
        model.eval()  # Set model to evaluation mode
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Set up data transforms
    base_transform = transforms.ToTensor()
    
    # Load the appropriate dataset based on the model's training dataset
    if dataset_name == 'quickdraw':
        print("Loading QuickDraw dataset for evaluation...")
        test_dataset = QuickDrawImageDataset(root='./data/quickdraw', train=False, download=True, 
                                           transform=base_transform, samples_per_category=5000)
    elif dataset_name == 'mnist':
        print("Loading MNIST dataset for evaluation...")
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)
    elif dataset_name == 'fashion':
        print("Loading FashionMNIST dataset for evaluation...")
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=base_transform)
    elif dataset_name == 'cifar':
        print("Loading CIFAR10 dataset for evaluation...")
        cifar_transform = transforms.Compose([
            transforms.Grayscale(),             # Convert RGB to grayscale
            transforms.Resize((28, 28)),        # Resize to 28x28
            transforms.ToTensor()               # Convert to tensor
        ])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
    elif dataset_name == 'combined':
        print("Loading Combined dataset for evaluation...")
        # Load MNIST
        MNIST_Test = datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)
        
        # Load FashionMNIST
        FashionMNIST_Test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=base_transform)
        
        # Load CIFAR10
        cifar_transform = transforms.Compose([
            transforms.Grayscale(),             # Convert RGB to grayscale
            transforms.Resize((28, 28)),        # Resize to 28x28
            transforms.ToTensor()               # Convert to tensor
        ])
        CIFAR10_Test = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
        
        # Load QuickDraw
        QuickDraw_Test = QuickDrawImageDataset(root='./data/quickdraw', train=False, download=True, 
                                             transform=base_transform, samples_per_category=2000)
        
        # Create combined dataset
        datasets_test = {'mnist': MNIST_Test, 'fashion_mnist': FashionMNIST_Test, 
                        'quickdraw': QuickDraw_Test, 'cifar': CIFAR10_Test}
        test_dataset = CombinedDataset(datasets_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Generate visualizations in the output directory
    print("Generating visualizations...")
    
    # Visualize latent space
    latent_plot_path = os.path.join(output_dir, 'tsne_latent_space_classs_moe.png')
    visualize_latent_space(model, test_dataloader, device, save_path=latent_plot_path, output_dir=output_dir)
    
    # Visualize reconstructions
    recon_plot_path = os.path.join(output_dir, 'reconstructions_moe.png')
    visualize_reconstructions(model, test_dataloader, device, save_path=recon_plot_path)
    
    # Visualize expert activation space - reuse embeddings from latent space visualization
    expert_activation_plot_path = os.path.join(output_dir, 'expert_activation_space.png')
    visualize_expert_activation_space(model, test_dataloader, device, save_path=expert_activation_plot_path, output_dir=output_dir)
    
    # Visualize expert correlation with class labels
    expert_correlation_plot_path = os.path.join(output_dir, 'expert_correlation.png')
    visualize_expert_correlation(model, test_dataloader, device, num_experts, save_path=expert_correlation_plot_path)
    
    # Visualize expert frequencies
    histogram_plot_path = os.path.join(output_dir, 'expert_histogram.png')
    visualize_expert_frequencies(model, test_dataloader, device, num_experts, save_path=histogram_plot_path)
    
    # Visualize expert specialization
    expert_specialization_plot_path = os.path.join(output_dir, 'expert_specialization.png')
    visualize_expert_specialization(model, test_dataloader, device, num_experts, save_path=expert_specialization_plot_path)

    # Visualize latent interpolation
    latent_interpolation_plot_path = os.path.join(output_dir, 'latent_interpolation.png')
    visualize_latent_interpolation(model, device, output_dir, num_points=20, save_path=latent_interpolation_plot_path)
    
    # Evaluate linear classification performance
    print("\n" + "="*50)
    print("EVALUATING LINEAR CLASSIFICATION PERFORMANCE")
    print("="*50)
    
    # Compute embeddings to get latent vectors and expert selections
    embeddings = compute_latent_embeddings(model, test_dataloader, device, output_dir, force_recompute=False)
    
    # Extract data for classification evaluation
    latent_vectors_path = os.path.join(output_dir, 'embeddings', 'latent_vectors.npy')
    all_labels = embeddings['plot_labels']
    selected_experts = embeddings['selected_experts']
    
    # Evaluate linear classification
    evaluation_results = evaluate_linear_classification(
        latent_vectors_path=latent_vectors_path,
        labels=all_labels,
        selected_experts=selected_experts
    )
    
    if evaluation_results:
        print("\nLinear classification evaluation completed!")
    else:
        print("\nLinear classification evaluation failed!")
    
    print("Visualization complete.")
    return model

def visualize_expert_specialization(model, dataloader, device, num_experts, save_path='expert_specialization.png', 
                                    max_samples_per_expert=6, min_samples_threshold=10):
    """
    Visualize what types of images each expert specializes in by showing representative samples.
    
    This function creates a comprehensive view showing:
    - Sample images that each expert processes most confidently
    - Both original and reconstructed versions
    - Statistics about expert activation
    - Graceful handling of unused experts
    
    Args:
        model: The VAE mixture of experts model
        dataloader: DataLoader containing the test dataset  
        device: Device to run computation on
        num_experts: Number of experts in the model
        save_path: Path to save the visualization
        max_samples_per_expert: Maximum number of sample images to show per expert
        min_samples_threshold: Minimum number of samples needed for an expert to be considered "active"
    """
    model.eval()
    
    # Collect data for each expert
    expert_data = {i: {'images': [], 'labels': [], 'confidences': [], 'reconstructions': []} 
                   for i in range(num_experts)}
    expert_counts = np.zeros(num_experts)
    
    # Process data and collect samples for each expert
    print("Collecting expert specialization data...")
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.cpu().numpy()
            
            # Get model outputs
            reconstructed_x, _, _, logits, _ = model(data)
            
            # Get expert selection probabilities and most likely expert for each sample
            expert_probs = torch.softmax(logits, dim=-1)
            selected_experts = torch.argmax(logits, dim=-1).cpu().numpy()
            max_confidences = torch.max(expert_probs, dim=-1)[0].cpu().numpy()
            
            # Store samples for each expert
            for i in range(data.shape[0]):
                expert_idx = selected_experts[i]
                expert_counts[expert_idx] += 1
                
                # Only keep samples if we haven't collected enough yet
                if len(expert_data[expert_idx]['images']) < max_samples_per_expert * 3:  # Collect more to select best
                    expert_data[expert_idx]['images'].append(data[i].cpu())
                    expert_data[expert_idx]['labels'].append(labels[i])
                    expert_data[expert_idx]['confidences'].append(max_confidences[i])
                    expert_data[expert_idx]['reconstructions'].append(reconstructed_x[i].cpu())
            
            # Break early if we have enough samples for all active experts
            if batch_idx > 50:  # Process reasonable amount of data
                break
    
    # Select best samples for each expert based on confidence
    for expert_idx in range(num_experts):
        if len(expert_data[expert_idx]['images']) > max_samples_per_expert:
            # Sort by confidence and take the most confident samples
            sorted_indices = np.argsort(expert_data[expert_idx]['confidences'])[::-1]
            top_indices = sorted_indices[:max_samples_per_expert]
            
            expert_data[expert_idx]['images'] = [expert_data[expert_idx]['images'][i] for i in top_indices]
            expert_data[expert_idx]['labels'] = [expert_data[expert_idx]['labels'][i] for i in top_indices]
            expert_data[expert_idx]['confidences'] = [expert_data[expert_idx]['confidences'][i] for i in top_indices]
            expert_data[expert_idx]['reconstructions'] = [expert_data[expert_idx]['reconstructions'][i] for i in top_indices]
    
    # Get category names if available
    category_names = get_category_names_from_dataloader(dataloader)
    
    # Identify active experts (those with enough samples)
    active_experts = [i for i in range(num_experts) if expert_counts[i] >= min_samples_threshold]
    inactive_experts = [i for i in range(num_experts) if expert_counts[i] < min_samples_threshold]
    
    print(f"Active experts: {len(active_experts)}/{num_experts}")
    if inactive_experts:
        print(f"Inactive experts (< {min_samples_threshold} samples): {inactive_experts}")
    
    # Calculate grid layout for active experts
    if len(active_experts) == 0:
        print("No active experts found!")
        return
    
    # Determine optimal grid layout
    if len(active_experts) <= 2:
        n_cols = len(active_experts)
        n_rows = 1
    elif len(active_experts) <= 6:
        n_cols = min(3, len(active_experts))
        n_rows = (len(active_experts) + n_cols - 1) // n_cols
    elif len(active_experts) <= 12:
        n_cols = 4
        n_rows = (len(active_experts) + n_cols - 1) // n_cols
    else:
        n_cols = 5
        n_rows = (len(active_experts) + n_cols - 1) // n_cols
        # For very large numbers of experts, limit the display
        if n_rows > 10:
            print(f"Too many active experts ({len(active_experts)}), showing top 50 by usage")
            # Sort by usage count and take top experts
            expert_usage = [(i, expert_counts[i]) for i in active_experts]
            expert_usage.sort(key=lambda x: x[1], reverse=True)
            active_experts = [x[0] for x in expert_usage[:50]]
            n_rows = 10
    
    # Calculate figure size - now with 6 samples per expert
    samples_per_row = min(max_samples_per_expert, 6)  # Show up to 6 samples per expert
    fig_width = n_cols * (samples_per_row * 1.5 + 2)  # Adjust width for 6 samples + title space
    fig_height = n_rows * 4.5  # Increase height per expert row for better spacing
    fig_width = min(fig_width, 25)  # Maximum width
    fig_height = min(fig_height, 30)  # Maximum height
    
    # Create the plot with better spacing
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create main grid with more spacing for titles
    main_gs = fig.add_gridspec(n_rows, n_cols, hspace=0.6, wspace=0.25, 
                              top=0.93, bottom=0.05, left=0.02, right=0.98)
    
    for plot_idx, expert_idx in enumerate(active_experts[:n_rows * n_cols]):
        expert_images = expert_data[expert_idx]['images']
        expert_labels = expert_data[expert_idx]['labels']
        expert_recons = expert_data[expert_idx]['reconstructions']
        
        if not expert_images:
            continue
            
        # Determine number of samples to show for this expert
        num_samples_to_show = min(len(expert_images), samples_per_row)
        
        # Get position in grid
        expert_row = plot_idx // n_cols
        expert_col = plot_idx % n_cols
        
        # Create nested grid for this expert's samples (reserve space for title)
        expert_outer_gs = main_gs[expert_row, expert_col].subgridspec(3, 1, 
                                                                     height_ratios=[0.15, 1, 1],
                                                                     hspace=0.1)
        
        # Title area
        title_ax = fig.add_subplot(expert_outer_gs[0, 0])
        title_ax.axis('off')
        expert_title = f"Expert {expert_idx}"
        stats_text = f"({int(expert_counts[expert_idx])} samples, {expert_counts[expert_idx]/np.sum(expert_counts)*100:.1f}%)"
        title_ax.text(0.5, 0.7, expert_title, ha='center', va='center', 
                     fontsize=12, weight='bold', transform=title_ax.transAxes)
        title_ax.text(0.5, 0.2, stats_text, ha='center', va='center', 
                     fontsize=9, transform=title_ax.transAxes)
        
        # Images grid
        images_gs = expert_outer_gs[1:, 0].subgridspec(2, num_samples_to_show, 
                                                       hspace=0.15, wspace=0.08)
        
        # Show original and reconstructed images
        for sample_idx in range(num_samples_to_show):
            # Original image
            ax_orig = fig.add_subplot(images_gs[0, sample_idx])
            img = expert_images[sample_idx].squeeze().numpy()
            ax_orig.imshow(img, cmap='gray')
            ax_orig.axis('off')
            
            # Add class label for all original images
            label = expert_labels[sample_idx]
            label_text = category_names.get(label, str(label)) if category_names else str(label)
            # Truncate long labels
            if len(str(label_text)) > 8:
                label_text = str(label_text)[:8] + "..."
            ax_orig.text(0.5, -0.1, label_text, ha='center', va='top', 
                       fontsize=8, transform=ax_orig.transAxes)
            
            # Reconstructed image
            ax_recon = fig.add_subplot(images_gs[1, sample_idx])
            recon_img = expert_recons[sample_idx].squeeze().numpy()
            ax_recon.imshow(recon_img, cmap='gray')
            ax_recon.axis('off')
            
            # Add "Recon" label only on first image for clarity
            if sample_idx == 0:
                ax_recon.text(0.5, -0.1, "Recon", ha='center', va='top', 
                            fontsize=8, transform=ax_recon.transAxes)
    
    # Add overall title with better positioning
    plt.suptitle(f'Expert Specialization Visualization - {len(active_experts)}/{num_experts} Active Experts', 
                fontsize=16, y=0.97)
    
    # Add text about inactive experts if any
    if inactive_experts:
        inactive_text = f"Inactive experts (< {min_samples_threshold} samples): {', '.join(map(str, inactive_experts[:10]))}"
        if len(inactive_experts) > 10:
            inactive_text += f" and {len(inactive_experts) - 10} more..."
        plt.figtext(0.02, 0.02, inactive_text, fontsize=8, style='italic')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Expert specialization visualization saved to {save_path}")
    plt.close()

    experts_statistics =  {
        'expert_counts': list(expert_counts),
        'active_experts': list(active_experts),
        'inactive_experts': list(inactive_experts),
        'num_active': len(active_experts),
        'num_inactive': len(inactive_experts),
    }
    # extract dir path from save path
    output_dir = os.path.dirname(save_path)
    with open(os.path.join(output_dir, 'active_experts_summary.json'), 'w') as f:
        json.dump(experts_statistics, f, indent=4)
    # Return statistics for potential further analysis
    return experts_statistics

def visualize_latent_interpolation(model, device, output_dir, num_points=10, save_path='latent_interpolation.png', force_recompute=False):
    """
    Visualize interpolation between two points in the latent space that are farthest apart.
    Shows the decoded images and expert selection along the interpolation path.
    
    Args:
        model: The VAE mixture of experts model
        device: Device to run computation on
        output_dir: Directory to load precomputed embeddings from
        num_points: Number of interpolation points (including endpoints)
        save_path: Path to save the visualization
        force_recompute: Whether to force recomputation of embeddings
    """
    # Ensure the model is in eval mode
    from scipy.spatial.distance import pdist, squareform
    model.eval()
    
    # Load precomputed embeddings
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    latent_path = os.path.join(embeddings_dir, 'latent_vectors.npy')
    latent_2d_path = os.path.join(embeddings_dir, 'latent_2d.npy')
    selected_experts_path = os.path.join(embeddings_dir, 'selected_experts.npy') 
    
    # Check if we have the needed embeddings
    if not os.path.exists(latent_path) or not os.path.exists(latent_2d_path):
        raise ValueError(f"Required embeddings not found in {embeddings_dir}. Run compute_latent_embeddings first.")
    
    # Load the embeddings
    latent_representations = np.load(latent_path)
    latent_2d = np.load(latent_2d_path)
    selected_experts = np.load(selected_experts_path)
    
    # Find the two points that are furthest apart in the 2D t-SNE space
    print("Finding the two points with maximum distance in projected space...")
    max_distance = 0
    point_a_idx, point_b_idx = 0, 0
    # Efficiently compute pairwise distances using broadcasting

    # Compute condensed distance matrix (upper triangle)
    dists = pdist(latent_2d)
    max_idx = np.argmax(dists)
    # Convert condensed index to 2D indices
    point_a_idx, point_b_idx = np.unravel_index(np.argmax(squareform(dists)), squareform(dists).shape)
    max_distance = dists[max_idx]
    
    # Get the latent vectors corresponding to these points
    point_a = latent_representations[point_a_idx]
    point_b = latent_representations[point_b_idx]
    
    print(f"Found points with maximum distance: {max_distance:.2f}")
    print(f"Point A index: {point_a_idx}, Point B index: {point_b_idx}")
    
    # Generate interpolation points
    alphas = np.linspace(0, 1, num_points)
    interpolated_points = np.array([point_a * (1 - alpha) + point_b * alpha for alpha in alphas])

    
    # Decode the interpolated points and get expert assignments
    with torch.no_grad():
        # Convert to torch tensors
        interpolated_tensors = torch.tensor(interpolated_points, dtype=torch.float32).to(device)
        
        # Decode each point and get expert assignments
        decoded_images = []
        expert_indices = []
        expert_probs = []
        
        # Process each interpolated point
        for z in interpolated_tensors:
            # Add batch dimension
            z_batch = z.unsqueeze(0)
            
            # Get expert logits from gating network
            logits = model.gating_network(z_batch)
            
            # Get soft assignment (probabilities for each expert)
            probs = torch.softmax(logits, dim=-1)
            
            # Get the most likely expert
            top_expert_idx = torch.argmax(logits, dim=-1)
            
            # Decode the latent vector through the model
            # Note: the model's forward method handles decoding with all experts
            reconstructed_x = model.decoders[top_expert_idx](z_batch)
            # Store results
            decoded_images.append(reconstructed_x.squeeze().cpu().numpy())
            expert_indices.append(top_expert_idx.item())
            expert_probs.append(probs.squeeze().cpu().numpy())
    
    # Create visualization
    fig = plt.figure(figsize=(num_points * 1.5, 10))
    
    # Create a colormap for experts
    if model.num_experts <= 10:
        cmap = plt.get_cmap('tab10', model.num_experts)
    else:
        cmap = plt.get_cmap('viridis', model.num_experts)
    
    # Plot settings
    plt.suptitle(f"Latent Space Interpolation Between Points {point_a_idx} and {point_b_idx}", fontsize=16)
    
    # Create a more balanced grid - make top row take half the height
    gs = fig.add_gridspec(6, num_points)
    
    # Plot the original 2D embedding with the selected points highlighted
    # Make it use 3 rows (half the height) and half the width
    ax_tsne = fig.add_subplot(gs[:3, :num_points//2])
    ax_tsne.scatter(latent_2d[:, 0], latent_2d[:, 1], s=5, alpha=0.5, c=cmap(selected_experts))
    ax_tsne.scatter(latent_2d[point_a_idx, 0], latent_2d[point_a_idx, 1], s=100, c=cmap(expert_indices[0]), marker='*', edgecolor='black', label=f"Point A (Exp {expert_indices[0]})")
    ax_tsne.scatter(latent_2d[point_b_idx, 0], latent_2d[point_b_idx, 1], s=100, c=cmap(expert_indices[-1]), marker='*', edgecolor='black', label=f"Point B (Exp {expert_indices[-1]})")

    ax_tsne.set_title("2D Latent Space")
    ax_tsne.legend(loc='best')
    ax_tsne.set_xlabel("Dimension 1")
    ax_tsne.set_ylabel("Dimension 2")
    
    # Plot the expert probabilities as a stacked area chart
    # Also use 3 rows (half the height) and half the width
    ax_experts = fig.add_subplot(gs[:3, num_points//2:])
    expert_probs_array = np.array(expert_probs)
    
    # Create the stacked area chart
    expert_labels = [f"Expert {i}" for i in range(model.num_experts)]
    ax_experts.stackplot(alphas, expert_probs_array.T, labels=expert_labels, colors=[cmap(i) for i in range(model.num_experts)])
    ax_experts.set_title("Expert Probabilities During Interpolation")
    ax_experts.set_xlabel("Interpolation (α)")
    ax_experts.set_ylabel("Probability")
    ax_experts.set_xlim(0, 1)
    ax_experts.set_ylim(0, 1)
    
    # Plot the decoded images in the bottom half
    for i in range(num_points):
        ax = fig.add_subplot(gs[3:5, i])
        ax.imshow(decoded_images[i], cmap='gray')
        ax.set_title(f"α={alphas[i]:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a colored border to indicate the expert
        ax.spines['bottom'].set_color(cmap(expert_indices[i]))
        ax.spines['top'].set_color(cmap(expert_indices[i]))
        ax.spines['left'].set_color(cmap(expert_indices[i]))
        ax.spines['right'].set_color(cmap(expert_indices[i]))
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['top'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
    
    # Add labels showing the dominant expert for each point
    for i in range(num_points):
        ax = fig.add_subplot(gs[5, i])
        ax.axis('off')
        ax.text(0.5, 0.5, f"Expert {expert_indices[i]}",
                ha='center', va='center',
                color=cmap(expert_indices[i]),
                fontweight='bold',
                fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Latent interpolation visualization saved to {save_path}")
    plt.close()
    return {
        'point_a_idx': point_a_idx,
        'point_b_idx': point_b_idx,
        'interpolated_points': interpolated_points,
        'decoded_images': decoded_images,
        'expert_indices': expert_indices,
        'expert_probs': expert_probs
    }

def evaluate_linear_classification(latent_vectors_path, labels, selected_experts, test_size=0.2, random_state=42):
    """
    Load latent vectors and evaluate linear classification performance using two different labeling approaches.
    
    Args:
        latent_vectors_path (str): Path to the saved latent vectors file
        labels (array-like): Ground truth labels for classification
        selected_experts (array-like): Expert selection indices for each sample
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing accuracy results for both labeling approaches
    """
    # Load latent vectors
    try:
        latent_vectors = np.load(latent_vectors_path)
        print(f"Loaded latent vectors with shape: {latent_vectors.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find latent vectors file at {latent_vectors_path}")
        return None
    
    # Ensure labels and latent vectors have compatible shapes
    if len(labels) != latent_vectors.shape[0]:
        print(f"Error: Number of labels ({len(labels)}) doesn't match number of samples ({latent_vectors.shape[0]})")
        return None
    
    if len(selected_experts) != latent_vectors.shape[0]:
        print(f"Error: Number of expert selections ({len(selected_experts)}) doesn't match number of samples ({latent_vectors.shape[0]})")
        return None
    
    print(f"Number of unique ground truth labels: {len(np.unique(labels))}")
    print(f"Number of unique expert selections: {len(np.unique(selected_experts))}")
    
    # Approach 1: Classification using ground truth labels
    print("\n=== Classification with Ground Truth Labels ===")
    try:
        X_train_gt, X_test_gt, y_train_gt, y_test_gt = train_test_split(
            latent_vectors, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        clf_gt = LogisticRegression(random_state=random_state, max_iter=1000)
        clf_gt.fit(X_train_gt, y_train_gt)
        
        y_pred_gt = clf_gt.predict(X_test_gt)
        accuracy_gt = accuracy_score(y_test_gt, y_pred_gt)
        print(f"Ground truth labels accuracy: {accuracy_gt:.4f}")
        print(f"Training set size: {X_train_gt.shape[0]}")
        print(f"Test set size: {X_test_gt.shape[0]}")
    except Exception as e:
        print(f"Error in ground truth classification: {e}")
        accuracy_gt = None
    
    # Approach 2: Classification using expert selections as labels
    print("\n=== Classification with Expert Selection Labels ===")
    try:
        X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
            latent_vectors, selected_experts, test_size=test_size, random_state=random_state, stratify=selected_experts
        )
        
        clf_exp = LogisticRegression(random_state=random_state, max_iter=1000)
        clf_exp.fit(X_train_exp, y_train_exp)
        
        y_pred_exp = clf_exp.predict(X_test_exp)
        accuracy_exp = accuracy_score(y_test_exp, y_pred_exp)
        print(f"Expert selection labels accuracy: {accuracy_exp:.4f}")
        print(f"Training set size: {X_train_exp.shape[0]}")
        print(f"Test set size: {X_test_exp.shape[0]}")
    except Exception as e:
        print(f"Error in expert selection classification: {e}")
        accuracy_exp = None
    
    # Summary
    print("\n=== Comparison Summary ===")
    if accuracy_gt is not None:
        print(f"Ground truth labels accuracy: {accuracy_gt:.4f}")
    else:
        print("Ground truth labels accuracy: Failed")
        
    if accuracy_exp is not None:
        print(f"Expert selection labels accuracy: {accuracy_exp:.4f}")
    else:
        print("Expert selection labels accuracy: Failed")
        
    if accuracy_gt is not None and accuracy_exp is not None:
        improvement = accuracy_exp - accuracy_gt
        print(f"Difference (Expert - Ground Truth): {improvement:.4f}")
        if improvement > 0:
            print("→ Expert selection provides better clustering of latent space!")
        elif improvement < 0:
            print("→ Ground truth labels provide better clustering of latent space!")
        else:
            print("→ Both approaches provide similar clustering quality!")
    
    return {
        'ground_truth_accuracy': accuracy_gt,
        'expert_selection_accuracy': accuracy_exp,
        'improvement': improvement if (accuracy_gt is not None and accuracy_exp is not None) else None
    }
    
if __name__ == "__main__":
    # Example usage
    model_dir = f'results/moe_ld32_ne10_1.0_dataset_percentage_4'
    visualize_pretrained_model(model_dir, None)

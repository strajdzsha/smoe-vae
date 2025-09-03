import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import os

from datasets import RotatedMNISTDataset, CombinedDataset, QuickDrawImageDataset
from torch.utils.data import Subset
from visualization_utils import visualize_latent_space, visualize_reconstructions, visualize_expert_activation_space, visualize_expert_frequencies, visualize_expert_correlation, visualize_expert_specialization, visualize_latent_interpolation

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

class ConvDecoderMid(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoderMid, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 7 * 7)  # Larger feature map

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # For 14x14 and 28x28

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 28x28
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 7, 7)

        x = self.upsample(x)           # 14x14
        x = F.elu(self.conv1(x))

        x = self.upsample(x)           # 28x28
        x = F.elu(self.conv2(x))

        x = torch.sigmoid(self.conv3(x))  # Final output
        return x

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 7, 7)  # Reshape to feature map
        x = F.relu(self.conv_transpose1(x))
        x = F.relu(self.conv_transpose2(x))
        x = torch.sigmoid(self.conv_transpose3(x)) # Use sigmoid for image pixel values [0, 1]
        return x

class ConvDecoderLite(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoderLite, self).__init__()
        self.fc = nn.Linear(latent_dim, 8 * 7 * 7)  # Smaller feature map
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 8, 7, 7)

        x = self.upsample(x)  # 14x14
        x = F.elu(self.conv1(x))

        x = self.upsample(x)  # 28x28
        x = torch.sigmoid(self.conv2(x))  # Final output
        return x

# Gating Network (Expert Selector)
class GatingNetwork(nn.Module):
    def __init__(self, latent_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_experts)  # Output probabilities for each expert

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        # Output probabilities for each expert
        expert_probs = F.softmax(self.fc3(x), dim=-1)
        return expert_probs

# VAE Model with Supervised Mixture of Experts
class VAE_SupervisedMixtureOfExperts(nn.Module):
    def __init__(self, latent_dim, num_experts, model_size='medium'):
        super(VAE_SupervisedMixtureOfExperts, self).__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts

        # Shared encoder
        self.encoder = ConvEncoder(latent_dim)

        # Multiple decoders (experts)
        if model_size == 'lite':
            self.decoders = nn.ModuleList([ConvDecoderLite(latent_dim) for _ in range(num_experts)])
        elif model_size == 'medium':
            self.decoders = nn.ModuleList([ConvDecoderMid(latent_dim) for _ in range(num_experts)])
        else:
            self.decoders = nn.ModuleList([ConvDecoder(latent_dim) for _ in range(num_experts)])

        # Gating network to get expert weights (trained supervised)
        self.gating_network = GatingNetwork(latent_dim, num_experts)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, labels=None):
        # Encode input to latent space
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        
        # Get expert probabilities from gating network
        expert_probs = self.gating_network(z)
        
        # Get the most likely expert for each sample
        expert_indices = torch.argmax(expert_probs, dim=-1)
        
        # Apply the selected expert to each sample
        # Run each decoder on the entire batch
        all_reconstructions = [decoder(z) for decoder in self.decoders]
        
        # Stack the reconstructions to create a tensor of shape [num_experts, batch_size, height, width]
        stacked_reconstructions = torch.stack(all_reconstructions)
        
        # Create indices for batch selection
        batch_indices = torch.arange(z.size(0), device=z.device)
        
        # Get reconstructions based on selected experts: [batch_size, height, width]
        reconstructed_x_list = [stacked_reconstructions[expert_indices[i], i] for i in range(z.size(0))]
        
        # Combine all outputs for the batch
        reconstructed_x = torch.stack(reconstructed_x_list)
        
        # Create logits from probabilities for compatibility
        logits = torch.log(expert_probs + 1e-8)
        
        return reconstructed_x, mu, log_var, logits, expert_probs

# Loss function for Supervised VAE MoE
def loss_function(reconstructed_x, x, mu, log_var, logits, num_experts, epoch, labels, beta=1.0, load_balancing_coeff=0.0, entropy_coeff=0.0):
    if x.dim() == 3:  # If input is [batch_size, channels, height, width]
        x = x.unsqueeze(1)  # Add channel dimension if missing
    if reconstructed_x.dim() == 3:
        reconstructed_x = reconstructed_x.unsqueeze(1)
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

    # KL divergence loss
    kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Expert selection loss (encourage the correct expert to be selected)
    # Convert logits back to probabilities
    expert_probs = F.softmax(logits, dim=-1)
    # Use cross-entropy loss for supervised expert selection
    expert_targets = labels % num_experts  # Ensure valid indices
    expert_selection_loss = F.cross_entropy(logits, expert_targets, reduction='sum')

    # Total loss
    total_loss = reconstruction_loss + beta * kl_divergence_loss + load_balancing_coeff * expert_selection_loss

    return total_loss, reconstruction_loss, beta * kl_divergence_loss, load_balancing_coeff * expert_selection_loss

# Training function
def train(model, dataloader, optimizer, device, epoch, beta=1.0, load_balancing_coeff=0.0, entropy_coeff=0.0):
    model.train()
    total_loss_epoch = 0
    total_reconstruction_loss = 0
    total_kl_divergence_loss = 0
    total_expert_selection_loss = 0
    avg_gating_accuracy = 0.0
    expert_probabilities = []
    progress_bar = tqdm(dataloader, desc="Train", leave=False)

    for batch_idx, (data, labels) in enumerate(progress_bar):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        reconstructed_x, mu, log_var, logits, expert_selection_output = model(data, labels)
        loss, recon_loss, kl_loss, expert_loss = loss_function(
            reconstructed_x, data, mu, log_var, logits, model.num_experts, epoch, labels, beta, load_balancing_coeff, entropy_coeff
        )

        # Calculate gating network accuracy
        expert_targets = labels % model.num_experts  # Ensure valid indices
        predicted_experts = torch.argmax(expert_selection_output, dim=1)
        gating_accuracy = (predicted_experts == expert_targets).float().mean().item()

        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        total_reconstruction_loss += recon_loss.item()
        total_kl_divergence_loss += kl_loss.item()
        total_expert_selection_loss += expert_loss.item()
        avg_gating_accuracy += gating_accuracy
        
        expert_probabilities.append(expert_selection_output.detach().cpu().numpy())

        progress_bar.set_postfix({
            'loss': loss.item() / len(data),
            'recon': recon_loss.item() / len(data),
            'kl': kl_loss.item() / len(data),
            'expert': expert_loss.item() / len(data),
            'acc': gating_accuracy
        })

    avg_total_loss = total_loss_epoch / len(dataloader.dataset)
    avg_reconstruction_loss = total_reconstruction_loss / len(dataloader.dataset)
    avg_kl_divergence_loss = total_kl_divergence_loss / len(dataloader.dataset)
    avg_expert_selection_loss = total_expert_selection_loss / len(dataloader.dataset)
    avg_gating_accuracy = avg_gating_accuracy / batch_idx


    print(f'Train set Average loss: {avg_total_loss:.4f}')
    print(f'  Reconstruction loss: {avg_reconstruction_loss:.4f}')
    print(f'  KL divergence loss: {avg_kl_divergence_loss:.4f}')
    print(f'  Expert selection loss: {avg_expert_selection_loss:.4f}')
    print(f'  Gating accuracy: {avg_gating_accuracy:.4f}')

    return avg_total_loss, avg_reconstruction_loss, avg_kl_divergence_loss, avg_expert_selection_loss, expert_probabilities

# Testing function
def test(model, dataloader, device, epoch, beta=1.0, load_balancing_coeff=0.0, entropy_coeff=0.0):
    model.eval()
    total_loss_epoch = 0
    total_reconstruction_loss = 0
    total_kl_divergence_loss = 0
    total_expert_selection_loss = 0
    avg_gating_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            reconstructed_x, mu, log_var, logits, expert_selection_output = model(data, labels)
            loss, recon_loss, kl_loss, expert_loss = loss_function(
                reconstructed_x, data, mu, log_var, logits, model.num_experts, epoch, labels, beta, load_balancing_coeff, entropy_coeff
            )

            # Calculate gating network accuracy
            expert_targets = labels % model.num_experts  # Ensure valid indices
            predicted_experts = torch.argmax(expert_selection_output, dim=1)
            gating_accuracy = (predicted_experts == expert_targets).float().mean().item()
            avg_gating_accuracy += gating_accuracy

            total_loss_epoch += loss.item()
            total_reconstruction_loss += recon_loss.item()
            total_kl_divergence_loss += kl_loss.item()
            total_expert_selection_loss += expert_loss.item()

    avg_total_loss = total_loss_epoch / len(dataloader.dataset)
    avg_reconstruction_loss = total_reconstruction_loss / len(dataloader.dataset)
    avg_kl_divergence_loss = total_kl_divergence_loss / len(dataloader.dataset)
    avg_expert_selection_loss = total_expert_selection_loss / len(dataloader.dataset)
    avg_gating_accuracy = avg_gating_accuracy / batch_idx

    print(f'Test set Average loss: {avg_total_loss:.4f}')
    print(f'  Reconstruction loss: {avg_reconstruction_loss:.4f}')
    print(f'  KL divergence loss: {avg_kl_divergence_loss:.4f}')
    print(f'  Expert selection loss: {avg_expert_selection_loss:.4f}')
    print(f'  Gating accuracy: {gating_accuracy:.4f}')

    return avg_total_loss, avg_reconstruction_loss, avg_kl_divergence_loss, avg_expert_selection_loss

# Main function to run the training and evaluation
def main(config=None):
    # Hyperparameters
    num_experts = config['num_experts'] if config else 10
    latent_dim = config['latent_dim'] if config else 32
    epochs = config['epochs'] if config else 10
    test_flag = config['test_flag'] if config else False
    dataset = config['dataset'] if config else 'mnist'
    results_dir_suffix = config['results_dir_suffix'] if config else '_supervised'
    batch_size = config['batch_size'] if config else 128
    learning_rate = config['learning_rate'] if config else 1e-3
    beta = config['beta'] if config else 1.0
    load_balancing_coeff = config['load_balancing_coeff']  # Not needed for supervised
    entropy_coeff = 0.0  # Not needed for supervised
    model_size = config['model_size'] if config else 'medium'
    dataset_percentage = config['dataset_percentage'] if config else 0.5

    print(config)
        
    results_dir_base = f"semestral-project-awareness/vae/snapshots/{dataset}/moe_supervised_ld{latent_dim}_ne{num_experts}{results_dir_suffix}"

    if test_flag:
        results_dir = f"{results_dir_base}_test"
    else:
        results_dir = results_dir_base

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(os.path.join(results_dir, 'hyperparams.txt'), 'w') as f:
        f.write(f"latent_dim={latent_dim}\n")
        f.write(f"num_experts={num_experts}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"beta={beta}\n")
        f.write(f"load_balancing_coeff={load_balancing_coeff}\n")
        f.write(f"entropy_coeff={entropy_coeff}\n")
        f.write(f"dataset={dataset}\n")
        f.write(f"model_size={model_size}\n")
        f.write(f"supervised=True\n")

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    base_transform = transforms.ToTensor()

    if dataset == 'combined':
        print("Loading Combined Dataset (MNIST + FashionMNIST + CIFAR10 + QuickDraw).")
        # Load MNIST
        MNIST_Train = datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
        MNIST_Test = datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)
        
        # Load FashionMNIST
        FashionMNIST_Train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=base_transform)
        FashionMNIST_Test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=base_transform)
        
        # Load CIFAR10
        CIFAR10_Train = datasets.CIFAR10(root='./data', train=True, download=True, transform=base_transform)
        CIFAR10_Test = datasets.CIFAR10(root='./data', train=False, download=True, transform=base_transform)

        # Load QuickDraw
        QuickDraw_Train = QuickDrawImageDataset(root='./data/quickdraw', train=True, download=True, transform=base_transform, samples_per_category=10000)
        QuickDraw_Test = QuickDrawImageDataset(root='./data/quickdraw', train=False, download=True, transform=base_transform, samples_per_category=2000)

        # Create combined datasets
        datasets_train = {'mnist': MNIST_Train, 'fashion_mnist': FashionMNIST_Train, 'quickdraw': QuickDraw_Train}
        datasets_test = {'mnist': MNIST_Test, 'fashion_mnist': FashionMNIST_Test, 'quickdraw': QuickDraw_Test}
        OriginalTrainDataset = CombinedDataset(datasets_train)
        OriginalTestDataset = CombinedDataset(datasets_test)
        
    elif dataset == 'fashion':
        print("Loading FashionMNIST dataset.")
        OriginalTrainDataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=base_transform)
        OriginalTestDataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=base_transform)
    elif dataset == 'cifar':
        print("Loading CIFAR10 dataset.")
        cifar_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        OriginalTrainDataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
        OriginalTestDataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
    elif dataset == 'quickdraw':
        print("Loading QuickDraw dataset.")
        categories = ['face', 'cat', 'snowflake', 'pencil', 'eye']
        samples_per_category = int(dataset_percentage * 70000)
        OriginalTrainDataset = QuickDrawImageDataset(root='./data/quickdraw', train=True, download=True, transform=base_transform, samples_per_category=samples_per_category, categories=categories)
        OriginalTestDataset = QuickDrawImageDataset(root='./data/quickdraw', train=False, download=True, transform=base_transform, samples_per_category=int(0.1*samples_per_category), categories=categories)
    else:
        print("Loading MNIST dataset.")
        OriginalTrainDataset = datasets.MNIST(root='./data', train=True, download=True, transform=base_transform)
        OriginalTestDataset = datasets.MNIST(root='./data', train=False, download=True, transform=base_transform)

    train_dataset = OriginalTrainDataset
    test_dataset = OriginalTestDataset
    train_shuffle = True
        
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create data loaders
    if test_flag:
        subsample_fraction = 0.01
        train_indices = np.random.choice(len(train_dataset), int(len(train_dataset) * subsample_fraction), replace=False)
        test_indices = np.random.choice(len(test_dataset), int(len(test_dataset) * subsample_fraction), replace=False)

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=train_shuffle)

    # Initialize model and optimizer
    model = VAE_SupervisedMixtureOfExperts(latent_dim, num_experts, model_size=model_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    test_losses = []
    train_recon_losses = []
    train_kl_losses = []
    train_expert_losses = []
    test_recon_losses = []
    test_kl_losses = []
    test_expert_losses = []

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_loss, train_recon, train_kl, train_expert, expert_probs = train(
            model, train_dataloader, optimizer, device, epoch, beta, load_balancing_coeff, entropy_coeff
        )
        test_loss, test_recon, test_kl, test_expert = test(
            model, test_dataloader, device, epoch, beta, load_balancing_coeff, entropy_coeff
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_recon_losses.append(train_recon)
        train_kl_losses.append(train_kl)
        train_expert_losses.append(train_expert)
        test_recon_losses.append(test_recon)
        test_kl_losses.append(test_kl)
        test_expert_losses.append(test_expert)

        # Save model state periodically
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
             model_save_path = os.path.join(results_dir, f'vae_moe_supervised_model_epoch_{epoch+1}.pth')
             torch.save(model.state_dict(), model_save_path)
             print(f"Model state saved to {model_save_path}")

    # Plot the entropy of expert selection
    expert_probs = np.concatenate(expert_probs, axis=0)
    expert_entropy = -np.sum(expert_probs * np.log(expert_probs + 1e-8), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(expert_entropy, bins=50, alpha=0.7)
    plt.title('Entropy of Expert Selection (Supervised)')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    entropy_plot_path = os.path.join(results_dir, 'expert_entropy.png')
    plt.savefig(entropy_plot_path)
    plt.close()
    print(f"Entropy plot saved to {entropy_plot_path}")
    
    # Save losses to .npy files for later analysis
    np.save(os.path.join(results_dir, 'train_recon_losses.npy'), train_recon_losses)
    np.save(os.path.join(results_dir, 'test_recon_losses.npy'), test_recon_losses)

    # Plot training and testing losses
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Total Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs+1), train_recon_losses, label='Train Recon Loss')
    plt.plot(range(1, epochs+1), test_recon_losses, label='Test Recon Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs+1), train_kl_losses, label='Train KL Loss')
    plt.plot(range(1, epochs+1), test_kl_losses, label='Test KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(range(1, epochs+1), train_expert_losses, label='Train Expert Selection Loss')
    plt.plot(range(1, epochs+1), test_expert_losses, label='Test Expert Selection Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Expert Selection Losses (Supervised)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_plot_path = os.path.join(results_dir, 'losses_detailed.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plots saved to {loss_plot_path}")
    plt.close()

    # Generate visualizations in the output directory
    print("Generating visualizations...")
    
    # Visualize latent space
    latent_plot_path = os.path.join(results_dir, 'tsne_latent_space_classs_moe.png')
    visualize_latent_space(model, test_dataloader, device, save_path=latent_plot_path, output_dir=results_dir)
    
    # Visualize reconstructions
    recon_plot_path = os.path.join(results_dir, 'reconstructions_moe.png')
    visualize_reconstructions(model, test_dataloader, device, save_path=recon_plot_path)
    
    # Visualize expert activation space - reuse embeddings from latent space visualization
    expert_activation_plot_path = os.path.join(results_dir, 'expert_activation_space.png')
    visualize_expert_activation_space(model, test_dataloader, device, save_path=expert_activation_plot_path, output_dir=results_dir)
    
    # Visualize expert correlation with class labels
    expert_correlation_plot_path = os.path.join(results_dir, 'expert_correlation.png')
    visualize_expert_correlation(model, test_dataloader, device, num_experts, save_path=expert_correlation_plot_path)
    
    # Visualize expert frequencies
    histogram_plot_path = os.path.join(results_dir, 'expert_histogram.png')
    visualize_expert_frequencies(model, test_dataloader, device, num_experts, save_path=histogram_plot_path)
    
    # Visualize expert specialization
    expert_specialization_plot_path = os.path.join(results_dir, 'expert_specialization.png')
    visualize_expert_specialization(model, test_dataloader, device, num_experts, save_path=expert_specialization_plot_path)

    # Visualize latent interpolation
    latent_interpolation_plot_path = os.path.join(results_dir, 'latent_interpolation.png')
    visualize_latent_interpolation(model, device, results_dir, num_points=20, save_path=latent_interpolation_plot_path)
    print("Visualization complete.")
    
    # Save the final model
    final_model_path = os.path.join(results_dir, 'vae_moe_supervised_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    # Load config from config.yaml
    import yaml
    with open('vae/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config = config['debug']

    main(config)
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
from visualization_utils import visualize_latent_space, visualize_reconstructions, visualize_expert_activation_space, visualize_expert_frequencies, visualize_expert_correlation

# # Define Convolutional Encoder (Same as before)
# class ConvEncoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(ConvEncoder, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 28x28 → 14x14
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 14x14 → 7x7
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2)
#         )
#         self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
#         self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.view(x.size(0), -1)
#         mu = self.fc_mu(x)
#         log_var = self.fc_logvar(x)
#         return mu, log_var

# # Define Decoder for MNIST digits (Same as before)
# class ConvDecoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(ConvDecoder, self).__init__()
#         self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

#         self.deconv1 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 → 14x14
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2)
#         )
#         self.deconv2 = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 14x14 → 28x28
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2)
#         )
#         self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

#     def forward(self, z):
#         x = self.fc(z)
#         x = x.view(x.size(0), 128, 7, 7)
#         x = self.deconv1(x)
#         x = self.deconv2(x)
#         x = torch.sigmoid(self.deconv3(x))  # Output in [0, 1]
#         return x

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

class ConvEncoderLite(nn.Module):
    def __init__(self, latent_dim):
        super(ConvEncoderLite, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)  # 14x14 -> 7x7
        self.fc_mu = nn.Linear(8 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(8 * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
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

# Define Decoder for MNIST digits (Same as before)
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

# Gating Network (Decoder Selector - now Expert Selector)
class GatingNetwork(nn.Module):
    def __init__(self, latent_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_experts)  # Output probabilities for each expert

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        # Output raw logits for Gumbel-Softmax
        logits = self.fc3(x)
        return logits

# VAE Model with Mixture of Experts Decoders
class VAE_MixtureOfExperts(nn.Module):
    def __init__(self, latent_dim, num_experts, model_size='medium', encoder_size='medium'):
        super(VAE_MixtureOfExperts, self).__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        # Temperature for Gumbel-Softmax (can be annealed)
        self.gumbel_tau = 1.0 # You might want to anneal this (e.g., decrease over time)

        # Shared encoder
        self.encoder = ConvEncoder(latent_dim)

        # Multiple decoders (experts)
        if model_size == 'lite':
            self.decoders = nn.ModuleList([ConvDecoderLite(latent_dim) for _ in range(num_experts)])
        elif model_size == 'medium':
            self.decoders = nn.ModuleList([ConvDecoderMid(latent_dim) for _ in range(num_experts)])
        else:
            self.decoders = nn.ModuleList([ConvDecoder(latent_dim) for _ in range(num_experts)])

        if encoder_size == 'lite':
            self.encoder = ConvEncoderLite(latent_dim)
        else:
            self.encoder = ConvEncoder(latent_dim)

        # Gating network to get expert weights
        self.gating_network = GatingNetwork(latent_dim, num_experts)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # Encode input to latent space
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        # Get expert logits from gating network
        logits = self.gating_network(z) # Shape: (batch_size, num_experts)

        # Get soft probabilities (weights) for each expert using softmax during training
        # During eval, we still use argmax for deterministic selection
        if self.training:
            # Soft gating: Use softmax probabilities as weights
            expert_probs = F.softmax(logits, dim=-1) # Shape: (batch_size, num_experts)

            # Calculate outputs from all experts
            # Stack outputs along a new dimension (dim=1)
            all_expert_outputs = torch.stack([decoder(z) for decoder in self.decoders], dim=1)
            # Shape: (batch_size, num_experts, C, H, W), e.g., (128, 10, 1, 28, 28)

            # Weight the outputs by the probabilities
            # Reshape probs for broadcasting: (batch_size, num_experts, 1, 1, 1)
            weights = expert_probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # Weighted sum across the expert dimension (dim=1)
            reconstructed_x = torch.sum(weights * all_expert_outputs, dim=1)
            # Shape: (batch_size, C, H, W), e.g., (128, 1, 28, 28)

            # Keep track of probabilities for loss calculation
            expert_selection_output = expert_probs # Use soft probs during training

        else:
            # Hard gating at inference: Pick the best expert deterministically
            indices = torch.argmax(logits, dim=-1) # Shape: (batch_size,)
            expert_selection_one_hot = F.one_hot(indices, num_classes=self.num_experts).float()
            # Shape: (batch_size, num_experts)

            # Compute output only for the selected expert for each sample (using loop for simplicity)
            reconstructed_x_list = []
            for i in range(z.size(0)): # Iterate over samples in the batch
                selected_expert_index = indices[i].item()
                latent_vector_sample = z[i:i+1] # Keep batch dim: (1, latent_dim)
                reconstruction_sample = self.decoders[selected_expert_index](latent_vector_sample) # Shape: (1, 1, 28, 28)
                reconstructed_x_list.append(reconstruction_sample)
            reconstructed_x = torch.cat(reconstructed_x_list, dim=0) # Shape: (batch_size, 1, 28, 28)

            # Keep track of one-hot selection for potential analysis during eval
            expert_selection_output = expert_selection_one_hot

        # Return logits needed for the load balancing loss, and the selection output (probs or one-hot)
        return reconstructed_x, mu, log_var, logits, expert_selection_output


# Loss function for VAE MoE - Using Gumbel-Softmax gating and load balancing loss
def loss_function(reconstructed_x, x, mu, log_var, logits, num_experts, epoch, beta=1.0, load_balancing_coeff=1.0, entropy_coeff=1.0):
    # Reconstruction loss (same as before, applied to the output of the selected expert)
    # Ensure x has the same shape as reconstructed_x if needed (e.g., channel dim)
    if x.dim() == 3: # Add channel dimension if missing (e.g., (batch, 28, 28) -> (batch, 1, 28, 28))
        x = x.unsqueeze(1)
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

    # KL divergence loss (same as before)
    kl_divergence_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Load balancing loss based on average expert probabilities (encourages uniform usage)
    # Calculate probabilities from logits
    expert_probs = F.softmax(logits, dim=-1) # Shape: (batch_size, num_experts)
    # Calculate average probability for each expert across the batch
    avg_expert_probs = torch.mean(expert_probs, dim=0) # Shape: (num_experts,)
    # Ideal uniform distribution
    ideal_dist = torch.full_like(avg_expert_probs, 1.0 / num_experts)
    # Calculate MSE between average probabilities and uniform distribution
    load_balancing_loss = F.mse_loss(avg_expert_probs, ideal_dist, reduction='sum')
    # Multiply by num_experts to scale similarly to some other load balancing formulations
    load_balancing_loss = load_balancing_loss * num_experts

    # Entropy-based Sharpness Loss: Penalize high entropy in per-sample distributions
    # H(p) = - sum(p_i * log(p_i)). Minimizing loss means minimizing entropy.
    # Add a small epsilon for numerical stability inside the log.
    entropy_per_sample = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8), dim=1)
    # Average entropy across the batch to get the batch entropy loss
    entropy_loss = torch.mean(entropy_per_sample) # Changed from torch.sum to torch.mean

    # Combine Load Balancing (uniform average usage) and Entropy Loss (individual sharpness)
    combined_gating_loss = load_balancing_coeff * load_balancing_loss + entropy_coeff * entropy_loss

    # Total loss
    total_loss = reconstruction_loss + beta * kl_divergence_loss + combined_gating_loss

    return total_loss, reconstruction_loss, beta * kl_divergence_loss, combined_gating_loss

# Training function
def train(model, dataloader, optimizer, device, epoch, beta=1.0, load_balancing_coeff=1.0, entropy_coeff=1.0):
    model.train()
    total_loss_epoch = 0
    total_reconstruction_loss = 0
    total_kl_divergence_loss = 0
    total_load_balancing_loss = 0
    expert_probabilities = []
    progress_bar = tqdm(dataloader, desc="Train", leave=False)

    for batch_idx, (data, _) in enumerate(progress_bar): # Labels are not needed for loss calculation anymore
        data = data.to(device)

        optimizer.zero_grad()

        # Model returns logits and expert_selection_output (probs during train, one-hot during eval)
        reconstructed_x, mu, log_var, logits, expert_selection_output = model(data)
        loss, recon_loss, kl_loss, lb_loss = loss_function(
            reconstructed_x, data, mu, log_var, logits, model.num_experts, epoch, beta, load_balancing_coeff, entropy_coeff
        )

        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        total_reconstruction_loss += recon_loss.item()
        total_kl_divergence_loss += kl_loss.item()
        total_load_balancing_loss += lb_loss.item()
        # Store expert probabilities for analysis
        expert_probabilities.append(expert_selection_output.detach().cpu().numpy())

        progress_bar.set_postfix({
            'loss': loss.item() / len(data),
            'recon': recon_loss.item() / len(data),
            'kl': kl_loss.item() / len(data),
            'lb': lb_loss.item() # Load balancing loss is per-batch, not per-sample
        })

    avg_total_loss = total_loss_epoch / len(dataloader.dataset)
    avg_reconstruction_loss = total_reconstruction_loss / len(dataloader.dataset)
    avg_kl_divergence_loss = total_kl_divergence_loss / len(dataloader.dataset)
    # Average load balancing loss over batches
    avg_load_balancing_loss = total_load_balancing_loss / len(dataloader.dataset)

    print(f'Train set Average loss: {avg_total_loss:.4f}')
    print(f'  Reconstruction loss: {avg_reconstruction_loss:.4f}')
    print(f'  KL divergence loss: {avg_kl_divergence_loss:.4f}')
    print(f'  Load Balancing loss: {avg_load_balancing_loss:.4f}')

    return avg_total_loss, avg_reconstruction_loss, avg_kl_divergence_loss, avg_load_balancing_loss, expert_probabilities

# Testing function
def test(model, dataloader, device, epoch, beta=1.0, load_balancing_coeff=1.0, entropy_coeff=1.0):
    model.eval()
    total_loss_epoch = 0
    total_reconstruction_loss = 0
    total_kl_divergence_loss = 0
    total_load_balancing_loss = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader): # Labels not needed
            data = data.to(device)

            # Model returns logits and expert_selection_output (probs during train, one-hot during eval)
            reconstructed_x, mu, log_var, logits, expert_selection_output = model(data)
            loss, recon_loss, kl_loss, lb_loss = loss_function(
                reconstructed_x, data, mu, log_var, logits, model.num_experts, epoch, beta, load_balancing_coeff, entropy_coeff
            )

            total_loss_epoch += loss.item()
            total_reconstruction_loss += recon_loss.item()
            total_kl_divergence_loss += kl_loss.item()
            total_load_balancing_loss += lb_loss.item()

    avg_total_loss = total_loss_epoch / len(dataloader.dataset)
    avg_reconstruction_loss = total_reconstruction_loss / len(dataloader.dataset)
    avg_kl_divergence_loss = total_kl_divergence_loss / len(dataloader.dataset)
    avg_load_balancing_loss = total_load_balancing_loss / len(dataloader.dataset) # Average over batches

    print(f'Test set Average loss: {avg_total_loss:.4f}')
    print(f'  Reconstruction loss: {avg_reconstruction_loss:.4f}')
    print(f'  KL divergence loss: {avg_kl_divergence_loss:.4f}')
    print(f'  Load Balancing loss: {avg_load_balancing_loss:.4f}')

    return avg_total_loss, avg_reconstruction_loss, avg_kl_divergence_loss, avg_load_balancing_loss

# Main function to run the training and evaluation
def main(config=None):
    # Hyperparameters
    num_experts = config['num_experts'] if config else 1
    latent_dim = config['latent_dim'] if config else 32
    epochs = config['epochs'] if config else 10
    test_flag = config['test_flag'] if config else False
    dataset = config['dataset'] if config else 'mnist' # Default dataset
    results_dir_suffix = config['results_dir_suffix'] if config else '' # Suffix for results directory
    batch_size = config['batch_size'] if config else 128 # Default batch size
    learning_rate = config['learning_rate'] if config else 1e-3 # Default learning rate
    beta = config['beta'] if config else 1.0 # Default weight for KL divergence term
    load_balancing_coeff = config['load_balancing_coeff'] if config else 1 # Default weight for load balancing loss
    entropy_coeff = config['entropy_coeff'] if config else 1 # Default weight for entropy loss
    model_size = config['model_size'] if config else True # Use smaller decoders
    dataset_percentage = config['dataset_percentage'] if config else 0.5 # Percentage of dataset to use
    encoder_size = config['encoder_size'] if config else 'medium' # Size of the encoder

    print(config)
        
    results_dir_base = f"semestral-project-awareness/vae/snapshots/{dataset}/dataset_size/moe_ld{latent_dim}_ne{num_experts}{results_dir_suffix}"

    if test_flag:
        results_dir = f"{results_dir_base}_test" # Directory to save results
    else:
        results_dir = results_dir_base

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)    # Save hyperparameters
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

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")    # Load dataset
    base_transform = transforms.ToTensor() # Basic transform for all datasets

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

        # print lens of each dataset
        print(f"MNIST Train size: {len(MNIST_Train)}")
        print(f"MNIST Test size: {len(MNIST_Test)}")
        print(f"FashionMNIST Train size: {len(FashionMNIST_Train)}")
        print(f"FashionMNIST Test size: {len(FashionMNIST_Test)}")
        print(f"CIFAR10 Train size: {len(CIFAR10_Train)}")
        print(f"CIFAR10 Test size: {len(CIFAR10_Test)}")
        print(f"QuickDraw Train size: {len(QuickDraw_Train)}")
        print(f"QuickDraw Test size: {len(QuickDraw_Test)}")
        
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
        # Additional transforms for CIFAR10: Convert to grayscale and resize to 28x28
        cifar_transform = transforms.Compose([
            transforms.Grayscale(),             # Convert RGB to grayscale
            transforms.Resize((28, 28)),        # Resize to 28x28
            transforms.ToTensor()               # Convert to tensor
        ])
        OriginalTrainDataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
        OriginalTestDataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
    elif dataset == 'quickdraw':
        print("Loading QuickDraw dataset.")
        categories = ['face', 'cat', 'snowflake', 'pencil', 'eye']
        samples_per_category = int(dataset_percentage * 70000)  # 70,000 is the default number of samples per category in QuickDraw
        OriginalTrainDataset = QuickDrawImageDataset(root='./data/quickdraw', train=True, download=True, transform=base_transform, samples_per_category=samples_per_category, categories=categories)
        OriginalTestDataset = QuickDrawImageDataset(root='./data/quickdraw', train=False, download=True, transform=base_transform, samples_per_category=int(0.1*samples_per_category), categories=categories)
    elif dataset == 'sinusoidal':
        print("Loading Sinusoidal2D dataset.")
        from datasets import Sinusoidal2DDataset
        # Get parameters from config
        mode = config.get('sinusoidal_mode', 'rotation')
        num_unique = config.get('num_unique', 5)
        samples_per_unique = config.get('samples_per_unique', 1000)
        img_size = config.get('img_size', 28)
        base_frequency = config.get('base_frequency', 1.0)
        noise_std = config.get('noise_std', 0.1)
        
        if mode == 'frequency':
            min_frequency = config.get('min_frequency', 0.5)
            max_frequency = config.get('max_frequency', 2.0)
            OriginalTrainDataset = Sinusoidal2DDataset(mode=mode,
                                                       num_unique=num_unique,
                                                       samples_per_unique=samples_per_unique,
                                                       img_size=img_size,
                                                       base_frequency=base_frequency,
                                                       noise_std=noise_std,
                                                       min_frequency=min_frequency,
                                                       max_frequency=max_frequency,
                                                       transform=base_transform)
            OriginalTestDataset = Sinusoidal2DDataset(mode=mode,
                                                      num_unique=num_unique,
                                                      samples_per_unique=int(samples_per_unique * 0.2),
                                                      img_size=img_size,
                                                      base_frequency=base_frequency,
                                                      noise_std=noise_std,
                                                      min_frequency=min_frequency,
                                                      max_frequency=max_frequency,
                                                      random_seed=config.get('random_seed', 42) + 1000,
                                                      transform=base_transform)
        else:
            OriginalTrainDataset = Sinusoidal2DDataset(mode=mode,
                                                       num_unique=num_unique,
                                                       samples_per_unique=samples_per_unique,
                                                       img_size=img_size,
                                                       base_frequency=base_frequency,
                                                       noise_std=noise_std,
                                                       transform=base_transform)
            OriginalTestDataset = Sinusoidal2DDataset(mode=mode,
                                                      num_unique=num_unique,
                                                      samples_per_unique=int(samples_per_unique * 0.2),
                                                      img_size=img_size,
                                                      base_frequency=base_frequency,
                                                      noise_std=noise_std,
                                                      random_seed=config.get('random_seed', 42) + 1000,
                                                      transform=base_transform)
        print(f"Train dataset: {len(OriginalTrainDataset)} samples")
        print(f"Test dataset: {len(OriginalTestDataset)} samples")
    
    elif dataset == 'synthetic':
        print("Loading Synthetic Cluster dataset.")
        num_clusters = config.get('num_clusters', 5)  # Number of clusters to generate
        samples_per_cluster = config.get('samples_per_cluster', 2000)  # Samples per cluster
        cluster_separation = config.get('cluster_separation', 8.0)  # Distance between cluster centers
        cluster_std = config.get('cluster_std', 2.0)  # Standard deviation for cluster center placement
        blob_std = config.get('blob_std', 1.5)  # Standard deviation for individual Gaussian blobs
        random_seed = config.get('random_seed', 42)  # Random seed for reproducibility
        
        print(f"  Generating {num_clusters} clusters with {samples_per_cluster} samples each")
        print(f"  Cluster separation: {cluster_separation}, Cluster std: {cluster_std}, Blob std: {blob_std}")
        
        # Create train dataset
        OriginalTrainDataset = SyntheticClusterDataset(
            num_clusters=num_clusters, 
            samples_per_cluster=samples_per_cluster,
            cluster_separation=cluster_separation,
            cluster_std=cluster_std,
            blob_std=blob_std,
            random_seed=random_seed,
            transform=base_transform
        )
        
        # Create test dataset with different seed and fewer samples
        OriginalTestDataset = SyntheticClusterDataset(
            num_clusters=num_clusters, 
            samples_per_cluster=int(samples_per_cluster * 0.2),  # 20% of train size for test
            cluster_separation=cluster_separation,
            cluster_std=cluster_std,
            blob_std=blob_std,
            random_seed=random_seed + 1000,  # Different seed for test set
            transform=base_transform
        )
        
        # Print cluster information
        cluster_info = OriginalTrainDataset.get_cluster_info()
        print(f"  Train dataset: {cluster_info['total_samples']} total samples")
        cluster_info_test = OriginalTestDataset.get_cluster_info()
        print(f"  Test dataset: {cluster_info_test['total_samples']} total samples")
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
    # Subsample both train and test datasets to 1%
    if test_flag:
        subsample_fraction = 0.01
        train_indices = np.random.choice(len(train_dataset), int(len(train_dataset) * subsample_fraction), replace=False)
        test_indices = np.random.choice(len(test_dataset), int(len(test_dataset) * subsample_fraction), replace=False)

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=train_shuffle)

    # Initialize model and optimizer
    model = VAE_MixtureOfExperts(latent_dim, num_experts, model_size=model_size, encoder_size=encoder_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    test_losses = []
    train_recon_losses = []
    train_kl_losses = []
    train_lb_losses = []
    test_recon_losses = []
    test_kl_losses = []
    test_lb_losses = []

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_loss, train_recon, train_kl, train_lb, expert_probs = train(
            model, train_dataloader, optimizer, device, epoch, beta, load_balancing_coeff, entropy_coeff
        )
        test_loss, test_recon, test_kl, test_lb = test(
            model, test_dataloader, device, epoch, beta, load_balancing_coeff, entropy_coeff
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_recon_losses.append(train_recon)
        train_kl_losses.append(train_kl)
        train_lb_losses.append(train_lb)
        test_recon_losses.append(test_recon)
        test_kl_losses.append(test_kl)
        test_lb_losses.append(test_lb)

        # # Save model state periodically
        # if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        #      model_save_path = os.path.join(results_dir, f'vae_moe_model_epoch_{epoch+1}.pth')
        #      torch.save(model.state_dict(), model_save_path)
        #      print(f"Model state saved to {model_save_path}")

    # # Plot the entropy of expert selection
    # expert_probs = np.concatenate(expert_probs, axis=0) # Shape: (num_samples, num_experts)
    # expert_entropy = -np.sum(expert_probs * np.log(expert_probs + 1e-8), axis=1) # Shape: (num_samples,)
    # plt.figure(figsize=(10, 6))
    # plt.hist(expert_entropy, bins=50, alpha=0.7)
    # plt.title('Entropy of Expert Selection')
    # plt.xlabel('Entropy')
    # plt.ylabel('Frequency')
    # entropy_plot_path = os.path.join(results_dir, 'expert_entropy.png')
    # plt.savefig(entropy_plot_path)
    # plt.close()
    # print(f"Entropy plot saved to {entropy_plot_path}")
    # # Save losses to .npy files for later analysis
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
    plt.plot(range(1, epochs+1), train_lb_losses, label='Train Load Balancing Loss')
    plt.plot(range(1, epochs+1), test_lb_losses, label='Test Load Balancing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CV^2)')
    plt.title('Load Balancing Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_plot_path = os.path.join(results_dir, 'losses_detailed.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plots saved to {loss_plot_path}")
    plt.close()

    # Visualize latent space using test data
    # Default: Color by digit
    latent_plot_path_digits = os.path.join(results_dir, 'tsne_latent_space_digits_moe.png')
    visualize_latent_space(model, test_dataloader, device, save_path=latent_plot_path_digits)

    # Visualize reconstructions using test data
    recon_plot_path = os.path.join(results_dir, 'reconstructions_moe.png')
    visualize_reconstructions(model, test_dataloader, device, save_path=recon_plot_path)

    # Visualize expert activation space using test data
    expert_activation_plot_path = os.path.join(results_dir, 'expert_activation_space.png')
    visualize_expert_activation_space(model, test_dataloader, device, save_path=expert_activation_plot_path)
    
    # Visualize expert correlation with digit labels
    expert_correlation_plot_path = os.path.join(results_dir, 'expert_correlation.png')
    visualize_expert_correlation(model, test_dataloader, device, num_experts, save_path=expert_correlation_plot_path)
    
    # Visualize expert frequencies
    histogram_plot_path = os.path.join(results_dir, 'expert_histogram.png')
    visualize_expert_frequencies(model, test_dataloader, device, num_experts, save_path=histogram_plot_path)
    # Save the final model
    final_model_path = os.path.join(results_dir, 'vae_moe_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    # Load config from config.yaml
    import yaml
    with open('vae/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Choose configuration: 'debug', 'train', or 'synthetic'
    config_name = 'synthetic'  # Change this to switch between configurations
    config = config[config_name]
    
    print(f"Using configuration: {config_name}")
    main(config)
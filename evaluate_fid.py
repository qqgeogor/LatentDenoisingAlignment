import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy import linalg
from train_ctrl_inv import Decoder, Encoder
import argparse
import multiprocessing
from torch.amp import autocast

# Set file system sharing strategy to avoid "Too many open files" error
torch.multiprocessing.set_sharing_strategy('file_system')
# Set start method to spawn to avoid shared memory issues
multiprocessing.set_start_method('spawn', force=True)

class FakeDataset(Dataset):
    def __init__(self, generator, num_samples, latent_dim, device, use_amp=True):
        self.generator = generator
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.device = device
        self.use_amp = use_amp
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        with torch.no_grad(), autocast(device_type='cuda', enabled=self.use_amp):
            z = torch.randn(1, self.latent_dim).to(self.device)
            fake_img, _ = self.generator(z)
            if isinstance(fake_img, tuple):
                fake_img = fake_img[0]
            # Apply the same transforms as real images
            fake_img = self.transform(fake_img)
            # Convert to float32 before returning to avoid pinning issues
            return fake_img.squeeze(0).float()

def get_activations(images, model, batch_size=50, dims=2048, device='cuda', use_amp=True):
    """Calculates the activations of the pool_3 layer for all images."""
    model.eval()
    n_batches = len(images) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    
    for i in tqdm(range(n_batches), desc='Calculating activations'):
        start = i * batch_size
        end = start + batch_size
        
        batch = images[start:end].to(device)
        
        with torch.no_grad(), autocast(enabled=use_amp):
            pred = model(batch)[0]
        
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = nn.AdaptiveAvgPool2d((1, 1))(pred)
        
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
    
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(dataloader, model, batch_size=50, dims=2048, device='cuda', use_amp=True):
    """Calculation of the statistics used by the FID."""
    model.eval()
    
    # Initialize accumulators for mean and covariance
    act_sum = np.zeros(dims)
    act_sq_sum = np.zeros((dims, dims))
    total_samples = 0
    
    for batch in tqdm(dataloader, desc='Calculating statistics'):
        # Handle both cases: when batch is (images, labels) or just images
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
            
        images = images.to(device)
        batch_size = images.size(0)
        total_samples += batch_size
        
        with torch.no_grad(), autocast(device_type='cuda', enabled=use_amp):
            # Get the features from the Inception model
            # Inception v3 returns a tuple (output, aux_output)
            # We need to get the features before the final classification layer
            pred = model(images)
            
            # Get the features from the pool3 layer
            if isinstance(pred, tuple):
                # Inception v3 returns a tuple (output, aux_output)
                features = pred[0]
            else:
                features = pred
            
            # Ensure we have the right shape [batch, channels, height, width]
            if len(features.shape) == 4:
                # Global average pooling if needed
                if features.shape[2] != 1 or features.shape[3] != 1:
                    features = nn.AdaptiveAvgPool2d((1, 1))(features)
                features = features.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
            
            # Convert to numpy and reshape
            features = features.cpu().data.numpy()
            
            # Accumulate sum and squared sum
            act_sum += features.sum(axis=0)
            act_sq_sum += np.dot(features.T, features)
    
    # Calculate mean and covariance
    mu = act_sum / total_samples
    sigma = (act_sq_sum / total_samples) - np.outer(mu, mu)
    
    return mu, sigma

def calculate_fid_incremental(real_loader, fake_loader, model, batch_size=50, device='cuda', use_amp=True):
    """Calculate FID incrementally between real and generated images."""
    # Calculate statistics for real images
    print("Calculating statistics for real images...")
    m1, s1 = calculate_activation_statistics(real_loader, model, batch_size, device=device, use_amp=use_amp)
    
    # Calculate statistics for generated images
    print("Calculating statistics for generated images...")
    m2, s2 = calculate_activation_statistics(fake_loader, model, batch_size, device=device, use_amp=use_amp)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value

def evaluate_fid(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable AMP if CUDA is available
    use_amp = torch.cuda.is_available()
    if use_amp:
        print("Using Automatic Mixed Precision (AMP)")
    
    # Load pretrained Inception model
    inception_model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
    # Remove the final classification layer
    inception_model.fc = torch.nn.Identity()
    inception_model.aux_logits = False
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    # Load pretrained GAN model
    generator = Decoder(latent_dim=args.latent_dim).to(device)
    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # Data preprocessing - ensure proper dimensions for Inception v3
    transform = transforms.Compose([
        transforms.Resize(299),  # Inception expects 299x299 images
        transforms.CenterCrop(299),  # Ensure exact 299x299 size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load real images
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    # Filter dataset if class is specified
    if args.cls != -1:
        indices = [i for i, label in enumerate(dataset.targets) if label == args.cls]
        dataset.data = dataset.data[indices]
        dataset.targets = [dataset.targets[i] for i in indices]
    
    # Create a subset of 10% of the dataset
    total_size = len(dataset)
    subset_size = total_size // 10
    indices = torch.randperm(total_size)[:subset_size]
    dataset = torch.utils.data.Subset(dataset, indices)
    print(f"Using {len(dataset)} samples (10% of the dataset) for evaluation")
    
    # Reduce number of workers to avoid shared memory issues
    num_workers = min(2, multiprocessing.cpu_count())
    
    # Reduce batch size if CUDA is available to prevent OOM
    if torch.cuda.is_available():
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory // 4
            memory_per_sample = 299 * 299 * 3 * 2  # 2 bytes per half-precision float
            max_batch_size = available_memory // memory_per_sample
            args.batch_size = min(args.batch_size, max_batch_size)
            print(f"Adjusted batch size to {args.batch_size} based on available GPU memory")
        except Exception as e:
            print(f"Could not determine GPU memory, using default batch size: {e}")
            args.batch_size = min(args.batch_size, 32)
    
    # Create real images dataloader
    real_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create fake images dataloader
    fake_dataset = FakeDataset(generator, len(dataset), args.latent_dim, device, use_amp)
    fake_loader = DataLoader(
        fake_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for fake dataset to avoid half-precision issues
    )
    
    # Calculate FID incrementally
    fid_value = calculate_fid_incremental(real_loader, fake_loader, inception_model, args.batch_size, device, use_amp)
    
    print(f"FID score: {fid_value:.2f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'fid_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"FID score: {fid_value:.2f}\n")
        f.write(f"Evaluated on {len(dataset)} samples (10% of the dataset)\n")
    
    return fid_value

def get_args_parser():
    parser = argparse.ArgumentParser('FID Evaluation for GAN')
    
    # Model parameters
    parser.add_argument('--checkpoint_path', default='', type=str,
                       help='Path to GAN checkpoint')
    parser.add_argument('--latent_dim', default=128, type=int)
    
    # Data parameters
    parser.add_argument('--data_path', default='c:/datasets', type=str)
    parser.add_argument('--batch_size', default=32, type=int)  # Reduced default batch size
    parser.add_argument('--cls', default=-1, type=int,
                       help='Class to evaluate (default: -1 for all classes)')
    
    # AMP parameters
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision')
    
    # Output parameters
    parser.add_argument('--output_dir', default='./output/fid_evaluation')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate_fid(args) 
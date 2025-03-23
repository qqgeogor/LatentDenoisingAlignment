import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
from timm.models.vision_transformer import Block, PatchEmbed
# from utils_ibot import SVDPatchPCANoise as PatchPCANoise
from timm.models.layers import trunc_normal_
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.utils.checkpoint
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
import math
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.optim import create_optimizer_v2
from karas_sampler import KarrasSampler, get_sigmas_karras
from einops import rearrange
import seaborn as sns
import torch.nn.functional as F
import contextlib
import utils_ibot as utils
from vit import MaskedAutoencoderViT
from copy import deepcopy
# Set matplotlib backend to avoid GUI dependencies
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')


def fast_logdet_svd(x):
    """Calculate log determinant using SVD."""
    u, s, v = torch.linalg.svd(x, full_matrices=False)
    return torch.sum(torch.log(s))


def fast_logdet_cholesky(x):
    """Calculate log determinant using Cholesky decomposition."""
    L = torch.linalg.cholesky(x)
    return 2 * torch.sum(torch.log(torch.diag(L)))


class SVDPatchPCANoise(nn.Module):
    """Module for applying PCA-based noise to image patches."""
    
    def __init__(self, patch_size=4, noise_scale=0.5, kernel='linear', gamma=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None

    def inverse_transform(self, x_components):
        B, N, C = x_components.shape
        x_components = x_components.reshape(B*N, C)
        return (x_components @ self.ema_eig_vecs.T).reshape(B, N, C)

    def forward(self, x, return_patches=False):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

        # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)

        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean

            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:, idx]

            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs
        
        noise_coeff = torch.randn(all_patches.size(0), self.valid_components).to(all_patches.device)
        scaled_noise = noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0)
        pca_noise = scaled_noise @ self.ema_eig_vecs.T

        # Calculate noise energy per patch
        noise_energy = torch.sum(pca_noise**2, dim=1)  # L2 norm squared per patch
        
        # Normalize to create weights - can use different normalization strategies
        patch_weights = noise_energy / noise_energy.max()  # Simple min-max normalization
        # Alternative: softmax-based weighting
        # patch_weights = F.softmax(noise_energy / temperature, dim=0)
        
        # Reshape weights to match the original patch dimensions
        patch_weights = patch_weights.reshape(B, num_patches_h * num_patches_w)
        
        # Store the weights for later use in the model
        self.patch_weights = patch_weights

        # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise

        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        if return_patches:
            components = all_patches @ self.ema_eig_vecs
            components = components * torch.sqrt(self.ema_eig_vals + 1e-8).unsqueeze(0)
            x_components = components.reshape_as(x_patches)
            return noisy_image, x_components
        else:
            return noisy_image


def R_nonorm(Z, eps=0.5, if_fast=True):
    """Compute the log-determinant term."""
    b = Z.size(-2)
    c = Z.size(-1)
    
    cov = Z.transpose(-2, -1) @ Z
    I = torch.eye(cov.size(-1)).to(Z.device)
    for i in range(len(Z.shape)-2):
        I = I.unsqueeze(0)
    alpha = c/(b*eps)
    
    cov = alpha * cov + I

    if if_fast:
        out = 0.5 * fast_logdet_cholesky(cov)
    else:
        out = 0.5 * torch.logdet(cov)
    return out.mean()


def weighted_simsiam_loss(z_pred, z_target, weights):
    z_pred = F.normalize(z_pred, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    loss_tcr = -R_nonorm(z_pred) * 1e-2
    # Weight the similarity based on patch importance
    cos_sim = torch.cosine_similarity(z_pred, z_target, dim=-1)
    loss_cos = 1 - (cos_sim)
    loss_sim = (loss_cos * weights)
    out = loss_tcr + loss_sim.mean()
    return out, loss_tcr, loss_cos.mean(),loss_sim.mean()



# Visualization and utility functions

def visualize_attention(model, images, save_path='attention_maps', layer_idx=-1):
    """Visualize attention maps for given images."""
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get attention weights
        attn_weights = model.get_attention_maps(images, layer_idx=layer_idx)
        
        # Average over heads
        attn_weights = attn_weights.mean(dim=1)  # [B, N, N]
        
        # Plot attention maps
        n_images = min(4, images.shape[0])
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        
        for i in range(n_images):
            # Plot original image
            img = images[i].cpu()
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0., 1.)
            axes[0, i].imshow(img.permute(1, 2, 0))
            axes[0, i].axis('off')
            axes[0, i].set_title('Original Image')
            
            # Plot attention map
            attn = attn_weights[i].cpu()
            sns.heatmap(attn.numpy(), ax=axes[1, i], cmap='viridis')
            axes[1, i].set_title('Attention Map')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'attention_map_layer_{layer_idx}.png'))
        plt.close()
        
        return attn_weights
    

def visualize_reconstruction(model, images, mask_ratio=0.75, save_path='reconstructions', pca_noiser=None):
    """Visualize original, masked, and reconstructed images."""
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    if pca_noiser is None:
        pca_noiser = SVDPatchPCANoise(patch_size=model.patch_size, noise_scale=(3**0.5), kernel='linear', gamma=1.0)
    
    noised_images, x_components = pca_noiser(images, return_patches=True)
    
    model.eval()
    with torch.no_grad():
        # Get reconstruction and mask
        latent, mask, ids_restore = model.forward_encoder(noised_images, mask_ratio)
        noised_x = torch.randn_like(noised_images) * model.sampler.sigma_max
        sigmas = get_sigmas_karras(
            1, model.sampler.sigma_min, model.sampler.sigma_max, 
            rho=model.sampler.rho, device="cpu"
        )
        
        pred1 = model.denoise(noised_x, latent, mask, ids_restore, sigmas[0])
        pred3 = pred2 = pred1
        
        # Create masked images
        masked_images = images.clone()
        
        # Reshape mask to match image dimensions
        patch_size = model.patch_size
        mask = mask.reshape(
            shape=(mask.shape[0], int(images.shape[2]/patch_size), int(images.shape[3]/patch_size))
        )
        mask = mask.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)
        masked_images = masked_images * (1 - mask.unsqueeze(1).float())
        
        # Normalize images for visualization
        def denormalize_image(img):
            img = img.cpu()
            # Denormalize from CIFAR-10 normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0., 1.)
            return img
        
        # Prepare images for grid
        images = denormalize_image(images)
        noised_images = denormalize_image(noised_images)
        pred1 = denormalize_image(pred1)
        pred2 = denormalize_image(pred2)  
        pred3 = denormalize_image(pred3)

        # Create image grid
        n_images = min(8, images.size(0))
        comparison = torch.cat([
            images[:n_images],
            noised_images[:n_images],
            pred1[:n_images],
            pred2[:n_images],
            pred3[:n_images],
        ])
        
        grid = make_grid(comparison, nrow=n_images, padding=2, normalize=False)
    
    model.train()
    return grid


def save_model(model, optimizer, scheduler, epoch, loss, save_dir='checkpoints'):
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    path = os.path.join(save_dir, f'mae_epoch_{epoch}.pth')
    torch.save(checkpoint, path)
    
    # Save latest checkpoint separately
    latest_path = os.path.join(save_dir, 'mae_latest.pth')
    torch.save(checkpoint, latest_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:  # Load scaler state if it exists
            scaler = GradScaler()
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        else:
            scaler = GradScaler()
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    return 0


def get_args_parser():
    """Configure command-line argument parser."""
    parser = argparse.ArgumentParser('MAE training for CIFAR-10', add_help=False)
    
    # Dataset arguments
    parser.add_argument(
        '--dataset', 
        default='cifar10', 
        type=str, 
        choices=['cifar10', 'tiny-imagenet', 'imagenet-100'],
        help='Dataset to use (cifar10, tiny-imagenet, or imagenet-100)'
    )
    parser.add_argument(
        '--data_path', 
        default='c:/dataset', 
        type=str,
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--noise_scale', 
        default=(3**0.5), 
        type=float,
        help='Noise scale for PCA noise'
    )
    parser.add_argument(
        '--ema_decay', 
        default=0.996, 
        type=float,
        help='EMA decay rate'
    )
    parser.add_argument(
        '--num_views', 
        default=5, 
        type=int,
        help='Number of views for centroid calculation'
    )
    
    # Model parameters
    parser.add_argument(
        '--model_name', 
        default='mae_base', 
        type=str,
        help='Name of the model configuration'
    )
    parser.add_argument(
        '--img_size', 
        default=32, 
        type=int,
        help='Input image size'
    )
    parser.add_argument(
        '--patch_size', 
        default=4, 
        type=int,
        help='Patch size for image tokenization'
    )
    parser.add_argument(
        '--embed_dim', 
        default=192, 
        type=int,
        help='Embedding dimension'
    )
    parser.add_argument(
        '--depth', 
        default=12, 
        type=int,
        help='Depth of transformer'
    )
    parser.add_argument(
        '--num_heads', 
        default=3, 
        type=int,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--decoder_embed_dim', 
        default=96, 
        type=int,
        help='Decoder embedding dimension'
    )
    parser.add_argument(
        '--decoder_depth', 
        default=4, 
        type=int,
        help='Depth of decoder'
    )
    parser.add_argument(
        '--decoder_num_heads', 
        default=3, 
        type=int,
        help='Number of decoder attention heads'
    )
    parser.add_argument(
        '--mlp_ratio', 
        default=4., 
        type=float,
        help='MLP hidden dim ratio'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', 
        default=1600, 
        type=int,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch_size', 
        default=128, 
        type=int,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--lr', 
        default=1.5e-4, 
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', 
        default=0.04, 
        type=float,
        help='Weight decay'
    )
    parser.add_argument(
        '--mask_ratio', 
        default=0.75, 
        type=float,
        help='Ratio of masked patches'
    )
    parser.add_argument(
        '--warmup_epochs', 
        default=10, 
        type=int,
        help='Number of epochs for warmup'
    )
    
    # System parameters
    parser.add_argument(
        '--num_workers', 
        default=8, 
        type=int,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device', 
        default='cuda',
        help='Device to use for training'
    )
    parser.add_argument(
        '--seed', 
        default=0, 
        type=int,
        help='Random seed'
    )
    parser.add_argument(
        '--use_amp', 
        action='store_true',
        default=True,
        help='Use mixed precision training'
    )
    parser.add_argument(
        '--use_checkpoint', 
        action='store_true',
        default=False,
        help='Use gradient checkpointing to save memory'
    )
    
    # Logging and saving
    parser.add_argument(
        '--output_dir', 
        default='/mnt/d/repo/output/mae_cifar10_hldae_dino',
        help='Path where to save checkpoints and logs'
    )
    parser.add_argument(
        '--save_freq', 
        default=10, 
        type=int,
        help='Frequency of saving checkpoints'
    )
    parser.add_argument(
        '--log_freq', 
        default=100, 
        type=int,
        help='Frequency of logging training progress'
    )
    
    # Resume training
    parser.add_argument(
        '--resume', 
        default='',
        help='Resume from checkpoint path'
    )
    parser.add_argument(
        '--start_epoch', 
        default=0, 
        type=int,
        help='Start epoch when resuming'
    )
    
    # LR schedule arguments
    parser.add_argument(
        '--min_lr', 
        default=1e-6, 
        type=float,
        help='Minimum learning rate after decay'
    )
    parser.add_argument(
        '--num_cycles', 
        default=1, 
        type=int,
        help='Number of cycles for cosine decay'
    )
    parser.add_argument(
        '--warmup_lr_init', 
        default=1e-6, 
        type=float,
        help='Initial learning rate for warmup'
    )
    
    # Optimizer arguments
    parser.add_argument(
        '--opt', 
        default='adamw', 
        type=str, 
        metavar='OPTIMIZER',
        help='Optimizer (default: "adamw")'
    )
    parser.add_argument(
        '--opt-eps', 
        default=1e-8, 
        type=float, 
        metavar='EPSILON',
        help='Optimizer Epsilon (default: 1e-8)'
    )
    parser.add_argument(
        '--opt-betas', 
        default=[0.5, 0.999], 
        type=float, 
        nargs='+',
        help='Optimizer Betas (default: [0.5, 0.999])'
    )
    parser.add_argument(
        '--clip-grad', 
        type=float, 
        default=None, 
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument(
        '--momentum', 
        type=float, 
        default=0.9, 
        metavar='M',
        help='SGD momentum (default: 0.9)'
    )

    return parser


def train_mae():
    """Main training function."""
    args = get_args_parser().parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preprocessing
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(
            root=args.data_path, 
            train=True,
            download=True, 
            transform=transform
        )
    else:  # tiny-imagenet or imagenet-100
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        
        # Load Tiny ImageNet or ImageNet-100 dataset using ImageFolder
        trainset = torchvision.datasets.ImageFolder(
            root=args.data_path,
            transform=transform
        )

    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers
    )

    # Initialize PCA noiser
    pca_noiser = SVDPatchPCANoise(
        patch_size=args.patch_size, 
        noise_scale=args.noise_scale,
        kernel='linear',
        gamma=1.0
    )

    # Set up schedulers for EMA momentum and weight decay
    momentum_scheduler = utils.cosine_scheduler(
        base_value=args.ema_decay, 
        final_value=1, 
        epochs=args.epochs, 
        niter_per_ep=len(trainloader), 
        warmup_epochs=0, 
        start_warmup_value=0
    )

    weight_decay_scheduler = utils.cosine_scheduler(
        base_value=args.weight_decay, 
        final_value=args.weight_decay*10, 
        epochs=args.epochs, 
        niter_per_ep=len(trainloader), 
        warmup_epochs=0, 
        start_warmup_value=0
    )

    # Initialize student model
    print('use_checkpoint:', args.use_checkpoint)
    print('use_amp:', args.use_amp)
    model = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        use_checkpoint=args.use_checkpoint
    ).to(device)

    # Initialize teacher model (for EMA)
    teacher_model = deepcopy(model)

    # Create optimizer with explicit betas
    optimizer = create_optimizer_v2(
        model,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        eps=args.opt_eps,
        betas=tuple(args.opt_betas) if args.opt_betas else (0.5, 0.999),
    )
    
    # Use timm's CosineLRScheduler
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,  # Total number of epochs
        lr_min=args.min_lr,
        warmup_t=args.warmup_epochs,  # Warmup epochs
        warmup_lr_init=1e-6,  # Initial warmup learning rate
        cycle_limit=args.num_cycles,  # Number of cycles
        t_in_epochs=True,  # Use epochs for scheduling
        warmup_prefix=True,  # Don't count warmup in cycle
    )

    # Initialize AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            teacher_model.load_state_dict(checkpoint['teacher_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {args.start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")

    # Main training loop
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Set learning rate for epoch
        scheduler.step(epoch)
        
        for i, (imgs, _) in enumerate(trainloader):
            imgs = imgs.to(device)
            noised_images = pca_noiser(imgs)
            optimizer.zero_grad()
            
            it = i + epoch * len(trainloader)
            momentum = momentum_scheduler[it]
            current_weight_decay = weight_decay_scheduler[it]
            # optimizer.param_groups[0]['weight_decay'] = current_weight_decay    

            # Forward pass with mixed precision
            with autocast() if args.use_amp else contextlib.nullcontext():
                
                with torch.no_grad():
                    # target = teacher_model.forward_feature(imgs)
                    target = teacher_model.patch_embed(imgs)
                    # print(h_target.shape)
                    # target = teacher_model.forward_predictor(h_target)
                # h_target = teacher_model.forward_feature(imgs)
                # target = teacher_model.forward_predictor(h_target)

                target = F.normalize(target, dim=-1)
                target = target.detach()

                h_view = model.forward_feature(noised_images)
                view = model.forward_predictor(h_view)
                view = F.normalize(view, dim=-1)

                view = view.reshape(-1, view.size(-1))
                target = target.reshape(-1, target.size(-1))
                patch_weights = pca_noiser.patch_weights.reshape(-1,1)
                dummy_patch_weights = torch.ones_like(patch_weights)    
                loss, loss_tcr, loss_cos, loss_sim = weighted_simsiam_loss(view, target, dummy_patch_weights)
                
            # Backward pass with gradient scaling if using AMP
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            # Update EMA teacher model
            with torch.no_grad():
                for param_q, param_k in zip(model.parameters(), teacher_model.parameters()):
                    param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

            # Log progress
            if i % args.log_freq == args.log_freq - 1:
                avg_loss = total_loss / num_batches
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f'Epoch: {epoch + 1}, Batch: {i + 1}, '
                    f'Loss: {avg_loss:.3f}, '
                    f'Loss_tcr: {loss_tcr:.3f}, '
                    f'Loss_cos: {loss_cos:.3f}, '
                    f'Loss_sim: {loss_sim:.3f}, '
                    f'Patch_weights: {patch_weights.mean(),patch_weights.max(),patch_weights.min()}, '
                    f'Momentum: {momentum:.5f}, '
                    f'Weight_decay: {current_weight_decay:.6f}, '
                    f'LR: {current_lr:.6f}'
                )
        
        # Calculate epoch loss
        epoch_loss = total_loss / num_batches
        
        # Save checkpoint and visualize
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'teacher_model_state_dict': teacher_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': epoch_loss,
                'args': args,
            }
            torch.save(save_dict, checkpoint_path)
            
        print(f'Epoch {epoch + 1} completed. Average loss: {epoch_loss:.3f}')


if __name__ == '__main__':
    train_mae()
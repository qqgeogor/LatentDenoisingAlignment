import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
from timm.models.vision_transformer import Block, PatchEmbed
from vit_transformer import SVDPatchPCANoise as PatchPCANoise
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


def simsiam_loss(z_pred, z_target):
    """Compute SimSiam loss between predictions and targets."""
    z_pred = F.normalize(z_pred, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    loss_tcr = -R_nonorm(z_pred) * 1e-2
    loss_sim = 1 - torch.cosine_similarity(z_pred, z_target, dim=-1).mean()
    out = loss_tcr + loss_sim
    return out


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with Vision Transformer backbone."""
    
    def __init__(
        self, 
        img_size=32, 
        patch_size=4, 
        in_chans=3,
        embed_dim=192, 
        depth=12, 
        num_heads=3,
        decoder_embed_dim=96, 
        decoder_depth=4, 
        decoder_num_heads=3,
        mlp_ratio=4., 
        norm_layer=nn.LayerNorm, 
        norm_pix_loss=False, 
        use_checkpoint=False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embed_dim = embed_dim
        self.patch_size = patch_size    
        self.num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        
        # Projection head for self-supervised learning
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim),
        )
        
        # Encoder components
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_decoder = PatchEmbed(img_size, patch_size, in_chans, decoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # Decoder components
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), 
            requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        
        # Initialize model weights and embeddings
        self.initialize_weights()
        
        # Diffusion sampler
        self.sampler = KarrasSampler(
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            num_steps=40
        )

    def initialize_weights(self):
        """Initialize model weights and positional embeddings."""
        # Initialize position embeddings with sinusoidal encodings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches**.5), 
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            int(self.patch_embed.num_patches**.5), 
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize tokens and other parameters
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Convert images to patches."""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """Convert patches back to images."""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """Apply random masking to embeddings."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_feature(self, x):
        """Forward pass to extract features."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x)  # Enable gradient checkpointing
            else:
                x = blk(x)
        x = self.norm(x)
        return x

    def forward_encoder(self, x, mask_ratio):
        """Forward pass through the encoder with masking."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        _, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        idx = 0
        for blk, noiser in zip(self.blocks, self.noisers):
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            idx += 1
        x = self.norm(x)

        return x, mask, ids_restore

    def sample(self, x):
        """Sample from the diffusion model."""
        return self.sampler.sample(x)

    def forward_decoder(self, x, noised_image, mask, ids_restore):
        """Forward pass through the decoder."""
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # Remove CLS token

        return x

    def forward_loss(self, imgs, pred, mask, weightings=None):
        """Compute reconstruction loss."""
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        if weightings is not None:
            loss = loss * weightings.view(-1, 1)
        loss = loss.mean()
        return loss

    def forward(self, imgs):
        """Forward pass for feature extraction."""
        latent = self.forward_feature(imgs)
        z = self.proj_head(latent)
        return z
    
    def denoise(self, noised_image, latent, mask, ids_restore, sigma):
        """Denoise an image using the model."""
        pred = self.forward_decoder(latent, noised_image, mask, ids_restore)
        pred = self.unpatchify(pred)
        return pred

    def get_attention_maps(self, x, layer_idx=-1):
        """Get attention maps from a specific transformer layer."""
        B = x.shape[0]
        
        # Get patches
        x = self.patch_embed(x)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, 1:, :]
        
        # Add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Pass through transformer blocks until target layer
        target_block = self.blocks[layer_idx]
        
        # Get attention weights from target block
        with torch.no_grad():
            # Forward pass until attention
            attn = target_block.attn
            qkv = attn.qkv(x)
            qkv = rearrange(qkv, 'b n (h d qkv) -> qkv b h n d', h=attn.num_heads, qkv=3)
            q, k, v = qkv[0], qkv[1], qkv[2]   # b h n d
            
            # Calculate attention weights
            attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
            attn_weights = attn_weights.softmax(dim=-1)  # b h n n
            
        return attn_weights


# Helper functions for positional embeddings

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sinusoidal position embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate 2D sinusoidal position embedding from grid."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sinusoidal position embedding from grid."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


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
        pca_noiser = PatchPCANoise(patch_size=model.patch_size, noise_scale=(3**0.5), kernel='linear', gamma=1.0)
    
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
            root=os.path.join(args.data_path, f'{args.dataset}/train'),
            transform=transform
        )

    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers
    )

    # Initialize PCA noiser
    pca_noiser = PatchPCANoise(
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
    teacher_model = MaskedAutoencoderViT(
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
            optimizer.zero_grad()
            
            it = i + epoch * len(trainloader)
            momentum = momentum_scheduler[it]
            current_weight_decay = weight_decay_scheduler[it]
            optimizer.param_groups[0]['weight_decay'] = current_weight_decay    

            # Forward pass with mixed precision
            with autocast() if args.use_amp else contextlib.nullcontext():
                num_views = args.num_views
                
                # Get target features
                target = model(imgs)
                target = F.normalize(target, dim=-1)
                
                # Generate all views at once
                all_views = [imgs]  # Include original image as first view
                for _ in range(args.num_views):
                    noised_imgs = pca_noiser(imgs, return_patches=False)
                    all_views.append(noised_imgs)
                
                # Concatenate all views into a single batch
                concatenated_views = torch.cat(all_views, dim=0)
                
                # Forward pass on concatenated batch
                all_features = model(concatenated_views)
                all_features = F.normalize(all_features, dim=-1)
                
                # Split features back into separate views
                split_features = torch.split(all_features, imgs.size(0), dim=0)
                
                # Original image features
                target = split_features[0]
                
                # Calculate centroid by averaging all views (including target)
                centroid = torch.zeros_like(target[:, 1:])
                for view_features in split_features:
                    centroid += view_features[:, 1:]
                centroid = centroid / len(split_features)
                centroid = centroid.transpose(0, 1)
                
                # Use last view features for cosine similarity loss calculation
                x = split_features[-1]
                
                # Calculate loss components
                loss_centroid = -R_nonorm(centroid, if_fast=False)
                loss = loss_centroid

                loss_cos = 1 - F.cosine_similarity(x, target, dim=-1)
                loss_cos = loss_cos.mean(-1).mean()

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
                    f'Loss_cos: {loss_cos:.3f}, '
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
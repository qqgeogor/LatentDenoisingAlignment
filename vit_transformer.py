# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from DINO and timm library:
GitHub - facebookresearch/dino: PyTorch code for Vision Transformers training with the Self-Supervis
pytorch-image-models/timm/models/vision_transformer.py at main · huggingface/pytorch-image-models
"""

import math
import torch
import torch.nn as nn

from functools import partial

from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import torch.nn.init as init

import torch
import torch.nn as nn

from functools import partial
import numpy as np
from timm.models.vision_transformer import Block, PatchEmbed
from timm.models.layers import trunc_normal_
import os
import torch.utils.checkpoint


import math
from karas_sampler import KarrasSampler
from einops import rearrange
import torch.nn.functional as F


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



def fast_logdet_svd(x):
    u, s, v = torch.linalg.svd(x, full_matrices=False)
    return torch.sum(torch.log(s))  # Just the log determinant

def fast_logdet_cholesky(x):
    L = torch.linalg.cholesky(x)
    return 2*torch.sum(torch.log(torch.diag(L)))  # Just the log determinant

class PatchPCAScaling(nn.Module):
    def __init__(self, patch_size=4, scale_low=0.5, scale_high=1.5):
        super().__init__()
        self.patch_size = patch_size
        self.scale_low = scale_low
        self.scale_high = scale_high

    def forward(self, x):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # ... existing patch extraction code ...
        x_patches = x.unfold(2, p, p).unfold(3, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)
        all_patches = x_patches.reshape(-1, C*p*p)

        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean
            
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
            
            threshold = 1e-6
            mask = S > (S[0] * threshold)
            valid_singular_values = mask.sum()
            if valid_singular_values == 0:
                return x

            # Random scaling of singular values
            scaling_factors = torch.rand(valid_singular_values, device=x.device) * \
                            (self.scale_high - self.scale_low) + self.scale_low
            scaled_S = S[:valid_singular_values] * scaling_factors

            # Reconstruct with scaled singular values
            reconstructed = (U[:, :valid_singular_values] * scaled_S.unsqueeze(0)) @ Vt[:valid_singular_values, :]
            modified_patches = reconstructed + mean

        # ... existing patch reconstruction code ...
        modified_patches = modified_patches.reshape_as(x_patches)
        modified_patches = modified_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        modified_patches = modified_patches.permute(0, 3, 1, 4, 2, 5)
        modified_image = modified_patches.reshape(B, C, H, W)

        return modified_image
    

class PatchPCAShuffle(nn.Module):
    def __init__(self, patch_size=4, noise_scale=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale

    def forward(self, x):
        if not self.training:
            return x

        if self.noise_scale==0:
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
        

        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean
            
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
            
            threshold = 1e-6
            mask = S > (S[0] * threshold)
            valid_singular_values = mask.sum()
            if valid_singular_values == 0:  # Handle case with no valid components
                return x

            # Shuffle top 5 singular values
            n_shuffle = min(5, valid_singular_values)  # Take top 5 or all available if less than 5
            shuffle_indices = torch.arange(n_shuffle)  # Indices 0-4 (top components)
            shuffle_indices = shuffle_indices[torch.randperm(n_shuffle)]  # Shuffle the top indices

            # Create shuffled version of S (only modify top components)
            S_shuffled = S.clone()
            S_shuffled[:n_shuffle] = S_shuffled[shuffle_indices]

            # Reconstruct with shuffled eigenvalues
            reconstructed = (U[:, :valid_singular_values] * S_shuffled.unsqueeze(0)) @ Vt[:valid_singular_values, :]
            modified_patches = reconstructed + mean

        # Reshape noise and add to original patches
        noisy_patches = modified_patches.reshape_as(x_patches)

        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)


        return noisy_image


# class PatchPCANoise(nn.Module):
#     def __init__(self, patch_size=4, noise_scale=0.5):
#         super().__init__()
#         self.patch_size = patch_size
#         self.noise_scale = noise_scale

#     def forward(self, x):
#         if not self.training:
#             return x

#         if self.noise_scale == 0:
#             return x

#         B, C, H, W = x.shape
#         p = self.patch_size
#         assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

#         # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
#         x_patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
#         x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)  # (B, H/p, W/p, C, p, p)
#         num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
#         x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)

#         # Flatten all patches across batch and spatial dimensions
#         all_patches = x_patches.reshape(-1, C*p*p)  # (B*num_patches_total, C*p*p)
        
#         with torch.no_grad():
#             mean = all_patches.mean(dim=0)
#             centered = all_patches - mean
            
#             # Compute covariance matrix
#             n_samples = centered.size(0)
#             cov = (centered.T @ centered) / (n_samples - 1)
            
#             # Use eigendecomposition
#             eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
#             # Sort in descending order since eigh returns ascending
#             eigenvalues = eigenvalues.flip(0)
#             eigenvectors = eigenvectors.flip(1)
            
#             # Filter small eigenvalues
#             threshold = eigenvalues[0] * 1e-6
#             mask = eigenvalues > threshold
#             valid_components = mask.sum()
            
#             if valid_components == 0:
#                 return x
                
#             eigenvalues = eigenvalues[:valid_components]
#             eigenvectors = eigenvectors[:, :valid_components]

            
#             # proj_data = all_patches @ eigenvectors
#             # # Generate noise in the eigenspace
#             # noise = torch.randn(all_patches.size(0), valid_components, device=x.device) * (
#             #     torch.sqrt(eigenvalues.max())
#             # )
            
#             # noise = noise * self.noise_scale + proj_data#*(1-self.noise_scale)
#             proj_data = all_patches @ eigenvectors
#             # Calculate original data norms
#             data_norms = torch.norm(proj_data, dim=1, keepdim=True)
            
#             # Generate normalized noise in eigenspace
#             noise = torch.randn_like(proj_data)
#             noise_norms = torch.norm(noise, dim=1, keepdim=True)
#             noise_normalized = noise / (noise_norms + 1e-8)  # Add epsilon to avoid division by zero
            
#             # Apply random noise scale between 0.8 and 1.2 times the base noise scale
#             current_noise_scale = self.noise_scale * (0.8 + 0.4 * torch.rand(B*num_patches_h*num_patches_w,device=x.device))
#             current_noise_scale = current_noise_scale.view(B*num_patches_h*num_patches_w,1)
#             # Scale noise to match data norms and apply noise scale
#             scaled_noise = noise_normalized * data_norms * current_noise_scale

#             # Apply random noise scale between 0.8 and 1.2 times the base noise scale
            
#             # current_noise_scale = self.noise_scale * (1 + 0.2 * torch.rand_like(eigenvalues))
            
#             # current_noise_scale = self.noise_scale * (0.8 + 0.4 * torch.rand(B*num_patches_h*num_patches_w, device=x.device))
#             # current_noise_scale = current_noise_scale.view(B*num_patches_h*num_patches_w,1)
#             # scaled_noise = noise * current_noise_scale * torch.sqrt(eigenvalues)

#             # scaled_noise = noise * self.noise_scale * torch.sqrt(eigenvalues)
            
#             # Combine with original data (maintain original scale while adding noise)
#             noise = proj_data + scaled_noise
                       
#             # Project noise to original space
#             pca_noise = (noise @ eigenvectors.T)

#             # Add noise to original patches
#             # noisy_patches = all_patches + pca_noise
#             noisy_patches = pca_noise

#         # Reconstruct noisy image from patches
#         noisy_patches = noisy_patches.reshape_as(x_patches)
#         noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
#         noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
#         noisy_image = noisy_patches.reshape(B, C, H, W)

#         return noisy_image


class PatchPCANoise(nn.Module):
    def __init__(self, patch_size=4, noise_scale=0.5,kernel='linear',gamma=1.0,alpha=0.995):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None
        self.alpha = alpha

    def forward(self, x,return_patches=False):
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
            cov = (centered.T @ centered) / (centered.size(0) - 1 + 1e-6)
            if self.ema_cov is None:
                self.ema_cov = cov
            else:
                self.ema_cov = self.ema_cov*self.alpha + cov*(1-self.alpha)
            eig_vals, eig_vecs = torch.linalg.eigh(self.ema_cov+1e-6*torch.eye(self.ema_cov.size(0)).to(self.ema_cov.device))
            # Reverse to get descending order
            eig_vals = eig_vals.flip(0)
            eig_vecs = eig_vecs.flip(1)
            valid_components = torch.sum(eig_vals > 1e-6)
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]

        # Generate PCA-space noise
        # noise_coeff = torch.randn_like(all_patches)  # (B*num_patches_total, C*p*p)
        noise_coeff = torch.randn(all_patches.size(0),valid_components).to(all_patches.device)  # (B*num_patches_total, C*p*p)
        scaled_noise = noise_coeff * (eig_vals.sqrt() * self.noise_scale).unsqueeze(0)
        # scaled_noise = noise_coeff * (self.noise_scale)
        pca_noise = scaled_noise @ eig_vecs.T

        # Reshape noise and add to original patches
        pca_noise = pca_noise.reshape_as(x_patches)
        noisy_patches = x_patches + pca_noise

        # Reconstruct noisy image from patches
        noisy_patches = noisy_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        noisy_patches = noisy_patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
        noisy_image = noisy_patches.reshape(B, C, H, W)

        if return_patches:
            components = all_patches @ eig_vecs
            components = components * torch.sqrt(eig_vals + 1e-8).unsqueeze(0)
            x_components = components.reshape_as(x_patches)
            return noisy_image,x_components
        else:
            return noisy_image


class SVDPatchPCANoise(nn.Module):
    def __init__(self, patch_size=4, noise_scale=0.5,kernel='linear',gamma=1.0):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale
        self.ema_cov = None


    def inverse_transform(self,x_components):
        B,N,C = x_components.shape
        x_components = x_components.reshape(B*N,C)
        # print('x_components',x_components.shape)
        return (x_components @ self.ema_eig_vecs.T).reshape(B,N,C)

    def forward(self, x,return_patches=False):
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
            # cov = (centered.T @ centered) / (centered.size(0))
            # cov = cov+1e-6*torch.eye(cov.size(0)).to(cov.device)

            # eig_vals, eig_vecs = torch.linalg.eigh(cov)
            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1 + 1e-6)
            eig_vecs = v.T

            idx = torch.argsort(eig_vals,descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:,idx]

            # print('eig_vals',eig_vals.shape)
            # print('eig_vecs',eig_vecs.shape)
            valid_components = torch.sum(eig_vals > 1e-6)
            self.valid_components = valid_components
            eig_vals = eig_vals[:valid_components]
            eig_vecs = eig_vecs[:, :valid_components]
            
            self.ema_eig_vals = eig_vals
            self.ema_eig_vecs = eig_vecs
            
            # # Reverse to get descending order
            # eig_vals = eig_vals.flip(0)
            # eig_vecs = eig_vecs.flip(1)
        
        # random_scale = torch.rand(all_patches.size(0)).unsqueeze(1).to(all_patches.device)* self.noise_scale
        # random_scale = random_scale.expand(-1,self.valid_components)
        # noise_coeff = torch.randn(all_patches.size(0),self.valid_components).to(all_patches.device)  # (B*num_patches_total, C*p*p)
        # scaled_noise = random_scale*noise_coeff * (self.ema_eig_vals.sqrt()).unsqueeze(0)

        
        noise_coeff = torch.randn(all_patches.size(0),self.valid_components).to(all_patches.device)  # (B*num_patches_total, C*p*p)
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
            return noisy_image,x_components
        else:
            return noisy_image
        
class SingularValueNoise(nn.Module):
    def __init__(self, patch_size=4,noise_scale=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale

    def forward(self, x):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # ... existing patch extraction code ...
        x_patches = x.unfold(2, p, p).unfold(3, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)
        all_patches = x_patches.reshape(-1, C*p*p)

        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean
            
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
            
            threshold = 1e-6
            mask = S > (S[0] * threshold)
            valid_singular_values = mask.sum()
            if valid_singular_values == 0:
                return x
            n_samples = B * num_patches_h * num_patches_w
            noise_coeff = torch.randn(B*num_patches_h*num_patches_w, valid_singular_values, device=x.device)
            # Add noise directly to singular values
            scaled_noise = noise_coeff * (
                (S / torch.sqrt(torch.tensor(n_samples - 1, dtype=torch.float, device=x.device))) 
                * self.noise_scale
            ).unsqueeze(0)
            S_scaled = S[:valid_singular_values] + scaled_noise
            
            # Reconstruct with modified singular values
            reconstructed = (U[:, :valid_singular_values] * S_scaled.unsqueeze(0)) @ Vt[:valid_singular_values, :]
            modified_patches = reconstructed + mean

        # ... existing patch reconstruction code ...
        modified_patches = modified_patches.reshape_as(x_patches)
        modified_patches = modified_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        modified_patches = modified_patches.permute(0, 3, 1, 4, 2, 5)
        modified_image = modified_patches.reshape(B, C, H, W)

        return modified_image
    

class LatentPatchPCANoise(nn.Module):
    def __init__(self, patch_size=4, noise_scale=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.noise_scale = noise_scale

    def forward(self, x):
        if not self.training:
            return x

        if self.noise_scale==0:
            return x
        x_cls,x_patches = x[:,:1],x[:,1:]
        B,N,C = x_patches.shape
        # Flatten all patches across batch and spatial dimensions
        all_patches = x_patches.reshape(B*N,C)
        
        # Compute PCA components
        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean
            n = centered.size(0)
            u, s, v = torch.linalg.svd(centered, full_matrices=False)
            eig_vals = (s**2)/(n-1)
            eig_vecs = v.T
            
            idx = torch.argsort(eig_vals, descending=True)
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:,idx]

            
        # Generate PCA-space noise
        noise_coeff = torch.randn_like(all_patches)  # (B*num_patches_total, C*p*p)
        scaled_noise = noise_coeff * (eig_vals.sqrt() * self.noise_scale).unsqueeze(0)
        pca_noise = scaled_noise @ eig_vecs.T
        
        pca_noise = pca_noise.reshape(B,N,C)

        noisy_patches = pca_noise + x_patches
        x = torch.cat([x_cls,noisy_patches],dim=1)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class ISTAAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., step_size=0.1,lambd=0.5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.step_size = step_size
        self.lambd = lambd

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.normalize(q,dim=-1)
        k = F.normalize(k,dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)

        
        attn = self.attn_drop(attn)
        
        # b,h,l,l * b h l,d -> b,h,l,d -> b,l,h,d -> b,l,c
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        weight = attn
        x = v
        
        x1 = (weight @ x) # b,h,l,d
        grad_1 = weight.transpose(-1,-2) @ x1
        grad_2 = weight.transpose(-1,-2) @ x

        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd
        x = F.relu(x + grad_update)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class ISTA(nn.Module):
    def __init__(self, dim, step_size=0.1,lambd=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = step_size
        self.lambd = lambd

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output
    


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim , bias=qkv_bias)
        self.k = nn.Linear(dim, dim , bias=qkv_bias)
        self.v = nn.Linear(dim, dim , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q,k,v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q(q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)
        k = self.k(k).reshape(B, M, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)   
        v = self.v(v).reshape(B, M, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x#, attn
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbedIbot(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        return self.proj(x)

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), return_all_tokens=False, 
                 init_values=0, use_mean_pooling=False, masked_im_modeling=False,pretrained_cfg=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens

        self.patch_embed = PatchEmbedIbot(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                init_values=init_values)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # masked image modeling
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at `interpolate_pos_encoding(x, pos_embed)` doesnt return correct dimension for images that is not squa
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        ))
        
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=None):
        B, nc, w, h = x.shape
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_tokens=None, mask=None):
        # mim
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return x
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_num_layers(self):
        return len(self.blocks)

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x

class PatchPCAScaling(nn.Module):
    def __init__(self, patch_size=4, scale_low=0.5, scale_high=1.5,noise_scale=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.noise_scale = noise_scale

    def forward(self, x):
        if not self.training:
            return x

        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"

        # Extract patches (B, C, H, W) -> (B, num_patches, C*p*p)
        x_patches = x.unfold(2, p, p).unfold(3, p, p)
        x_patches = x_patches.permute(0, 2, 3, 1, 4, 5)
        num_patches_h, num_patches_w = x_patches.size(1), x_patches.size(2)
        x_patches = x_patches.reshape(B, num_patches_h * num_patches_w, C * p * p)
        all_patches = x_patches.reshape(-1, C*p*p)

        with torch.no_grad():
            mean = all_patches.mean(dim=0)
            centered = all_patches - mean
            
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
            
            threshold = 1e-6
            mask = S > (S[0] * threshold)
            valid_singular_values = mask.sum()
            if valid_singular_values == 0:
                return x

            # Generate random scale for this batch
            # current_scale = self.base_scale + torch.rand(1, device=x.device) * (2*self.scale_range) - self.scale_range

            # Random scaling of singular values
            scaling_factors = torch.rand(valid_singular_values, device=x.device) * \
                            (self.scale_high - self.scale_low) + self.scale_low
            scaled_S = S[:valid_singular_values] * scaling_factors

            # Reconstruct with scaled singular values
            reconstructed = (U[:, :valid_singular_values] * scaled_S.unsqueeze(0)) @ Vt[:valid_singular_values, :]
            modified_patches = reconstructed + mean

        # Reshape and reconstruct the image
        modified_patches = modified_patches.reshape_as(x_patches)
        modified_patches = modified_patches.reshape(B, num_patches_h, num_patches_w, C, p, p)
        modified_patches = modified_patches.permute(0, 3, 1, 4, 2, 5)
        modified_image = modified_patches.reshape(B, C, H, W)

        return modified_image

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model

class VitAutoEncoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192, depth=12, 
                 num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        # Encoder
        self.encoder = VisionTransformer(
            img_size=[img_size], patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
            return_all_tokens=True  # Return all tokens including [CLS]
        )
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                norm_layer=norm_layer,)
            for _ in range(4)])

        self.decoder_norm = norm_layer(embed_dim)
        self.decoder_proj = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward(self, x):
        # Encode
        x = self.encoder(x,return_all_tokens=True)  # Returns all tokens including [CLS]

        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)
            
        x = self.decoder_norm(x)
        x = self.decoder_proj(x)
        x = x[:, 1:]
        output = self.unpatchify(x)  # Reshape to image
        
        return output  # Apply tanh to match original range [-1, 1]
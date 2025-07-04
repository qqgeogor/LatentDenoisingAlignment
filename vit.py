import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
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
# from ddpm_sampler import DDPMSampler
import torch.nn.functional as F
from einops import rearrange
import seaborn as sns
from masks.utils import apply_masks

import os
os.environ['MPLBACKEND'] = 'Agg'  # Set this before importing matplotlib
import matplotlib.pyplot as plt


def R_nonorm(Z, eps=0.5):
    """Compute the log-determinant term."""
    b = Z.size(-2)
    c = Z.size(-1)
    
    cov = Z.transpose(-2, -1) @ Z
    I = torch.eye(cov.size(-1)).to(Z.device)
    for i in range(len(Z.shape)-2):
        I = I.unsqueeze(0)
    alpha = c/(b*eps)
    
    cov = alpha * cov + I


    out = 0.5 * torch.logdet(cov)
    return out.mean()


class InvarianceFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_flows=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_flows = n_flows
        
        # Create a series of invertible coupling layers
        self.flows = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dim) 
            for _ in range(n_flows)
        ])
    
    def forward(self, x):
        """Forward transformation: x -> z"""
        z = x
        log_det_sum = 0
        
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det
            
        return z, log_det_sum
    
    def inverse(self, z):
        """Inverse transformation: z -> x"""
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x
    
    def forward_loss(self, x):
        """Compute the flow-based loss using change of variables formula"""
        # Forward pass to get latent z and log determinant
        z, log_det = self.forward(x)
        
        # Prior likelihood (assuming standard normal prior)
        prior_ll = -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
        
        # Total loss = negative log likelihood
        loss = -(prior_ll + log_det)
        
        # Add total correlation regularization
        tcr_loss = -R_nonorm(F.normalize(z, dim=-1))
        
        total_loss = loss.mean() + tcr_loss
        return total_loss, loss.mean(), tcr_loss

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Split dimensions in half
        self.split_dim = input_dim // 2
        
        # Scale and translation networks
        self.net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (input_dim - self.split_dim) * 2)  # Output scale and translation
        )
        
    def forward(self, x):
        """Forward pass: x -> z"""
        x1, x2 = torch.split(x, [self.split_dim, self.input_dim - self.split_dim], dim=-1)
        
        h = self.net(x1)
        scale, translate = torch.chunk(h, 2, dim=-1)
        scale = torch.tanh(scale)  # Bound scaling factor
        
        z2 = x2 * torch.exp(scale) + translate
        z = torch.cat([x1, z2], dim=-1)
        
        # Log determinant of the transformation
        log_det = scale.sum(dim=-1)
        
        return z, log_det
    
    def inverse(self, z):
        """Inverse pass: z -> x"""
        z1, z2 = torch.split(z, [self.split_dim, self.input_dim - self.split_dim], dim=-1)
        
        h = self.net(z1)
        scale, translate = torch.chunk(h, 2, dim=-1)
        scale = torch.tanh(scale)
        
        x2 = (z2 - translate) * torch.exp(-scale)
        x = torch.cat([z1, x2], dim=-1)
        
        return x
    


class PatchAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim if latent_dim is not None else hidden_dim // 2
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        z = self.encoder(x)
        z = F.normalize(z,dim=-1)
        return z
        
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def prior_ll(self, z):
        '''
        Compute the prior log likelihood of the latent code , assuming standard normal prior
        '''
        return -0.5 * (z**2 + math.log(2 * math.pi)).sum(dim=1)
    

    
    def forward_loss(self, x):
        """
        Compute the AE loss for the given input.
        
        Args:
            x: Input tensor
            
        Returns:
            A tuple containing:
            - total_loss: The combined reconstruction and KL divergence loss
            - recon_loss: The reconstruction loss component
            - kl_loss: The KL divergence loss component
        """
        # Forward pass through the VAE
        x_recon, z = self.forward(x)
        
        tcr_loss = -R_nonorm(F.normalize(z,dim=-1))
        
        # Compute reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=-1).mean()
        # recon_loss = (x_recon - x).pow(2).mean(-1).mean()
        
        # Compute KL divergence
        kl_loss = -self.prior_ll(z).mean()
        
        # Combine losses
        total_loss = recon_loss + kl_loss + tcr_loss
        
        # Return average loss over batch
        return total_loss, recon_loss, kl_loss,tcr_loss,z

class PatchVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim if latent_dim is not None else hidden_dim // 2
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Variational part
        self.fc_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar,z
    
    def kl_divergence(self, mu, logvar):
        # KL divergence between N(mu, sigma) and N(0, 1)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    def forward_loss(self, x):
        """
        Compute the VAE loss for the given input.
        
        Args:
            x: Input tensor
            
        Returns:
            A tuple containing:
            - total_loss: The combined reconstruction and KL divergence loss
            - recon_loss: The reconstruction loss component
            - kl_loss: The KL divergence loss component
        """
        # Forward pass through the VAE
        x_recon, mu, logvar,z = self.forward(x)
        
        tcr_loss = -R_nonorm(F.normalize(z,dim=-1))

        # Compute reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=-1).mean()
        # recon_loss = (x_recon - x).pow(2).mean(-1).mean()
        
        # Compute KL divergence
        kl_loss = self.kl_divergence(mu, logvar).mean()
        
        # Combine losses
        total_loss = recon_loss + kl_loss + tcr_loss
        
        # Return average loss over batch
        return total_loss, recon_loss, kl_loss,tcr_loss

class DyTanh(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3,
                 decoder_embed_dim=96, decoder_depth=4, decoder_num_heads=3,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, use_checkpoint=False):
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
        
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim),
        )
        
        
        self.discriminator_head = nn.Linear(embed_dim,1)
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_decoder = PatchEmbed(img_size, patch_size, in_chans, decoder_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.initialize_weights()
        # self.sampler = DDPMSampler()

    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                          int(self.patch_embed.num_patches**.5), 
                                          cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 
                                                   int(self.patch_embed.num_patches**.5), 
                                                   cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize tokens and other parameters
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def random_masking(self, x, mask_ratio):
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
    

    def forward_predictor(self, x, masks_x=None, masks=None):

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.decoder_embed(x)

        # -- add positional embedding to x tokens
        if masks_x is not None:        
            x_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1)
            # print(x_pos_embed.shape)

            x += apply_masks(x_pos_embed, masks_x)
        else:
            x += self.decoder_pos_embed

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        if masks is not None:        
            pos_embs = self.decoder_pos_embed.repeat(B, 1, 1)
            pos_embs = apply_masks(pos_embs, masks)
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.proj_head(x)

        return x
    
    

    def forward_feature(self, x,masks=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        if masks is not None:
            x = apply_masks(x, masks)
        
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
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x)  # Enable gradient checkpointing
            else:
                x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    
    # def forward_predictor(self, x):
    #     x = self.decoder_embed(x)

    #     x = x + self.decoder_pos_embed

    #     for blk in self.decoder_blocks:
    #         if self.use_checkpoint and self.training:
    #             x = torch.utils.checkpoint.checkpoint(blk, x)  # Enable gradient checkpointing
    #         else:
    #             x = blk(x)
    #     x = self.decoder_norm(x)
    #     x = self.proj_head(x)
    #     # x = self.decoder_pred(x)
    #     x = x[:, 1:, :]  # Remove CLS token
        

    #     return x


    def forward_decoder(self, x,noised_image=None, mask=None,ids_restore=None):
        x = self.decoder_embed(x)
        
        
        if noised_image is not None:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

            x_dec = self.patch_embed_decoder(noised_image)

            x_ = (1-mask.unsqueeze(-1)) * x_ + mask.unsqueeze(-1) * x_dec
        
            
            x = torch.cat([x[:, :1, :], x_], dim=1)

            x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x)  # Enable gradient checkpointing
            else:
                x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # Remove CLS token

        return x


    def forward_generator(self, x):
        x = self.decoder_embed(x)
        
        L = self.num_patches
        
        mask_tokens = self.mask_token.repeat(x.shape[0], L, 1)
        
        x = torch.cat([x.unsqueeze(1), mask_tokens], dim=1)
        
        x = x + self.decoder_pos_embed
        
        
        for blk in self.decoder_blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x)  # Enable gradient checkpointing
            else:
                x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # Remove CLS token
        x = self.unpatchify(x)

        return x


    def forward_loss(self, imgs, pred, mask,weightings=None):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        if weightings is not None:
            loss = loss * weightings.view(-1,1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        
        noised_image,t = self.sampler.add_noise(imgs)
        weightings = torch.sqrt(self.sampler.alphas_cumprod[t]) / (1 - self.sampler.alphas_cumprod[t])

        weightings = weightings.view(-1, 1)  # Reshape to match the batch size
        weightings = torch.ones_like(weightings)
        
        # print('snrs',snrs.min(),snrs.max())
        # print('sigma',sigma.min(),sigma.max())
        # exit()
        
        pred = self.forward_decoder(latent, noised_image, mask, ids_restore)
        loss = self.forward_loss(imgs, pred, mask,weightings)
        return loss, pred, mask
    
    
    def denoise(self, noised_image, latent, mask,ids_restore):
        
        pred = self.forward_decoder(latent, noised_image, mask, ids_restore)
        pred = self.unpatchify(pred)
        return pred
    

    def get_attention_maps(self, x, layer_idx=-1):
        """
        Get attention maps from a specific transformer layer
        Args:
            x: Input tensor
            layer_idx: Index of transformer layer to visualize (-1 for last layer)
        Returns:
            attention_maps: [B, H, N, N] attention weights
        """
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
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
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

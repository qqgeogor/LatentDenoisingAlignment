import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
from timm.models.vision_transformer import Block, PatchEmbed
from utils_ibot import SVDPatchPCANoise as PatchPCANoise
from timm.models.layers import trunc_normal_
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.utils.checkpoint
from torch.amp import autocast, GradScaler
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
# Set matplotlib backend to avoid GUI dependencies
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
# from masks.multiblock import MaskCollator

from masks.random import MaskCollator
from masks.utils import apply_masks


def fast_logdet_svd(x):
    """Calculate log determinant using SVD."""
    u, s, v = torch.linalg.svd(x, full_matrices=False)
    return torch.sum(torch.log(s))


def fast_logdet_cholesky(x):
    """Calculate log determinant using Cholesky decomposition."""
    L = torch.linalg.cholesky(x)
    return 2 * torch.sum(torch.log(torch.diag(L)))



def R_nonorm(Z, eps=0.5, if_fast=False):
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


def simsiam_loss(z_pred, z_target,lambda1=1e-2):
    z_pred = F.normalize(z_pred, dim=-1)
    z_target = F.normalize(z_target, dim=-1)
    loss_tcr = -R_nonorm(z_pred) * lambda1
    cos_sim = torch.cosine_similarity(z_pred, z_target, dim=-1)
    loss_cos = 1 - cos_sim
    out = loss_tcr + loss_cos.mean()
    return out, loss_tcr, loss_cos.mean()


def visualize_reconstruction(model, images, mask_ratio=0.75, save_path='reconstructions',noise_scale=1.,pca_noiser=None):
    """Visualize original, masked, and reconstructed images"""
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    if pca_noiser is None:
        pca_noiser = PatchPCANoise(patch_size=model.patch_size, noise_scale=noise_scale)
    noised_images,pca_noise = pca_noiser(images,return_patches=True)
    model.eval()
    with torch.no_grad():
        # Get reconstruction

        z = model.forward_feature(noised_images)
        
        pred1 = model.forward_decoder(z)
        
        pred1 = model.unpatchify(pred1)
        
        pred2 = pca_noise
        pred3 = noised_images - pred1
 
        
        # Normalize images for visualization
        def denormalize_image(img):
            img = img.cpu()
            # Denormalize from CIFAR-10 normalization
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0., 1.)
            return img
        
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
        default=((1/3)**0.5), 
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

    parser.add_argument(
        '--enc_mask_scale',
        type=float,
        default=(0.85, 1.0),
        help='Enc mask scale'
    )   
    parser.add_argument(
        '--pred_mask_scale',
        type=float,
        default=(0.15, 0.2),
        help='Pred mask scale'
    )

    parser.add_argument(
        '--aspect_ratio',
        type=float,
        default=(0.75, 1.5),
        help='Aspect ratio'
    )
    
    parser.add_argument(
        '--nenc',
        type=int,
        default=1,
        help='Number of enc'
    )   
    parser.add_argument(
        '--npred',
        type=int,
        default=4,
        help='Number of pred'
    )

    parser.add_argument(
        '--min_keep',
        type=int,
        default=4,
        help='Minimum keep'
    )
    
    parser.add_argument(
        '--allow_overlap',
        type=bool,
        default=False,
        help='Allow overlap'
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
    
    # Initialize mask collator
    mask_collator = MaskCollator(
        ratio=(0.75,0.75),
        input_size=(args.img_size, args.img_size),
        patch_size=args.patch_size,
    )
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
            transform=transform,
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
            transform=transform,
        )

    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size,
        shuffle=True, 
        collate_fn=mask_collator,
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

    imgs = next(iter(trainloader))[0][0].to(device)
    grid = visualize_reconstruction(model,imgs,pca_noiser=pca_noiser,save_path=args.output_dir)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, f'reconstruction_epoch_{args.start_epoch}.png'))
    plt.close()

    # Main training loop
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Set learning rate for epoch
        scheduler.step(epoch)
        
        for i, (batch, masks_enc, masks_pred) in enumerate(trainloader):
            # imgs = batch.to(device)
            # print(len(batch))
            imgs,labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # for mask_enc in masks_enc:
            #     print(mask_enc.shape) 
            #     print(mask_enc,mask_enc.max(),mask_enc.min()) 



            # for mask_pred in masks_pred:
            #     print(mask_pred.shape)
            #     print(mask_pred,mask_pred.max(),mask_pred.min())
            
            # print('batch.shape:', len(batch))
            # print('masks_enc.shape:', len(masks_enc))
            # print('masks_pred.shape:', len(masks_pred))
            # exit()

            optimizer.zero_grad()
            
            it = i + epoch * len(trainloader)
            momentum = momentum_scheduler[it]
            current_weight_decay = weight_decay_scheduler[it]
            # optimizer.param_groups[0]['weight_decay'] = current_weight_decay    

            # Forward pass with mixed precision
            with autocast('cuda') if args.use_amp else contextlib.nullcontext():
                
                noised_image,pca_noise = pca_noiser(imgs,return_patches=True)
                z = model.forward_feature(noised_image,masks=masks_enc)

                pred = model.forward_decoder(z)

                target = model.patchify(pca_noise)
                target = apply_masks(target,masks_enc)

                loss = (pred-target)**2
                loss = loss.mean(-1).mean()

                
                
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
                    f'Momentum: {momentum:.5f}, '
                    f'Weight_decay: {current_weight_decay:.6f}, '
                    f'LR: {current_lr:.6f}'
                )
        
        # Calculate epoch loss
        epoch_loss = total_loss / num_batches
        grid = visualize_reconstruction(model,imgs,pca_noiser=pca_noiser,save_path=args.output_dir)
        plt.figure(figsize=(15, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, f'reconstruction_epoch_{args.start_epoch}.png'))
        plt.close()
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
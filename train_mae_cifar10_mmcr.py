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
from utils_ibot import R_nonorm
import utils_ibot as utils
from vit import MaskedAutoencoderViT
import torch.nn.functional as F
from tqdm import tqdm

def mcr_nv_loss(ps):
    N = ps.shape[0]
    C = ps.shape[-1]
    B = ps.shape[-2]

    
    centroid = ps.mean(dim=0)

    loss_mmcr = -R_nonorm(centroid)

    return loss_mmcr.mean()



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

def visualize_reconstruction(model, images, mask_ratio=0.75, save_path='reconstructions'):
    """Visualize original, masked, and reconstructed images"""
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get reconstruction and mask
        loss, pred, mask = model(images, mask_ratio)
        
        # Convert predictions to images
        pred = model.unpatchify(pred)
        
        # Create masked images
        masked_images = images.clone()
        
        # Reshape mask to match image dimensions
        patch_size = model.patch_size
        mask = mask.reshape(shape=(mask.shape[0], int(images.shape[2]/patch_size), int(images.shape[3]/patch_size)))
        mask = mask.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2)
        masked_images = masked_images * (1 - mask.unsqueeze(1).float())
        
        # Normalize images for visualization
        def normalize_image(img):
            img = img.cpu()
            # Denormalize from CIFAR-10 normalization
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0., 1.)
            return img
        
        # Prepare images for grid
        images = normalize_image(images)
        masked_images = normalize_image(masked_images)
        pred = normalize_image(pred)
        
        # Create image grid
        n_images = min(8, images.size(0))
        comparison = torch.cat([
            images[:n_images],
            masked_images[:n_images],
            pred[:n_images]
        ])
        
        grid = make_grid(comparison, nrow=n_images, padding=2, normalize=False)
        return grid

def save_model(model, optimizer, scheduler, epoch, loss, save_dir='checkpoints'):
    """Save model checkpoint"""
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
    """Load model checkpoint"""
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
    parser = argparse.ArgumentParser('MAE training for CIFAR-10', add_help=False)
    
    # Add dataset arguments
    parser.add_argument('--dataset', default='tiny-imagenet', type=str, choices=['cifar10', 'tiny-imagenet','imagenet-100'],
                        help='Dataset to use (cifar10 or tiny-imagenet or imagenet-100 )')
    parser.add_argument('--data_path', default='c:/dataset', type=str,
                        help='Path to dataset root directory')

    # Model parameters
    parser.add_argument('--model_name', default='mae_base', type=str,
                        help='Name of the model configuration')
    parser.add_argument('--img_size', default=32, type=int,
                        help='Input image size')
    parser.add_argument('--patch_size', default=4, type=int,
                        help='Patch size for image tokenization')
    parser.add_argument('--embed_dim', default=192, type=int,
                        help='Embedding dimension')
    parser.add_argument('--depth', default=12, type=int,
                        help='Depth of transformer')
    parser.add_argument('--num_heads', default=3, type=int,
                        help='Number of attention heads')
    parser.add_argument('--decoder_embed_dim', default=96, type=int,
                        help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='Depth of decoder')
    parser.add_argument('--decoder_num_heads', default=3, type=int,
                        help='Number of decoder attention heads')
    parser.add_argument('--mlp_ratio', default=4., type=float,
                        help='MLP hidden dim ratio')
    
    # Training parameters
    parser.add_argument('--epochs', default=1600, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--lr', default=1.5e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='Weight decay')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Ratio of masked patches')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Number of epochs for warmup')
    
    # System parameters
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='Use gradient checkpointing to save memory')
    
    # Logging and saving
    parser.add_argument('--output_dir', default='F:/output/mae_cifar10_ddpm',
                        help='Path where to save checkpoints and logs')
    parser.add_argument('--save_freq', default=20, type=int,
                        help='Frequency of saving checkpoints')
    parser.add_argument('--log_freq', default=100, type=int,
                        help='Frequency of logging training progress')
    parser.add_argument('--num_views', default=32, type=int,
                        help='Number of views for multi-view training')
    # Resume training
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint path')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Start epoch when resuming')
    
    # Update LR schedule arguments
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate after decay')
    parser.add_argument('--num_cycles', default=1, type=int,
                        help='Number of cycles for cosine decay')
    parser.add_argument('--warmup_lr_init', default=1e-6, type=float,
                        help='Initial learning rate for warmup')
    
    # Add optimizer arguments
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=[0.5, 0.999], type=float, nargs='+',
                        help='Optimizer Betas (default: [0.9, 0.999])')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    return parser

# Add MultiViewTransform class
class MultiViewTransform:
    
    def __init__(self, base_transform,n_views=20):
        self.n_views = n_views
        self.base_transform = base_transform

    def __call__(self, x):
        views = []
        for _ in range(self.n_views):
            views.append(self.base_transform(x))
        return views


def train_mae():
    args = get_args_parser().parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    transform=MultiViewTransform(transform,args.num_views)
    # Load Tiny ImageNet dataset using ImageFolder
    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(args.data_path),
        transform=transform
    )





    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # Initialize model
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

    # Create optimizer with explicit betas
    optimizer = create_optimizer_v2(
        model,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        eps=args.opt_eps,
        betas=tuple(args.opt_betas) if args.opt_betas else (0.9, 0.999),  # Provide default tuple
    )
    

    # Add learning rate scheduler
    lr_scheduler = utils.cosine_scheduler(
        base_value=args.lr, final_value=1e-6, epochs=args.epochs, niter_per_ep=len(trainloader), warmup_epochs=5, start_warmup_value=1e-6
    )

    weight_decay_scheduler = utils.cosine_scheduler(
        base_value=args.weight_decay, final_value=args.weight_decay*10, epochs=args.epochs, niter_per_ep=len(trainloader), warmup_epochs=0, start_warmup_value=0
    )


    # Initialize AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {args.start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")


    
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        
        for i, (images, _) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            # img1, img2 = images[0].to(device), images[1].to(device)  # Unpack the two views
            views = [img.to(device) for img in images]
            it = i + epoch * len(trainloader)
            lr = lr_scheduler[it]
            weight_decay = weight_decay_scheduler[it]
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[0]['weight_decay'] = weight_decay

            # Compute loss
            optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with autocast(enabled=args.use_amp):
               # Process all views in a single batch for efficiency
                cat_views = torch.cat(views, dim=0)
                cat_features = model.forward_feature(cat_views)[:,0]
                cat_projections = model.proj_head(cat_features)
                
                # Split the results back into separate views
                ps = cat_projections.chunk(len(views), dim=0)
                ps = torch.stack(ps, dim=0)
                ps = F.normalize(ps,dim=-1)
                
                
                mmcr_loss = mcr_nv_loss(ps)

                loss_cos = (1 - F.cosine_similarity(ps[0], ps[-1], dim=-1).mean())

                
                # Final loss
                loss = mmcr_loss  # + loss_gp*args.gp_weight
            
            # Use scaler for backward and optimizer step if AMP is enabled
            num_batches +=1
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            
            if i % args.log_freq == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'Loss_mmcr: {mmcr_loss.item():.4f}, '
                      f'Loss: {loss.item():.4f},  Loss_cos: {loss_cos.item():.4f}, '
                      f'LR: {lr:.6f}, WD: {weight_decay:.6f}')
            
        epoch_loss = total_loss / num_batches
        

        # Save checkpoint and visualize
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': epoch_loss,
                'args': args,
            }
            torch.save(save_dict, checkpoint_path)
            
        
        print(f'Epoch {epoch + 1} completed. Average loss: {epoch_loss:.3f}')

if __name__ == '__main__':
    train_mae() 
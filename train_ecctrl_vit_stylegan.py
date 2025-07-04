import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from utils_ibot import SVDPCANoise,cosine_scheduler
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from contextlib import nullcontext
from vit_ibot import MaskedAutoencoderViT
from stylegan import create_conditional_stylegan

# Set matplotlib backend to avoid GUI dependencies
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')


def zero_centered_gradient_penalty(samples, critics):
    grad, = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True)
    return grad.square().sum([1, 2, 3])


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
def R(Z,eps=0.5):
    c = Z.shape[-1]
    b = Z.shape[-2]
    
    Z = F.normalize(Z, p=2, dim=-1)
    cov = Z.T @ Z
    I = torch.eye(cov.size(-1)).to(Z.device)
    alpha = c/(b*eps)
    
    cov = alpha * cov +  I

    out = 0.5*torch.logdet(cov)
    return out.mean()


def R_nonorm(Z,eps=0.5):
    c = Z.shape[-1]
    b = Z.shape[-2]
    
    Z = Z
    cov = Z.T @ Z
    I = torch.eye(cov.size(-1)).to(Z.device)
    alpha = c/(b*eps)
    
    cov = alpha * cov +  I

    out = 0.5*torch.logdet(cov)
    return out.mean()

def mcr(Z1,Z2):
    return R(torch.cat([Z1,Z2],dim=0))-0.5*R(Z1)-0.5*R(Z2)



# Add SimSiam loss function
def simsiam_loss(p1, p2, h1, h2):

    loss_tcr = -R(p1).mean()
    loss_tcr *=1e-2

    # Negative cosine similarity
    loss_cos = (F.cosine_similarity(h1, p2.detach(), dim=-1).mean() + 
             F.cosine_similarity(h2, p1.detach(), dim=-1).mean()) * 0.5
    
    loss_cos = 1-loss_cos

    return loss_cos,loss_tcr

def tcr_loss(Z1,Z2):
    Z1 = F.normalize(Z1,p=2,dim=-1)
    Z2 = F.normalize(Z2,p=2,dim=-1)
    Z = (Z1+Z2)/2
    return R_nonorm(Z)

# Add a proper reshape layer
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, up=False):
        super().__init__()
        
        self.up = up
        
        # Main branch
        layers = []
        if up:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.main_branch = nn.Sequential(*layers)
        
        # Shortcut branch
        shortcut_layers = []
        if up:
            shortcut_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if in_channels != out_channels or stride != 1:
            shortcut_layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1))
            shortcut_layers.append(nn.BatchNorm2d(out_channels))
        
        self.shortcut = nn.Sequential(*shortcut_layers) if shortcut_layers else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_branch(x)
        return F.relu(out + identity)

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Main branch
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut branch with downsampling
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return F.relu(self.main_branch(x) + self.shortcut(x))

class Decoder(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        # Initial dense layer from 128-dim latent to 4x4x256
        self.dense = nn.Linear(latent_dim, 4 * 4 * 256)
        self.reshape = Reshape((256, 4, 4))
        
        # Three ResBlocks with upsampling (up 256)
        self.resblock1 = ResBlock(256, 256, up=True)  # 4x4 -> 8x8
        self.resblock2 = ResBlock(256, 256, up=True)  # 8x8 -> 16x16
        self.resblock3 = ResBlock(256, 256, up=True)  # 16x16 -> 32x32
        
        # Final layers: BN, ReLU, 3x3 conv, Tanh
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
         # Variational part
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        self.projection = nn.Linear(latent_dim*2,latent_dim)


    def kl_divergence(self, mu, logvar):
        # KL divergence between N(mu, sigma) and N(0, 1)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, z):
        # mu = self.fc_mu(z)
        # logvar = self.fc_logvar(z)
        # z_reparam = self.reparameterize(mu,logvar)
        # kld = self.kl_divergence(mu,logvar).mean()


        # n = torch.randn_like(z)
        # kld = n.mean()
        
        # z = self.projection(torch.cat([z,n],dim=-1))
        kld = z.mean()
        x = self.dense(z)
        x = self.reshape(x)  # -> 4x4x256
        x = self.resblock1(x)  # -> 8x8x256
        x = self.resblock2(x)  # -> 16x16x256
        x = self.resblock3(x)  # -> 32x32x256
        x = self.bn(x)
        x = self.relu(x)
        x = self.final_conv(x)  # -> 32x32x3
        x = self.tanh(x)
        return x,kld
# Now let's update the Encoder and Decoder with these specific ResBlock implementations:


# Modify training function
def train_ebm_gan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize AMP scaler
    scaler = GradScaler()
    
    
    
    # Data preprocessing
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])
        ])
        
        transform = MultiViewTransform(transform,n_views=2)
        
        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                              download=True, transform=transform)
    elif args.dataset == 'imagenet100':  # imagenet100
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        transform = MultiViewTransform(transform,n_views=2)
        
        # Load Tiny ImageNet dataset using ImageFolder
        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_path),
            transform=transform
        )
    
    # Filter the dataset to only include class 1
    if args.cls!=-1:
        class_1_indices = [i for i, label in enumerate(trainset.targets) if label == args.cls]
        trainset.data = trainset.data[class_1_indices]
        trainset.targets = [trainset.targets[i] for i in class_1_indices]
    

    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # Initialize models
    generator = create_conditional_stylegan(resolution=args.img_size, conditioning_dim=args.embed_dim)
    generator = generator.to(device)

    # discriminator = ResNetEnergyNet(img_channels=3, hidden_dim=64).to(device)
    discriminator = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=0,
    ).to(device)

    teacher_discriminator = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=0,
    ).to(device)    
    teacher_discriminator.load_state_dict(discriminator.state_dict())

    # Optimizers
    g_optimizer = torch.optim.Adam(
        generator.parameters(), 
        lr=args.g_lr, 
        betas=(args.g_beta1, args.g_beta2)
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), 
        lr=args.d_lr, 
        betas=(args.g_beta1, args.g_beta2)
    )
    
    # Add Cosine Annealing schedulers
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        d_optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    
    momentum_scheduler = cosine_scheduler(
        base_value=args.ema_weight_start, final_value=args.ema_weight_end, 
        epochs=args.epochs, niter_per_ep=len(trainloader))

    start_epoch = 0
    pca_noiser = SVDPCANoise(noise_scale=0.5,kernel='linear',gamma=1.0)
    # Add checkpoint loading logic
    if args.resume:
        checkpoint_path = args.resume
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            teacher_discriminator.load_state_dict(checkpoint['teacher_discriminator_state_dict'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
            d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch,args.epochs):
        generator.train()
        discriminator.train()
        teacher_discriminator.eval()
        for i, (views, _) in enumerate(tqdm(trainloader)):
            real_samples,aug_samples = views
            batch_size = real_samples.size(0)
            real_samples = real_samples.to(device)
            aug_samples = aug_samples.to(device)
            it = i + epoch * len(trainloader)
            momentum = momentum_scheduler[it]

            # Train Discriminator
            for _ in range(args.n_critic):
                d_optimizer.zero_grad()
                
                with autocast() if args.use_amp else nullcontext():
                    real_samples = real_samples.detach().requires_grad_(True)

                    z_context = discriminator.forward_feature(real_samples)
                    z = z_context[:,0]
                    # z = discriminator.head(z)
                    with torch.no_grad():
                        z_anchor = teacher_discriminator.forward_feature(aug_samples.detach())
                        z_anchor = z_anchor[:,0]
                        z_anchor = z_anchor.detach()


                    
                    batch_size = z.shape[0]

                    fake_samples = generator.generate(batch_size, device, conditioning=z)
                    fake_samples = fake_samples.detach().requires_grad_(True)
                    
                    z_fake = discriminator.forward_feature(fake_samples)
                    z_fake = z_fake[:,0]
                    # z_fake = discriminator.head(z_fake)
                    
                    real_energy = F.cosine_similarity(z,z_anchor,dim=-1)
  

                    fake_energy = F.cosine_similarity(z_fake,z_anchor,dim=-1)
                    

                    realistic_logits = real_energy - fake_energy
                    
                    
                    d_loss = F.softplus(-realistic_logits/args.temperature)
                    
                    
                    
                    loss_tcr = -R(z).mean() * args.tcr_weight
                    
                    r1 = zero_centered_gradient_penalty(real_samples, real_energy)
                    r2 = zero_centered_gradient_penalty(fake_samples, fake_energy)

                    d_loss = d_loss + args.gp_weight/2 * (r1 + r2)
                    d_loss = d_loss.mean() + loss_tcr 
                if args.use_amp:
                    scaler.scale(d_loss).backward()
                    scaler.step(d_optimizer)
                    scaler.update()
                else:
                    d_loss.backward()
                    d_optimizer.step()

            with torch.no_grad():
                for param_q, param_k in zip(discriminator.parameters(), teacher_discriminator.parameters()):
                    param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data

            # Train Generator
            g_optimizer.zero_grad()
            
            with autocast() if args.use_amp else nullcontext():
                # Generate new fake samples
                z = z.detach()
                z_context = discriminator.forward_feature(real_samples)
                z = z_context[:,0]
                # z = discriminator.head(z)
                with torch.no_grad():
                    z_anchor = teacher_discriminator.forward_feature(aug_samples.detach())
                    z_anchor = z_anchor[:,0]
                    z_anchor = z_anchor.detach()
                    
                # z_noised = pca_noiser(z)
                
                fake_samples = generator.generate(batch_size, device, conditioning=z)

                z_fake = discriminator.forward_feature(fake_samples)
                z_fake = z_fake[:,0]
                # z_fake = discriminator.head(z_fake)
                
                real_energy = F.cosine_similarity(z,z_anchor,dim=-1)
                fake_energy = F.cosine_similarity(z_fake,z_anchor,dim=-1)

                loss_tgr = -R(z_fake).mean() * args.tcr_weight
                

                realistic_logits = fake_energy - real_energy
                g_loss = F.softplus(-realistic_logits/args.temperature)

                g_loss = g_loss.mean() + loss_tgr
            if args.use_amp:
                scaler.scale(g_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                g_loss.backward()
                g_optimizer.step()

            if i % args.log_freq == 0:
                current_g_lr = g_optimizer.param_groups[0]['lr']
                current_d_lr = d_optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, '
                      f'r1: {r1.mean().item():.4f}, r2: {r2.mean().item():.4f}, '
                      f'loss_tcr: {loss_tcr.item():.4f}, '
                      f'loss_tgr: {loss_tgr.item():.4f}, '
                      f'Real Energy: {real_energy.mean().item():.4f}, '
                      f'Fake Energy: {fake_energy.mean().item():.4f}, '
                      f'G_LR: {current_g_lr:.6f}, D_LR: {current_d_lr:.6f}'
                      )
        
        # Step the schedulers at the end of each epoch
        g_scheduler.step()
        d_scheduler.step()
        
        save_gan_samples(generator, discriminator,pca_noiser, epoch, args.output_dir, device,real_samples=real_samples)
    
        # Save samples and model checkpoints
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'teacher_discriminator_state_dict': teacher_discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': g_scheduler.state_dict(),
                'd_scheduler_state_dict': d_scheduler.state_dict(),
            }, os.path.join(args.output_dir, f'ebm_gan_checkpoint_{epoch}.pth'))

def save_gan_samples(generator, discriminator,pca_noiser, epoch, output_dir, device, n_samples=36,real_samples=None):
    generator.eval()
    if real_samples is not None:
        real_samples = real_samples[:n_samples]
        batch_size = real_samples.size(0)
    else:
        return
    with torch.no_grad():
        
        z_context = discriminator.forward_feature(real_samples.detach())
        z = z_context[:,0]

        # z_noised = pca_noiser(z)
        fake_samples = generator.generate(batch_size, device, conditioning=z)
        # Changed 'range' to 'value_range'
        grid = make_grid(fake_samples, nrow=6, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'gan_samples_epoch_{epoch}.png'))


        # Changed 'range' to 'value_range'
        grid = make_grid(real_samples, nrow=6, normalize=True, value_range=(-1, 1))
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'gan_samples_epoch_{epoch}_real.png'))



        plt.close()
    generator.train()

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Compute gradient penalty for improved training stability"""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def get_args_parser():
    parser = argparse.ArgumentParser('EBM-GAN training for CIFAR-10')
    
    # dataset parameters
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset to train on')
    
    # Add GAN-specific parameters
    
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--g_lr', default=1e-4, type=float)
    parser.add_argument('--d_lr', default=1e-4, type=float)
    parser.add_argument('--n_critic', default=1, type=int,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--gp_weight', default=0.05, type=float,
                        help='Weight of gradient penalty')
    
    # Modify learning rates
    parser.add_argument('--g_beta1', default=0.5, type=float,
                        help='Beta1 for generator optimizer')
    parser.add_argument('--g_beta2', default=0.999, type=float,
                        help='Beta2 for generator optimizer')
    parser.add_argument('--temperature', default=1, type=float,
                        help='Temperature for softplus')
    
    parser.add_argument('--ema_weight_start', default=0.9, type=float,
                        help='EMA weight start')
    parser.add_argument('--ema_weight_end', default=1., type=float,
                        help='EMA weight end')
    
    parser.add_argument('--cls', default=-1, type=int,
                        help='Class to train on')
    
    # Existing parameters
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    parser.add_argument('--data_path', default='c:/datasets', type=str)
    parser.add_argument('--output_dir', default='/mnt/d/repo/output/cifar10-ebm-gan-r3gan-ctrl')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    
    # Add learning rate scheduling parameters
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate for cosine annealing')

    # Add dataset parameters
    # vit parameters
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num_heads', default=3, type=int)
    parser.add_argument('--mlp_ratio', default=4., type=float)
    parser.add_argument('--embed_dim', default=192, type=int)
    parser.add_argument('--norm_layer', default=nn.LayerNorm, type=nn.Module)
    parser.add_argument('--decoder_depth', default=4, type=int)
    parser.add_argument('--decoder_embed_dim', default=96, type=int)

    ## tcr parameters
    parser.add_argument('--tcr_weight', default=0.01, type=float,
                        help='Weight of tcr loss')
    

    
    # Add checkpoint loading parameter
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to checkpoint to resume training from')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm_gan(args) 
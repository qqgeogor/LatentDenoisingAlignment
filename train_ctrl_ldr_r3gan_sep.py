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
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from contextlib import nullcontext
# Set matplotlib backend to avoid GUI dependencies
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

def zero_centered_gradient_penalty(samples, critics):
    grad, = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True)
    return grad.square().sum([1, 2, 3])



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

class Encoder(nn.Module):
    def __init__(self, img_channels=3,latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        # Initial ResBlock down with 128 channels
        self.resblock1 = ResBlockDown(img_channels, 128)  # 32x32 -> 16x16
        self.resblock2 = ResBlockDown(128, 128)          # 16x16 -> 8x8
        
        # Regular ResBlocks with 128 channels
        self.resblock3 = ResBlock(128, 128)              # 8x8
        self.resblock4 = ResBlock(128, 128)              # 8x8
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Global sum pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final dense layer to 128-dim latent space
        self.dense = nn.Linear(128, latent_dim)
        
        self.head = nn.Sequential(
            nn.Linear(latent_dim,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,1),
        )
    
    def net(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x
    
    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x
    
    

# Modify training function
def train_ebm_gan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize AMP scaler
    scaler = GradScaler()

    # Data preprocessing
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                          download=True, transform=transform)
    
    # Filter the dataset to only include class 1
    if args.cls!=-1:
        class_1_indices = [i for i, label in enumerate(trainset.targets) if label == args.cls]
        trainset.data = trainset.data[class_1_indices]
        trainset.targets = [trainset.targets[i] for i in class_1_indices]
    

    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # Initialize models
    embedder = Encoder(latent_dim=args.latent_dim).to(device)
    
    generator = Decoder(latent_dim=args.latent_dim).to(device)
    # discriminator = ResNetEnergyNet(img_channels=3, hidden_dim=64).to(device)
    discriminator = Encoder(latent_dim=args.latent_dim).to(device)
    
    # Optimizers
    g_optimizer = torch.optim.AdamW(
        list(generator.parameters())+list(embedder.parameters()), 
        lr=args.g_lr, 
        betas=(args.g_beta1, args.g_beta2)
    )
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(), 
        lr=args.d_lr, 
        betas=(0.5, 0.999)
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
    start_epoch = 0
    
    # Add checkpoint loading logic
    if args.resume:
        checkpoint_path = args.resume
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
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
        
        for i, (real_samples, _) in enumerate(tqdm(trainloader)):
            batch_size = real_samples.size(0)
            real_samples = real_samples.to(device)
            
            # Train Discriminator
            for _ in range(args.n_critic):
                d_optimizer.zero_grad()
                
                with autocast() if args.use_amp else nullcontext():
                    # Generate fake samples
                    z = embedder.net(real_samples.detach()).squeeze()

                    real_samples = real_samples.detach().requires_grad_(True)
                    fake_samples,_ = generator(z)
                    fake_samples = fake_samples.detach().requires_grad_(True)
                    # Compute energies
                    real_energy = discriminator(real_samples)
                    fake_energy = discriminator(fake_samples)
                    
                    realistic_logits = real_energy - fake_energy
                    d_loss = F.softplus(-realistic_logits)

                    # loss_tcr = -R(z).mean()
                    # loss_tcr /=200

                    r1 = zero_centered_gradient_penalty(real_samples, real_energy)
                    r2 = zero_centered_gradient_penalty(fake_samples, fake_energy)

                    d_loss = d_loss + args.gp_weight/2 * (r1 + r2)
                    d_loss = d_loss.mean() #+ loss_tcr
                
                if args.use_amp:
                    scaler.scale(d_loss).backward()
                    scaler.step(d_optimizer)
                    scaler.update()
                else:
                    d_loss.backward()
                    d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            
            with autocast() if args.use_amp else nullcontext():
                # Generate new fake samples
                z = embedder.net(real_samples.detach()).squeeze()

                loss_tcr = -R(z).mean()
                loss_tcr /=200

                fake_samples,loss_kld = generator(z)
                fake_energy = discriminator(fake_samples)
                real_energy = discriminator(real_samples)
                z_fake = embedder.net(fake_samples).squeeze()
                
                loss_cos = 1 - F.cosine_similarity(z_fake,z,dim=-1).mean()
                loss_mse = F.mse_loss(fake_samples,real_samples).mean()

                realistic_logits = fake_energy - real_energy
                g_loss = F.softplus(-realistic_logits)
                g_loss = g_loss.mean() + loss_tcr #+ loss_cos # + loss_kld 
            
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
                      f'loss_cos: {loss_cos.item():.4f}, '
                      f'loss_kld: {loss_kld.item():.4f}, '
                      f'loss_mse: {loss_mse.item():.4f}, '
                      f'Real Energy: {real_energy.mean().item():.4f}, '
                      f'Fake Energy: {fake_energy.mean().item():.4f}, '
                      f'G_LR: {current_g_lr:.6f}, D_LR: {current_d_lr:.6f}'
                      )
        
        # Step the schedulers at the end of each epoch
        g_scheduler.step()
        d_scheduler.step()
        
        real_samples = next(iter(trainloader))[0].to(device)
        save_gan_samples(generator, embedder, epoch, args.output_dir, device,real_samples=real_samples)

        # Save samples and model checkpoints
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'embedder_state_dict': embedder.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': g_scheduler.state_dict(),
                'd_scheduler_state_dict': d_scheduler.state_dict(),
            }, os.path.join(args.output_dir, f'ebm_gan_checkpoint_{epoch}.pth'))

def save_gan_samples(generator, discriminator, epoch, output_dir, device, n_samples=36,real_samples=None):
    generator.eval()
    discriminator.eval()
    real_samples = real_samples[:n_samples]
    batch_size = real_samples.size(0)
    with torch.no_grad():
        
        z = discriminator.net(real_samples.detach()).squeeze()

        fake_samples,_ = generator(z)
        
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
    discriminator.train()

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
    
    # Add checkpoint loading parameter
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to checkpoint to resume training from')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm_gan(args) 
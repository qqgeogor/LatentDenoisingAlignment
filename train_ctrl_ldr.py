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
# Set matplotlib backend to avoid GUI dependencies
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
from contextlib import nullcontext


def mcr2(Z1,Z2):
    loss_expd = R(torch.cat([Z1,Z2],dim=0))
    loss_comp = 0.5*R(Z1)+0.5*R(Z2)
    total_loss = loss_expd - loss_comp
    return total_loss,loss_expd,loss_comp
    
def zero_centered_gradient_penalty(samples, critics):
    grad, = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True)
    return grad.square().sum([1, 2, 3])

class EnergyNet(nn.Module):
    def __init__(self, img_channels=3, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            # Initial conv: [B, 3, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(img_channels, hidden_dim, 4, 2, 1),
            # nn.GroupNorm(8, hidden_dim),  # Add normalization
            nn.LeakyReLU(0.2),
            
            # [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            # nn.GroupNorm(8, hidden_dim * 2),  # Add normalization
            nn.LeakyReLU(0.2),
            
            # [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            # nn.GroupNorm(8, hidden_dim * 4),  # Add normalization
            nn.LeakyReLU(0.2),
            
            # [B, 256, 4, 4] -> [B, 512, 2, 2]
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            # nn.GroupNorm(8, hidden_dim * 8),  # Add normalization
            nn.LeakyReLU(0.2),
            
            # Final conv to scalar energy: [B, 512, 2, 2] -> [B, 1, 1, 1]
            nn.Conv2d(hidden_dim * 8, 128, 2, 1, 0)
        )

        self.head = nn.Linear(128, 1)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        z = self.net(x).squeeze()
        logits = self.head(z)
        # print(x.shape)
        # logits = self.head(logits)
        # Add regularization term to prevent collapse
        # reg_term = 0.1 * (logits ** 2).mean()
        # logits = logits# + reg_term
        #logits = -F.logsigmoid(logits)
        return logits



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


class ResNetEnergyNet(nn.Module):
    def __init__(self, img_channels=3, hidden_dim=64):
        super().__init__()
        
        # Initial conv layer
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # ResNet blocks with downsampling
        self.layer1 = ResBlock(hidden_dim, hidden_dim * 2, stride=2)
        self.layer2 = ResBlock(hidden_dim * 2, hidden_dim * 4, stride=2)
        self.layer3 = ResBlock(hidden_dim * 4, hidden_dim * 8, stride=2)
        self.layer4 = ResBlock(hidden_dim * 8, hidden_dim * 8, stride=2)
        
        # Final energy output
        self.energy_head = nn.Sequential(
            nn.Conv2d(hidden_dim * 8, hidden_dim * 4, 2, 1, 0),
            nn.GroupNorm(8, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 4, 1, 1, 1, 0)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.energy_head(x).squeeze()
        energy = -F.logsigmoid(logits)
        return energy
    
class LangevinSampler:
    def __init__(self, n_steps=60, step_size=10.0, noise_scale=0.005):
        self.n_steps = n_steps
        self.step_size = step_size
        self.noise_scale = noise_scale
    
    def sample(self, model, x_init, return_trajectory=False):
        model.eval()
        # Ensure x requires gradients
        x = x_init.clone().detach().requires_grad_(True)
        trajectory = [x.clone().detach()] if return_trajectory else None
        
        for _ in range(self.n_steps):
            # Ensure x requires gradients at each step
            if not x.requires_grad:
                x.requires_grad_(True)
                
            # Compute energy gradient
            energy = model(x)
            if isinstance(energy, torch.Tensor):
                energy = energy.sum()
            
            # Compute gradients
            if x.grad is not None:
                x.grad.zero_()
            grad = torch.autograd.grad(energy, x, create_graph=False, retain_graph=True)[0]
            
            # Langevin dynamics update
            noise = torch.randn_like(x) * self.noise_scale
            x = x.detach()  # Detach from computation graph
            x = x - self.step_size * grad + noise  # Update x
            x.requires_grad_(True)  # Re-enable gradients
            x = torch.clamp(x, -1, 1)  # Keep samples in valid range
            
            if return_trajectory:
                trajectory.append(x.clone().detach())
        
        return (x.detach(), trajectory) if return_trajectory else x.detach()

# Add a proper reshape layer
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

# Add Generator class
class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=64):
        super().__init__()

                
        # Variational part
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)


        self.net = nn.Sequential(
            # Initial projection
            nn.Linear(latent_dim, hidden_dim * 8 * 4 * 4),
            nn.LeakyReLU(0.2),
            
            # Reshape layer instead of lambda
            Reshape((hidden_dim * 8, 4, 4)),
            
            # [4x4] -> [8x8]
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            # [8x8] -> [16x16]
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            # [16x16] -> [32x32]
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            # Final layer
            nn.ConvTranspose2d(hidden_dim, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
        
    def kl_divergence(self, mu, logvar):
        # KL divergence between N(mu, sigma) and N(0, 1)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    

    def forward(self, z):
        if use_vae:
                
            mu = self.fc_mu(z)
            logvar = self.fc_logvar(z)
            z = self.reparameterize(mu,logvar)
            kld = self.kl_divergence(mu,logvar).mean()
            return self.net(z),kld
        else:
            return self.net(z),None


    # def forward(self, z):
    #     mu = self.fc_mu(z)
    #     logvar = self.fc_logvar(z)
    #     z = self.reparameterize(mu,logvar)
    #     kld = self.kl_divergence(mu,logvar).mean()
    #     return self.net(z),kld

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
    def __init__(self):
        super().__init__()
        
        # Initial dense layer from 128-dim latent to 4x4x256
        self.dense = nn.Linear(128, 4 * 4 * 256)
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
        
    def forward(self, z):
        x = self.dense(z)
        x = self.reshape(x)  # -> 4x4x256
        x = self.resblock1(x)  # -> 8x8x256
        x = self.resblock2(x)  # -> 16x16x256
        x = self.resblock3(x)  # -> 32x32x256
        x = self.bn(x)
        x = self.relu(x)
        x = self.final_conv(x)  # -> 32x32x3
        x = self.tanh(x)
        return x
# Now let's update the Encoder and Decoder with these specific ResBlock implementations:

class Encoder(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        
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
        self.dense = nn.Linear(128, 128)
    
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
        return self.net(x)
    


# Modify training function
def train_ebm_gan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize AMP scaler
    scaler = GradScaler(enabled=args.use_amp)

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
    generator = Decoder().to(device)
    discriminator = Encoder(img_channels=3).to(device)
    
    # Optimizers
    g_optimizer = torch.optim.AdamW(
        generator.parameters(), 
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
                
                # Use autocast for mixed precision training
                with autocast(enabled=args.use_amp) if args.use_amp else nullcontext():
                    z = discriminator(real_samples.detach()).squeeze()
                    fake_samples = generator(z)
                    fake_samples = fake_samples.detach()

                    z_real = discriminator(real_samples).squeeze()
                    z_fake = discriminator(fake_samples).squeeze()

                    d_mcr2, d_expd, d_comp = mcr2(z_real, z_fake)
                    d_loss = -d_mcr2

                if args.use_amp:    
                    scaler.scale(d_loss).backward()
                    scaler.step(d_optimizer)
                    scaler.update()
                else:
                    d_loss.backward()
                    d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            # Use autocast for mixed precision training
            with autocast(enabled=args.use_amp) if args.use_amp else nullcontext()  :
                z = discriminator(real_samples.detach()).squeeze()
                fake_samples = generator(z)

                z_real = discriminator(real_samples).squeeze()
                z_fake = discriminator(fake_samples).squeeze()
                g_mcr2, g_expd, g_comp = mcr2(z_real, z_fake)
                g_loss = g_mcr2
            
            if args.use_amp:
                # Scale loss and backward pass
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
                      f'D_expd: {d_expd.item():.4f}, D_comp: {d_comp.item():.4f}, '
                      f'G_expd: {g_expd.item():.4f}, G_comp: {g_comp.item():.4f}, '
                      f'G_LR: {current_g_lr:.6f}, D_LR: {current_d_lr:.6f}'
                      )
        
        # Step the schedulers at the end of each epoch
        g_scheduler.step()
        d_scheduler.step()
        
        real_samples = next(iter(trainloader))[0].to(device)
        save_gan_samples(generator, discriminator, epoch, args.output_dir, device,real_samples=real_samples)
    
        # Save samples and model checkpoints
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
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
        
        z = discriminator(real_samples.detach()).squeeze()

        fake_samples = generator(z)
        
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
    
    parser.add_argument('--use_amp', action='store_true')
    # Existing parameters
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    parser.add_argument('--data_path', default='c:/datasets', type=str)
    parser.add_argument('--output_dir', default='/mnt/d/repo/output/cifar10-ebm-gan-r3gan-ctrl')
    parser.add_argument('--num_workers', default=4, type=int)
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
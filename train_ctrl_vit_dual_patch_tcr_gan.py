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
from utils_ibot import SVDPCANoise,cosine_scheduler
import numpy as np
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import random
# Set matplotlib backend to avoid GUI dependencies
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

from vit_ibot_registry import MaskedAutoencoderViT

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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.GroupNorm(8, out_channels)
            )
    
    def forward(self, x):
        out = F.leaky_relu(self.gn1(self.conv1(x)), 0.2)
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.2)
        return out


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
class GeneratorProjector(nn.Module):
    def __init__(self, latent_dim=100,):
        super().__init__()
        self.net = nn.Sequential(
            # Initial projection
            nn.Linear(latent_dim, latent_dim * 4 * 4),
            nn.LeakyReLU(0.2),
            
            # Reshape layer instead of lambda
            Reshape((latent_dim, 4, 4)),
            
            # [4x4] -> [8x8]
            nn.ConvTranspose2d(latent_dim, latent_dim , 4, 2, 1),
            nn.BatchNorm2d(latent_dim ),
            nn.LeakyReLU(0.2),
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
    
    def forward(self, z):
        z_up = self.net(z)
        b,c,h,w = z_up.shape
        z_up = z_up.reshape(b,c,-1).transpose(-2,-1)
        z_up = torch.cat([z.unsqueeze(1),z_up],dim=1)
        
        return z_up

# Add Generator class
class Generator(nn.Module):
    def __init__(self, latent_dim=100, hidden_dim=64):
        super().__init__()
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
                
    def forward(self, z):
        

        return self.net(z)

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_ebm_gan(args):
    # Set random seed at the start of training
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize GradScaler for AMP
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    if args.dataset == 'imagenet100':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.95, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # image net normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
    
    # Load CIFAR-10
    if args.dataset == 'imagenet100':
        print('dataset',args.dataset)
        print('dataset loaded',args.data_path)

        trainset = torchvision.datasets.ImageFolder(root=args.data_path, transform=transform)
        
    else:
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
    generator =  MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=0,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        use_checkpoint=args.use_checkpoint
    ).to(device)



    # discriminator = ResNetEnergyNet(img_channels=3, hidden_dim=64).to(device)
    discriminator = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=4,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        use_checkpoint=args.use_checkpoint
    ).to(device)
    
    teacher_discriminator = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=4,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        use_checkpoint=args.use_checkpoint
    ).to(device)

    teacher_discriminator.load_state_dict(discriminator.state_dict())

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
    pca_noiser = SVDPCANoise(noise_scale=args.noise_scale, kernel='linear', gamma=1.0)

    
    momentum_scheduler = cosine_scheduler(
        base_value=args.base_momentum, final_value=args.final_momentum, 
        epochs=args.epochs, niter_per_ep=len(trainloader))
    
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
            
            it = i + epoch * len(trainloader)
            momentum = momentum_scheduler[it]
            # Train Discriminator
            for _ in range(args.n_critic):  # Train discriminator more frequently
                d_optimizer.zero_grad()
                
                with autocast():
                    # Generate fake samples
                    real_samples = real_samples.detach().requires_grad_(True)
                    h_real, mask, ids_restore = discriminator.forward_encoder_predictor_restored(real_samples,mask_ratio=0.75)
                    z_real = discriminator.proj_head(h_real)
                    mask = mask.float()
                    fake_samples = generator.forward_decoder(h_real)

                    fake_samples = generator.patchify(real_samples)*(1-mask.unsqueeze(-1)) + fake_samples*(mask.unsqueeze(-1))
                    
                    fake_samples = generator.unpatchify(fake_samples).detach().requires_grad_(True)
                    
                    mask = mask.bool()
                    z_real_mask = z_real[:,1:][mask,:]
                    z_real = z_real[:,0]
                    # # Compute energies
                    with torch.no_grad():
                        z_anchor = teacher_discriminator.forward_feature(real_samples).detach()
                        z_anchor_mask = z_anchor[:,1:][mask,:]

                        z_anchor = z_anchor[:,0].detach()
                        z_anchor = F.layer_norm(z_anchor, (z_anchor.shape[1],))

                    z_fake = discriminator.forward_feature(fake_samples)
                    z_fake = discriminator.proj_head(z_fake)
                    mask = mask.bool()
                    z_fake = z_fake[:,0]
                    
                    real_energy = F.cosine_similarity(z_real, z_anchor, dim=1)
                    fake_energy = F.cosine_similarity(z_fake, z_anchor, dim=1)
                    
                    loss_cos = 1 - F.cosine_similarity(z_real, z_anchor, dim=1)
                    # # Compute loss
                    realistic_logits = real_energy - fake_energy
                    loss_adv = F.softplus(-realistic_logits/args.temperature).mean()

                    loss_smooth = F.smooth_l1_loss(z_real_mask, z_anchor_mask,reduction='none').mean(dim=1)
                    loss_smooth = loss_smooth.mean()
                    r1 = zero_centered_gradient_penalty(real_samples, real_energy)
                    r2 = zero_centered_gradient_penalty(fake_samples, fake_energy)

                    # TCR loss  
                    loss_tcr = -R(z_real).mean()
                    loss_tcr *= 1e-2

                    d_loss = loss_adv *args.d_adv_weight + loss_smooth + loss_tcr*args.tcr_weight + (args.gp_weight/2)*(r1+r2).mean()
                # Scale and backpropagate
                scaler_d.scale(d_loss).backward()
                scaler_d.step(d_optimizer)
                scaler_d.update()

            with torch.no_grad():
                for param_q, param_k in zip(discriminator.parameters(), teacher_discriminator.parameters()):
                    param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)
            
            # Train Generator
            g_optimizer.zero_grad()
            
            with autocast():
                # Generate new fake samples
                real_samples = real_samples
                h_real, mask, ids_restore = discriminator.forward_encoder_predictor_restored(real_samples)
                z_real = discriminator.proj_head(h_real)
                mask = mask.float()
                fake_samples = generator.forward_decoder(h_real)
                
                recon_loss = F.smooth_l1_loss(fake_samples, generator.patchify(real_samples),reduction='none').mean(-1)
                recon_loss = recon_loss.mean()
                
                fake_samples = generator.patchify(real_samples)*(1-mask.unsqueeze(-1)) + fake_samples*(mask.unsqueeze(-1))
                
                fake_samples = generator.unpatchify(fake_samples)
                
                mask = mask.bool()
                z_real = z_real[:,0]
                
                # # Compute energies
                with torch.no_grad():
                    z_anchor = teacher_discriminator.forward_feature(real_samples).detach()
                    z_anchor = z_anchor[:,0]
                    z_anchor = F.layer_norm(z_anchor, (z_anchor.shape[1],))

                z_fake = discriminator.forward_feature(fake_samples)
                z_fake = discriminator.proj_head(z_fake)
                mask = mask.bool()
                z_fake = z_fake[:,0]
                real_energy = F.cosine_similarity(z_real, z_anchor, dim=1)
                fake_energy = F.cosine_similarity(z_fake, z_anchor, dim=1)
                
                # # Compute loss
                realistic_logits = fake_energy - real_energy
                loss_adv = F.softplus(-realistic_logits/args.temperature).mean()
                

                
                g_loss = recon_loss + loss_adv *args.g_adv_weight


            
            # Scale and backpropagate
            scaler_g.scale(g_loss).backward()
            scaler_g.step(g_optimizer)
            scaler_g.update()
            
            if i % args.log_freq == 0:
                current_g_lr = g_optimizer.param_groups[0]['lr']
                current_d_lr = d_optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'D_loss: {d_loss.item():.4f}, '
                      f'G_loss: {g_loss.item():.4f}, ' 
                      f'r1: {r1.mean().item():.4f}, '
                      f'r2: {r2.mean().item():.4f}, '
                      f'loss_cos: {loss_cos.mean().item():.4f}, '
                      f'adv_loss: {loss_adv.item():.4f}, '
                      f'loss_tcr: {loss_tcr.item():.4f}, '
                      f'Real Energy: {real_energy.mean().item():.4f}, '
                      f'Fake Energy: {fake_energy.mean().item():.4f}, '
                      f'G_LR: {current_g_lr:.6f}, D_LR: {current_d_lr:.6f}'
                      )
        
        # Step the schedulers at the end of each epoch
        g_scheduler.step()
        d_scheduler.step()
        
        real_samples = next(iter(trainloader))[0].to(device)
        save_gan_samples(generator, discriminator,pca_noiser, epoch, args.output_dir, real_samples=real_samples)
        
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

def save_gan_samples(generator, discriminator,pca_noiser, epoch, output_dir, n_samples=36,real_samples=None):
    generator.eval()
    discriminator.eval()
    real_samples = real_samples[:n_samples]
    batch_size = real_samples.size(0)
    with torch.no_grad():
        
        h_real,_,_ = discriminator.forward_encoder_predictor_restored(real_samples)
        
        fake_samples = generator.forward_decoder(h_real)
        fake_samples = generator.unpatchify(fake_samples)
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
    
    parser.add_argument('--dataset', default='imagenet100', type=str,
                        help='Dataset to train on')
    
    
    # Add GAN-specific parameters
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--g_lr', default=1e-4, type=float)
    parser.add_argument('--d_lr', default=1e-4, type=float)
    parser.add_argument('--n_critic', default=1, type=int,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--gp_weight', default=0.05, type=float,
                        help='Weight of gradient penalty')
    parser.add_argument('--noise_scale', default=0.5, type=float,
                        help='Noise scale for PCA noise')
    parser.add_argument('--d_adv_weight', default=0.2, type=float,
                        help='Weight of adversarial loss')
    parser.add_argument('--g_adv_weight', default=0.2, type=float,
                        help='Weight of adversarial loss')
    parser.add_argument('--base_momentum', default=0.996, type=float,
                        help='Base momentum for momentum scheduler')
    parser.add_argument('--final_momentum', default=1.0, type=float,
                        help='Final momentum for momentum scheduler')
    
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
    parser.add_argument('--tcr_weight', default=0, type=float)
    parser.add_argument('--global_weight', default=0, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)


    # model parameters
    parser.add_argument('--img_size', default=32, type=int) 
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--embed_dim', default=192, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num_heads', default=3, type=int)
    parser.add_argument('--decoder_embed_dim', default=192, type=int)
    parser.add_argument('--decoder_depth', default=4, type=int)
    parser.add_argument('--decoder_num_heads', default=3, type=int)
    parser.add_argument('--mlp_ratio', default=4., type=float)
    parser.add_argument('--use_checkpoint', default=False, type=bool)
    
    # Add learning rate scheduling parameters
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate for cosine annealing')
    
    # Add checkpoint loading parameter
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to checkpoint to resume training from')
    
    # Add seed parameter
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm_gan(args) 
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
from PIL import Image
from tqdm import tqdm
import argparse
import torch.optim as optim
import torch.nn.functional as F
from contextlib import nullcontext
from vit_ibot_registry import MaskedAutoencoderViT,get_2d_sincos_pos_embed
from timm.models.vision_transformer import Block,PatchEmbed,DropPath
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# Set matplotlib backend to avoid GUI dependencies
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
from utils_ibot import SVDPCANoise,DataAugmentationiBOT

def knn_predict(train_features, train_labels, test_features, k=20, num_classes=10):
    """
    Predict labels using k-nearest neighbors
    """
    # Normalize the features
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    
    # Convert to numpy for sklearn
    train_features_np = train_features.cpu().numpy()
    test_features_np = test_features.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features_np, train_labels_np)
    
    # Get probabilities for each class
    probs = knn.predict_proba(test_features_np)
    
    return torch.from_numpy(probs).float()

def create_tsne_plot(features, labels, probs, epoch, output_dir,sample_size=1000):
    """
    Create and save t-SNE visualization of features with KNN probabilities
    """
    # Use a subset of data for visualization if needed
    if sample_size < len(features):
        indices = np.random.choice(len(features), sample_size, replace=False)
        features = features[indices]
        labels = labels[indices]
        probs = probs[indices]
        
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Create main plot with true labels
    plt.figure(figsize=(20, 10))
    
    # Plot with true labels
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f't-SNE visualization with true labels (Epoch {epoch})')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    # Plot with KNN confidence
    plt.subplot(1, 2, 2)
    confidence = np.max(probs, axis=1)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=confidence, cmap='viridis')
    plt.colorbar(scatter, label='KNN Confidence')
    plt.title(f't-SNE visualization with KNN confidence (Epoch {epoch})')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tsne_epoch_{epoch}.png'))
    plt.close()

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        return x



class DenoisingBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.block = CrossAttentionBlock(embed_dim, num_heads=3, mlp_ratio=4, qkv_bias=True)   
        
    
    def forward(self, x, z_t):
        x = torch.cat([z_t.unsqueeze(1), x], dim=1)
        x = self.block(x)[:,0]
        return x
    



class BaseFeatureExtractor(nn.Module):
    def __init__(self, embed_dim,args,device):
        super().__init__()
        # Models initialization
        self.patch_embed = PatchEmbed(img_size=args.img_size, patch_size=args.patch_size, in_chans=3, embed_dim=embed_dim).to(device)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.initialize_weights()


    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed[:,1:,:]
        return x
    
    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                          int(self.patch_embed.num_patches**.5), 
                                          cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize tokens and other parameters
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)




class ContrastiveModel(nn.Module):
    def __init__(self,args=None,device='cuda'):
        super().__init__()
        self.args = args    
        self.device = device
        self.T = args.T
        T = args.T
        self.embed_dim = args.embed_dim
        self.feature_dim = args.embed_dim
        self.lr = args.lr
        embed_dim = args.embed_dim 
        self.base_feature_extractor = BaseFeatureExtractor(embed_dim,args,device).to(device)
        self.mlps = nn.ModuleList([DenoisingBlock(embed_dim).to(device) for _ in range(T)])
        
        
        




class LatentDenoisingAlignment(nn.Module):
    def __init__(self,args=None,device='cuda'):
        super().__init__()
        T = args.T 
        self.T = T
        embed_dim = args.embed_dim
        self.embed_dim = args.embed_dim
        self.feature_dim = args.embed_dim
        self.device = device
        self.args = args
        lr = args.lr
        # Noise schedule (linear)
        self.alpha = torch.linspace(1.0, 0.1, T).to(device)
        
        
        self.contrastive_model = ContrastiveModel(args,device)
        self.teacher_model = ContrastiveModel(args,device)
        
        # Optimizers
        self.cnn_optimizer = optim.AdamW(self.contrastive_model.base_feature_extractor.parameters(), lr=lr)
        self.mlp_optimizers = [optim.AdamW(mlp.parameters(), lr=lr) for mlp in self.contrastive_model.mlps]
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self,x):
        x = self.contrastive_model.base_feature_extractor(x)
        return x

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.contrastive_model.base_feature_extractor.load_state_dict(checkpoint['base_feature_extractor_state_dict'])
        for i, mlp in enumerate(self.contrastive_model.mlps):
            mlp.load_state_dict(checkpoint['mlps_state_dict'][i])
        self.cnn_optimizer.load_state_dict(checkpoint['cnn_optimizer_state_dict'])
        for i, ml_optimizer in enumerate(self.mlp_optimizers):
            ml_optimizer.load_state_dict(checkpoint['mlp_optimizers_state_dict'][i])
        self.alpha = checkpoint['alpha']
        self.T = checkpoint['T']
        self.embed_dim = checkpoint['embed_dim']
        self.feature_dim = checkpoint['feature_dim']

   

    
            
    def train_model(self, train_loader, epochs, log_freq=100, save_freq=10, output_dir='./checkpoints', use_amp=False):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create test data loader
        test_transform = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if self.args.dataset == 'cifar10':
            testset = torchvision.datasets.CIFAR10(root=self.args.data_path, train=False,
                                                 download=True, transform=test_transform)
        else:
            testset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.args.data_path, 'val'),
                transform=test_transform
            )
        
        test_loader = DataLoader(testset, batch_size=self.args.batch_size,
                               shuffle=False, num_workers=self.args.num_workers)

        lr_scheduler = cosine_scheduler(
            base_value=1e-4, final_value=1e-6, 
            epochs=epochs, niter_per_ep=len(train_loader),warmup_epochs=10, start_warmup_value=1e-5
        )       
        momentum_scheduler = cosine_scheduler(
            base_value=0.996, final_value=1., 
            epochs=epochs, niter_per_ep=len(train_loader))


        wd_scheduler = cosine_scheduler(
            base_value=0.04, final_value=0.4, 
            epochs=epochs, niter_per_ep=len(train_loader), warmup_epochs=0, start_warmup_value=0.04)
        
        
        # Initialize AMP GradScaler if using mixed precision
        scaler = GradScaler() if use_amp else None
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i, (views, label) in enumerate(train_loader):
                x = views[0].to(self.device)
                x_aug = views[1].to(self.device)
                # current_batch_size = x.shape[0]
                it = i + epoch * len(train_loader)

                current_lr = lr_scheduler[it]
                current_momentum = momentum_scheduler[it]
                current_wd = wd_scheduler[it]

                self.cnn_optimizer.param_groups[0]['lr'] = current_lr
                self.cnn_optimizer.param_groups[0]['weight_decay'] = current_wd
                for ml_optimizer in self.mlp_optimizers:
                    ml_optimizer.param_groups[0]['lr'] = current_lr
                    ml_optimizer.param_groups[0]['weight_decay'] = current_wd
                
                eps = torch.randn(x.shape[0],self.embed_dim,device=self.device)
                with torch.no_grad():
                    u_y = self.predict(x_aug,self.teacher_model,eps).detach()
                    u_y = F.layer_norm(u_y, (u_y.shape[-1],))
                
                # Forward diffusion (add noise progressively)
                z = [u_y]
                for t in range(1, self.T + 1):
                    
                    z_t = torch.sqrt(self.alpha[t-1]) * z[-1] + torch.sqrt(1 - self.alpha[t-1]) * eps
                    z.append(z_t)
            
                # Extract image features once
                self.cnn_optimizer.zero_grad()
                for opt in self.mlp_optimizers:
                    opt.zero_grad()
                
                # Use autocast for mixed precision training if requested
                with autocast() if use_amp else nullcontext():
                    x_features = self.contrastive_model.base_feature_extractor(x)
                    
                    # Train MLPs independently
                    losses = []
                    for t in range(self.T):
                        # Each MLP tries to denoise from its specific noise level
                        u_hat = self.contrastive_model.mlps[t](x_features, z[t+1].detach())
                        loss_tcr = - R(u_hat)/100
                        loss_mse = (u_hat - u_y.detach()) ** 2
                        loss = loss_mse.mean(-1).mean() #+ loss_tcr
                        loss_cos = 1 - F.cosine_similarity(u_hat,u_y.detach(),dim=-1).mean()

                        losses.append(loss)

                    # Optimize all models
                    total_loss = sum(losses)
                
                # Use scaler for AMP if enabled
                if use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(self.cnn_optimizer)
                    for opt in self.mlp_optimizers:
                        scaler.step(opt)
                    scaler.update()
                else:
                    total_loss.backward()
                    self.cnn_optimizer.step()
                    for opt in self.mlp_optimizers:
                        opt.step()

                epoch_loss += total_loss.item()
                batch_count += 1
                
                # Log progress at specified frequency
                if i % log_freq == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(train_loader)}], Loss: {total_loss.item():.4f}, Loss_tcr: {loss_tcr.item():.4f}, Loss_cos: {loss_cos.item():.4f}")


                with torch.no_grad():
                    momentum = 0.996
                    for param_q,param_k in zip(self.contrastive_model.parameters(),self.teacher_model.parameters()):
                        param_k.data = param_k.data * momentum + param_q.data * (1 - momentum)  
                    
                    
            # Epoch summary
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

            # At the end of each epoch, evaluate on test set
            self.eval()
            with torch.no_grad():
                
                # Extract features from train set
                all_train_features = []
                all_train_labels = []
                for views, labels in train_loader:
                    images = views[0].to(self.device)
                    features = self.predict(images,self.contrastive_model)
                    all_train_features.append(features)
                    all_train_labels.append(labels)
            
                # Extract features from test set
                all_test_features = []
                all_test_labels = []
                for images, labels in test_loader:
                    images = images.to(self.device)
                    features = self.predict(images,self.contrastive_model)
                    all_test_features.append(features)
                    all_test_labels.append(labels)
                
                # Concatenate all features and labels
                all_test_features = torch.cat(all_test_features, dim=0)
                all_test_labels = torch.cat(all_test_labels, dim=0)
                all_train_features = torch.cat(all_train_features, dim=0)
                all_train_labels = torch.cat(all_train_labels, dim=0)
                # Get KNN probabilities
                knn_probs = knn_predict(all_train_features, all_train_labels, all_test_features)
                
                # Create and save t-SNE plot
                create_tsne_plot(
                    all_test_features.cpu().numpy(),
                    all_test_labels.cpu().numpy(),
                    knn_probs.numpy(),
                    epoch,
                    output_dir
                )
            
            self.train()
            
            # Save model checkpoint at specified frequency
            if (epoch + 1) % save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'base_feature_extractor_state_dict': self.contrastive_model.base_feature_extractor.state_dict(),
                    'mlps_state_dict': [mlp.state_dict() for mlp in self.contrastive_model.mlps],
                    'cnn_optimizer_state_dict': self.cnn_optimizer.state_dict(),
                    'mlp_optimizers_state_dict': [opt.state_dict() for opt in self.mlp_optimizers],
                    'alpha': self.alpha,
                    'T': self.T,
                    'embed_dim': self.embed_dim,
                    'feature_dim': self.feature_dim
                }
                torch.save(checkpoint, os.path.join(output_dir, f'lda_checkpoint_epoch_{epoch+1}.pth'))

    def predict(self, x,model,z_t=None):
        x = x.to(self.device)
        # Start from random noise
        x_features = model.base_feature_extractor(x)
        b,n,c = x_features.shape
        if z_t is None:
            z_t = torch.randn(b,c,device=self.device)
        
        
        # Iteratively denoise
        for t in reversed(range(self.T)):
            u_hat = model.mlps[t](x_features, z_t)
            # Optional: add some noise proportional to the step
            if t > 0:  # Skip noise at the last step for better results
                z_t = torch.sqrt(self.alpha[t]) * u_hat + torch.sqrt(1 - self.alpha[t]) * torch.randn_like(u_hat)
            else:
                z_t = u_hat
                
        return z_t

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                for i in range(len(x)):
                    pred = self.predict(x[i],self.contrastive_model)
                    if pred.item() == y[i].item():
                        correct += 1
                    total += 1
                    
        accuracy = 100.0 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def visualize_predictions(self, test_loader, num_samples=5):
        x_test, y_test = next(iter(test_loader))
        
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            # Get a sample
            x, y = x_test[i], y_test[i]
            
            # Get prediction
            pred = self.predict(x,self.contrastive_model)
            
            # Plot
            plt.subplot(1, num_samples, i+1)
            plt.imshow(x.squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"True: {y.item()}\nPred: {pred.item()}", 
                     color='green' if pred.item() == y.item() else 'red')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig('predictions.png')
        plt.close()
        print(f"Saved visualizations to 'predictions.png'")



def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


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
        x = self.forward_feature(x)
        x = self.head(x)
        return x
    

# Modify training function
def train_ebm_gan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize AMP scaler
    d_scaler = GradScaler()
    g_scaler = GradScaler()
    
    # Load CIFAR-10
    if args.dataset=='cifar10':
       
        transform = DataAugmentationiBOT(
            global_crops_scale=args.global_crops_scale,
            local_crops_scale=args.local_crops_scale,
            global_crops_number=args.global_crops_number,
            local_crops_number=args.local_crops_number,
            img_size=(args.img_size,args.img_size)
        )
        
        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                          download=True, transform=transform)
    else:

        transform = DataAugmentationiBOT(
            global_crops_scale=args.global_crops_scale,
            local_crops_scale=args.local_crops_scale,
            global_crops_number=args.global_crops_number,
            local_crops_number=args.local_crops_number,
            img_size=(args.img_size,args.img_size)
        )
        
        # transform = MultiViewTransform(transform,n_views=2)
        # Load Tiny ImageNet or ImageNet-100 dataset using ImageFolder
        trainset = torchvision.datasets.ImageFolder(
            root=args.data_path,
            transform=transform,
        )
    # Filter the dataset to only include class 1
    if args.cls!=-1:
        class_1_indices = [i for i, label in enumerate(trainset.targets) if label == args.cls]
        trainset.data = trainset.data[class_1_indices]
        trainset.targets = [trainset.targets[i] for i in class_1_indices]
    

    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)

    # # Initialize models
    # generator = Decoder(latent_dim=args.latent_dim).to(device)
    # discriminator = Encoder(latent_dim=args.latent_dim).to(device)
    # teacher_discriminator = Encoder(latent_dim=args.latent_dim).to(device)
    # # checkpoint = torch.load('../../autodl-tmp/output_cl_dino_all/ebm_gan_checkpoint_1000.pth')
    # # teacher_discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Initialize models
    generator = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=0,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        use_checkpoint=args.use_checkpoint
    ).to(device)

    # generator_projector = GeneratorProjector(latent_dim=192).to(device)
    


    # discriminator = ResNetEnergyNet(img_channels=3, hidden_dim=64).to(device)
    discriminator = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=0,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        use_checkpoint=args.use_checkpoint
    ).to(device)
    
    teacher_discriminator =  MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=0,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
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
    
    
    start_epoch = 0
    
    latent_denoising_alignment = LatentDenoisingAlignment(args=args)
    
    pca_noiser = SVDPCANoise(noise_scale=args.noise_scale, kernel='linear', gamma=1.0) if args.noise_scale>0 else nn.Identity()
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


        latent_denoising_alignment.train_model(trainloader,args.epochs)
        
def save_gan_samples(generator, discriminator, epoch, output_dir, device, n_samples=36,real_samples=None,pca_noiser=None):
    generator.eval()
    real_samples = real_samples[:n_samples]
    batch_size = real_samples.size(0)
    with torch.no_grad():
        
        z = discriminator.forward_feature(real_samples.detach())
        b,n,c = z.shape
        z = z[:,0]
        mask_tokens = generator.mask_token.repeat(z.shape[0],n-1, 1)
        z_noised = torch.cat([z.unsqueeze(1), mask_tokens], dim=1)
        z_noised = z_noised + generator.decoder_pos_embed

        
        fake_samples= generator.forward_decoder(z_noised)
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
    
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset to train on')
    
    parser.add_argument('--T', default=12, type=int)
    # Add GAN-specific parameters
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--g_lr', default=1e-4, type=float)
    parser.add_argument('--d_lr', default=1e-4, type=float)
    parser.add_argument('--n_critic', default=1, type=int,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--gp_weight', default=0.05, type=float,
                        help='Weight of gradient penalty')
    parser.add_argument('--noise_scale', default=0.0, type=float,
                        help='Weight of gradient penalty')
    
    # Modify learning rates
    parser.add_argument('--g_beta1', default=0.5, type=float,
                        help='Beta1 for generator optimizer')
    parser.add_argument('--g_beta2', default=0.999, type=float,
                        help='Beta2 for generator optimizer')
    
    parser.add_argument('--cls', default=-1, type=int,
                        help='Class to train on')
    parser.add_argument('--adv_weight', default=0.1, type=float,
                        help='Weight of adversarial loss')
    
    # Existing parameters
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    parser.add_argument('--data_path', default='c:/datasets', type=str)
    parser.add_argument('--output_dir', default='/mnt/d/repo/output/cifar10-ebm-gan-r3gan-ctrl')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    
    # Add learning rate scheduling parameters
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate for cosine annealing')
    
    parser.add_argument('--weight_decay', default=0.04, type=float,
                        help='Weight decay for cosine annealing')
    parser.add_argument('--final_weight_decay', default=0.4, type=float,
                        help='Final weight decay for cosine annealing')
    parser.add_argument('--ema_momentum', default=0.996, type=float,
                        help='EMA momentum for teacher model')


    # vit parameters 
    parser.add_argument('--img_size', default=32, type=int) 
    parser.add_argument('--patch_size', default=4, type=int)    
    parser.add_argument('--embed_dim', default=192, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num_heads', default=3, type=int)
    parser.add_argument('--decoder_embed_dim', default=192, type=int)
    parser.add_argument('--decoder_depth', default=12, type=int)
    parser.add_argument('--decoder_num_heads', default=3, type=int)
    parser.add_argument('--mlp_ratio', default=4., type=float)
    parser.add_argument('--use_checkpoint', action='store_true')

    ## augmentation params
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.25, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.25 1." for example)""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.25),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping. When disabling multi-crop (--global_crops_number 1), we
        recommand using a narrower range of scale ("--local_crops_scale 0.05 0.25" for example)""")
    parser.add_argument('--global_crops_number', type=int, default=2,
        help='Number of global crops')
    parser.add_argument('--local_crops_number', type=int, default=10,
        help='Number of local crops')
    
    
    # Add checkpoint loading parameter
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to checkpoint to resume training from')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ebm_gan(args) 
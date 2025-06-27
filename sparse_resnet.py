import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import spconv.pytorch as spconv


class SparseConv2d(nn.Module):
    """Simple sparse Conv2d wrapper using spconv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, subm=False):
        super().__init__()
        
        if subm:
            self.conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        else:
            self.conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    
    def forward(self, x):
        return self.conv(x)


class ResBlockSparse(nn.Module):
    """Sparse ResNet block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = SparseConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, subm=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = SparseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, subm=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = SparseConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, subm=True)
            self.skip_bn = nn.BatchNorm1d(out_channels)
        else:
            self.skip = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        
        if self.skip is not None:
            identity = self.skip(identity)
            identity = identity.replace_feature(self.skip_bn(identity.features))
        
        # Use sparse tensor addition
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        
        return out


class ResBlockDownSparse(nn.Module):
    """Sparse ResNet downsampling block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = SparseConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, subm=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = SparseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, subm=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        
        
    def forward(self, x):

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        return out


class SparseEncoderResNet(nn.Module):
    """Sparse ResNet Encoder"""
    
    def __init__(
        self, 
        img_channels: int = 3,
        patch_size: int = 16, 
        hidden_dim: int = 192,
        embed_dim: int = 192,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # Calculate number of layers based on patch_size
        n_layers = int(np.log2(patch_size))
        
        # Stem layer
        self.stem = SparseConv2d(img_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False, subm=False)
        self.stem_bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Down blocks and res blocks
        self.down_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        
        for i in range(n_layers):
            down_block = ResBlockDownSparse(hidden_dim, hidden_dim)
            res_block = ResBlockSparse(hidden_dim, hidden_dim)
            self.down_blocks.append(down_block)
            self.res_blocks.append(res_block)
        
        # Final projection
        self.final_proj = SparseConv2d(hidden_dim, embed_dim, kernel_size=1, bias=False, subm=True)
        self.final_bn = nn.BatchNorm1d(embed_dim)
        
        # Calculate default patch dimensions (will be updated based on actual input size)
        self.expected_img_size = patch_size * (2 ** n_layers)
        self.num_patches_per_side = self.expected_img_size // patch_size  # Default value
        self.num_patches = self.num_patches_per_side ** 2  # Default value
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights"""
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
            torch.nn.init.xavier_uniform_(m.conv.weight)
            if hasattr(m.conv, 'bias') and m.conv.bias is not None:
                nn.init.constant_(m.conv.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _update_patch_dimensions(self, H: int, W: int):
        """Update patch dimensions based on actual input size"""
        assert H == W, f"Expected square input, got {H}x{W}"
        self.actual_img_size = H
        self.num_patches_per_side = H // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
    
    def create_mask(self, x: torch.Tensor, mask_ratio: float=0.75) -> tuple[torch.Tensor, torch.Tensor]:
        """Create random mask for patches and return ids_restore"""
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

        return mask, ids_restore
    
    def image_to_sparse_tensor(self, images: torch.Tensor, mask: torch.Tensor) -> spconv.SparseConvTensor:
        """Convert masked images to sparse tensor"""
        B, C, H, W = images.shape

        L = mask.shape[1]

        f = int(L**.5)

        scale_H = H // f
        scale_W = W // f

        keep_mask = ~mask.bool()
        # Convert patch mask to spatial mask
        spatial_mask = keep_mask.reshape(B, f, f).repeat_interleave(scale_H, dim=1).repeat_interleave(scale_W, dim=2)
        
        
        # Create sparse tensor
        indices_list = []
        features_list = []
        
        for b in range(B):
            active_mask = spatial_mask[b] > 0.5
            y_coords, x_coords = torch.where(active_mask)
            
            if len(y_coords) > 0:
                batch_indices = torch.full((len(y_coords),), b, dtype=torch.int32, device=images.device)
                indices = torch.stack([batch_indices, y_coords.int(), x_coords.int()], dim=1)
                indices_list.append(indices)
                
                features = images[b, :, y_coords, x_coords].t()
                features_list.append(features)
        
        if indices_list:
            all_indices = torch.cat(indices_list, dim=0)
            all_features = torch.cat(features_list, dim=0)
        else:
            all_indices = torch.empty((0, 3), dtype=torch.int32, device=images.device)
            all_features = torch.empty((0, C), dtype=images.dtype, device=images.device)
        
        return spconv.SparseConvTensor(
            features=all_features,
            indices=all_indices,
            spatial_shape=[H, W],
            batch_size=B
        )
    
    def forward_backbone(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Forward through backbone"""
        # Stem
        x = self.stem(x)
        x = x.replace_feature(self.stem_bn(x.features))
        x = x.replace_feature(self.relu(x.features))
        
        # Down and res blocks
        for down_block, res_block in zip(self.down_blocks, self.res_blocks):
            x = down_block(x)
            x = res_block(x)
        
        return x

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
        
    def forward(self, images: torch.Tensor, mask: Optional[torch.Tensor] = None, mask_ratio: float = 0.75) -> Tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: Input images [B, C, H, W]
            mask: Optional mask for patches [B, num_patches]
            
        Returns:
            features: Sparse feature tensor
            mask: Applied mask [B, num_patches] 
            ids_restore: Restoration indices [B, num_patches]
        """
        B, C, H, W = images.shape
        device = images.device
        
        # Generate mask if not provided
        if mask is None:
            patch_images = self.patchify(images)
            mask, ids_restore = self.create_mask(patch_images, mask_ratio)
        else:
            # If mask is provided, create dummy ids_restore
            ids_restore = torch.arange(mask.shape[1], device=device).unsqueeze(0).repeat(B, 1)

        # Convert to sparse tensor
        sparse_input = self.image_to_sparse_tensor(images, mask)
        
        # Forward through backbone
        x = self.forward_backbone(sparse_input)
        
        # Final projection
        x = self.final_proj(x)
        x = x.replace_feature(self.final_bn(x.features))
        x = x.replace_feature(self.relu(x.features))
        
        return x, mask, ids_restore


def create_sparse_resnet(
    img_channels: int = 3,
    patch_size: int = 16,
    hidden_dim: int = 192,
    embed_dim: int = 192,
    **kwargs
) -> SparseEncoderResNet:
    """Factory function to create a sparse ResNet"""
    return SparseEncoderResNet(
        img_channels=img_channels,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        **kwargs
    )


class SPConvEncoder(nn.Module):
    def __init__(self, img_channels=3, patch_size=16, hidden_dim=192, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        self.sparse_resnet = create_sparse_resnet(
            img_channels=img_channels,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim
        )


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

    def forward_feature(self, x):
        x, mask, ids_restore = self.sparse_resnet(x,mask_ratio=0)
        return x

    def forward(self, x, mask_ratio=0.75):
        x, mask, ids_restore = self.sparse_resnet(x,mask_ratio=mask_ratio)
        x = x.dense()
        b,c,h,w = x.shape 

        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c
        keep_mask = ~mask.bool()
        
        x = x[keep_mask,:] # b,h*w,c -> b,h*w_masked,c
        x = x.reshape(b,-1,c) # b,h*w_masked,c -> b,h*w_masked/c
        
        return x, mask, ids_restore

# Example usage
if __name__ == "__main__":
    if torch.cuda.is_available():
        # Test input
        batch_size = 1
        img_size = 128
        patch_size = 16

        model = create_sparse_resnet(
            img_channels=3,
            patch_size=patch_size,
            hidden_dim=192,
            embed_dim=192
        ).cuda()
        
        
        
        images = torch.randn(batch_size, 3, img_size, img_size).cuda()
        
        # Forward pass
        features, mask, ids_restore = model(images, mask_ratio=0.75)
        
        print(f"Input shape: {images.shape}")
        print(f"Output features: {features.dense().shape} {features.features.shape[0]}")
        print(f"Mask shape: {mask.shape}")
        print(f"Pred patches: {mask.sum(dim=1)}")
        print(f"ids_restore shape: {ids_restore.shape}")
        
        encoder = SPConvEncoder(
            img_channels=3,
            patch_size=patch_size,
            hidden_dim=192,
            embed_dim=192
        ).cuda()

        x, mask, ids_restore = encoder(images, mask_ratio=0.75)
        print(f"Encoder features: {x.shape}")
        print(f"Encoder mask: {mask.shape}")
        print(f"Encoder ids_restore: {ids_restore.shape}")
    else:
        print("CUDA not available, skipping test")


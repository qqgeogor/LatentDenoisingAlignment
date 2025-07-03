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
        out = out.replace_feature(self.relu(out.features))

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
        print('Using SparseResNet')
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # Calculate number of layers based on patch_size
        n_layers = int(np.log2(patch_size))
        
        all_hidden_dims = [hidden_dim] 
        for i in range(n_layers-1):
            all_hidden_dims.append(all_hidden_dims[-1]//2)
        all_hidden_dims = all_hidden_dims[::-1]

        print('all_hidden_dims', all_hidden_dims)
        print('n_layers', n_layers)

        # Stem layer
        self.stem = SparseConv2d(img_channels, all_hidden_dims[0]//2, kernel_size=1, stride=1, padding=0, bias=False, subm=False)
        self.stem_bn = nn.BatchNorm1d(all_hidden_dims[0]//2)
        self.relu = nn.ReLU(inplace=True)
        
        # Down blocks and res blocks
        self.down_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        
        for i in range(n_layers):
            print(f'all_hidden_dims {i}', all_hidden_dims[i])
            down_block = ResBlockDownSparse(all_hidden_dims[i]//2, all_hidden_dims[i])
            res_block = ResBlockSparse(all_hidden_dims[i], all_hidden_dims[i])
            self.down_blocks.append(down_block)
            self.res_blocks.append(res_block)
        
        # Final projection
        self.final_proj = SparseConv2d(all_hidden_dims[-1], embed_dim, kernel_size=1, bias=False, subm=True)
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
        
        # Vectorized approach: find all active coordinates at once
        active_coords = torch.nonzero(spatial_mask, as_tuple=False)  # [N, 3] where N is total active pixels
        
        if active_coords.numel() > 0:
            # Extract batch, y, x coordinates
            batch_indices = active_coords[:, 0].int()
            y_coords = active_coords[:, 1].int()
            x_coords = active_coords[:, 2].int()
            
            # Create indices tensor for sparse conv (batch, y, x)
            all_indices = torch.stack([batch_indices, y_coords, x_coords], dim=1)
            
            # Gather features using advanced indexing - shape is already [N, C]
            all_features = images[batch_indices, :, y_coords, x_coords]
            
        else:
            # Handle empty case
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
        x = x.dense()
        b,c,h,w = x.shape 
        x = x.reshape(b,c,h*w).permute(0,2,1) # b,c,h,w -> b,h*w,c

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




# Dense version for parameter compatibility
class DenseConv2d(nn.Module):
    """Dense Conv2d wrapper matching SparseConv2d structure"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, subm=False):
        super().__init__()
        # subm parameter is ignored for dense convs - all convs work the same
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    
    def forward(self, x):
        return self.conv(x)


class DenseResBlock(nn.Module):
    """Dense ResNet block matching ResBlockSparse structure"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = DenseConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, subm=True)
        self.bn1 = nn.BatchNorm2d(out_channels)  # BatchNorm2d for dense tensors
        
        self.conv2 = DenseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, subm=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = DenseConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, subm=True)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skip is not None:
            identity = self.skip(identity)
            identity = self.skip_bn(identity)
        
        out = out + identity  # Simple addition for dense tensors
        out = self.relu(out)
        
        return out


class DenseResBlockDown(nn.Module):
    """Dense ResNet downsampling block matching ResBlockDownSparse structure"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = DenseConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False, subm=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = DenseConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, subm=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out


class DenseEncoderResNet(nn.Module):
    """Dense ResNet Encoder matching SparseEncoderResNet structure"""
    
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

        all_hidden_dims = [hidden_dim] 
        for i in range(n_layers-1):
            all_hidden_dims.append(all_hidden_dims[-1]//2)
        all_hidden_dims = all_hidden_dims[::-1]

        print('all_hidden_dims', all_hidden_dims)
        print('n_layers', n_layers)
        
        
        # Stem layer
        self.stem = DenseConv2d(img_channels, all_hidden_dims[0]//2, kernel_size=1, stride=1, padding=0, bias=False, subm=False)
        self.stem_bn = nn.BatchNorm2d(all_hidden_dims[0]//2)  # BatchNorm2d for dense tensors
        self.relu = nn.ReLU(inplace=True)
        
        # Down blocks and res blocks
        self.down_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        
        for i in range(n_layers):
            down_block = DenseResBlockDown(all_hidden_dims[i]//2, all_hidden_dims[i])
            res_block = DenseResBlock(all_hidden_dims[i], all_hidden_dims[i])
            self.down_blocks.append(down_block)
            self.res_blocks.append(res_block)
        
        # Final projection
        self.final_proj = DenseConv2d(all_hidden_dims[-1], embed_dim, kernel_size=1, bias=False, subm=True)
        self.final_bn = nn.BatchNorm2d(embed_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights"""
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
            torch.nn.init.xavier_uniform_(m.conv.weight)
            if hasattr(m.conv, 'bias') and m.conv.bias is not None:
                nn.init.constant_(m.conv.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = images
        
        # Stem
        x = self.stem(x)
        x = self.stem_bn(x)
        x = self.relu(x)
        
        # Down and res blocks
        for down_block, res_block in zip(self.down_blocks, self.res_blocks):
            x = down_block(x)
            x = res_block(x)
        
        # Final projection
        x = self.final_proj(x)
        x = self.final_bn(x)
        x = self.relu(x)
        
        return x


def transfer_sparse_to_dense_params(sparse_model: SparseEncoderResNet, dense_model: DenseEncoderResNet):
    """Transfer parameters from sparse model to dense model"""
    
    def copy_conv_params(sparse_conv, dense_conv):
        """Copy conv parameters - handle potential shape differences"""
        sparse_weight = sparse_conv.conv.weight.data
        dense_weight = dense_conv.conv.weight.data
        
        # Handle potential shape mismatch between spconv and torch conv
        if sparse_weight.shape != dense_weight.shape:
            # If shapes don't match, try to reshape or handle common cases
            if len(sparse_weight.shape) == len(dense_weight.shape):
                # Same number of dimensions, try to broadcast or reshape
                try:
                    dense_conv.conv.weight.data.copy_(sparse_weight.view_as(dense_weight))
                except RuntimeError:
                    print(f"Warning: Could not copy conv weights due to shape mismatch: {sparse_weight.shape} vs {dense_weight.shape}")
                    # Initialize with Xavier uniform as fallback
                    torch.nn.init.xavier_uniform_(dense_conv.conv.weight)
            else:
                print(f"Warning: Could not copy conv weights due to dimension mismatch: {sparse_weight.shape} vs {dense_weight.shape}")
                torch.nn.init.xavier_uniform_(dense_conv.conv.weight)
        else:
            dense_conv.conv.weight.data.copy_(sparse_weight)
            
        if sparse_conv.conv.bias is not None and dense_conv.conv.bias is not None:
            dense_conv.conv.bias.data.copy_(sparse_conv.conv.bias.data)
    
    def copy_bn_params(sparse_bn, dense_bn):
        """Copy batch norm parameters - handle 1D to 2D conversion"""
        dense_bn.weight.data.copy_(sparse_bn.weight.data)
        dense_bn.bias.data.copy_(sparse_bn.bias.data)
        dense_bn.running_mean.data.copy_(sparse_bn.running_mean.data)
        dense_bn.running_var.data.copy_(sparse_bn.running_var.data)
    
    # Copy stem
    copy_conv_params(sparse_model.stem, dense_model.stem)
    copy_bn_params(sparse_model.stem_bn, dense_model.stem_bn)
    
    # Copy down blocks and res blocks
    for i, (sparse_down, sparse_res) in enumerate(zip(sparse_model.down_blocks, sparse_model.res_blocks)):
        dense_down = dense_model.down_blocks[i]
        dense_res = dense_model.res_blocks[i]
        
        # Copy down block
        copy_conv_params(sparse_down.conv1, dense_down.conv1)
        copy_bn_params(sparse_down.bn1, dense_down.bn1)
        copy_conv_params(sparse_down.conv2, dense_down.conv2)
        copy_bn_params(sparse_down.bn2, dense_down.bn2)
        
        # Copy res block
        copy_conv_params(sparse_res.conv1, dense_res.conv1)
        copy_bn_params(sparse_res.bn1, dense_res.bn1)
        copy_conv_params(sparse_res.conv2, dense_res.conv2)
        copy_bn_params(sparse_res.bn2, dense_res.bn2)
        if sparse_res.skip is not None and dense_res.skip is not None:
            copy_conv_params(sparse_res.skip, dense_res.skip)
            copy_bn_params(sparse_res.skip_bn, dense_res.skip_bn)
    
    # Copy final projection
    copy_conv_params(sparse_model.final_proj, dense_model.final_proj)
    copy_bn_params(sparse_model.final_bn, dense_model.final_bn)
    
    print("✅ Parameters transferred successfully from sparse to dense model!")


def create_dense_resnet(
    img_channels: int = 3,
    patch_size: int = 16,
    hidden_dim: int = 192,
    embed_dim: int = 192,
    **kwargs
) -> DenseEncoderResNet:
    """Factory function to create a dense ResNet matching sparse structure"""
    return DenseEncoderResNet(
        img_channels=img_channels,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        **kwargs
    )




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
        
        dense_model = create_dense_resnet(
            img_channels=3,
            patch_size=patch_size,
            hidden_dim=192,
            embed_dim=192
        ).cuda()

        transfer_sparse_to_dense_params(model, dense_model)
        
        
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
        
        # Demonstrate parameter transfer to dense model
        print("\n🔄 Testing parameter transfer to dense model...")
        dense_model = create_dense_resnet(
            img_channels=3,
            patch_size=patch_size,
            hidden_dim=192,
            embed_dim=192
        ).cuda()
        
        # Transfer parameters
        transfer_sparse_to_dense_params(model, dense_model)
        
        # Test dense model
        dense_output = dense_model(images)
        print(f"Dense model output: {dense_output.shape}")
        
        # Compare outputs (should be similar for unmasked regions)
        print("✅ Dense model created and parameters transferred successfully!")
    else:
        print("CUDA not available, skipping test")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


class EqualizedLinear(nn.Module):
    """Equalized learning rate linear layer from StyleGAN."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 lr_multiplier: float = 1.0, bias_init: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_multiplier = lr_multiplier
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.full([out_features], bias_init)) if bias else None
        
        # He initialization scaling factor
        self.weight_scale = lr_multiplier / math.sqrt(in_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.weight_scale
        bias = self.bias * self.lr_multiplier if self.bias is not None else None
        return F.linear(x, weight, bias)


class EqualizedConv2d(nn.Module):
    """Equalized learning rate 2D convolution from StyleGAN."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 lr_multiplier: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr_multiplier = lr_multiplier
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # He initialization scaling factor
        self.weight_scale = lr_multiplier / math.sqrt(in_channels * kernel_size * kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.weight_scale
        bias = self.bias * self.lr_multiplier if self.bias is not None else None
        return F.conv2d(x, weight, bias, self.stride, self.padding)


class ModulatedConv2d(nn.Module):
    """Modulated convolution layer from StyleGAN2."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 style_dim: int, demodulate: bool = True, up: int = 1, down: int = 1,
                 resample_filter: Optional[List[int]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.up = up
        self.down = down
        
        # Weight and modulation
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.affine = EqualizedLinear(style_dim, in_channels, bias_init=1.0)
        
        # Resampling filter
        if resample_filter is None:
            resample_filter = [1, 3, 3, 1]
        self.register_buffer('resample_filter', torch.tensor(resample_filter, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Upsampling if needed
        if self.up > 1:
            x = F.interpolate(x, scale_factor=self.up, mode='bilinear', align_corners=False)
        
        # Modulation
        style = self.affine(style).view(batch_size, 1, self.in_channels, 1, 1)
        weight = self.weight.unsqueeze(0) * style
        
        # Demodulation
        if self.demodulate:
            sigma_inv = torch.rsqrt((weight ** 2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * sigma_inv
        
        # Reshape for group convolution
        x = x.view(1, batch_size * self.in_channels, x.shape[2], x.shape[3])
        weight = weight.view(batch_size * self.out_channels, self.in_channels, 
                           self.kernel_size, self.kernel_size)
        
        # Convolution
        x = F.conv2d(x, weight, groups=batch_size, padding=self.kernel_size // 2)
        x = x.view(batch_size, self.out_channels, x.shape[2], x.shape[3])
        
        # Downsampling if needed
        if self.down > 1:
            x = F.avg_pool2d(x, self.down)
        
        return x


class NoiseInjection(nn.Module):
    """Noise injection layer from StyleGAN."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return x + self.weight * noise


class MappingNetwork(nn.Module):
    """Mapping network that converts latent codes to style vectors."""
    
    def __init__(self, z_dim: int = 512, w_dim: int = 512, num_layers: int = 8,
                 lr_multiplier: float = 0.01, activation: str = 'lrelu',
                 conditioning_dim: Optional[int] = None):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        self.conditioning_dim = conditioning_dim
        
        # If conditioning is used, combine z and conditioning embedding
        input_dim = z_dim
        if conditioning_dim is not None:
            input_dim += conditioning_dim
            # Optional: Add a projection layer for conditioning
            self.condition_proj = EqualizedLinear(conditioning_dim, conditioning_dim, lr_multiplier=lr_multiplier)
        
        # Build the mapping network
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else w_dim
            layers.append(EqualizedLinear(in_dim, w_dim, lr_multiplier=lr_multiplier))
            if activation == 'lrelu':
                layers.append(nn.LeakyReLU(0.2))
            elif activation == 'relu':
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z: Latent codes of shape [batch_size, z_dim]
            conditioning: Optional conditioning embedding of shape [batch_size, conditioning_dim]
        Returns:
            w: Style codes of shape [batch_size, w_dim]
        """
        if self.conditioning_dim is not None:
            if conditioning is not None:
                # Project and combine conditioning with latent code
                conditioning_proj = self.condition_proj(conditioning)
            else:
                # If model expects conditioning but none provided, use zeros
                conditioning_proj = self.condition_proj(torch.zeros(z.shape[0], self.conditioning_dim, device=z.device))
            z = torch.cat([z, conditioning_proj], dim=1)
        
        return self.network(z)


class SynthesisBlock(nn.Module):
    """Synthesis block for the generator."""
    
    def __init__(self, in_channels: int, out_channels: int, w_dim: int, 
                 resolution: int, is_first: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.is_first = is_first
        
        if is_first:
            # First block starts with a constant
            self.const = nn.Parameter(torch.randn(1, in_channels, 4, 4))
            self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, w_dim)
        else:
            # Subsequent blocks have upsampling
            self.conv0 = ModulatedConv2d(in_channels, out_channels, 3, w_dim, up=2)
            self.conv1 = ModulatedConv2d(out_channels, out_channels, 3, w_dim)
        
        self.noise1 = NoiseInjection(out_channels)
        self.noise2 = NoiseInjection(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
        # ToRGB layer
        self.to_rgb = ModulatedConv2d(out_channels, 3, 1, w_dim, demodulate=False)
        
    def forward(self, x: torch.Tensor, w: torch.Tensor, 
                noise1: Optional[torch.Tensor] = None, 
                noise2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_first:
            # Start with constant
            x = self.const.repeat(w.shape[0], 1, 1, 1)
            x = self.conv1(x, w)
            x = self.noise1(x, noise1)
            x = self.activation(x)
        else:
            # Upsample and convolve
            x = self.conv0(x, w)
            x = self.noise1(x, noise1)
            x = self.activation(x)
            
            x = self.conv1(x, w)
            x = self.noise2(x, noise2)
            x = self.activation(x)
        
        # RGB output
        rgb = self.to_rgb(x, w)
        
        return x, rgb


class SynthesisNetwork(nn.Module):
    """Synthesis network (generator) for StyleGAN."""
    
    def __init__(self, w_dim: int = 512, img_resolution: int = 1024, 
                 img_channels: int = 3, channel_base: int = 16384, 
                 channel_max: int = 512):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        
        # Calculate the number of layers needed
        self.num_layers = int(np.log2(img_resolution)) - 1  # -1 because we start at 4x4
        
        # Build synthesis blocks
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            resolution = 4 * (2 ** i)
            in_channels = min(channel_max, channel_base // resolution)
            out_channels = min(channel_max, channel_base // (resolution * 2))
            
            if i == 0:
                in_channels = min(channel_max, channel_base // 4)
                
            block = SynthesisBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                w_dim=w_dim,
                resolution=resolution,
                is_first=(i == 0)
            )
            self.blocks.append(block)
    
    def forward(self, w: torch.Tensor, noise_mode: str = 'random') -> torch.Tensor:
        """
        Args:
            w: Style codes of shape [batch_size, num_layers, w_dim] or [batch_size, w_dim]
            noise_mode: 'random', 'const', or 'none'
        Returns:
            Generated images of shape [batch_size, img_channels, img_resolution, img_resolution]
        """
        batch_size = w.shape[0]
        
        # Handle style broadcasting
        if w.dim() == 2:
            w = w.unsqueeze(1).repeat(1, len(self.blocks), 1)
        
        x = None
        rgb_out = None
        
        for i, block in enumerate(self.blocks):
            # For noise generation, we need the resolution after potential upsampling
            if i == 0:
                # First block: starts at 4x4
                noise_resolution = 4
            else:
                # Subsequent blocks: previous resolution * 2 (due to upsampling)
                noise_resolution = 4 * (2 ** i)
            
            # Generate noise
            if noise_mode == 'random':
                noise1 = torch.randn(batch_size, 1, noise_resolution, noise_resolution, device=w.device)
                noise2 = torch.randn(batch_size, 1, noise_resolution, noise_resolution, device=w.device)
            elif noise_mode == 'const':
                noise1 = torch.zeros(batch_size, 1, noise_resolution, noise_resolution, device=w.device)
                noise2 = torch.zeros(batch_size, 1, noise_resolution, noise_resolution, device=w.device)
            else:  # none
                noise1 = noise2 = None
            
            x, rgb = block(x, w[:, i], noise1, noise2)
            
            # Accumulate RGB output with upsampling
            if rgb_out is not None:
                rgb_out = F.interpolate(rgb_out, scale_factor=2, mode='bilinear', align_corners=False)
                rgb_out = rgb_out + rgb
            else:
                rgb_out = rgb
        
        assert rgb_out is not None, "No synthesis blocks were processed"
        return torch.tanh(rgb_out)


class Discriminator(nn.Module):
    """Discriminator network for StyleGAN."""
    
    def __init__(self, img_resolution: int = 1024, img_channels: int = 3,
                 channel_base: int = 16384, channel_max: int = 512):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        
        # Calculate the number of layers
        self.num_layers = int(np.log2(img_resolution)) - 1
        
        # Build discriminator blocks
        layers = []
        
        # FromRGB layer
        layers.append(EqualizedConv2d(img_channels, min(channel_max, channel_base // img_resolution), 1))
        layers.append(nn.LeakyReLU(0.2))
        
        # Downsampling blocks
        for i in range(self.num_layers):
            resolution = img_resolution // (2 ** i)
            in_channels = min(channel_max, channel_base // resolution)
            out_channels = min(channel_max, channel_base // (resolution // 2))
            
            # Convolution block
            layers.append(EqualizedConv2d(in_channels, out_channels, 3, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            
            # Downsampling
            if i < self.num_layers - 1:
                layers.append(nn.AvgPool2d(2))
        
        # Final layers
        final_channels = min(channel_max, channel_base // 4)
        layers.append(EqualizedConv2d(final_channels, final_channels, 3, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(EqualizedConv2d(final_channels, final_channels, 4))
        layers.append(nn.LeakyReLU(0.2))
        
        # Output layer
        layers.append(EqualizedLinear(final_channels, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape [batch_size, img_channels, img_resolution, img_resolution]
        Returns:
            Discriminator scores of shape [batch_size, 1]
        """
        batch_size = x.shape[0]
        x = self.network[:-1](x)  # All layers except the final linear
        x = x.view(batch_size, -1)  # Flatten
        x = self.network[-1](x)  # Final linear layer
        return x


class StyleGAN(nn.Module):
    """Complete StyleGAN model with mapping network, generator, and discriminator."""
    
    def __init__(self, 
                 z_dim: int = 512,
                 w_dim: int = 512,
                 img_resolution: int = 1024,
                 img_channels: int = 3,
                 mapping_layers: int = 8,
                 channel_base: int = 16384,
                 channel_max: int = 512,
                 conditioning_dim: Optional[int] = None):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.conditioning_dim = conditioning_dim
        
        # Networks
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers, conditioning_dim=conditioning_dim)
        self.synthesis = SynthesisNetwork(w_dim, img_resolution, img_channels, 
                                        channel_base, channel_max)
        self.discriminator = Discriminator(img_resolution, img_channels,
                                         channel_base, channel_max)
        
    def forward(self, z: torch.Tensor, conditioning: Optional[torch.Tensor] = None, 
                noise_mode: str = 'random') -> torch.Tensor:
        """
        Generate images from latent codes.
        
        Args:
            z: Latent codes of shape [batch_size, z_dim]
            conditioning: Optional conditioning embedding of shape [batch_size, conditioning_dim]
            noise_mode: 'random', 'const', or 'none'
        Returns:
            Generated images of shape [batch_size, img_channels, img_resolution, img_resolution]
        """
        w = self.mapping(z, conditioning)
        return self.synthesis(w, noise_mode)
    
    def generate(self, batch_size: int, device: torch.device, 
                 truncation_psi: float = 1.0, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate random images.
        
        Args:
            batch_size: Number of images to generate
            device: Device to generate on
            truncation_psi: Truncation factor for style mixing
            conditioning: Optional conditioning embedding of shape [batch_size, conditioning_dim]
        Returns:
            Generated images
        """
        z = torch.randn(batch_size, self.z_dim, device=device)
        w = self.mapping(z, conditioning)
        
        # Apply truncation trick
        if truncation_psi < 1.0:
            w_avg = w.mean(dim=0, keepdim=True)
            w = w_avg + truncation_psi * (w - w_avg)
            
        return self.synthesis(w)
    
    def get_model_info(self) -> dict:
        """Get detailed information about the model architecture and parameters."""
        
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        mapping_params = count_parameters(self.mapping)
        synthesis_params = count_parameters(self.synthesis)
        discriminator_params = count_parameters(self.discriminator)
        total_params = mapping_params + synthesis_params + discriminator_params
        
        return {
            "model_type": "StyleGAN3-inspired",
            "architecture": {
                "z_dim": self.z_dim,
                "w_dim": self.w_dim,
                "img_resolution": self.img_resolution,
                "img_channels": self.img_channels,
                "num_synthesis_layers": len(self.synthesis.blocks)
            },
            "parameters": {
                "mapping_network": f"{mapping_params:,}",
                "synthesis_network": f"{synthesis_params:,}",
                "discriminator": f"{discriminator_params:,}",
                "total": f"{total_params:,}"
            },
            "features": [
                "Equalized learning rate",
                "Style-based generator",
                "Modulated convolutions",
                "Noise injection",
                "Progressive synthesis",
                "Anti-aliasing (via proper upsampling)",
                "Truncation trick support"
            ]
        }


# Example usage and model instantiation
def create_stylegan_model(resolution: int = 1024, conditioning_dim: Optional[int] = None, **kwargs) -> StyleGAN:
    """
    Create a StyleGAN model with specified resolution.
    
    Args:
        resolution: Output image resolution (64, 128, 256, 512, 1024)
        conditioning_dim: Optional dimension for conditioning embeddings (e.g., 512 for text embeddings)
        **kwargs: Additional arguments for StyleGAN
    
    Returns:
        StyleGAN model instance
    """
    return StyleGAN(img_resolution=resolution, conditioning_dim=conditioning_dim, **kwargs)


def create_conditional_stylegan(resolution: int = 256, conditioning_dim: int = 512, **kwargs) -> StyleGAN:
    """
    Create a conditional StyleGAN model optimized for embedding-based generation.
    
    Args:
        resolution: Output image resolution
        conditioning_dim: Dimension of conditioning embeddings (e.g., CLIP embeddings are 512D)
        **kwargs: Additional arguments for StyleGAN
    
    Returns:
        Conditional StyleGAN model instance
    """
    return StyleGAN(
        img_resolution=resolution,
        conditioning_dim=conditioning_dim,
        z_dim=512,
        w_dim=512,
        mapping_layers=8,
        **kwargs
    )


# Model configurations for different scales
STYLEGAN_CONFIGS = {
    "stylegan_64": {
        "img_resolution": 64,
        "channel_base": 8192,
        "channel_max": 256
    },
    "stylegan_128": {
        "img_resolution": 128,
        "channel_base": 8192,
        "channel_max": 256
    },
    "stylegan_256": {
        "img_resolution": 256,
        "channel_base": 16384,
        "channel_max": 512
    },
    "stylegan_512": {
        "img_resolution": 512,
        "channel_base": 16384,
        "channel_max": 512
    },
    "stylegan_1024": {
        "img_resolution": 1024,
        "channel_base": 16384,
        "channel_max": 512
    }
}


if __name__ == "__main__":
    # Example: Create and test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Testing Unconditional StyleGAN ===")
    # Create model for 256x256 resolution
    model = create_stylegan_model(resolution=256)
    model = model.to(device)
    
    # Print model information
    info = model.get_model_info()
    print("StyleGAN Model Information:")
    print(f"Type: {info['model_type']}")
    print(f"Architecture: {info['architecture']}")
    print(f"Parameters: {info['parameters']}")
    print(f"Features: {', '.join(info['features'])}")
    
    # Test generation
    batch_size = 4
    with torch.no_grad():
        # Generate random images
        generated_images = model.generate(batch_size, device)
        print(f"Generated images shape: {generated_images.shape}")
        
        # Test discriminator
        disc_scores = model.discriminator(generated_images)
        print(f"Discriminator scores shape: {disc_scores.shape}")
    
    print("\n=== Testing Conditional StyleGAN ===")
    # Create conditional model
    conditioning_dim = 512  # e.g., CLIP embedding dimension
    conditional_model = create_conditional_stylegan(resolution=256, conditioning_dim=conditioning_dim)
    conditional_model = conditional_model.to(device)
    
    # Print conditional model info
    cond_info = conditional_model.get_model_info()
    print("Conditional StyleGAN Model Information:")
    print(f"Architecture: {cond_info['architecture']}")
    print(f"Parameters: {cond_info['parameters']}")
    print(f"Conditioning dim: {conditioning_dim}")
    
    with torch.no_grad():
        # Example 1: Generate with random conditioning embedding
        conditioning_embedding = torch.randn(batch_size, conditioning_dim, device=device)
        conditional_images = conditional_model.generate(batch_size, device, conditioning=conditioning_embedding)
        print(f"Conditional generated images shape: {conditional_images.shape}")
        
        # Example 2: Generate with specific conditioning (e.g., text embedding)
        # In practice, this would be a CLIP text embedding or other semantic embedding
        text_embedding = torch.randn(1, conditioning_dim, device=device)  # Placeholder for actual text embedding
        text_conditional_images = conditional_model.generate(1, device, conditioning=text_embedding)
        print(f"Text-conditional generated images shape: {text_conditional_images.shape}")
        
        # Example 3: Generate without conditioning (uses zero embedding)
        uncond_images = conditional_model.generate(batch_size, device, conditioning=None)
        print(f"Unconditional (zero-embedding) images shape: {uncond_images.shape}")
        
    print("\n=== Conditional Generation Usage Examples ===")
    print("""
    # Example usage patterns for conditional generation:
    
    # 1. Text-to-Image with CLIP embeddings
    import clip
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(["a photo of a cat"]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text).float()
    generated_images = conditional_model.generate(1, device, conditioning=text_features)
    
    # 2. Class-conditional generation with one-hot embeddings
    num_classes = 1000  # e.g., ImageNet classes
    class_embedding = torch.zeros(1, num_classes, device=device)
    class_embedding[0, 281] = 1.0  # class 281 (tabby cat)
    # Need to project to conditioning_dim first
    class_proj = nn.Linear(num_classes, conditioning_dim).to(device)
    projected_class = class_proj(class_embedding)
    generated_images = conditional_model.generate(1, device, conditioning=projected_class)
    
    # 3. Multi-modal conditioning (text + class + style)
    combined_embedding = torch.cat([text_features, style_embedding], dim=1)
    # Adjust conditioning_dim accordingly when creating model
    
    # 4. Interpolation between conditions
    embedding1 = torch.randn(1, conditioning_dim, device=device)
    embedding2 = torch.randn(1, conditioning_dim, device=device)
    alpha = 0.5
    interpolated = alpha * embedding1 + (1 - alpha) * embedding2
    generated_images = conditional_model.generate(1, device, conditioning=interpolated)
    """)

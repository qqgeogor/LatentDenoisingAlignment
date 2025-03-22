import torch
import numpy as np
from typing import Callable, Optional, Tuple
import math


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

class DDPMSampler:
    """
    Implementation of Denoising Diffusion Probabilistic Models (DDPM) sampler
    """
    def __init__(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_steps: int = 1000,
        device: str = "cuda"
    ):
        """
        Args:
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            num_steps: Number of diffusion steps
            device: Device to store schedule tensors
        """
        self.num_steps = num_steps
        self.device = device
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        
        # Calculate alphas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        
        # Calculate diffusion parameters
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def add_noise(
        self,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to image according to DDPM schedule
        """
        if noise is None:
            noise = torch.randn_like(x)
            
        if timesteps is None:
            timesteps = torch.randint(0, self.num_steps, (x.shape[0],), device=x.device)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        noisy = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy, timesteps

    @torch.no_grad()
    def sample(
        self,
        model,images,mask_ratio=0.75,latent=None,mask=None,ids_restore=None
    ) -> torch.Tensor:
        """
        Generate samples using DDPM sampling
        """
        
        noise = torch.randn_like(images)
        
        if latent is None:
            latent,mask,ids_restore = model.forward_encoder(images,mask_ratio)
        
        x = noise
        
        for t in reversed(range(self.num_steps)):

            # Predict x0 (denoised image)
            predicted_x0 = model.denoise(x,latent,mask,ids_restore)
            
            # Get alpha values for current timestep
            alpha = self.alphas[t].view(-1,1,1,1)
            alpha_cumprod = self.alphas_cumprod[t].view(-1,1,1,1)
            beta = self.betas[t].view(-1,1,1,1)
            
            # No noise for last step
            noise = torch.randn_like(x) if t > 0 else 0
            
            
            # # Get alpha values for current timestep
            # alpha = self.alphas[t].view(-1,1,1,1)
            # alpha_cumprod = self.alphas_cumprod[t].view(-1,1,1,1)
            # alpha_cumprod_prev = self.alphas_cumprod_prev[t].view(-1,1,1,1)
            # beta = self.betas[t].view(-1,1,1,1)
            
            
            # # Direct x0 sampling using the reverse process posterior mean
            # x = (
            #     (torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod)) * predicted_x0 +
            #     (torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)) * x
            # ) + torch.sqrt(beta) * noise

            # Convert x0 prediction to noise prediction
            predicted_noise = (x - torch.sqrt(alpha_cumprod) * predicted_x0) / torch.sqrt(1 - alpha_cumprod)
            
            # Compute next sample
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
            # x = (torch.sqrt(1/alpha) * (x - (beta/torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
            #     + torch.sqrt(beta) * noise
            # )
         
        out = (1-mask.unsqueeze(-1)) * model.patchify(images) + mask.unsqueeze(-1) * model.patchify(x)
        out = model.unpatchify(out)

        return out,mask


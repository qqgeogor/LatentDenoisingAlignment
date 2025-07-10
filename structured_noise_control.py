import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Union
import matplotlib.pyplot as plt


class StructuredNoiseController:
    """
    Control noise injection using decomposition methods (PCA, ICA, NMF) to respect
    the semantic structure of conditioning embeddings.
    """
    
    def __init__(self, decomposition_method: str = 'pca', n_components: Optional[int] = None):
        self.decomposition_method = decomposition_method
        self.n_components = n_components
        self.decomposer = None
        self.scaler = StandardScaler()
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        self.semantic_importance_weights = None
        
    def fit(self, conditioning_embeddings: torch.Tensor):
        """
        Fit the decomposition model on a dataset of conditioning embeddings.
        
        Args:
            conditioning_embeddings: [N, embedding_dim] - Collection of embeddings
        """
        # Convert to numpy for sklearn
        embeddings_np = conditioning_embeddings.detach().cpu().numpy()
        
        # Standardize the embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings_np)
        
        # Choose decomposition method
        if self.decomposition_method == 'pca':
            self.decomposer = PCA(n_components=self.n_components)
            transformed = self.decomposer.fit_transform(embeddings_scaled)
            
            # Store eigenvalues for noise scaling
            self.eigenvalues_ = self.decomposer.explained_variance_
            self.explained_variance_ratio_ = self.decomposer.explained_variance_ratio_
            
        elif self.decomposition_method == 'ica':
            self.decomposer = FastICA(n_components=self.n_components, random_state=42)
            transformed = self.decomposer.fit_transform(embeddings_scaled)
            
            # For ICA, estimate importance from component variance
            self.eigenvalues_ = np.var(transformed, axis=0)
            self.explained_variance_ratio_ = self.eigenvalues_ / np.sum(self.eigenvalues_)
            
        elif self.decomposition_method == 'nmf':
            # NMF requires non-negative data
            embeddings_positive = embeddings_scaled - embeddings_scaled.min() + 1e-6
            self.decomposer = NMF(n_components=self.n_components, random_state=42)
            transformed = self.decomposer.fit_transform(embeddings_positive)
            
            # Estimate importance from component magnitude
            self.eigenvalues_ = np.mean(transformed, axis=0)
            self.explained_variance_ratio_ = self.eigenvalues_ / np.sum(self.eigenvalues_)
        
        # Compute semantic importance weights (higher for more important components)
        self._compute_semantic_importance_weights()
        
        print(f"✅ Fitted {self.decomposition_method.upper()} on {len(embeddings_np)} embeddings")
        print(f"   Components: {len(self.explained_variance_ratio_)}")
        print(f"   Cumulative variance explained: {np.cumsum(self.explained_variance_ratio_)[:10]}")
        
    def _compute_semantic_importance_weights(self):
        """
        Compute importance weights for each component.
        Higher weights = more semantically important = less noise allowed
        """
        # Method 1: Based on explained variance (more variance = more important)
        variance_weights = self.explained_variance_ratio_
        
        # Method 2: Exponential decay (first components are most important)
        decay_weights = np.exp(-np.arange(len(variance_weights)) * 0.1)
        
        # Method 3: Rank-based importance
        ranks = np.argsort(-variance_weights) + 1
        rank_weights = 1.0 / ranks
        
        # Combine methods (you can adjust these weights)
        self.semantic_importance_weights = (
            0.5 * variance_weights +
            0.3 * decay_weights +
            0.2 * rank_weights
        )
        
        # Normalize to [0, 1]
        self.semantic_importance_weights = (
            self.semantic_importance_weights / self.semantic_importance_weights.max()
        )
    
    def add_structured_noise(self, 
                           conditioning_embedding: torch.Tensor,
                           noise_strength: float = 0.1,
                           noise_strategy: str = 'adaptive',
                           preserve_top_k: int = 10) -> torch.Tensor:
        """
        Add structured noise to conditioning embedding in decomposed space.
        
        Args:
            conditioning_embedding: [batch_size, embedding_dim]
            noise_strength: Overall noise magnitude
            noise_strategy: 'adaptive', 'uniform', 'importance_weighted', 'selective'
            preserve_top_k: Number of top components to preserve (for 'selective')
        
        Returns:
            noisy_embedding: [batch_size, embedding_dim]
        """
        device = conditioning_embedding.device
        batch_size = conditioning_embedding.shape[0]
        
        # Convert to numpy and standardize
        emb_np = conditioning_embedding.detach().cpu().numpy()
        emb_scaled = self.scaler.transform(emb_np)
        
        # Transform to decomposed space
        if self.decomposition_method == 'nmf':
            emb_positive = emb_scaled - emb_scaled.min() + 1e-6
            emb_transformed = self.decomposer.transform(emb_positive)
        else:
            emb_transformed = self.decomposer.transform(emb_scaled)
        
        # Generate noise in decomposed space
        noise_transformed = self._generate_decomposed_noise(
            emb_transformed, noise_strength, noise_strategy, preserve_top_k
        )
        
        # Add noise in decomposed space
        noisy_transformed = emb_transformed + noise_transformed
        
        # Transform back to original space
        if self.decomposition_method == 'nmf':
            noisy_scaled = self.decomposer.inverse_transform(noisy_transformed)
            # Restore original scale
            noisy_emb = noisy_scaled + emb_scaled.min() - 1e-6
        else:
            noisy_scaled = self.decomposer.inverse_transform(noisy_transformed)
            noisy_emb = noisy_scaled
        
        # Inverse standardization
        noisy_original = self.scaler.inverse_transform(noisy_emb)
        
        return torch.tensor(noisy_original, dtype=conditioning_embedding.dtype, device=device)
    
    def _generate_decomposed_noise(self, emb_transformed, noise_strength, noise_strategy, preserve_top_k):
        """Generate noise in the decomposed space according to strategy"""
        
        batch_size, n_components = emb_transformed.shape
        
        if noise_strategy == 'uniform':
            # Uniform noise across all components
            noise = np.random.normal(0, noise_strength, (batch_size, n_components))
            
        elif noise_strategy == 'adaptive':
            # Noise proportional to eigenvalue magnitude (more noise where there's natural variation)
            noise_scales = np.sqrt(self.eigenvalues_) * noise_strength
            noise = np.random.normal(0, 1, (batch_size, n_components))
            noise = noise * noise_scales[np.newaxis, :]
            
        elif noise_strategy == 'importance_weighted':
            # Less noise for more important components
            # High importance = low noise
            noise_scales = noise_strength * (1.0 - self.semantic_importance_weights)
            noise = np.random.normal(0, 1, (batch_size, n_components))
            noise = noise * noise_scales[np.newaxis, :]
            
        elif noise_strategy == 'selective':
            # No noise for top-k most important components
            noise = np.random.normal(0, noise_strength, (batch_size, n_components))
            # Zero out top-k components
            top_k_indices = np.argsort(-self.semantic_importance_weights)[:preserve_top_k]
            noise[:, top_k_indices] = 0
            
        elif noise_strategy == 'hierarchical':
            # Different noise levels for different importance tiers
            noise = np.random.normal(0, 1, (batch_size, n_components))
            
            # Define tiers based on cumulative variance
            cumvar = np.cumsum(self.explained_variance_ratio_)
            tier1 = cumvar <= 0.5  # Top 50% variance - minimal noise
            tier2 = (cumvar > 0.5) & (cumvar <= 0.8)  # Next 30% - medium noise
            tier3 = cumvar > 0.8   # Remaining 20% - high noise
            
            noise[:, tier1] *= noise_strength * 0.1  # 10% of base noise
            noise[:, tier2] *= noise_strength * 0.5  # 50% of base noise  
            noise[:, tier3] *= noise_strength * 1.0  # 100% of base noise
            
        else:
            raise ValueError(f"Unknown noise strategy: {noise_strategy}")
            
        return noise
    
    def interpolate_in_decomposed_space(self, 
                                      emb1: torch.Tensor, 
                                      emb2: torch.Tensor, 
                                      alpha: float = 0.5,
                                      component_weights: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Interpolate between embeddings in decomposed space with optional component weighting.
        """
        device = emb1.device
        
        # Transform to decomposed space
        emb1_np = self.scaler.transform(emb1.detach().cpu().numpy())
        emb2_np = self.scaler.transform(emb2.detach().cpu().numpy())
        
        if self.decomposition_method == 'nmf':
            emb1_np = emb1_np - emb1_np.min() + 1e-6
            emb2_np = emb2_np - emb2_np.min() + 1e-6
        
        emb1_transformed = self.decomposer.transform(emb1_np)
        emb2_transformed = self.decomposer.transform(emb2_np)
        
        # Interpolate with optional component weighting
        if component_weights is not None:
            # Weighted interpolation per component
            interpolated = (1 - alpha) * emb1_transformed + alpha * emb2_transformed
            interpolated = interpolated * component_weights[np.newaxis, :]
        else:
            # Standard linear interpolation
            interpolated = (1 - alpha) * emb1_transformed + alpha * emb2_transformed
        
        # Transform back
        if self.decomposition_method == 'nmf':
            interpolated_scaled = self.decomposer.inverse_transform(interpolated)
            interpolated_scaled = interpolated_scaled + emb1_np.min() - 1e-6
        else:
            interpolated_scaled = self.decomposer.inverse_transform(interpolated)
        
        interpolated_original = self.scaler.inverse_transform(interpolated_scaled)
        
        return torch.tensor(interpolated_original, dtype=emb1.dtype, device=device)
    
    def analyze_components(self, top_k: int = 20) -> Dict:
        """Analyze the learned components and their importance"""
        
        analysis = {
            'explained_variance_ratio': self.explained_variance_ratio_[:top_k],
            'eigenvalues': self.eigenvalues_[:top_k],
            'semantic_importance': self.semantic_importance_weights[:top_k],
            'cumulative_variance': np.cumsum(self.explained_variance_ratio_)[:top_k]
        }
        
        print(f"\n📊 **{self.decomposition_method.upper()} Component Analysis (Top {top_k})**")
        print("-" * 60)
        for i in range(min(top_k, 10)):
            print(f"Component {i:2d}: "
                  f"Variance={self.explained_variance_ratio_[i]:.3f}, "
                  f"Importance={self.semantic_importance_weights[i]:.3f}, "
                  f"CumVar={np.cumsum(self.explained_variance_ratio_)[i]:.3f}")
        
        return analysis


class StructuredStyleGAN(nn.Module):
    """StyleGAN with structured noise control for conditioning embeddings"""
    
    def __init__(self, stylegan_model, noise_controller: StructuredNoiseController):
        super().__init__()
        self.stylegan = stylegan_model
        self.noise_controller = noise_controller
        
    def forward(self, z: torch.Tensor, conditioning: torch.Tensor,
                noise_strength: float = 0.1, noise_strategy: str = 'adaptive') -> torch.Tensor:
        """
        Generate with structured noise applied to conditioning
        """
        # Apply structured noise to conditioning
        noisy_conditioning = self.noise_controller.add_structured_noise(
            conditioning, noise_strength, noise_strategy
        )
        
        # Generate with noisy conditioning
        return self.stylegan(z, noisy_conditioning)
    
    def generate_variants(self, conditioning: torch.Tensor, 
                         num_variants: int = 5,
                         noise_strength: float = 0.1,
                         noise_strategy: str = 'adaptive') -> torch.Tensor:
        """
        Generate multiple variants of the same conditioning with structured noise
        """
        variants = []
        
        for _ in range(num_variants):
            # Different random z for each variant
            z = torch.randn(1, self.stylegan.z_dim, device=conditioning.device)
            
            # Apply structured noise to conditioning
            noisy_conditioning = self.noise_controller.add_structured_noise(
                conditioning, noise_strength, noise_strategy
            )
            
            # Generate image
            image = self.stylegan(z, noisy_conditioning)
            variants.append(image)
        
        return torch.cat(variants, dim=0)


def demonstrate_structured_noise():
    """Demonstrate structured noise control with different decomposition methods"""
    
    print("🧬 **Structured Noise Control Demonstration**\n")
    
    # Create synthetic conditioning embeddings (e.g., CLIP embeddings)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Simulate different types of conditioning embeddings
    n_samples = 1000
    embedding_dim = 512
    
    # Create embeddings with some structure
    # Components 0-50: high variance (visual features)
    # Components 51-200: medium variance (semantic features)  
    # Components 201-511: low variance (noise/irrelevant)
    
    embeddings = []
    for i in range(n_samples):
        emb = torch.zeros(embedding_dim)
        emb[:50] = torch.randn(50) * 2.0      # High variance visual features
        emb[50:200] = torch.randn(150) * 1.0  # Medium variance semantic
        emb[200:] = torch.randn(312) * 0.1    # Low variance noise
        embeddings.append(emb)
    
    embeddings = torch.stack(embeddings)
    print(f"Created {n_samples} synthetic embeddings of dimension {embedding_dim}")
    
    # Test different decomposition methods
    methods = ['pca', 'ica']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"🔬 **Testing {method.upper()} Decomposition**")
        print('='*50)
        
        # Create and fit noise controller
        controller = StructuredNoiseController(
            decomposition_method=method, 
            n_components=100  # Use top 100 components
        )
        controller.fit(embeddings)
        
        # Analyze components
        analysis = controller.analyze_components(top_k=15)
        
        # Test different noise strategies
        test_embedding = embeddings[:1]  # Single test embedding
        noise_strategies = ['uniform', 'adaptive', 'importance_weighted', 'selective', 'hierarchical']
        
        print(f"\n🎛️ **Noise Strategy Comparison**")
        print("-" * 40)
        
        for strategy in noise_strategies:
            # Apply structured noise
            noisy_emb = controller.add_structured_noise(
                test_embedding, 
                noise_strength=0.2, 
                noise_strategy=strategy
            )
            
            # Measure change
            change_magnitude = torch.norm(noisy_emb - test_embedding).item()
            
            # Measure semantic preservation (cosine similarity)
            semantic_preservation = F.cosine_similarity(
                test_embedding, noisy_emb, dim=1
            ).item()
            
            print(f"{strategy:20s}: Change={change_magnitude:.3f}, "
                  f"Preservation={semantic_preservation:.3f}")


def create_training_pipeline():
    """Create a training pipeline that uses structured noise for better conditioning control"""
    
    class StructuredConditionalLoss(nn.Module):
        def __init__(self, noise_controller: StructuredNoiseController):
            super().__init__()
            self.noise_controller = noise_controller
            
        def forward(self, generated_images, original_conditioning, 
                   noisy_conditioning, feature_extractor):
            """
            Compute loss that encourages semantic preservation despite noise
            """
            # Extract features from generated images
            gen_features = feature_extractor(generated_images)
            
            # Similarity to original conditioning (should be high)
            original_sim = F.cosine_similarity(gen_features, original_conditioning, dim=1)
            
            # Similarity to noisy conditioning (should be moderate)
            noisy_sim = F.cosine_similarity(gen_features, noisy_conditioning, dim=1)
            
            # Loss: maintain similarity to original while allowing controlled variation
            semantic_preservation_loss = -original_sim.mean()
            controlled_variation_loss = F.mse_loss(noisy_sim, 
                                                  torch.full_like(noisy_sim, 0.8))  # Target 80% similarity
            
            return semantic_preservation_loss + 0.5 * controlled_variation_loss
    
    return StructuredConditionalLoss


# Best practices guide
STRUCTURED_NOISE_GUIDE = """
🎯 **STRUCTURED NOISE CONTROL - BEST PRACTICES**

1. **Decomposition Method Selection:**
   ✅ PCA: Best for capturing variance-based importance
   ✅ ICA: Good for finding independent semantic directions  
   ✅ NMF: Useful for non-negative sparse representations
   ✅ Choose based on your embedding type and use case

2. **Component Analysis:**
   ✅ Use top 80-90% variance components (discard noise)
   ✅ Analyze component importance patterns
   ✅ Validate semantic meaning of top components
   ✅ Monitor cumulative variance explained

3. **Noise Strategy Selection:**
   ✅ 'adaptive': For natural variation respecting embedding structure
   ✅ 'importance_weighted': For conservative semantic preservation
   ✅ 'selective': For preserving specific semantic aspects
   ✅ 'hierarchical': For multi-level diversity control

4. **Hyperparameter Tuning:**
   ✅ noise_strength: 0.05-0.3 (start small)
   ✅ preserve_top_k: 5-20% of components
   ✅ Monitor semantic preservation scores
   ✅ Adjust based on generated sample quality

5. **Training Integration:**
   ✅ Pre-fit decomposition on large embedding dataset
   ✅ Use structured noise during training for regularization
   ✅ Include semantic preservation losses
   ✅ Monitor conditioning consistency metrics

6. **Quality Assurance:**
   ✅ Measure semantic preservation (cosine similarity)
   ✅ Evaluate diversity within semantic constraints
   ✅ Test interpolation quality in decomposed space
   ✅ Validate component interpretability
"""


if __name__ == "__main__":
    print("🧬 **Structured Noise Control for Conditional StyleGAN**\n")
    print(STRUCTURED_NOISE_GUIDE)
    
    # Run demonstration
    demonstrate_structured_noise()
    
    print("\n" + "="*60)
    print("💡 **KEY BENEFITS OF STRUCTURED NOISE**")
    print("="*60)
    print("""
    1. 🎯 **Semantic Preservation**: Noise respects embedding structure
    2. 🎨 **Controlled Diversity**: Variation in semantically meaningful directions
    3. 🧠 **Adaptive Control**: Noise scales with natural variation patterns
    4. 🔒 **Conditioning Respect**: Important semantic aspects preserved
    5. 🎛️ **Fine-grained Control**: Different strategies for different needs
    6. 🔬 **Interpretability**: Can analyze which components affect what
    
    This approach significantly improves upon naive noise addition by ensuring
    that diversity doesn't come at the cost of semantic coherence!
    """) 
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCANoiseController:
    """
    Use PCA decomposition to add structured noise to conditioning embeddings.
    This prevents noise from dominating important semantic directions.
    """
    
    def __init__(self):
        self.pca = None
        self.scaler = StandardScaler()
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, conditioning_embeddings):
        """Fit PCA on conditioning embeddings to learn semantic structure"""
        embeddings_np = conditioning_embeddings.detach().cpu().numpy()
        embeddings_scaled = self.scaler.fit_transform(embeddings_np)
        
        self.pca = PCA(n_components=min(100, embeddings_scaled.shape[1]))
        self.pca.fit(embeddings_scaled)
        
        self.eigenvalues_ = self.pca.explained_variance_
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        
        print(f"PCA fitted: {len(self.explained_variance_ratio_)} components")
        print(f"Variance explained: {np.sum(self.explained_variance_ratio_):.3f}")
        
    def add_structured_noise(self, conditioning_embedding, noise_strength=0.1):
        """Add noise scaled by eigenvalues in PCA space"""
        device = conditioning_embedding.device
        
        # Transform to PCA space
        emb_np = conditioning_embedding.detach().cpu().numpy()
        emb_scaled = self.scaler.transform(emb_np)
        emb_pca = self.pca.transform(emb_scaled)
        
        # Generate noise scaled by sqrt(eigenvalues)
        # More noise where there's natural variation, less where it's constrained
        noise_scales = np.sqrt(self.eigenvalues_) * noise_strength
        noise_pca = np.random.normal(0, 1, emb_pca.shape) * noise_scales[np.newaxis, :]
        
        # Add noise in PCA space
        noisy_pca = emb_pca + noise_pca
        
        # Transform back to original space
        noisy_scaled = self.pca.inverse_transform(noisy_pca)
        noisy_original = self.scaler.inverse_transform(noisy_scaled)
        
        return torch.tensor(noisy_original, dtype=conditioning_embedding.dtype, device=device)


def demonstrate_benefits():
    """Show why PCA-based noise is better than uniform noise"""
    
    print("🧬 PCA-Based Structured Noise vs Uniform Noise\n")
    
    # Create synthetic embeddings with structure
    n_samples = 500
    embedding_dim = 256
    
    embeddings = []
    for i in range(n_samples):
        emb = torch.zeros(embedding_dim)
        emb[:50] = torch.randn(50) * 2.0    # High variance semantic features
        emb[50:100] = torch.randn(50) * 1.0  # Medium variance features
        emb[100:] = torch.randn(156) * 0.1   # Low variance noise
        embeddings.append(emb)
    
    embeddings_tensor = torch.stack(embeddings)
    test_embedding = embeddings_tensor[:1]
    
    # Fit PCA controller
    controller = PCANoiseController()
    controller.fit(embeddings_tensor)
    
    # Compare noise methods
    noise_strength = 0.2
    
    # Method 1: Uniform noise (naive approach)
    uniform_noise = torch.randn_like(test_embedding) * noise_strength
    uniform_noisy = test_embedding + uniform_noise
    
    # Method 2: PCA-structured noise (our approach)
    pca_noisy = controller.add_structured_noise(test_embedding, noise_strength)
    
    # Measure semantic preservation
    uniform_similarity = F.cosine_similarity(test_embedding, uniform_noisy, dim=1).item()
    pca_similarity = F.cosine_similarity(test_embedding, pca_noisy, dim=1).item()
    
    print(f"Semantic Preservation (cosine similarity):")
    print(f"  Uniform noise:     {uniform_similarity:.4f}")
    print(f"  PCA-structured:    {pca_similarity:.4f}")
    print(f"  Improvement:       {((pca_similarity - uniform_similarity) / uniform_similarity * 100):+.1f}%")
    
    # Measure change magnitude
    uniform_change = torch.norm(uniform_noisy - test_embedding).item()
    pca_change = torch.norm(pca_noisy - test_embedding).item()
    
    print(f"\nChange Magnitude:")
    print(f"  Uniform noise:     {uniform_change:.4f}")
    print(f"  PCA-structured:    {pca_change:.4f}")
    
    # Show component importance
    print(f"\nTop 10 PCA Components (eigenvalues):")
    for i in range(10):
        print(f"  PC{i+1:2d}: {controller.eigenvalues_[i]:.3f} "
              f"({controller.explained_variance_ratio_[i]:.3f} variance)")
    
    print(f"\n✨ Key Benefits of PCA-Structured Noise:")
    print("1. 🎯 Preserves semantic meaning better")
    print("2. 🔒 Less noise in constrained (important) directions") 
    print("3. 🎨 More variation in natural directions")
    print("4. 📊 Adaptive to embedding structure")
    print("5. 🧠 Prevents noise from overwhelming conditioning")


if __name__ == "__main__":
    demonstrate_benefits() 
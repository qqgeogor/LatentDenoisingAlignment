import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

# NoProp / Latent Denoising Alignment Implementation
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 128)  # MNIST: 28x28 → 1600-dim
        )

    def forward(self, x):
        return self.features(x)

class DenoisingMLP(nn.Module):
    def __init__(self, feature_dim, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + embed_dim, 256),  # Input: image features + noisy label
            nn.ReLU(),
            nn.Linear(256, embed_dim)                # Output: denoised label
        )

    def forward(self, x_features, z_t):
        combined = torch.cat([x_features, z_t], dim=1)
        return self.mlp(combined)

class LatentDenoisingAlignment:
    def __init__(self, T=10, embed_dim=10, feature_dim=128, lr=0.001, device='cuda'):
        self.T = T
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.device = device
        
        # Noise schedule (linear)
        self.alpha = torch.linspace(1.0, 0.1, T).to(device)
        
        # Models initialization
        self.cnn = CNN().to(device)
        self.mlps = nn.ModuleList([DenoisingMLP(feature_dim, embed_dim).to(device) for _ in range(T)])
        
        # Optimizers
        self.cnn_optimizer = optim.Adam(self.cnn.parameters(), lr=lr)
        self.mlp_optimizers = [optim.Adam(mlp.parameters(), lr=lr) for mlp in self.mlps]

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                current_batch_size = x.shape[0]
                
                # One-hot encode the labels
                u_y = torch.zeros(current_batch_size, self.embed_dim, device=self.device)
                u_y.scatter_(1, y.unsqueeze(1), 1)

                # Forward diffusion (add noise progressively)
                z = [u_y]
                for t in range(1, self.T + 1):
                    eps = torch.randn_like(u_y)
                    z_t = torch.sqrt(self.alpha[t-1]) * z[-1] + torch.sqrt(1 - self.alpha[t-1]) * eps
                    z.append(z_t)

                # Extract image features once
                x_features = self.cnn(x)
                
                # Train MLPs independently
                losses = []
                for t in range(self.T):
                    # Each MLP tries to denoise from its specific noise level
                    u_hat = self.mlps[t](x_features, z[t+1].detach())
                    loss = torch.mean((u_hat - u_y) ** 2)
                    losses.append(loss)

                # Optimize all models
                total_loss = sum(losses)
                
                self.cnn_optimizer.zero_grad()
                for opt in self.mlp_optimizers:
                    opt.zero_grad()
                
                total_loss.backward()
                
                self.cnn_optimizer.step()
                for opt in self.mlp_optimizers:
                    opt.step()

                epoch_loss += total_loss.item()
                batch_count += 1

            # Epoch summary
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    def predict(self, x):
        x = x.to(self.device)
        # Start from random noise
        z_t = torch.randn(1, self.embed_dim, device=self.device)
        
        # Iteratively denoise
        for t in reversed(range(self.T)):
            x_features = self.cnn(x.unsqueeze(0))
            u_hat = self.mlps[t](x_features, z_t)
            # Optional: add some noise proportional to the step
            if t > 0:  # Skip noise at the last step for better results
                z_t = torch.sqrt(self.alpha[t]) * u_hat + torch.sqrt(1 - self.alpha[t]) * torch.randn_like(u_hat)
            else:
                z_t = u_hat
                
        return torch.argmax(z_t)

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                for i in range(len(x)):
                    pred = self.predict(x[i])
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
            pred = self.predict(x)
            
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

def main():
    parser = argparse.ArgumentParser(description='Latent Denoising Alignment for MNIST')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--T', type=int, default=10, help='diffusion steps')
    parser.add_argument('--embed-dim', type=int, default=10, help='embedding dimension')
    parser.add_argument('--feature-dim', type=int, default=128, help='feature dimension')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.ToTensor()
    
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = LatentDenoisingAlignment(
        T=args.T,
        embed_dim=args.embed_dim,
        feature_dim=args.feature_dim,
        lr=args.lr,
        device=device
    )
    
    # Train model
    model.train(train_loader, args.epochs)
    
    # Evaluate model
    accuracy = model.evaluate(test_loader)
    
    # Visualize some predictions
    model.visualize_predictions(test_loader)
    
if __name__ == "__main__":
    main() 
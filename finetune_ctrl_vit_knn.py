import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from vit_ibot import MaskedAutoencoderViT
from torch.cuda.amp import autocast
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image

def extract_features(model, dataloader, device, use_amp=True):
    """Extract features from a dataset using the given model"""
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # Get embeddings with optional AMP
            with autocast(enabled=use_amp):
                # Use encoder to get features
                features = model.forward_feature(images)[:,0]
                embeddings = F.normalize(features, p=2, dim=1)
            
            # Store features and labels
            all_features.append(embeddings.cpu())
            all_labels.append(labels)
    
    # Concatenate all features and labels
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels

def knn_classifier(train_features, train_labels, test_features, test_labels, k_values):
    """Evaluate using KNN with different k values"""
    results = {}
    
    # Normalize features for cosine similarity
    train_features = F.normalize(train_features, p=2, dim=1)
    test_features = F.normalize(test_features, p=2, dim=1)
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity = torch.mm(test_features, train_features.t())
    
    for k in k_values:
        print(f"Evaluating with k={k}...")
        # Get top k similar samples
        _, indices = similarity.topk(k, dim=1)
        
        # Get labels of top k neighbors
        topk_labels = train_labels[indices]
        
        # Predict label based on majority vote
        predictions = torch.mode(topk_labels, dim=1).values
        
        # Compute accuracy
        correct = (predictions == test_labels).sum().item()
        accuracy = 100 * correct / test_labels.size(0)
        
        results[k] = accuracy
        print(f"k={k}, Accuracy: {accuracy:.2f}%")
    
    return results

def visualize_features(features, labels, output_dir, method='both', n_components=2, perplexity=30, sample_size=1000):
    """Visualize features using PCA or t-SNE"""
    # Convert to numpy for sklearn
    features = features.numpy()
    labels = labels.numpy()
    
    # Use a subset of data for visualization if needed
    if sample_size < len(features):
        indices = np.random.choice(len(features), sample_size, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create a colormap
    colors = plt.cm.tab10(np.arange(10))
    
    # PCA visualization
    if method in ['pca', 'both']:
        print("Performing PCA...")
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        for i in range(10):
            plt.scatter(
                reduced_features[labels == i, 0],
                reduced_features[labels == i, 1],
                c=[colors[i]],
                label=class_names[i],
                alpha=0.6,
                s=30
            )
        
        plt.legend()
        plt.title(f'PCA visualization (explained variance: {sum(pca.explained_variance_ratio_):.2f})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
        plt.close()
    
    # t-SNE visualization
    if method in ['tsne', 'both']:
        print("Performing t-SNE...")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=1000, verbose=1)
        reduced_features = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        for i in range(10):
            plt.scatter(
                reduced_features[labels == i, 0],
                reduced_features[labels == i, 1],
                c=[colors[i]],
                label=class_names[i],
                alpha=0.6,
                s=30
            )
        
        plt.legend()
        plt.title('t-SNE visualization of features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
        plt.close()

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.dataset == 'cifar10':
        # Data preprocessing
        transform = transforms.Compose([
            transforms.Resize(int(args.img_size*1.14),interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Load CIFAR-10
        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                            download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        
        testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                            download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        
    else:
        transform = transforms.Compose([
            # transforms.Resize(args.img_size),
            transforms.Resize(int(args.img_size*1.14)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Load ImageNet-100
        trainset = torchvision.datasets.ImageFolder(
                root=args.data_path,
                transform=transform,
            )
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
        
        testset = torchvision.datasets.ImageFolder(root=args.data_path,
                                            transform=transform)
        testloader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    


    # Load pretrained model
    model = MaskedAutoencoderViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=args.norm_pix_loss,
        use_checkpoint=False
    ).to(device)
    
    if args.pretrained_path:    
        print(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['discriminator_state_dict'], strict=True)
    else:
        print("No pretrained model provided. Using randomly initialized model.")

    # Extract features
    print("Extracting training features...")
    train_features, train_labels = extract_features(model, trainloader, device, use_amp=args.use_amp)
    
    print("Extracting test features...")
    test_features, test_labels = extract_features(model, testloader, device, use_amp=args.use_amp)
    
    # Visualize features
    if args.visualize_features:
        print("Visualizing features...")
        visualize_features(
            test_features, 
            test_labels, 
            args.output_dir, 
            method=args.vis_method,
            perplexity=args.tsne_perplexity,
            sample_size=args.vis_sample_size
        )
    
    # Evaluate using KNN
    results = knn_classifier(train_features, train_labels, test_features, test_labels, args.k_values)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'knn_results.txt')
    with open(results_path, 'w') as f:
        for k, acc in results.items():
            f.write(f"k={k}, Accuracy: {acc:.2f}%\n")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xlabel('k value')
    plt.ylabel('Accuracy (%)')
    plt.title('KNN Classification Accuracy with Different k Values')
    plt.xticks(list(results.keys()))
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'knn_results.png'))
    plt.close()
    
    print(f"Results saved to {args.output_dir}")
    
    # Return best k and accuracy
    best_k = max(results, key=results.get)
    print(f"Best result: k={best_k}, Accuracy: {results[best_k]:.2f}%")
    
    return results

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('CIFAR-10 KNN Evaluation for Control Leader')
    
    # Model parameters
    parser.add_argument('--pretrained_path', default='', 
                       type=str, help='Path to pretrained model')
    parser.add_argument('--latent_dim', default=128, type=int)
    
    # Data parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # KNN parameters
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 5, 10, 20, 50, 100, 200],
                       help='Values of k to try for KNN')
    
    # Output parameters
    parser.add_argument('--output_dir', default='./output/ctrl_ldr_knn_evaluation')
    
    # AMP parameter
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision')
    
    # Visualization parameters
    parser.add_argument('--visualize_features', action='store_true', default=False,
                       help='Visualize features using PCA and t-SNE')
    parser.add_argument('--vis_method', type=str, default='both', choices=['pca', 'tsne', 'both'],
                       help='Visualization method to use')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                       help='Perplexity parameter for t-SNE')
    parser.add_argument('--vis_sample_size', type=int, default=1000,
                       help='Number of samples to use for visualization')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=32,
                       help='Image size')
    parser.add_argument('--patch_size', type=int, default=4,
                       help='Patch size')
    parser.add_argument('--in_chans', type=int, default=3,
                       help='Number of input channels')
    parser.add_argument('--embed_dim', type=int, default=192,
                       help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12,
                       help='Depth of the model')
    parser.add_argument('--num_heads', type=int, default=3,
                       help='Number of attention heads')
    parser.add_argument('--decoder_embed_dim', type=int, default=96,
                       help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', type=int, default=0,
                       help='Decoder depth')
    parser.add_argument('--decoder_num_heads', type=int, default=3,
                       help='Number of decoder attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.,
                       help='Ratio of MLP hidden dimension to embedding dimension')
    parser.add_argument('--norm_pix_loss', action='store_true', default=False,
                       help='Normalize pixel values before computing the loss')
    parser.add_argument('--num_register_tokens', type=int, default=0,
                       help='Number of register tokens')
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset to use')
    

    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = evaluate_model(args) 
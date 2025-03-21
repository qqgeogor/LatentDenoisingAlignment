import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path
from tqdm import tqdm
from vit import MaskedAutoencoderViT
from torch.cuda.amp import autocast, GradScaler

class CifarClassifier(nn.Module):
    def __init__(self, backbone, num_classes=10,freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(192,num_classes)
        )
        
    def forward(self, x):
        # Get features from the backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.forward_feature(x)[:,0]
        else:
            features = self.backbone.forward_feature(x)[:,0]
        
        # Pass through classifier
        return self.classifier(features)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize GradScaler for AMP
    scaler = GradScaler(enabled=args.use_amp)

    if args.img_size == 32:
        # Data preprocessing for training
        train_transform = transforms.Compose([
                # transforms.Resize(int(args.img_size*1.14),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # Data preprocessing for training
        train_transform = transforms.Compose([
            transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.img_size, padding=int(args.img_size * 0.125)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])    
    # Data preprocessing for testing
    test_transform = transforms.Compose([
        transforms.Resize(int(args.img_size),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                          download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                           shuffle=True, num_workers=args.num_workers)
    
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                         download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers)

    # Load pretrained SimSiam model
    backbone = MaskedAutoencoderViT(img_size=args.img_size, patch_size=args.patch_size, in_chans=3,
                 embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
                 decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
                 mlp_ratio=args.mlp_ratio, norm_layer=nn.LayerNorm, norm_pix_loss=False, use_checkpoint=False).to(device)
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path)
        backbone.load_state_dict(checkpoint['model_state_dict'],strict=False)
    
    # Create classifier
    model = CifarClassifier(backbone,freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(tqdm(trainloader)):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass with autocast、
            args.use_amp = True
            with autocast(enabled=args.use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if i % args.log_freq == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(trainloader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Evaluate (no need for autocast in eval mode)
        test_acc = evaluate(model, testloader, device)
        print(f'Epoch [{epoch}/{args.epochs}], Test Accuracy: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                'accuracy': test_acc,
            }, os.path.join(args.output_dir, 'best_classifier.pth'))
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                'accuracy': test_acc,
            }, os.path.join(args.output_dir, f'classifier_checkpoint_{epoch}.pth'))

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser('CIFAR-10 Classifier Finetuning')
    
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--encoder_type', default='student', type=str)
    parser.add_argument('--freeze_backbone', action='store_true',default=False,
                       help='Freeze backbone during training')
    # System parameters
    parser.add_argument('--data_path', default='c:/dataset', type=str)
    parser.add_argument('--output_dir', default='F:/output/cifar10-classifier')
    parser.add_argument('--pretrained_path', default='', 
                       type=str, help='Path to pretrained SimSiam model')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    


    # MAE parameters
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--embed_dim', default=192, type=int)
    parser.add_argument('--depth', default=12, type=int)
    parser.add_argument('--num_heads', default=3, type=int)
    parser.add_argument('--decoder_embed_dim', default=96, type=int)
    parser.add_argument('--decoder_depth', default=4, type=int)
    parser.add_argument('--decoder_num_heads', default=3, type=int)
    parser.add_argument('--mlp_ratio', default=4., type=float)
    

    # Add AMP argument
    parser.add_argument('--use_amp', action='store_true',
                       help='Use Automatic Mixed Precision training')
    
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    finetune(args) 
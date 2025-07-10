import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Union


class ImageToBEVProjector:
    """
    Projects image features to Bird's Eye View (BEV) feature map using camera parameters.
    
    This class handles the transformation from image coordinates to BEV coordinates
    using intrinsic and extrinsic camera parameters. Supports both single-scale and
    multi-scale FPN features.
    """
    
    def __init__(self, 
                 intrinsic: np.ndarray, 
                 extrinsic: np.ndarray,
                 bev_range: Tuple[float, float, float, float],
                 bev_resolution: float,
                 height_range: Tuple[float, float] = (-3.0, 1.0),
                 height_samples: int = 8):
        """
        Initialize the projector.
        
        Args:
            intrinsic: Camera intrinsic matrix (3x3)
            extrinsic: Camera extrinsic matrix (4x4) - world to camera transformation
            bev_range: BEV range in meters (x_min, x_max, y_min, y_max)
            bev_resolution: BEV resolution in meters per pixel
            height_range: Height range to sample in meters (z_min, z_max)
            height_samples: Number of height samples to use
        """
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.bev_range = bev_range
        self.bev_resolution = bev_resolution
        self.height_range = height_range
        self.height_samples = height_samples
        
        # Calculate BEV grid dimensions
        self.bev_width = int((bev_range[1] - bev_range[0]) / bev_resolution)
        self.bev_height = int((bev_range[3] - bev_range[2]) / bev_resolution)
        
        # Pre-compute BEV grid coordinates
        self._compute_bev_coordinates()
    
    def _compute_bev_coordinates(self):
        """Pre-compute BEV grid coordinates in world space."""
        x_coords = np.linspace(self.bev_range[0], self.bev_range[1], self.bev_width)
        y_coords = np.linspace(self.bev_range[2], self.bev_range[3], self.bev_height)
        z_coords = np.linspace(self.height_range[0], self.height_range[1], self.height_samples)
        
        # Create meshgrid for BEV coordinates
        X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Store BEV coordinates
        self.bev_x = X
        self.bev_y = Y
        self.z_coords = z_coords
    
    def world_to_camera(self, world_points: np.ndarray) -> np.ndarray:
        """
        Transform world coordinates to camera coordinates.
        
        Args:
            world_points: World coordinates (Nx4 homogeneous coordinates)
            
        Returns:
            Camera coordinates (Nx3)
        """
        # Apply extrinsic transformation
        camera_points = (self.extrinsic @ world_points.T).T
        return camera_points[:, :3]  # Remove homogeneous coordinate
    
    def camera_to_image(self, camera_points: np.ndarray) -> np.ndarray:
        """
        Project camera coordinates to image coordinates.
        
        Args:
            camera_points: Camera coordinates (Nx3)
            
        Returns:
            Image coordinates (Nx2) in pixels
        """
        # Project to image plane
        image_points = (self.intrinsic @ camera_points.T).T
        
        # Convert to pixel coordinates
        image_coords = image_points[:, :2] / image_points[:, 2:3]
        return image_coords
    
    def project_to_bev(self, 
                      image_features: torch.Tensor, 
                      image_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Project image features to BEV feature map.
        
        Args:
            image_features: Image features (C, H, W)
            image_shape: Image dimensions (height, width)
            
        Returns:
            BEV feature map (C, bev_height, bev_width)
        """
        device = image_features.device
        C, H, W = image_features.shape
        
        # Initialize BEV feature map
        bev_features = torch.zeros(C, self.bev_height, self.bev_width, device=device)
        count_map = torch.zeros(self.bev_height, self.bev_width, device=device)
        
        # For each height sample
        for z in self.z_coords:
            # Create world coordinates for current height
            world_coords = np.stack([
                self.bev_x.flatten(),
                self.bev_y.flatten(),
                np.full(self.bev_x.size, z),
                np.ones(self.bev_x.size)
            ], axis=1)
            
            # Transform to camera coordinates
            camera_coords = self.world_to_camera(world_coords)
            
            # Filter points in front of camera
            valid_depth = camera_coords[:, 2] > 0
            if not np.any(valid_depth):
                continue
            
            # Project to image coordinates
            image_coords = self.camera_to_image(camera_coords)
            
            # Filter points within image bounds
            valid_x = (image_coords[:, 0] >= 0) & (image_coords[:, 0] < W)
            valid_y = (image_coords[:, 1] >= 0) & (image_coords[:, 1] < H)
            valid_mask = valid_depth & valid_x & valid_y
            
            if not np.any(valid_mask):
                continue
            
            # Get valid image coordinates
            valid_image_coords = image_coords[valid_mask]
            valid_bev_indices = np.where(valid_mask)[0]
            
            # Convert to tensor
            valid_image_coords = torch.from_numpy(valid_image_coords).float().to(device)
            
            # Normalize coordinates for grid_sample
            norm_coords = torch.zeros_like(valid_image_coords)
            norm_coords[:, 0] = 2.0 * valid_image_coords[:, 0] / (W - 1) - 1.0  # x
            norm_coords[:, 1] = 2.0 * valid_image_coords[:, 1] / (H - 1) - 1.0  # y
            
            # Sample features from image
            norm_coords = norm_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
            image_features_expanded = image_features.unsqueeze(0)  # (1, C, H, W)
            
            sampled_features = F.grid_sample(
                image_features_expanded, 
                norm_coords, 
                mode='bilinear', 
                padding_mode='zeros',
                align_corners=True
            )
            
            sampled_features = sampled_features.squeeze(0).squeeze(1)  # (C, N)
            
            # Map back to BEV coordinates
            bev_y_indices = valid_bev_indices // self.bev_width
            bev_x_indices = valid_bev_indices % self.bev_width
            
            # Accumulate features
            for i, (by, bx) in enumerate(zip(bev_y_indices, bev_x_indices)):
                bev_features[:, by, bx] += sampled_features[:, i]
                count_map[by, bx] += 1
        
        # Average features where multiple samples exist
        count_map = torch.clamp(count_map, min=1)
        bev_features = bev_features / count_map.unsqueeze(0)
        
        return bev_features
    
    def project_fpn_to_bev(self, 
                          fpn_features: List[torch.Tensor], 
                          fpn_scales: List[float],
                          original_image_shape: Tuple[int, int],
                          fusion_strategy: str = 'weighted_sum') -> torch.Tensor:
        """
        Project FPN features to BEV feature map.
        
        Args:
            fpn_features: List of FPN feature tensors [(C, H1, W1), (C, H2, W2), ...]
            fpn_scales: List of scales for each FPN level (e.g., [1/4, 1/8, 1/16, 1/32])
            original_image_shape: Original image dimensions (height, width)
            fusion_strategy: Strategy for fusing multi-scale features
                - 'weighted_sum': Weighted sum based on scale
                - 'max_pool': Take maximum across scales
                - 'avg_pool': Average across scales
                - 'concat': Concatenate features from all scales
                - 'adaptive': Adaptive fusion based on distance
            
        Returns:
            BEV feature map (C, bev_height, bev_width) or (C*num_scales, bev_height, bev_width) for concat
        """
        if len(fpn_features) != len(fpn_scales):
            raise ValueError("Number of FPN features must match number of scales")
        
        device = fpn_features[0].device
        C = fpn_features[0].shape[0]
        
        # Project each FPN level to BEV
        bev_features_list = []
        for i, (features, scale) in enumerate(zip(fpn_features, fpn_scales)):
            # Calculate feature image shape
            feature_h, feature_w = features.shape[1], features.shape[2]
            
            # Project to BEV
            bev_feat = self.project_to_bev(features, original_image_shape)
            bev_features_list.append(bev_feat)
        
        # Fuse features according to strategy
        if fusion_strategy == 'weighted_sum':
            return self._weighted_sum_fusion(bev_features_list, fpn_scales)
        elif fusion_strategy == 'max_pool':
            return self._max_pool_fusion(bev_features_list)
        elif fusion_strategy == 'avg_pool':
            return self._avg_pool_fusion(bev_features_list)
        elif fusion_strategy == 'concat':
            return self._concat_fusion(bev_features_list)
        elif fusion_strategy == 'adaptive':
            return self._adaptive_fusion(bev_features_list, fpn_scales)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def _weighted_sum_fusion(self, bev_features_list: List[torch.Tensor], 
                           fpn_scales: List[float]) -> torch.Tensor:
        """Weighted sum fusion based on FPN scales."""
        # Higher resolution features (smaller scales) get higher weights
        weights = torch.tensor([1.0 / scale for scale in fpn_scales])
        weights = weights / weights.sum()  # Normalize
        
        device = bev_features_list[0].device
        weights = weights.to(device)
        
        fused_features = torch.zeros_like(bev_features_list[0])
        for i, bev_feat in enumerate(bev_features_list):
            fused_features += weights[i] * bev_feat
        
        return fused_features
    
    def _max_pool_fusion(self, bev_features_list: List[torch.Tensor]) -> torch.Tensor:
        """Max pooling fusion across scales."""
        stacked_features = torch.stack(bev_features_list, dim=0)  # (num_scales, C, H, W)
        fused_features, _ = torch.max(stacked_features, dim=0)
        return fused_features
    
    def _avg_pool_fusion(self, bev_features_list: List[torch.Tensor]) -> torch.Tensor:
        """Average pooling fusion across scales."""
        stacked_features = torch.stack(bev_features_list, dim=0)  # (num_scales, C, H, W)
        fused_features = torch.mean(stacked_features, dim=0)
        return fused_features
    
    def _concat_fusion(self, bev_features_list: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate features from all scales."""
        return torch.cat(bev_features_list, dim=0)  # (C*num_scales, H, W)
    
    def _adaptive_fusion(self, bev_features_list: List[torch.Tensor], 
                        fpn_scales: List[float]) -> torch.Tensor:
        """Adaptive fusion based on distance from camera."""
        device = bev_features_list[0].device
        
        # Create distance map from camera
        bev_center_x = (self.bev_range[0] + self.bev_range[1]) / 2
        bev_center_y = (self.bev_range[2] + self.bev_range[3]) / 2
        
        x_coords = torch.linspace(self.bev_range[0], self.bev_range[1], self.bev_width, device=device)
        y_coords = torch.linspace(self.bev_range[2], self.bev_range[3], self.bev_height, device=device)
        
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Distance from camera (assuming camera at origin)
        distances = torch.sqrt(X**2 + Y**2)
        
        # Normalize distances
        max_dist = distances.max()
        normalized_distances = distances / max_dist
        
        # Create adaptive weights based on distance
        # Closer regions use higher resolution features, farther regions use lower resolution
        fused_features = torch.zeros_like(bev_features_list[0])
        
        for i, (bev_feat, scale) in enumerate(zip(bev_features_list, fpn_scales)):
            # Higher resolution features for closer regions
            if i == 0:  # Highest resolution
                weight_map = torch.exp(-2 * normalized_distances)
            elif i == len(bev_features_list) - 1:  # Lowest resolution
                weight_map = torch.exp(-2 * (1 - normalized_distances))
            else:  # Middle resolutions
                optimal_distance = i / (len(bev_features_list) - 1)
                weight_map = torch.exp(-4 * (normalized_distances - optimal_distance)**2)
            
            # Apply weights
            fused_features += weight_map.unsqueeze(0) * bev_feat
        
        return fused_features
    
    def project_fpn_to_multi_scale_bev(self, 
                                     fpn_features: List[torch.Tensor], 
                                     fpn_scales: List[float],
                                     original_image_shape: Tuple[int, int],
                                     bev_scales: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Project FPN features to multiple BEV scales.
        
        Args:
            fpn_features: List of FPN feature tensors
            fpn_scales: List of scales for each FPN level
            original_image_shape: Original image dimensions
            bev_scales: List of BEV scales (resolutions) to generate
            
        Returns:
            Dictionary mapping BEV scale names to BEV feature maps
        """
        if bev_scales is None:
            bev_scales = [0.1, 0.2, 0.4]  # Default multi-scale BEV
        
        multi_scale_bev = {}
        
        for bev_scale in bev_scales:
            # Create projector for this BEV scale
            scale_projector = ImageToBEVProjector(
                intrinsic=self.intrinsic,
                extrinsic=self.extrinsic,
                bev_range=self.bev_range,
                bev_resolution=bev_scale,
                height_range=self.height_range,
                height_samples=self.height_samples
            )
            
            # Project FPN features to this BEV scale
            bev_features = scale_projector.project_fpn_to_bev(
                fpn_features, fpn_scales, original_image_shape, 
                fusion_strategy='weighted_sum'
            )
            
            multi_scale_bev[f'bev_{bev_scale}'] = bev_features
        
        return multi_scale_bev
    
    def visualize_fpn_projection(self, 
                               image: np.ndarray,
                               fpn_features: List[torch.Tensor],
                               fpn_scales: List[float],
                               original_image_shape: Tuple[int, int],
                               save_path: Optional[str] = None):
        """
        Visualize FPN projection results.
        
        Args:
            image: Input image (H, W, 3)
            fpn_features: List of FPN feature tensors
            fpn_scales: List of FPN scales
            original_image_shape: Original image dimensions
            save_path: Optional path to save visualization
        """
        # Project FPN features with different fusion strategies
        strategies = ['weighted_sum', 'max_pool', 'avg_pool', 'adaptive']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot different fusion strategies
        for i, strategy in enumerate(strategies):
            bev_features = self.project_fpn_to_bev(
                fpn_features, fpn_scales, original_image_shape, 
                fusion_strategy=strategy
            )
            
            if strategy == 'concat':
                # For concatenated features, show average across all channels
                bev_vis = bev_features.mean(dim=0).cpu().numpy()
            else:
                bev_vis = bev_features.mean(dim=0).cpu().numpy()
            
            im = axes[i+1].imshow(bev_vis, cmap='viridis', origin='lower')
            axes[i+1].set_title(f'BEV - {strategy.replace("_", " ").title()}')
            axes[i+1].set_xlabel('X (meters)')
            axes[i+1].set_ylabel('Y (meters)')
            
            # Set proper axis labels
            x_ticks = np.linspace(0, self.bev_width-1, 5)
            y_ticks = np.linspace(0, self.bev_height-1, 5)
            x_labels = np.linspace(self.bev_range[0], self.bev_range[1], 5)
            y_labels = np.linspace(self.bev_range[2], self.bev_range[3], 5)
            
            axes[i+1].set_xticks(x_ticks)
            axes[i+1].set_yticks(y_ticks)
            axes[i+1].set_xticklabels([f'{x:.1f}' for x in x_labels])
            axes[i+1].set_yticklabels([f'{y:.1f}' for y in y_labels])
            
            plt.colorbar(im, ax=axes[i+1])
        
        # Hide unused subplot
        if len(strategies) < 5:
            axes[5].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_projection(self, 
                           image: np.ndarray,
                           bev_features: torch.Tensor,
                           save_path: Optional[str] = None):
        """
        Visualize the projection results.
        
        Args:
            image: Input image (H, W, 3)
            bev_features: BEV feature map (C, bev_height, bev_width)
            save_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot BEV feature map (average across channels)
        bev_vis = bev_features.mean(dim=0).cpu().numpy()
        im = axes[1].imshow(bev_vis, cmap='viridis', origin='lower')
        axes[1].set_title('BEV Feature Map')
        axes[1].set_xlabel('X (meters)')
        axes[1].set_ylabel('Y (meters)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1])
        
        # Set proper axis labels
        x_ticks = np.linspace(0, self.bev_width-1, 5)
        y_ticks = np.linspace(0, self.bev_height-1, 5)
        x_labels = np.linspace(self.bev_range[0], self.bev_range[1], 5)
        y_labels = np.linspace(self.bev_range[2], self.bev_range[3], 5)
        
        axes[1].set_xticks(x_ticks)
        axes[1].set_yticks(y_ticks)
        axes[1].set_xticklabels([f'{x:.1f}' for x in x_labels])
        axes[1].set_yticklabels([f'{y:.1f}' for y in y_labels])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_sample_fpn_data():
    """Create sample FPN data for testing."""
    # Sample camera intrinsic parameters
    intrinsic = np.array([
        [721.5377, 0.0, 609.5593],
        [0.0, 721.5377, 172.8540],
        [0.0, 0.0, 1.0]
    ])
    
    # Sample camera extrinsic parameters
    pitch = -10 * np.pi / 180  # 10 degrees down
    translation = np.array([0, 0, 1.65])  # 1.65m high
    
    rotation = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation
    
    # Create sample image
    image = np.random.rand(480, 640, 3) * 255
    image = image.astype(np.uint8)
    
    # Create sample FPN features at different scales
    # Typical FPN outputs: P3, P4, P5, P6 with scales 1/8, 1/16, 1/32, 1/64
    fpn_features = [
        torch.randn(256, 60, 80),   # P3: 1/8 scale
        torch.randn(256, 30, 40),   # P4: 1/16 scale  
        torch.randn(256, 15, 20),   # P5: 1/32 scale
        torch.randn(256, 8, 10),    # P6: 1/64 scale (optional)
    ]
    
    fpn_scales = [1/8, 1/16, 1/32, 1/64]
    
    return intrinsic, extrinsic, image, fpn_features, fpn_scales


def validate_fpn_projection():
    """Validate the FPN projection implementation."""
    print("Creating sample FPN data...")
    intrinsic, extrinsic, image, fpn_features, fpn_scales = create_sample_fpn_data()
    
    print("Initializing projector...")
    projector = ImageToBEVProjector(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        bev_range=(-20, 20, -10, 30),
        bev_resolution=0.2,
        height_range=(-2, 2),
        height_samples=16
    )
    
    print("Projecting FPN features to BEV...")
    
    # Test different fusion strategies
    strategies = ['weighted_sum', 'max_pool', 'avg_pool', 'adaptive']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} fusion...")
        bev_features = projector.project_fpn_to_bev(
            fpn_features, fpn_scales, image.shape[:2], 
            fusion_strategy=strategy
        )
        
        print(f"  BEV shape: {bev_features.shape}")
        print(f"  Non-zero features: {torch.sum(bev_features != 0).item()}")
        print(f"  Feature range: [{bev_features.min():.3f}, {bev_features.max():.3f}]")
    
    # Test multi-scale BEV
    print("\nTesting multi-scale BEV...")
    multi_scale_bev = projector.project_fpn_to_multi_scale_bev(
        fpn_features, fpn_scales, image.shape[:2]
    )
    
    for scale_name, bev_feat in multi_scale_bev.items():
        print(f"  {scale_name}: {bev_feat.shape}")
    
    # Visualize results
    print("\nVisualizing FPN projection results...")
    projector.visualize_fpn_projection(
        image, fpn_features, fpn_scales, image.shape[:2], 
        "fpn_bev_projection_result.png"
    )
    
    return projector, fpn_features, fpn_scales


def create_sample_data():
    """Create sample data for testing."""
    # Sample camera intrinsic parameters (typical camera)
    intrinsic = np.array([
        [721.5377, 0.0, 609.5593],
        [0.0, 721.5377, 172.8540],
        [0.0, 0.0, 1.0]
    ])
    
    # Sample camera extrinsic parameters (camera mounted on vehicle)
    # This represents a camera mounted 1.65m high, tilted down 10 degrees
    pitch = -10 * np.pi / 180  # 10 degrees down
    translation = np.array([0, 0, 1.65])  # 1.65m high
    
    # Create rotation matrix
    rotation = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # Create extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation
    
    # Create sample image (synthetic)
    image = np.random.rand(376, 1241, 3) * 255
    image = image.astype(np.uint8)
    
    # Create sample image features (could be CNN features)
    image_features = torch.randn(64, 376, 1241)  # 64 channels
    
    return intrinsic, extrinsic, image, image_features


def validate_projection():
    """Validate the projection implementation."""
    print("Creating sample data...")
    intrinsic, extrinsic, image, image_features = create_sample_data()
    
    print("Initializing projector...")
    projector = ImageToBEVProjector(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        bev_range=(-20, 20, -10, 30),  # x_min, x_max, y_min, y_max
        bev_resolution=0.2,  # 20cm resolution
        height_range=(-2, 2),  # Sample from -2m to 2m height
        height_samples=16
    )
    
    print("Projecting image features to BEV...")
    bev_features = projector.project_to_bev(image_features, image.shape[:2])
    
    print(f"BEV feature map shape: {bev_features.shape}")
    print(f"BEV resolution: {projector.bev_resolution}m per pixel")
    print(f"BEV dimensions: {projector.bev_width} x {projector.bev_height}")
    
    # Visualize results
    print("Visualizing projection results...")
    projector.visualize_projection(image, bev_features, "bev_projection_result.png")
    
    # Additional validation
    print("\nValidation results:")
    print(f"Non-zero BEV features: {torch.sum(bev_features != 0).item()}")
    print(f"Feature range: [{bev_features.min():.3f}, {bev_features.max():.3f}]")
    print(f"Mean feature value: {bev_features.mean():.3f}")
    
    return projector, bev_features


if __name__ == "__main__":
    # Test regular projection
    print("=== Regular Projection Test ===")
    projector, bev_features = validate_projection()
    
    # Test FPN projection
    print("\n=== FPN Projection Test ===")
    projector_fpn, fpn_features, fpn_scales = validate_fpn_projection()
    
    print("\nValidation completed successfully!") 
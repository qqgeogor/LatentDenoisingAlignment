import numpy as np
import torch
import cv2
from image_to_bev_projection import ImageToBEVProjector


def load_kitti_style_calibration():
    """
    Example of loading KITTI-style camera calibration parameters.
    In real usage, you would load these from calibration files.
    """
    # KITTI-style intrinsic parameters (P2 matrix)
    # These are example values - replace with your actual calibration
    intrinsic = np.array([
        [721.5377, 0.0, 609.5593],
        [0.0, 721.5377, 172.8540],
        [0.0, 0.0, 1.0]
    ])
    
    # KITTI-style extrinsic parameters (Tr_velo_to_cam)
    # Transform from velodyne (LiDAR) to camera coordinates
    # In practice, you need camera-to-world transformation
    R = np.array([
        [7.533745e-03, -9.999714e-01, -6.166020e-04],
        [1.480249e-02, 7.280733e-04, -9.998902e-01],
        [9.998621e-01, 7.523790e-03, 1.480755e-02]
    ])
    
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
    
    # Create 4x4 transformation matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    
    return intrinsic, extrinsic


def demo_front_camera_projection():
    """Demo with front-facing camera parameters."""
    print("=== Front Camera BEV Projection Demo ===")
    
    # Load calibration
    intrinsic, extrinsic = load_kitti_style_calibration()
    
    # Create projector for front camera
    projector = ImageToBEVProjector(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        bev_range=(-30, 30, 0, 60),  # Wider range for front camera
        bev_resolution=0.1,  # 10cm resolution
        height_range=(-2, 3),  # Ground to 3m height
        height_samples=20
    )
    
    # Simulate image features (e.g., from a CNN backbone)
    image_features = torch.randn(256, 370, 1226)  # Typical CNN feature size
    
    # Project to BEV
    bev_features = projector.project_to_bev(image_features, (370, 1226))
    
    print(f"BEV feature map shape: {bev_features.shape}")
    print(f"BEV covers: {projector.bev_range[0]}m to {projector.bev_range[1]}m (x)")
    print(f"            {projector.bev_range[2]}m to {projector.bev_range[3]}m (y)")
    
    return projector, bev_features


def demo_multi_camera_fusion():
    """Demo with multiple camera views."""
    print("\n=== Multi-Camera BEV Fusion Demo ===")
    
    # Define cameras (front, left, right, rear)
    cameras = {
        'front': {
            'intrinsic': np.array([
                [721.5377, 0.0, 609.5593],
                [0.0, 721.5377, 172.8540],
                [0.0, 0.0, 1.0]
            ]),
            'extrinsic': np.eye(4),  # Identity for front camera
            'bev_range': (-15, 15, 0, 30)
        },
        'left': {
            'intrinsic': np.array([
                [721.5377, 0.0, 609.5593],
                [0.0, 721.5377, 172.8540],
                [0.0, 0.0, 1.0]
            ]),
            'extrinsic': create_camera_extrinsic(yaw=np.pi/2),  # 90 degrees left
            'bev_range': (-30, 0, -15, 15)
        },
        'right': {
            'intrinsic': np.array([
                [721.5377, 0.0, 609.5593],
                [0.0, 721.5377, 172.8540],
                [0.0, 0.0, 1.0]
            ]),
            'extrinsic': create_camera_extrinsic(yaw=-np.pi/2),  # 90 degrees right
            'bev_range': (0, 30, -15, 15)
        }
    }
    
    # BEV parameters
    bev_resolution = 0.2
    unified_bev_range = (-30, 30, -15, 30)
    
    # Create unified BEV feature map
    bev_height = int((unified_bev_range[3] - unified_bev_range[2]) / bev_resolution)
    bev_width = int((unified_bev_range[1] - unified_bev_range[0]) / bev_resolution)
    unified_bev = torch.zeros(256, bev_height, bev_width)
    
    for camera_name, camera_params in cameras.items():
        print(f"Processing {camera_name} camera...")
        
        # Create projector for this camera
        projector = ImageToBEVProjector(
            intrinsic=camera_params['intrinsic'],
            extrinsic=camera_params['extrinsic'],
            bev_range=camera_params['bev_range'],
            bev_resolution=bev_resolution,
            height_range=(-2, 2),
            height_samples=16
        )
        
        # Simulate image features
        image_features = torch.randn(256, 370, 1226)
        
        # Project to BEV
        camera_bev = projector.project_to_bev(image_features, (370, 1226))
        
        # Merge into unified BEV (simple averaging)
        # In practice, you might use more sophisticated fusion strategies
        x_start = int((camera_params['bev_range'][0] - unified_bev_range[0]) / bev_resolution)
        x_end = x_start + camera_bev.shape[2]
        y_start = int((camera_params['bev_range'][2] - unified_bev_range[2]) / bev_resolution)
        y_end = y_start + camera_bev.shape[1]
        
        unified_bev[:, y_start:y_end, x_start:x_end] += camera_bev
    
    print(f"Unified BEV shape: {unified_bev.shape}")
    
    return unified_bev


def create_camera_extrinsic(x=0, y=0, z=1.65, roll=0, pitch=0, yaw=0):
    """Create camera extrinsic matrix from position and orientation."""
    # Create rotation matrix
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = R_z @ R_y @ R_x
    
    # Create extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = [x, y, z]
    
    return extrinsic


def demo_with_real_image():
    """Demo with a real image (if available)."""
    print("\n=== Real Image Demo ===")
    
    try:
        # Try to load a real image
        image = cv2.imread("sample_image.jpg")
        if image is None:
            print("No real image found, creating synthetic image...")
            image = np.random.rand(480, 640, 3) * 255
            image = image.astype(np.uint8)
        else:
            print("Loaded real image")
            
        # Convert BGR to RGB for matplotlib
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create realistic camera parameters for a dashcam
        intrinsic = np.array([
            [800, 0, 320],    # fx, 0, cx
            [0, 800, 240],    # 0, fy, cy
            [0, 0, 1]         # 0, 0, 1
        ])
        
        # Camera mounted on car hood, 1.2m high, tilted down 5 degrees
        pitch = -5 * np.pi / 180
        extrinsic = create_camera_extrinsic(z=1.2, pitch=pitch)
        
        # Create projector
        projector = ImageToBEVProjector(
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            bev_range=(-20, 20, 0, 40),
            bev_resolution=0.25,
            height_range=(-1, 2),
            height_samples=12
        )
        
        # Create image features (simulate CNN features)
        # In practice, these would come from a trained CNN
        image_features = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Project to BEV
        bev_features = projector.project_to_bev(image_features, image.shape[:2])
        
        # Visualize
        projector.visualize_projection(image, bev_features, "real_image_bev.png")
        
        print(f"Real image BEV projection completed")
        
    except Exception as e:
        print(f"Error in real image demo: {e}")


if __name__ == "__main__":
    # Run demos
    demo_front_camera_projection()
    demo_multi_camera_fusion()
    demo_with_real_image()
    
    print("\n=== Usage Tips ===")
    print("1. Adjust bev_range based on your application needs")
    print("2. Tune bev_resolution for desired precision vs. speed")
    print("3. Set height_range to cover expected object heights")
    print("4. Use more height_samples for better accuracy")
    print("5. For real applications, calibrate your cameras properly")
    print("6. Consider using learned features from pre-trained CNNs") 
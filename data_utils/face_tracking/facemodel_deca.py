import torch
import torch.nn as nn
import numpy as np
import os
import sys
from typing import Dict, Optional

# Add DECA to path if it exists
DECA_PATH = os.path.join(os.path.dirname(__file__), 'DECA')
if os.path.exists(DECA_PATH) and DECA_PATH not in sys.path:
    sys.path.insert(0, DECA_PATH)

try:
    from decalib.deca import DECA
    from decalib.datasets import datasets
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg
    DECA_AVAILABLE = True
except ImportError:
    DECA_AVAILABLE = False
    print("Warning: DECA not available. Run: bash scripts/setup_deca.sh")


class Face_3DMM_DECA(nn.Module):
    """
    DECA-based 3D Morphable Model for face tracking.
    DECA provides better expression capture and head pose estimation compared to BFM.
    """
    
    def __init__(self, device='cuda', deca_dir=None):
        super(Face_3DMM_DECA, self).__init__()
        
        if not DECA_AVAILABLE:
            raise ImportError("DECA is not installed. Please run: pip install git+https://github.com/yfeng95/DECA.git")
        
        self.device = device
        
        # Initialize DECA model
        if deca_dir is None:
            deca_dir = os.path.join(os.path.dirname(__file__), 'deca_model')
        
        # Setup DECA configuration
        deca_cfg.model.use_tex = True
        deca_cfg.rasterizer_type = 'pytorch3d'
        deca_cfg.model.topology_path = os.path.join(deca_dir, 'data', 'head_template.obj')
        # Ensure we use the standard FLAME landmark embedding from DECA
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_dir, 'data', 'landmark_embedding.npy')
        
        # Initialize DECA
        self.deca = DECA(config=deca_cfg, device=device)
        
        # DECA parameters
        # Shape: 100 dimensions (identity)
        # Expression: 50 dimensions (more than BFM's 79)
        # Pose: 6 dimensions (global rotation and jaw pose)
        # Texture: 50 dimensions
        # Light: 27 dimensions (SH lighting)
        
        self.n_shape = 100
        self.n_exp = 50
        self.n_pose = 6
        self.n_tex = 50
        self.n_light = 27
        
        # Get FLAME model parameters
        self.flame = self.deca.flame
        
        # Face vertices count (FLAME has 5023 vertices)
        self.point_num = 5023
        
        # Landmarks are produced via FLAME with the configured landmark embedding
        
    def encode_image(self, images):
        """
        Encode images to get DECA parameters.
        
        Args:
            images: torch.Tensor [B, C, H, W], normalized to [-1, 1]
        
        Returns:
            codedict: Dictionary containing all DECA parameters
        """
        with torch.no_grad():
            codedict = self.deca.encode(images)
        return codedict
    
    def decode(self, codedict, return_landmarks=True):
        """
        Decode DECA parameters to get 3D face geometry.
        
        Args:
            codedict: Dictionary with keys 'shape', 'exp', 'pose', 'tex', 'light', 'cam'
            return_landmarks: Whether to return facial landmarks
        
        Returns:
            vertices: [B, V, 3] 3D face vertices
            landmarks (optional): [B, 68, 3] 3D facial landmarks
        """
        opdict, visdict = self.deca.decode(codedict, render=False)
        
        vertices = opdict['verts']
        
        if return_landmarks:
            landmarks = opdict['landmarks3d']
            return vertices, landmarks
        
        return vertices
    
    def forward_geo(self, shape_params, exp_params, pose_params=None, return_landmarks=False):
        """
        Forward pass to get 3D geometry from parameters.
        
        Args:
            shape_params: [B, 100] identity parameters
            exp_params: [B, 50] expression parameters  
            pose_params: [B, 6] pose parameters (optional, defaults to neutral)
            return_landmarks: If True, also return landmarks
        
        Returns:
            geometry: [B, V, 3] 3D face vertices
            landmarks3d: [B, 68, 3] (only if return_landmarks=True)
        """
        batch_size = shape_params.shape[0]
        
        if pose_params is None:
            pose_params = torch.zeros(batch_size, self.n_pose).to(self.device)
        
        # Use FLAME to generate geometry
        # FLAME returns 3 values: vertices, landmarks2d, landmarks3d
        vertices, landmarks2d, landmarks3d = self.flame(
            shape_params=shape_params,
            expression_params=exp_params,
            pose_params=pose_params
        )
        
        if return_landmarks:
            return vertices, landmarks3d
        
        return vertices
    
    def forward_geo_sub(self, shape_params, exp_params, sub_index, pose_params=None):
        """
        Forward pass to get sub-mesh geometry.
        
        Args:
            shape_params: [B, 100] identity parameters
            exp_params: [B, 50] expression parameters
            sub_index: [N] vertex indices to extract
            pose_params: [B, 6] pose parameters (optional)
        
        Returns:
            geometry: [B, N, 3] 3D vertices at sub_index
        """
        full_geometry = self.forward_geo(shape_params, exp_params, pose_params)
        
        # Extract sub vertices
        sub_geometry = full_geometry[:, sub_index.long(), :]
        
        return sub_geometry
    
    def get_landmarks(self, vertices):
        """
        Extract 68 facial landmarks from vertices.
        
        Args:
            vertices: [B, V, 3] 3D face vertices
        
        Returns:
            landmarks: [B, 68, 3] 3D facial landmarks
        """
        # FLAME model provides landmark extraction via seletec_3d68 method
        landmarks = self.flame.seletec_3d68(vertices)
        
        return landmarks
    
    def project_vertices(self, vertices, camera_params):
        """
        Project 3D vertices to 2D image space.
        
        Args:
            vertices: [B, V, 3] 3D vertices
            camera_params: [B, 3] camera parameters [scale, tx, ty]
        
        Returns:
            projected: [B, V, 2] 2D projected vertices
        """
        # Extract camera parameters
        scale = camera_params[:, 0:1, None]  # [B, 1, 1]
        tx = camera_params[:, 1:2, None]      # [B, 1, 1]
        ty = camera_params[:, 2:3, None]      # [B, 1, 1]
        
        # Orthographic projection
        projected = vertices[:, :, :2] * scale
        projected[:, :, 0:1] = projected[:, :, 0:1] + tx
        projected[:, :, 1:2] = projected[:, :, 1:2] + ty
        
        return projected
    
    def compute_shape_from_params(self, shape_params):
        """
        Compute mean shape from identity parameters.
        
        Args:
            shape_params: [B, 100] identity parameters
        
        Returns:
            mean_shape: [B, V, 3] mean face shape
        """
        batch_size = shape_params.shape[0]
        exp_params = torch.zeros(batch_size, self.n_exp).to(self.device)
        pose_params = torch.zeros(batch_size, self.n_pose).to(self.device)
        
        vertices = self.forward_geo(shape_params, exp_params, pose_params)
        
        return vertices
    
    def save_params(self, save_path, codedict):
        """
        Save DECA parameters to file.
        
        Args:
            save_path: Path to save parameters
            codedict: Dictionary of DECA parameters
        """
        torch.save(codedict, save_path)
    
    def load_params(self, load_path):
        """
        Load DECA parameters from file.
        
        Args:
            load_path: Path to load parameters from
        
        Returns:
            codedict: Dictionary of DECA parameters
        """
        codedict = torch.load(load_path)
        return codedict


class DECAWrapper:
    """
    Wrapper class for easier DECA usage with the existing pipeline.
    Provides compatibility layer between DECA and the original BFM interface.
    """
    
    def __init__(self, device='cuda'):
        self.model = Face_3DMM_DECA(device=device)
        self.device = device
    
    def extract_params_from_image(self, image_path):
        """
        Extract DECA parameters from a single image.
        
        Args:
            image_path: Path to input image
        
        Returns:
            params: Dictionary of extracted parameters
        """
        import cv2
        
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (DECA input size)
        img = cv2.resize(img, (224, 224))
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Encode with DECA
        codedict = self.model.encode_image(img_tensor)
        
        return codedict
    
    def extract_params_from_landmarks(self, landmarks, image_shape):
        """
        Fit DECA parameters from 2D landmarks using optimization.
        
        Args:
            landmarks: [68, 2] 2D facial landmarks
            image_shape: (H, W) image dimensions
        
        Returns:
            params: Optimized DECA parameters
        """
        # Initialize parameters
        shape_params = torch.zeros(1, self.model.n_shape, requires_grad=True, device=self.device)
        exp_params = torch.zeros(1, self.model.n_exp, requires_grad=True, device=self.device)
        pose_params = torch.zeros(1, self.model.n_pose, requires_grad=True, device=self.device)
        cam_params = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True, device=self.device)
        
        # Convert landmarks to tensor
        landmarks_tensor = torch.from_numpy(landmarks).float().to(self.device).unsqueeze(0)
        
        # Optimizer
        optimizer = torch.optim.Adam([shape_params, exp_params, pose_params, cam_params], lr=0.01)
        
        # Optimization loop
        for iter in range(200):
            optimizer.zero_grad()
            
            # Generate vertices
            vertices = self.model.forward_geo(shape_params, exp_params, pose_params)
            
            # Get landmarks from vertices
            pred_landmarks_3d = self.model.get_landmarks(vertices)
            
            # Project to 2D
            pred_landmarks_2d = self.model.project_vertices(pred_landmarks_3d, cam_params)
            
            # Normalize to image space
            h, w = image_shape
            pred_landmarks_2d[:, :, 0] = (pred_landmarks_2d[:, :, 0] + 1) * w / 2
            pred_landmarks_2d[:, :, 1] = (pred_landmarks_2d[:, :, 1] + 1) * h / 2
            
            # Landmark loss
            loss = torch.mean((pred_landmarks_2d - landmarks_tensor) ** 2)
            
            # Regularization
            loss += 0.001 * torch.mean(shape_params ** 2)
            loss += 0.001 * torch.mean(exp_params ** 2)
            
            loss.backward()
            optimizer.step()
        
        # Return optimized parameters
        codedict = {
            'shape': shape_params.detach(),
            'exp': exp_params.detach(),
            'pose': pose_params.detach(),
            'cam': cam_params.detach()
        }
        
        return codedict


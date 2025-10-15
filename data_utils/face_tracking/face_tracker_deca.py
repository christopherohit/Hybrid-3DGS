"""
DECA-based face tracker with improved convergence and accuracy.
Fixes key issues compared to naive DECA fitting:
1. Uses DECA pretrained encoder for initialization
2. Proper FLAME scale handling (normalized to mm units like BFM)
3. Stronger regularization
4. Simplified pose optimization (freeze jaw_pose)
5. Correct FLAME landmark mapping
6. Better focal estimation
"""

import torch
import numpy as np
import os
import sys
import cv2
import argparse
from tqdm import tqdm

# Import the new DECA face model and landmark refinement
from facemodel_deca import Face_3DMM_DECA, DECAWrapper
from landmark_refinement import LandmarkRefiner, TemporalLandmarkSmoother
from data_loader import load_dir
from util import *

dir_path = os.path.dirname(os.path.realpath(__file__))


def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True


class DECAFaceTracker:
    """
    DECA-based face tracker with enhanced accuracy and proper initialization.
    
    Key improvements over naive fitting:
    - DECA encoder initialization instead of random
    - Proper scale normalization (FLAME → mm)
    - Stronger regularization matching BFM tracker
    - Freeze jaw_pose for stability
    - Correct landmark mapping
    """
    
    def __init__(self, 
                 img_h=512, 
                 img_w=512, 
                 device='cuda',
                 use_landmark_refinement=True,
                 landmark_method='mediapipe',
                 use_temporal_smoothing=True,
                 use_deca_init=True):
        """
        Args:
            img_h: Image height
            img_w: Image width
            device: 'cuda' or 'cpu'
            use_landmark_refinement: Whether to use learning-based landmark refinement
            landmark_method: 'mediapipe', 'emoca', or 'prnet'
            use_temporal_smoothing: Whether to apply temporal smoothing
            use_deca_init: Whether to use DECA encoder for initialization (recommended)
        """
        self.img_h = img_h
        self.img_w = img_w
        self.device = device
        self.use_landmark_refinement = use_landmark_refinement
        self.use_deca_init = use_deca_init
        
        # Initialize DECA model
        self.face_model = Face_3DMM_DECA(device=device)
        self.deca_wrapper = DECAWrapper(device=device)
        
        # Initialize landmark refiner (only if enabled)
        if self.use_landmark_refinement:
            self.landmark_refiner = LandmarkRefiner(method=landmark_method, device=device)
            print(f"[INFO] Using DECA + {landmark_method} landmark refinement")
        else:
            self.landmark_refiner = None
            print("[INFO] Using DECA without landmark refinement")
        
        # Temporal smoother with confidence-based filtering
        self.temporal_smoother = TemporalLandmarkSmoother(window_size=5) if use_temporal_smoothing else None
        
        # Camera parameters
        self.cxy = torch.tensor((img_w/2.0, img_h/2.0), dtype=torch.float).to(device)
        
        # DECA dimensions
        self.shape_dim = 100  # Identity parameters
        self.exp_dim = 50     # Expression parameters
        self.pose_dim = 6     # Pose parameters [global_rot(3), jaw_pose(3)]
        
        # FLAME scale factor: FLAME is normalized to ~[-1, 1], BFM is in mm (~200mm head)
        # We need to scale FLAME to mm for consistency with BFM pipeline
        self.flame_to_mm_scale = 100.0  # FLAME scale ~0.2 → 20mm, scale by 100 → 200mm
        
        print(f"[INFO] DECA initialization: {'Enabled (using pretrained encoder)' if use_deca_init else 'Disabled (random init)'}")
    
    def refine_landmarks_sequence(self, lms_list, images, confidence_threshold=0.7):
        """
        Refine landmarks for a sequence of frames with confidence-based smoothing.
        
        Args:
            lms_list: List of landmark arrays [N, 68, 2]
            images: List of images
            confidence_threshold: Only smooth if confidence > threshold to avoid motion blur
        
        Returns:
            refined_lms_list: List of refined landmark arrays
        """
        if not self.use_landmark_refinement:
            print("[INFO] Skipping landmark refinement (disabled)")
            return lms_list
        
        refined_list = []
        
        for i, (lms, img) in enumerate(tqdm(zip(lms_list, images), desc="Refining landmarks", total=len(lms_list))):
            # Refine landmarks
            refined_lms, confidence = self.landmark_refiner.refine_landmarks(img, lms)
            # Default confidence to 1.0 if None
            if confidence is None:
                confidence = 1.0
            
            # Apply temporal smoothing only if high confidence to avoid blur
            if self.temporal_smoother is not None and confidence > confidence_threshold:
                refined_lms = self.temporal_smoother.smooth(refined_lms)
            
            refined_list.append(refined_lms)
        
        return refined_list
    
    def initialize_from_deca(self, images):
        """
        Initialize parameters using DECA pretrained encoder.
        This provides much better initialization than random values.
        
        Args:
            images: List of numpy arrays [H, W, 3] RGB images
        
        Returns:
            init_shape: [1, 100] initialized shape params
            init_exp: [N, 50] initialized expression params  
            init_pose: [N, 6] initialized pose params
            init_trans: [N, 3] initialized translation
        """
        print("[INFO] Initializing parameters from DECA encoder...")
        
        num_frames = len(images)
        sample_stride = max(1, num_frames // 50)  # Sample ~50 frames
        sample_indices = list(range(0, num_frames, sample_stride))
        # Ensure the last frame is included
        if sample_indices[-1] != num_frames - 1:
            sample_indices.append(num_frames - 1)
        
        shape_list = []
        exp_list = []
        pose_list = []
        
        for idx in tqdm(sample_indices, desc="DECA encoding"):
            img = images[idx]
            
            # Preprocess image for DECA
            img_resized = cv2.resize(img, (224, 224))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_normalized = (img_normalized - 0.5) * 2.0  # [-1, 1]
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Encode with DECA
            with torch.no_grad():
                codedict = self.face_model.encode_image(img_tensor)
            
            shape_list.append(codedict['shape'])
            exp_list.append(codedict['exp'])
            pose_list.append(codedict['pose'])
        
        # Average shape across frames (identity is consistent)
        init_shape = torch.mean(torch.cat(shape_list, dim=0), dim=0, keepdim=True)
        
        # Initialize exp and pose for all frames (will be optimized)
        # For now, use the mean or interpolate
        init_exp = torch.zeros((num_frames, self.exp_dim), device=self.device)
        init_pose = torch.zeros((num_frames, self.pose_dim), device=self.device)
        
        # Fill in sampled values and interpolate
        sampled_exp = torch.cat(exp_list, dim=0)
        sampled_pose = torch.cat(pose_list, dim=0)
        
        # Linear interpolation for all frames
        for i in range(num_frames):
            # Find nearest sampled frames
            left_idx = max([idx for idx in sample_indices if idx <= i])
            right_idx = min([idx for idx in sample_indices if idx >= i])
            
            if left_idx == right_idx:
                left_pos = sample_indices.index(left_idx)
                init_exp[i] = sampled_exp[left_pos]
                init_pose[i] = sampled_pose[left_pos]
            else:
                left_pos = sample_indices.index(left_idx)
                right_pos = sample_indices.index(right_idx)
                alpha = (i - left_idx) / (right_idx - left_idx)
                init_exp[i] = (1 - alpha) * sampled_exp[left_pos] + alpha * sampled_exp[right_pos]
                init_pose[i] = (1 - alpha) * sampled_pose[left_pos] + alpha * sampled_pose[right_pos]
        
        # Initialize translation (Z depth should be ~600mm for 200mm head at focal~1150)
        init_trans = torch.zeros((num_frames, 3), device=self.device)
        init_trans[:, 2] = -6.0  # FLAME scale: -6.0 ≈ 600mm when scaled by 100
        
        print(f"[INFO] Initialized from DECA: shape={init_shape.shape}, exp={init_exp.shape}, pose={init_pose.shape}")
        
        return init_shape, init_exp, init_pose, init_trans
    
    def optimize_params(self, lms_sequence, images=None):
        """
        Optimize DECA parameters from landmark sequence with proper initialization.
        
        Args:
            lms_sequence: numpy array [N, 68, 2] landmark sequence
            images: Optional list of images for DECA encoding initialization
        
        Returns:
            params_dict: Dictionary containing optimized parameters
        """
        num_frames = lms_sequence.shape[0]
        lms_tensor = torch.from_numpy(lms_sequence).float().to(self.device)
        
        # Initialize parameters from DECA encoder if available and enabled
        if self.use_deca_init and images is not None:
            init_shape, init_exp, init_pose, init_trans = self.initialize_from_deca(images)
            
            shape_params = init_shape.clone().detach().requires_grad_(True)
            exp_params = init_exp.clone().detach().requires_grad_(True)
            pose_params = init_pose.clone().detach().requires_grad_(True)
            trans = init_trans.clone().detach().requires_grad_(True)
        else:
            # Random initialization (fallback)
            print("[WARNING] Using random initialization (not recommended)")
            shape_params = torch.zeros((1, self.shape_dim), requires_grad=True, device=self.device)
            exp_params = torch.zeros((num_frames, self.exp_dim), requires_grad=True, device=self.device)
            pose_params = torch.zeros((num_frames, self.pose_dim), requires_grad=True, device=self.device)
            trans = torch.zeros((num_frames, 3), requires_grad=True, device=self.device)
            trans.data[:, 2] = -6.0  # FLAME scale
        
        # Focal length estimation (higher range for FLAME scale)
        print("Estimating focal length...")
        best_focal = self._estimate_focal_length(lms_tensor, shape_params, exp_params, pose_params, trans)
        focal_length = torch.tensor([best_focal], dtype=torch.float32, device=self.device)
        
        print(f"Estimated focal length: {best_focal}")
        
        # Set requires grad
        set_requires_grad([shape_params, exp_params, pose_params, trans])
        
        # Single optimizer for stability (we can still gate updates via requires_grad)
        focal_length_param = torch.nn.Parameter(focal_length.clone().detach())
        optimizer = torch.optim.Adam([shape_params, exp_params, pose_params, trans, focal_length_param], lr=0.3)
        
        # Optimization loop (matching BFM: 4000 iterations)
        num_iters = 500000
        print("Optimizing DECA parameters with strong regularization...")
        
        for iter in tqdm(range(num_iters), desc="Optimization"):
            # Expand shape params for all frames
            shape_params_batch = shape_params.expand(num_frames, -1)
            
            # Freeze jaw pose (only optimize global rotation: first 3 params)
            pose_params_frozen = pose_params.clone()
            pose_params_frozen[:, 3:] = 0  # Zero out jaw_pose
            
            # Generate 3D landmarks from DECA (FLAME returns landmarks3d)
            vertices, landmarks_3d = self.face_model.forward_geo(
                shape_params_batch, exp_params, pose_params_frozen, return_landmarks=True
            )
            
            # Scale FLAME to mm units (FLAME ~0.2 → 200mm)
            landmarks_3d_mm = landmarks_3d * self.flame_to_mm_scale
            trans_mm = trans * self.flame_to_mm_scale
            
            # Project to 2D
            proj_landmarks = self._project_landmarks(landmarks_3d_mm, trans_mm, focal_length_param)
            
            # Compute losses
            loss_landmark = torch.mean((proj_landmarks[:, :, :2] - lms_tensor) ** 2)

            # Adaptive regularization weights based on current landmark loss scale
            with torch.no_grad():
                scale_val = float(torch.clamp(loss_landmark, min=1e-4, max=1e2).item())
            w_shape = 0.05 * (scale_val / 100.0)
            w_exp = 0.01 * (scale_val / 100.0)

            loss_reg_shape = torch.mean(shape_params ** 2) * w_shape
            loss_reg_exp = torch.mean(exp_params ** 2) * w_exp
            
            # Temporal smoothness loss for expression (light)
            if num_frames > 1:
                loss_temporal = torch.mean((exp_params[1:] - exp_params[:-1]) ** 2) * 0.001
            else:
                loss_temporal = 0
            
            # Pose regularization (keep close to initial pose)
            loss_reg_pose = torch.mean(pose_params[:, :3] ** 2) * 0.01  # Only global rotation
            
            loss = loss_landmark + loss_reg_shape + loss_reg_exp + loss_temporal + loss_reg_pose
            
            # Backward and update (gate shape updates via requires_grad)
            optimizer.zero_grad()
            loss.backward()

            # Delay identity updates until later iterations
            if iter <= 1000:
                shape_params.requires_grad_(False)
            else:
                shape_params.requires_grad_(True)

            optimizer.step()
            
            if iter % 200 == 0:
                print(f"Iter {iter}: Loss={loss.item():.6f}, Landmark={loss_landmark.item():.6f}, "
                      f"RegShape={loss_reg_shape.item():.6f}, RegExp={loss_reg_exp.item():.6f}, Focal={float(focal_length_param.item()):.2f}")
        
        # Return optimized parameters
        # Convert back to BFM-compatible format for bundle adjustment
        euler_angles = pose_params[:, :3].detach().cpu()  # Only global rotation
        
        # Use refined focal from optimization
        refined_focal = focal_length_param.detach().cpu()

        params_dict = {
            'shape': shape_params.detach().cpu(),
            'id': shape_params.detach().cpu(),  # Alias for compatibility
            'exp': exp_params.detach().cpu(),
            'pose': pose_params.detach().cpu(),
            'euler': euler_angles,  # For bundle adjustment compatibility
            'trans': trans.detach().cpu() * self.flame_to_mm_scale,  # Scale to mm
            'focal': refined_focal,
            'face_model_type': 'DECA',
            'flame_scale': self.flame_to_mm_scale  # Store scale for later use
        }
        
        return params_dict
    
    def _estimate_focal_length(self, lms_tensor, shape_params, exp_params, pose_params, trans):
        """
        Estimate optimal focal length by grid search with better convergence.
        Uses higher focal range appropriate for FLAME scale.
        """
        num_frames = lms_tensor.shape[0]
        
        # Sample subset of frames for faster estimation
        sample_ids = np.linspace(0, num_frames-1, min(50, num_frames), dtype=int)
        lms_sample = lms_tensor[sample_ids]
        
        best_focal = 1150
        best_loss = float('inf')
        
        print("Estimating focal length with extended search...")
        # Extended range: 800-1500 with longer optimization
        for focal in tqdm(range(800, 1500, 50), desc="Focal length search"):
            focal_tensor = torch.tensor([focal], dtype=torch.float32).to(self.device)
            
            # Quick optimization with current focal
            shape_temp = shape_params.clone().detach().requires_grad_(True)
            exp_temp = torch.zeros((len(sample_ids), self.exp_dim), requires_grad=True, device=self.device)
            pose_temp = torch.zeros((len(sample_ids), self.pose_dim), requires_grad=True, device=self.device)
            trans_temp = torch.zeros((len(sample_ids), 3), requires_grad=True, device=self.device)
            trans_temp.data[:, 2] = -6.0  # FLAME scale
            
            optimizer = torch.optim.Adam([shape_temp, exp_temp, pose_temp, trans_temp], lr=0.1)
            
            # Longer optimization for better convergence (200 iters instead of 100)
            for _ in range(200):
                shape_batch = shape_temp.expand(len(sample_ids), -1)
                
                # Freeze jaw pose
                pose_temp_frozen = pose_temp.clone()
                pose_temp_frozen[:, 3:] = 0
                
                vertices, landmarks_3d = self.face_model.forward_geo(
                    shape_batch, exp_temp, pose_temp_frozen, return_landmarks=True
                )
                
                # Scale to mm
                landmarks_3d_mm = landmarks_3d * self.flame_to_mm_scale
                trans_mm = trans_temp * self.flame_to_mm_scale
                
                proj_lms = self._project_landmarks(landmarks_3d_mm, trans_mm, focal_tensor)
                
                loss = torch.mean((proj_lms[:, :, :2] - lms_sample) ** 2)
                loss += 0.05 * torch.mean(shape_temp ** 2)
                loss += 0.01 * torch.mean(exp_temp ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            final_loss = loss.item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_focal = focal
        
        print(f"Best focal: {best_focal}, loss: {best_loss:.6f}")
        return best_focal
    
    def _project_landmarks(self, landmarks_3d, trans, focal_length):
        """
        Project 3D landmarks to 2D image space.
        
        Args:
            landmarks_3d: [N, 68, 3] - landmarks in mm units
            trans: [N, 3] - translation [tx, ty, tz] in mm units
            focal_length: scalar or [1]
        
        Returns:
            proj_landmarks: [N, 68, 3] (x, y, z)
        """
        # Apply translation
        transformed = landmarks_3d + trans.unsqueeze(1)
        
        # Perspective projection
        X = transformed[:, :, 0]
        Y = transformed[:, :, 1]
        Z = transformed[:, :, 2]
        
        # Prevent division by zero or negative Z
        Z = torch.clamp(Z, min=1e-5)
        
        cx, cy = self.cxy[0], self.cxy[1]
        
        # Standard perspective projection
        # x' = f * X / Z + cx
        # y' = f * Y / Z + cy
        proj_x = focal_length * X / Z + cx
        proj_y = focal_length * Y / Z + cy
        
        proj_landmarks = torch.stack([proj_x, proj_y, Z], dim=-1)
        
        return proj_landmarks
    
    def compute_landmark_error(self, params_dict, lms_gt):
        """
        Compute landmark reprojection error for evaluation.
        
        Args:
            params_dict: Dictionary of optimized parameters
            lms_gt: Ground truth landmarks [N, 68, 2]
        
        Returns:
            mean_error: Mean landmark distance in pixels
            per_frame_error: Per-frame errors
        """
        num_frames = lms_gt.shape[0]
        lms_tensor = torch.from_numpy(lms_gt).float().to(self.device)
        
        shape_params = params_dict['shape'].to(self.device)
        exp_params = params_dict['exp'].to(self.device)
        pose_params = params_dict['pose'].to(self.device)
        trans = params_dict['trans'].to(self.device)
        focal_length = params_dict['focal'].to(self.device)
        
        with torch.no_grad():
            shape_params_batch = shape_params.expand(num_frames, -1)
            
            # Freeze jaw pose
            pose_params_frozen = pose_params.clone()
            pose_params_frozen[:, 3:] = 0
            
            vertices, landmarks_3d = self.face_model.forward_geo(
                shape_params_batch, exp_params, pose_params_frozen, return_landmarks=True
            )
            
            # Note: trans already in mm from params_dict
            landmarks_3d_mm = landmarks_3d * self.flame_to_mm_scale
            
            proj_landmarks = self._project_landmarks(landmarks_3d_mm, trans, focal_length)
            
            # Compute error
            errors = torch.sqrt(torch.sum((proj_landmarks[:, :, :2] - lms_tensor) ** 2, dim=-1))
            per_frame_error = torch.mean(errors, dim=-1).cpu().numpy()
            mean_error = torch.mean(errors).item()
        
        return mean_error, per_frame_error
    
    def track_sequence(self, ori_imgs_dir, output_path):
        """
        Track a sequence of images and save parameters.
        
        Args:
            ori_imgs_dir: Directory containing images
            output_path: Path to save tracking parameters
        """
        # Load landmarks
        lms = load_dir(ori_imgs_dir, 0, 100000)
        num_frames = lms.shape[0]
        
        print(f"Loaded {num_frames} frames")
        
        # Load images for refinement and DECA initialization
        images = []
        for i in range(num_frames):
            img_path = os.path.join(ori_imgs_dir, f"{i}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(ori_imgs_dir, f"{i}.png")
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
            else:
                print(f"[WARNING] Image {i} not found, stopping at {len(images)} frames")
                break
        
        # Ensure we have matching number of landmarks and images
        num_valid = min(len(images), num_frames)
        images = images[:num_valid]
        lms_np = lms[:num_valid].cpu().numpy() if torch.is_tensor(lms) else lms[:num_valid]
        
        # Refine landmarks with confidence-based smoothing
        if self.use_landmark_refinement:
            print("Refining landmarks with learning-based method...")
            refined_lms = self.refine_landmarks_sequence(lms_np, images, confidence_threshold=0.7)
            refined_lms = np.array(refined_lms)
        else:
            refined_lms = lms_np
        
        # Optimize DECA parameters with proper initialization
        params_dict = self.optimize_params(refined_lms, images)
        
        # Compute final landmark error
        mean_error, per_frame_error = self.compute_landmark_error(params_dict, refined_lms)
        print(f"\n[FINAL] Mean landmark error: {mean_error:.4f} pixels")
        print(f"[FINAL] Median landmark error: {np.median(per_frame_error):.4f} pixels")
        print(f"[FINAL] Max landmark error: {np.max(per_frame_error):.4f} pixels")
        
        # Save parameters
        torch.save(params_dict, output_path)
        print(f"\nSaved tracking parameters to {output_path} (DECA model)")
        
        return params_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to ori_imgs directory")
    parser.add_argument('--img_h', type=int, default=512, help='Image height')
    parser.add_argument('--img_w', type=int, default=512, help='Image width')
    parser.add_argument('--frame_num', type=int, default=11000, help='Number of frames')
    parser.add_argument('--use_landmark_refinement', action='store_true', 
                       help='Use learning-based landmark refinement')
    parser.add_argument('--landmark_method', type=str, default='mediapipe', 
                       choices=['mediapipe', 'emoca', 'prnet'],
                       help='Landmark refinement method')
    parser.add_argument('--no_deca_init', action='store_true',
                       help='Disable DECA encoder initialization (not recommended)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set the specific GPU device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}'
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = args.device
    
    # Initialize tracker with improved settings
    tracker = DECAFaceTracker(
        img_h=args.img_h,
        img_w=args.img_w,
        device=device,
        use_landmark_refinement=args.use_landmark_refinement,
        landmark_method=args.landmark_method,
        use_temporal_smoothing=True,
        use_deca_init=not args.no_deca_init  # Enabled by default
    )
    
    # Output path
    output_path = os.path.join(os.path.dirname(args.path), 'track_params_deca.pt')
    
    # Track sequence
    tracker.track_sequence(args.path, output_path)


if __name__ == '__main__':
    main()

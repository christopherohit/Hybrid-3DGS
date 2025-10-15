"""
Enhanced face tracker with optional landmark refinement.
Works with BFM, DECA, FLAME, and other 3DMM models.
"""

import torch
import numpy as np
import os
import sys
import cv2
import argparse
import glob
from tqdm import tqdm

# Add current directory to path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from facemodel import Face_3DMM
from data_loader import load_dir
from util import *

# Import landmark refinement
try:
    from landmark_refinement import LandmarkRefiner, TemporalLandmarkSmoother
    REFINEMENT_AVAILABLE = True
except ImportError:
    REFINEMENT_AVAILABLE = False
    print("Warning: Landmark refinement not available")


def set_requires_grad(tensor_list):
    for tensor in tensor_list:
        tensor.requires_grad = True


class EnhancedFaceTracker:
    """
    Enhanced face tracker with optional learning-based landmark refinement.
    Supports BFM, DECA, FLAME, and other 3DMM models.
    """
    
    def __init__(self, 
                 modelpath,
                 img_h=512, 
                 img_w=512,
                 device='cuda',
                 use_landmark_refinement=False,
                 landmark_method='mediapipe'):
        """
        Args:
            modelpath: Path to 3DMM model
            img_h: Image height
            img_w: Image width
            device: 'cuda' or 'cpu'
            use_landmark_refinement: Whether to use learning-based refinement
            landmark_method: 'mediapipe', 'emoca', or 'prnet'
        """
        self.img_h = img_h
        self.img_w = img_w
        self.device = device
        self.use_landmark_refinement = use_landmark_refinement
        
        # 3DMM model parameters (BFM default)
        self.id_dim = 100
        self.exp_dim = 79
        self.tex_dim = 100
        self.point_num = 34650
        
        # Initialize 3DMM model
        self.model_3dmm = Face_3DMM(
            modelpath, 
            self.id_dim, 
            self.exp_dim, 
            self.tex_dim, 
            self.point_num,
            device=device
        )
        
        # Load landmark info
        lands_info = np.loadtxt(
            os.path.join(modelpath, 'lands_info.txt'), 
            dtype=np.int32
        )
        self.lands_info = torch.as_tensor(lands_info).to(device)
        
        # Camera parameters
        self.cxy = torch.tensor((img_w/2.0, img_h/2.0), dtype=torch.float).to(device)
        
        # Initialize landmark refiner (optional)
        if self.use_landmark_refinement:
            if not REFINEMENT_AVAILABLE:
                raise ImportError("Landmark refinement requested but not available")
            self.landmark_refiner = LandmarkRefiner(method=landmark_method, device=device)
            self.temporal_smoother = TemporalLandmarkSmoother(window_size=5)
            print(f"[INFO] Using BFM + {landmark_method} landmark refinement")
        else:
            self.landmark_refiner = None
            self.temporal_smoother = None
            print("[INFO] Using BFM without landmark refinement")
    
    def refine_landmarks_sequence(self, lms_list, ori_imgs_dir):
        """
        Refine landmarks for a sequence of frames.
        
        Args:
            lms_list: numpy array [N, 68, 2] of landmarks
            ori_imgs_dir: Directory containing original images
        
        Returns:
            refined_lms: numpy array [N, 68, 2] of refined landmarks
        """
        if not self.use_landmark_refinement:
            print("[INFO] Skipping landmark refinement (disabled)")
            return lms_list
        
        num_frames = lms_list.shape[0]
        refined_list = []
        
        # Load images
        print("[INFO] Loading images for refinement...")
        images = []
        for i in range(num_frames):
            img_path = os.path.join(ori_imgs_dir, f"{i}.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
            else:
                break
        
        print(f"[INFO] Refining {len(images)} frames...")
        for i, (lms, img) in enumerate(tqdm(zip(lms_list, images), desc="Refining landmarks", total=len(images))):
            # Refine landmarks
            refined_lms, confidence = self.landmark_refiner.refine_landmarks(img, lms)
            
            # Apply temporal smoothing
            if self.temporal_smoother is not None:
                refined_lms = self.temporal_smoother.smooth(refined_lms)
            
            refined_list.append(refined_lms)
        
        return np.array(refined_list)
    
    def optimize_focal_length(self, lms, id_para, exp_para, euler_angle, trans):
        """
        Estimate optimal focal length by grid search.
        """
        sel_ids = np.arange(0, lms.shape[0], 10)
        sel_num = sel_ids.shape[0]
        
        arg_focal = 1150
        arg_landis = 1e5
        
        print("[INFO] Estimating focal length...")
        for focal in tqdm(range(500, 1500, 50), desc="Focal length search"):
            id_para_temp = lms.new_zeros((1, self.id_dim), requires_grad=True)
            exp_para_temp = lms.new_zeros((sel_num, self.exp_dim), requires_grad=True)
            euler_angle_temp = lms.new_zeros((sel_num, 3), requires_grad=True)
            trans_temp = lms.new_zeros((sel_num, 3), requires_grad=True)
            trans_temp.data[:, 2] -= 600
            focal_length = lms.new_zeros(1, requires_grad=False)
            focal_length.data += focal
            
            set_requires_grad([id_para_temp, exp_para_temp, euler_angle_temp, trans_temp])
            
            optimizer_id = torch.optim.Adam([id_para_temp], lr=.3)
            optimizer_exp = torch.optim.Adam([exp_para_temp], lr=.3)
            optimizer_frame = torch.optim.Adam([euler_angle_temp, trans_temp], lr=.3)
            
            iter_num = 2000
            for iter in range(iter_num):
                id_para_batch = id_para_temp.expand(sel_num, -1)
                geometry = self.model_3dmm.forward_geo_sub(
                    id_para_batch, exp_para_temp, self.lands_info[-51:].long())
                proj_geo = forward_transform(
                    geometry, euler_angle_temp, trans_temp, focal_length, self.cxy)
                loss_lan = cal_lan_loss(
                    proj_geo[:, :, :2], lms[sel_ids, -51:, :].detach())
                loss_regid = torch.mean(id_para_temp*id_para_temp)*8
                loss_regexp = torch.mean(exp_para_temp*exp_para_temp)*0.5
                loss = loss_lan + loss_regid + loss_regexp
                
                optimizer_id.zero_grad()
                optimizer_exp.zero_grad()
                optimizer_frame.zero_grad()
                loss.backward()
                
                if iter > 1000:
                    optimizer_id.step()
                    optimizer_exp.step()
                optimizer_frame.step()
            
            if loss_lan.item() < arg_landis:
                arg_landis = loss_lan.item()
                arg_focal = focal
        
        print(f"[INFO] Optimal focal length: {arg_focal}")
        return arg_focal
    
    def track_sequence(self, ori_imgs_dir, output_path):
        """
        Track face sequence with optional landmark refinement.
        
        Args:
            ori_imgs_dir: Directory containing images and landmarks
            output_path: Path to save tracking parameters
        """
        print("[INFO] Loading landmarks...")
        start_id = 0
        end_id = 100000
        
        lms = load_dir(ori_imgs_dir, start_id, end_id)
        lms = lms.to(self.device)
        num_frames = lms.shape[0]
        
        print(f"[INFO] Loaded {num_frames} frames")
        
        # Apply landmark refinement if enabled
        if self.use_landmark_refinement:
            lms_np = lms.cpu().numpy()
            refined_lms_np = self.refine_landmarks_sequence(lms_np, ori_imgs_dir)
            lms = torch.from_numpy(refined_lms_np).to(self.device)
            print(f"[INFO] Landmark refinement completed")
        
        # Initialize parameters
        id_para = lms.new_zeros((1, self.id_dim), requires_grad=True)
        exp_para = lms.new_zeros((num_frames, self.exp_dim), requires_grad=True)
        euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
        trans = lms.new_zeros((num_frames, 3), requires_grad=True)
        trans.data[:, 2] -= 600
        
        # Estimate focal length
        arg_focal = self.optimize_focal_length(lms, id_para, exp_para, euler_angle, trans)
        
        # Full optimization with optimal focal length
        focal_length = lms.new_zeros(1, requires_grad=False)
        focal_length.data += arg_focal
        
        set_requires_grad([id_para, exp_para, euler_angle, trans])
        
        optimizer_id = torch.optim.Adam([id_para], lr=.3)
        optimizer_exp = torch.optim.Adam([exp_para], lr=.3)
        optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=.3)
        
        iter_num = 2000
        print("[INFO] Optimizing 3DMM parameters...")
        
        for iter in tqdm(range(iter_num), desc="Optimization"):
            id_para_batch = id_para.expand(num_frames, -1)
            geometry = self.model_3dmm.forward_geo_sub(
                id_para_batch, exp_para, self.lands_info[-51:].long())
            proj_geo = forward_transform(
                geometry, euler_angle, trans, focal_length, self.cxy)
            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], lms[:, -51:, :].detach())
            loss_regid = torch.mean(id_para*id_para)*8
            loss_regexp = torch.mean(exp_para*exp_para)*0.5
            loss = loss_lan + loss_regid + loss_regexp
            
            optimizer_id.zero_grad()
            optimizer_exp.zero_grad()
            optimizer_frame.zero_grad()
            loss.backward()
            
            if iter > 1000:
                optimizer_id.step()
                optimizer_exp.step()
            optimizer_frame.step()
        
        print(f"[INFO] Final loss: {loss_lan.item():.6f}")
        
        # Save parameters
        torch.save({
            'id': id_para.detach().cpu(),
            'exp': exp_para.detach().cpu(),
            'euler': euler_angle.detach().cpu(),
            'trans': trans.detach().cpu(),
            'focal': focal_length.detach().cpu(),
            'face_model_type': 'BFM'
        }, output_path)
        
        print(f"[INFO] Saved tracking parameters to {output_path} (BFM model)")


def main():
    parser = argparse.ArgumentParser(description='Enhanced face tracker with optional landmark refinement')
    parser.add_argument("--path", type=str, required=True, help="Path to ori_imgs directory")
    parser.add_argument('--img_h', type=int, default=512, help='Image height')
    parser.add_argument('--img_w', type=int, default=512, help='Image width')
    parser.add_argument('--frame_num', type=int, default=11000, help='Number of frames')
    parser.add_argument('--use_landmark_refinement', action='store_true',
                       help='Use learning-based landmark refinement')
    parser.add_argument('--landmark_method', type=str, default='mediapipe',
                       choices=['mediapipe', 'emoca', 'prnet'],
                       help='Landmark refinement method')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}'
        print(f"[INFO] Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = 'cpu'
        print("[INFO] CUDA not available, using CPU")
    
    # Initialize tracker
    modelpath = os.path.join(dir_path, '3DMM')
    tracker = EnhancedFaceTracker(
        modelpath=modelpath,
        img_h=args.img_h,
        img_w=args.img_w,
        device=device,
        use_landmark_refinement=args.use_landmark_refinement,
        landmark_method=args.landmark_method
    )
    
    # Output path
    output_path = os.path.join(os.path.dirname(args.path), 'track_params.pt')
    
    # Track sequence
    tracker.track_sequence(args.path, output_path)


if __name__ == '__main__':
    main()


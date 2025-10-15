"""
Utility script to compare BFM and DECA face tracking results.
Provides comprehensive metrics including:
- Landmark reprojection error
- Parameter statistics
- Temporal smoothness
- Visual comparison
"""

import torch
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import face models
from facemodel import Face_3DMM as Face_3DMM_BFM
from facemodel_deca import Face_3DMM_DECA
from data_loader import load_dir
from util import *


class TrackerComparator:
    """
    Compare BFM and DECA tracking results.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize both face models
        dir_path = os.path.dirname(os.path.realpath(__file__))
        
        # BFM model
        try:
            self.bfm_model = Face_3DMM_BFM(
                os.path.join(dir_path, '3DMM'),
                id_dim=100, exp_dim=79, tex_dim=100, 
                point_num=34650, device=device
            )
            self.bfm_available = True
        except:
            print("[WARNING] BFM model not available")
            self.bfm_available = False
        
        # DECA model
        try:
            self.deca_model = Face_3DMM_DECA(device=device)
            self.deca_available = True
        except:
            print("[WARNING] DECA model not available")
            self.deca_available = False
        
        # Landmark indices for BFM
        try:
            lands_info = np.loadtxt(os.path.join(dir_path, '3DMM', 'lands_info.txt'), dtype=np.int32)
            self.bfm_lands_info = torch.as_tensor(lands_info[-51:]).to(device)
        except:
            self.bfm_lands_info = None
    
    def load_tracking_params(self, param_path):
        """
        Load tracking parameters from file.
        """
        params = torch.load(param_path, map_location=self.device)
        return params
    
    def compute_landmark_error_bfm(self, params, lms_gt, img_h=512, img_w=512):
        """
        Compute landmark error for BFM tracking.
        
        Args:
            params: BFM tracking parameters
            lms_gt: Ground truth landmarks [N, 68, 2]
        
        Returns:
            mean_error: Mean landmark error in pixels
            per_frame_error: Per-frame errors [N]
        """
        if not self.bfm_available or self.bfm_lands_info is None:
            return None, None
        
        num_frames = lms_gt.shape[0]
        lms_tensor = torch.from_numpy(lms_gt).float().to(self.device)
        
        id_para = params['id'].to(self.device)
        exp_para = params['exp'][:num_frames].to(self.device)
        euler = params['euler'][:num_frames].to(self.device)
        trans = params['trans'][:num_frames].to(self.device)
        focal = params['focal'].to(self.device)
        
        cxy = torch.tensor((img_w/2.0, img_h/2.0), dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            id_para_batch = id_para.expand(num_frames, -1)
            
            # Generate geometry
            geometry = self.bfm_model.forward_geo_sub(
                id_para_batch, exp_para, self.bfm_lands_info.long()
            )
            
            # Project to 2D
            proj_geo = forward_transform(geometry, euler, trans, focal, cxy)
            
            # Compute error (BFM uses 51 landmarks, compare with last 51 of 68)
            errors = torch.sqrt(torch.sum((proj_geo[:, :, :2] - lms_tensor[:, -51:, :]) ** 2, dim=-1))
            per_frame_error = torch.mean(errors, dim=-1).cpu().numpy()
            mean_error = torch.mean(errors).item()
        
        return mean_error, per_frame_error
    
    def compute_landmark_error_deca(self, params, lms_gt, img_h=512, img_w=512):
        """
        Compute landmark error for DECA tracking.
        
        Args:
            params: DECA tracking parameters
            lms_gt: Ground truth landmarks [N, 68, 2]
        
        Returns:
            mean_error: Mean landmark error in pixels
            per_frame_error: Per-frame errors [N]
        """
        if not self.deca_available:
            return None, None
        
        num_frames = lms_gt.shape[0]
        lms_tensor = torch.from_numpy(lms_gt).float().to(self.device)
        
        shape_params = params['shape'].to(self.device)
        exp_params = params['exp'][:num_frames].to(self.device)
        pose_params = params['pose'][:num_frames].to(self.device)
        trans = params['trans'][:num_frames].to(self.device)
        focal = params['focal'].to(self.device)
        
        cxy = torch.tensor((img_w/2.0, img_h/2.0), dtype=torch.float).to(self.device)
        
        # Get FLAME scale if available
        flame_scale = params.get('flame_scale', 100.0)
        
        with torch.no_grad():
            shape_params_batch = shape_params.expand(num_frames, -1)
            
            # Freeze jaw pose
            pose_params_frozen = pose_params.clone()
            pose_params_frozen[:, 3:] = 0
            
            # Generate geometry
            vertices, landmarks_3d = self.deca_model.forward_geo(
                shape_params_batch, exp_params, pose_params_frozen, return_landmarks=True
            )
            
            # Scale to mm
            landmarks_3d_mm = landmarks_3d * flame_scale
            
            # Project to 2D
            proj_landmarks = self._project_landmarks_deca(landmarks_3d_mm, trans, focal, cxy)
            
            # Compute error (DECA uses all 68 landmarks)
            errors = torch.sqrt(torch.sum((proj_landmarks[:, :, :2] - lms_tensor) ** 2, dim=-1))
            per_frame_error = torch.mean(errors, dim=-1).cpu().numpy()
            mean_error = torch.mean(errors).item()
        
        return mean_error, per_frame_error
    
    def _project_landmarks_deca(self, landmarks_3d, trans, focal_length, cxy):
        """Helper function for DECA landmark projection."""
        transformed = landmarks_3d + trans.unsqueeze(1)
        
        X = transformed[:, :, 0]
        Y = transformed[:, :, 1]
        Z = transformed[:, :, 2]
        Z = torch.clamp(Z, min=1e-5)
        
        proj_x = focal_length * X / Z + cxy[0]
        proj_y = focal_length * Y / Z + cxy[1]
        
        return torch.stack([proj_x, proj_y, Z], dim=-1)
    
    def compute_temporal_smoothness(self, params):
        """
        Compute temporal smoothness of expression parameters.
        Lower is smoother.
        """
        if 'exp' in params:
            exp_params = params['exp']
            if len(exp_params) > 1:
                diff = exp_params[1:] - exp_params[:-1]
                smoothness = torch.mean(torch.abs(diff)).item()
                return smoothness
        return None
    
    def compare_tracking(self, bfm_params_path, deca_params_path, lms_path, img_h=512, img_w=512):
        """
        Compare BFM and DECA tracking results.
        
        Args:
            bfm_params_path: Path to BFM tracking parameters
            deca_params_path: Path to DECA tracking parameters
            lms_path: Path to directory with landmarks (ori_imgs)
            img_h: Image height
            img_w: Image width
        
        Returns:
            results: Dictionary with comparison metrics
        """
        print("=" * 80)
        print("TRACKING COMPARISON: BFM vs DECA")
        print("=" * 80)
        
        # Load landmarks
        lms = load_dir(lms_path, 0, 100000)
        lms_np = lms.cpu().numpy() if torch.is_tensor(lms) else lms
        num_frames = lms_np.shape[0]
        print(f"\nLoaded {num_frames} frames of landmarks")
        
        results = {}
        
        # Load and evaluate BFM
        if os.path.exists(bfm_params_path):
            print("\n--- BFM Tracker ---")
            bfm_params = self.load_tracking_params(bfm_params_path)
            
            bfm_mean_error, bfm_per_frame = self.compute_landmark_error_bfm(
                bfm_params, lms_np, img_h, img_w
            )
            
            bfm_smoothness = self.compute_temporal_smoothness(bfm_params)
            
            if bfm_mean_error is not None:
                print(f"Mean landmark error: {bfm_mean_error:.4f} pixels")
                print(f"Median landmark error: {np.median(bfm_per_frame):.4f} pixels")
                print(f"Max landmark error: {np.max(bfm_per_frame):.4f} pixels")
                print(f"Temporal smoothness: {bfm_smoothness:.6f}")
                
                results['bfm'] = {
                    'mean_error': bfm_mean_error,
                    'median_error': np.median(bfm_per_frame),
                    'max_error': np.max(bfm_per_frame),
                    'smoothness': bfm_smoothness,
                    'per_frame_error': bfm_per_frame,
                    'num_params': bfm_params['id'].shape[1] + bfm_params['exp'].shape[1]
                }
        else:
            print(f"\n[WARNING] BFM params not found: {bfm_params_path}")
        
        # Load and evaluate DECA
        if os.path.exists(deca_params_path):
            print("\n--- DECA Tracker (Improved) ---")
            deca_params = self.load_tracking_params(deca_params_path)
            
            deca_mean_error, deca_per_frame = self.compute_landmark_error_deca(
                deca_params, lms_np, img_h, img_w
            )
            
            deca_smoothness = self.compute_temporal_smoothness(deca_params)
            
            if deca_mean_error is not None:
                print(f"Mean landmark error: {deca_mean_error:.4f} pixels")
                print(f"Median landmark error: {np.median(deca_per_frame):.4f} pixels")
                print(f"Max landmark error: {np.max(deca_per_frame):.4f} pixels")
                print(f"Temporal smoothness: {deca_smoothness:.6f}")
                print(f"FLAME scale: {deca_params.get('flame_scale', 'N/A')}")
                
                results['deca'] = {
                    'mean_error': deca_mean_error,
                    'median_error': np.median(deca_per_frame),
                    'max_error': np.max(deca_per_frame),
                    'smoothness': deca_smoothness,
                    'per_frame_error': deca_per_frame,
                    'num_params': deca_params['shape'].shape[1] + deca_params['exp'].shape[1]
                }
        else:
            print(f"\n[WARNING] DECA params not found: {deca_params_path}")
        
        # Summary comparison
        if 'bfm' in results and 'deca' in results:
            print("\n" + "=" * 80)
            print("SUMMARY COMPARISON")
            print("=" * 80)
            print(f"{'Metric':<30} {'BFM':<20} {'DECA':<20} {'Winner'}")
            print("-" * 80)
            
            bfm_mean = results['bfm']['mean_error']
            deca_mean = results['deca']['mean_error']
            winner = 'BFM' if bfm_mean < deca_mean else 'DECA'
            diff_pct = abs(bfm_mean - deca_mean) / bfm_mean * 100
            print(f"{'Mean Landmark Error (px)':<30} {bfm_mean:<20.4f} {deca_mean:<20.4f} {winner} ({diff_pct:.1f}% diff)")
            
            bfm_med = results['bfm']['median_error']
            deca_med = results['deca']['median_error']
            winner = 'BFM' if bfm_med < deca_med else 'DECA'
            print(f"{'Median Landmark Error (px)':<30} {bfm_med:<20.4f} {deca_med:<20.4f} {winner}")
            
            bfm_smooth = results['bfm']['smoothness']
            deca_smooth = results['deca']['smoothness']
            winner = 'BFM' if bfm_smooth < deca_smooth else 'DECA'
            print(f"{'Temporal Smoothness':<30} {bfm_smooth:<20.6f} {deca_smooth:<20.6f} {winner}")
            
            print(f"{'Num Parameters':<30} {results['bfm']['num_params']:<20} {results['deca']['num_params']:<20}")
            
            print("=" * 80)
        
        return results
    
    def plot_comparison(self, results, output_path='comparison.png'):
        """
        Plot comparison between BFM and DECA.
        """
        if 'bfm' not in results or 'deca' not in results:
            print("[WARNING] Cannot plot comparison, missing data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Per-frame landmark error
        ax = axes[0, 0]
        frames = np.arange(len(results['bfm']['per_frame_error']))
        ax.plot(frames, results['bfm']['per_frame_error'], label='BFM', alpha=0.7)
        ax.plot(frames, results['deca']['per_frame_error'], label='DECA', alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Landmark Error (pixels)')
        ax.set_title('Per-Frame Landmark Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax = axes[0, 1]
        ax.hist(results['bfm']['per_frame_error'], bins=50, alpha=0.5, label='BFM', density=True)
        ax.hist(results['deca']['per_frame_error'], bins=50, alpha=0.5, label='DECA', density=True)
        ax.set_xlabel('Landmark Error (pixels)')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Bar chart comparison
        ax = axes[1, 0]
        metrics = ['Mean Error', 'Median Error', 'Max Error']
        bfm_values = [results['bfm']['mean_error'], results['bfm']['median_error'], results['bfm']['max_error']]
        deca_values = [results['deca']['mean_error'], results['deca']['median_error'], results['deca']['max_error']]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, bfm_values, width, label='BFM', alpha=0.7)
        ax.bar(x + width/2, deca_values, width, label='DECA', alpha=0.7)
        ax.set_ylabel('Error (pixels)')
        ax.set_title('Landmark Error Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        TRACKING COMPARISON SUMMARY
        
        BFM Tracker:
        - Mean Error: {results['bfm']['mean_error']:.4f} px
        - Median Error: {results['bfm']['median_error']:.4f} px
        - Smoothness: {results['bfm']['smoothness']:.6f}
        - Parameters: {results['bfm']['num_params']}
        
        DECA Tracker:
        - Mean Error: {results['deca']['mean_error']:.4f} px
        - Median Error: {results['deca']['median_error']:.4f} px
        - Smoothness: {results['deca']['smoothness']:.6f}
        - Parameters: {results['deca']['num_params']}
        
        Improvement:
        - Error diff: {abs(results['bfm']['mean_error'] - results['deca']['mean_error']):.4f} px
        - Better: {'DECA' if results['deca']['mean_error'] < results['bfm']['mean_error'] else 'BFM'}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare BFM and DECA tracking results')
    parser.add_argument('--bfm_params', type=str, required=True, help='Path to BFM tracking parameters')
    parser.add_argument('--deca_params', type=str, required=True, help='Path to DECA tracking parameters')
    parser.add_argument('--lms_path', type=str, required=True, help='Path to ori_imgs directory with landmarks')
    parser.add_argument('--img_h', type=int, default=512, help='Image height')
    parser.add_argument('--img_w', type=int, default=512, help='Image width')
    parser.add_argument('--output_plot', type=str, default='tracking_comparison.png', help='Output plot path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}'
    else:
        device = args.device
    
    # Initialize comparator
    comparator = TrackerComparator(device=device)
    
    # Compare tracking
    results = comparator.compare_tracking(
        args.bfm_params,
        args.deca_params,
        args.lms_path,
        args.img_h,
        args.img_w
    )
    
    # Plot comparison
    if results:
        comparator.plot_comparison(results, args.output_plot)


if __name__ == '__main__':
    main()


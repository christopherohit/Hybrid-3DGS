#!/usr/bin/env python3
"""
Practical script to run improved DECA face tracking and compare with BFM.
This demonstrates the key improvements:
1. DECA encoder initialization
2. Proper FLAME scale handling
3. Strong regularization
4. Frozen jaw pose
5. Better focal estimation
"""

import os
import sys
import argparse
import torch

from face_tracker_deca import DECAFaceTracker
from compare_trackers import TrackerComparator


def run_improved_deca_tracking(data_path, output_dir, use_deca_init=True, 
                                 use_landmark_refinement=False, gpu_id=0):
    """
    Run improved DECA tracking on a video sequence.
    
    Args:
        data_path: Path to ori_imgs directory
        output_dir: Directory to save results
        use_deca_init: Use DECA encoder for initialization (recommended)
        use_landmark_refinement: Use MediaPipe/EMOCA for landmark refinement
        gpu_id: GPU ID to use
    """
    print("=" * 80)
    print("RUNNING IMPROVED DECA FACE TRACKING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data path: {data_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  DECA initialization: {use_deca_init}")
    print(f"  Landmark refinement: {use_landmark_refinement}")
    print(f"  GPU ID: {gpu_id}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if torch.cuda.is_available():
        device = f'cuda:{gpu_id}'
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}\n")
    else:
        device = 'cpu'
        print("Using CPU\n")
    
    # Initialize improved DECA tracker
    tracker = DECAFaceTracker(
        img_h=512,
        img_w=512,
        device=device,
        use_landmark_refinement=use_landmark_refinement,
        landmark_method='mediapipe',
        use_temporal_smoothing=True,
        use_deca_init=use_deca_init
    )
    
    # Output path
    output_path = os.path.join(output_dir, 'track_params_deca.pt')
    
    print("Starting face tracking...")
    print("-" * 80)
    
    # Run tracking
    params = tracker.track_sequence(data_path, output_path)
    
    print("\n" + "=" * 80)
    print("TRACKING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    
    return output_path


def compare_with_bfm(data_path, deca_params_path, bfm_params_path=None, gpu_id=0):
    """
    Compare DECA tracking results with BFM.
    
    Args:
        data_path: Path to ori_imgs directory  
        deca_params_path: Path to DECA tracking parameters
        bfm_params_path: Path to BFM tracking parameters (optional)
        gpu_id: GPU ID to use
    """
    # Auto-detect BFM params if not provided
    if bfm_params_path is None:
        data_dir = os.path.dirname(data_path)
        bfm_params_path = os.path.join(data_dir, 'track_params.pt')
    
    if not os.path.exists(bfm_params_path):
        print(f"\n[WARNING] BFM parameters not found at: {bfm_params_path}")
        print("Skipping comparison. Run BFM tracking first if you want to compare.")
        return
    
    print("\n" + "=" * 80)
    print("COMPARING WITH BFM TRACKER")
    print("=" * 80)
    
    # Set device
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # Initialize comparator
    comparator = TrackerComparator(device=device)
    
    # Run comparison
    results = comparator.compare_tracking(
        bfm_params_path=bfm_params_path,
        deca_params_path=deca_params_path,
        lms_path=data_path,
        img_h=512,
        img_w=512
    )
    
    # Generate comparison plot
    if results and 'bfm' in results and 'deca' in results:
        output_plot = os.path.join(os.path.dirname(deca_params_path), 'tracking_comparison.png')
        comparator.plot_comparison(results, output_plot)


def main():
    parser = argparse.ArgumentParser(
        description='Run improved DECA face tracking and optionally compare with BFM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run improved DECA tracking with all enhancements
  python run_improved_tracking.py --path data/Obama/ori_imgs
  
  # Run with landmark refinement
  python run_improved_tracking.py --path data/Obama/ori_imgs --use_landmark_refinement
  
  # Run without DECA init (not recommended, for testing)
  python run_improved_tracking.py --path data/Obama/ori_imgs --no_deca_init
  
  # Run and compare with existing BFM results
  python run_improved_tracking.py --path data/Obama/ori_imgs --compare_bfm
        """
    )
    
    parser.add_argument('--path', type=str, required=True,
                       help='Path to ori_imgs directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as input)')
    parser.add_argument('--use_landmark_refinement', action='store_true',
                       help='Use MediaPipe landmark refinement')
    parser.add_argument('--no_deca_init', action='store_true',
                       help='Disable DECA encoder initialization (not recommended)')
    parser.add_argument('--compare_bfm', action='store_true',
                       help='Compare results with BFM tracking')
    parser.add_argument('--bfm_params', type=str, default=None,
                       help='Path to BFM tracking parameters (auto-detected if not provided)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.path)
    
    # Run improved DECA tracking
    deca_params_path = run_improved_deca_tracking(
        data_path=args.path,
        output_dir=args.output_dir,
        use_deca_init=not args.no_deca_init,
        use_landmark_refinement=args.use_landmark_refinement,
        gpu_id=args.gpu_id
    )
    
    # Compare with BFM if requested
    if args.compare_bfm:
        compare_with_bfm(
            data_path=args.path,
            deca_params_path=deca_params_path,
            bfm_params_path=args.bfm_params,
            gpu_id=args.gpu_id
        )
    
    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print(f"\nDECA tracking parameters: {deca_params_path}")
    if args.compare_bfm:
        print(f"Comparison plot: {os.path.join(args.output_dir, 'tracking_comparison.png')}")
    print()


if __name__ == '__main__':
    main()


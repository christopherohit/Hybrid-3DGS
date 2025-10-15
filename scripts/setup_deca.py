"""
Setup script for DECA (Dense Expression Capture Model)
This script downloads and configures DECA for use with SyncTalk
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path


def print_status(message, status="info"):
    """Print colored status messages."""
    colors = {
        "info": "\033[94m",
        "success": "\033[92m",
        "warning": "\033[93m",
        "error": "\033[91m",
        "reset": "\033[0m"
    }
    
    color = colors.get(status, colors["info"])
    reset = colors["reset"]
    
    print(f"{color}[{status.upper()}] {message}{reset}")


def run_command(cmd, description=""):
    """Run a shell command and check for errors."""
    if description:
        print_status(f"{description}...", "info")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def download_file(url, destination, description=""):
    """Download a file with progress."""
    if description:
        print_status(f"Downloading {description}...", "info")
    
    try:
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as e:
        print_status(f"Failed to download: {e}", "error")
        return False


def main():
    print_status("Starting DECA setup for SyncTalk", "info")
    print("=" * 60)
    print()
    print_status("Make sure you have activated the synctalk conda environment:", "warning")
    print("    conda activate synctalk")
    print()
    
    # Check if in correct environment
    import sys
    if 'synctalk' not in sys.prefix.lower() and 'synctalk' not in sys.executable.lower():
        print_status("Warning: You may not be in the synctalk conda environment!", "warning")
        print_status(f"Current Python: {sys.executable}", "info")
        print()
    else:
        print_status("✓ Running in synctalk environment", "success")
        print()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    deca_dir = project_root / "data_utils" / "face_tracking" / "deca_model"
    
    # Create directories
    deca_dir.mkdir(parents=True, exist_ok=True)
    (deca_dir / "data").mkdir(exist_ok=True)
    
    # Step 1: Install dependencies
    print_status("Step 1/6: Installing dependencies", "info")
    deps = ["chumpy", "pyyaml", "yacs", "kornia", "roma", "scikit-image", "cython", "gdown"]
    for dep in deps:
        success, _ = run_command(f"pip install {dep}", f"Installing {dep}")
        if not success:
            print_status(f"Warning: Failed to install {dep}", "warning")
    
    # Step 2: Clone DECA repository
    print_status("Step 2/6: Cloning DECA repository", "info")
    face_tracking_dir = project_root / "data_utils" / "face_tracking"
    deca_repo_dir = face_tracking_dir / "DECA"
    
    if not deca_repo_dir.exists():
        success, _ = run_command(
            f"cd {face_tracking_dir} && git clone https://github.com/yfeng95/DECA.git",
            "Cloning DECA"
        )
        if success:
            print_status("DECA repository cloned successfully", "success")
        else:
            print_status("Failed to clone DECA", "error")
    else:
        print_status("DECA repository already exists", "success")
    
    # Step 3: Setup FLAME model (manual download required)
    print_status("Step 3/6: Setting up FLAME model", "info")
    data_dir = deca_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print_status(
        "FLAME model requires manual download from https://flame.is.tue.mpg.de/",
        "warning"
    )
    print_status(
        "After registration, download 'FLAME 2020' and place these files in: " + str(data_dir),
        "info"
    )
    print("  Required files:")
    print("  - generic_model.pkl")
    print("  - FLAME_texture.npz")
    print()
    
    if (data_dir / "generic_model.pkl").exists():
        print_status("FLAME model files found", "success")
    else:
        print_status("FLAME model files not found (will need manual download)", "warning")
    
    # Step 4: Download DECA pretrained weights
    print_status("Step 4/6: Downloading DECA pretrained weights", "info")
    
    deca_model_tar = deca_dir / "deca_model.tar"
    
    if not (deca_dir / "deca_model.pth").exists():
        print_status(
            "Downloading DECA model from Google Drive (may take a few minutes)...",
            "info"
        )
        
        # Try using gdown
        success, _ = run_command(
            f"pip install gdown && gdown 1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje -O {deca_model_tar}",
            "Downloading DECA weights"
        )
        
        if success and deca_model_tar.exists():
            # Extract tar file
            with tarfile.open(deca_model_tar, 'r') as tar:
                tar.extractall(deca_dir)
            deca_model_tar.unlink()
            print_status("DECA weights downloaded successfully", "success")
        else:
            print_status("Could not download DECA weights automatically", "warning")
            print_status(
                "Please download manually from: https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje",
                "info"
            )
    else:
        print_status("DECA model already exists", "success")
    
    # Step 5: Download head template
    print_status("Step 5/6: Downloading head template", "info")
    
    template_path = data_dir / "head_template.obj"
    
    if not template_path.exists():
        template_url = "https://raw.githubusercontent.com/yfeng95/DECA/master/data/head_template.obj"
        if download_file(template_url, template_path, "head template"):
            print_status("Head template downloaded successfully", "success")
        else:
            print_status("Failed to download head template", "warning")
    else:
        print_status("Head template already exists", "success")
    
    # Step 6: Verify installation
    print_status("Step 6/6: Verifying DECA installation", "info")
    
    verify_script = f"""
import sys
import os
sys.path.insert(0, '{deca_repo_dir}')
try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    
    try:
        from decalib.deca import DECA
        print("✓ DECA repository available")
    except ImportError as e:
        print("✗ DECA import failed:", e)
        sys.exit(1)
    
    try:
        import chumpy
        print("✓ Chumpy installed")
    except ImportError:
        print("⚠ Chumpy not found - install: pip install chumpy")
    
    try:
        import kornia
        print("✓ Kornia installed")
    except ImportError:
        print("⚠ Kornia not found - install: pip install kornia")
    
    print("\\n✓ DECA setup verification passed!")
    
except Exception as e:
    print(f"✗ Verification failed: {{e}}")
    sys.exit(1)
"""
    
    success, output = run_command(f'python -c "{verify_script}"', "Verifying installation")
    
    if success:
        print("\n" + "=" * 60)
        print_status("DECA setup completed successfully!", "success")
        print()
        print_status("IMPORTANT: Manual step required!", "warning")
        print("Download FLAME model from: https://flame.is.tue.mpg.de/")
        print("(Requires registration)")
        print()
        print(f"Place these files in: {data_dir}/")
        print("  - generic_model.pkl")
        print("  - FLAME_texture.npz")
        print()
        print("After completing the manual step:")
        print("1. Process your video with DECA:")
        print("   python data_utils/process.py data/<ID>/<ID>.mp4 --use_deca --landmark_method mediapipe")
        print()
        print("2. Train with DECA features:")
        print("   python main.py data/<ID> --workspace model/trial_deca -O --iters 60000 --asr_model ave")
    else:
        print_status("Installation verification failed", "error")
        print("\nPlease check the error messages above and:")
        print("1. Ensure PyTorch is installed: pip install torch torchvision")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Make sure you're in the synctalk conda environment: conda activate synctalk")
    
    print("=" * 60)


if __name__ == "__main__":
    main()


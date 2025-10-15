"""
Learning-based landmark refinement module.
Supports multiple methods: MediaPipe, EMOCA, PRNet
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from typing import Optional, Union, Tuple

# MediaPipe for fast landmark detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"Warning: MediaPipe not available: {e}")

# EMOCA for emotion-aware landmark refinement
try:
    from gdl.models.DECA import ExpDECA as EMOCA
    EMOCA_AVAILABLE = True
except ImportError:
    EMOCA_AVAILABLE = False
    print("Warning: EMOCA (ExpDECA) not available. Make sure gdl is installed.")


class LandmarkRefiner:
    """
    Base class for landmark refinement methods.
    """
    
    def __init__(self, method='mediapipe', device='cuda'):
        """
        Args:
            method: 'mediapipe', 'emoca', or 'prnet'
            device: 'cuda' or 'cpu'
        """
        self.method = method
        self.device = device
        
        if method == 'mediapipe':
            self.refiner = MediaPipeLandmarkRefiner(device)
        elif method == 'emoca':
            self.refiner = EMOCALandmarkRefiner(device)
        elif method == 'prnet':
            self.refiner = PRNetLandmarkRefiner(device)
        else:
            raise ValueError(f"Unknown refinement method: {method}")
    
    def refine_landmarks(self, image, initial_landmarks):
        """
        Refine landmarks using the selected method.
        
        Args:
            image: numpy array [H, W, 3] or torch.Tensor
            initial_landmarks: numpy array [68, 2] or torch.Tensor
        
        Returns:
            refined_landmarks: numpy array [68, 2] refined landmarks
            confidence: float, confidence score
        """
        return self.refiner.refine_landmarks(image, initial_landmarks)
    
    def batch_refine(self, images, initial_landmarks_list):
        """
        Batch refinement for multiple images.
        
        Args:
            images: list of images
            initial_landmarks_list: list of landmark arrays
        
        Returns:
            refined_landmarks_list: list of refined landmark arrays
            confidences: list of confidence scores
        """
        refined_list = []
        confidence_list = []
        
        for img, lms in zip(images, initial_landmarks_list):
            refined, conf = self.refine_landmarks(img, lms)
            refined_list.append(refined)
            confidence_list.append(conf)
        
        return refined_list, confidence_list


class MediaPipeLandmarkRefiner:
    """
    MediaPipe-based landmark refinement.
    Fast and accurate for real-time applications.
    """
    
    def __init__(self, device='cuda'):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available")
        
        self.device = device
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe 468 landmarks to 68 landmarks mapping
        self.landmark_mapping = self._get_68_landmark_indices()
    
    def _get_68_landmark_indices(self):
        """
        Map MediaPipe 468 landmarks to standard 68 facial landmarks.
        """
        # Mapping based on MediaPipe documentation
        mapping = [
            # Jaw (0-16)
            152, 234, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
            # Left eyebrow (17-21)
            107, 66, 105, 63, 70,
            # Right eyebrow (22-26)
            336, 296, 334, 293, 300,
            # Nose (27-35)
            168, 6, 197, 195, 5, 4, 1, 19, 94,
            # Left eye (36-41)
            33, 160, 158, 133, 153, 144,
            # Right eye (42-47)
            263, 387, 385, 362, 380, 373,
            # Mouth outer (48-59)
            61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
            # Mouth inner (60-67)
            78, 81, 13, 311, 308, 402, 14, 178
        ]
        
        return mapping[:68]  # Ensure we have exactly 68 landmarks
    
    def refine_landmarks(self, image, initial_landmarks):
        """
        Refine landmarks using MediaPipe.
        
        Args:
            image: numpy array [H, W, 3] RGB image
            initial_landmarks: numpy array [68, 2]
        
        Returns:
            refined_landmarks: numpy array [68, 2]
            confidence: float
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Process image with MediaPipe
        results = self.face_mesh.process(image)
        
        if not results.multi_face_landmarks:
            # No face detected, return initial landmarks
            return initial_landmarks, 0.0
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Extract 68 landmarks from 468
        refined_landmarks = np.zeros((68, 2), dtype=np.float32)
        
        for i, idx in enumerate(self.landmark_mapping):
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                refined_landmarks[i, 0] = lm.x * w
                refined_landmarks[i, 1] = lm.y * h
        
        # Blend with initial landmarks for robustness
        alpha = 0.7  # Weight for MediaPipe landmarks
        blended_landmarks = alpha * refined_landmarks + (1 - alpha) * initial_landmarks
        
        return blended_landmarks, 0.9
    
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


class EMOCALandmarkRefiner:
    """
    EMOCA-based landmark refinement.
    Better for emotion-aware facial analysis.
    """
    
    def __init__(self, device='cuda'):
        if not EMOCA_AVAILABLE:
            print("Warning: EMOCA not available, falling back to simple refinement")
            self.model = None
        else:
            self.device = device
            # Initialize EMOCA model
            try:
                # EMOCA is already imported as ExpDECA at the top
                # Note: Proper initialization requires config - this is simplified
                # config_path = "path/to/emoca/config.yaml"
                # checkpoint_path = "path/to/emoca/checkpoint.ckpt"
                # config = OmegaConf.load(config_path)
                # self.model = load_model(checkpoint_path, config)
                print("Note: EMOCA/ExpDECA requires proper config for initialization")
                print("Using fallback refinement instead")
            except Exception as e:
                print(f"Failed to load EMOCA: {e}")
                self.model = None
    
    def refine_landmarks(self, image, initial_landmarks):
        """
        Refine landmarks using EMOCA.
        
        Args:
            image: numpy array [H, W, 3]
            initial_landmarks: numpy array [68, 2]
        
        Returns:
            refined_landmarks: numpy array [68, 2]
            confidence: float
        """
        if self.model is None:
            # Fallback: return initial landmarks
            return initial_landmarks, 0.5
        
        # Preprocess image for EMOCA
        img_tensor = self._preprocess_image(image)
        
        # Run EMOCA
        with torch.no_grad():
            output = self.model.encode(img_tensor)
        
        # Extract landmarks from EMOCA output
        if 'landmarks' in output:
            landmarks_3d = output['landmarks'].cpu().numpy()[0]
            
            # Project to 2D (EMOCA provides 68 landmarks)
            refined_landmarks = landmarks_3d[:, :2]
            
            # Denormalize to image space
            h, w = image.shape[:2]
            refined_landmarks[:, 0] = (refined_landmarks[:, 0] + 1) * w / 2
            refined_landmarks[:, 1] = (refined_landmarks[:, 1] + 1) * h / 2
            
            return refined_landmarks, 0.95
        else:
            return initial_landmarks, 0.5
    
    def _preprocess_image(self, image):
        """
        Preprocess image for EMOCA.
        """
        # Resize to 224x224
        img = cv2.resize(image, (224, 224))
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_tensor


class PRNetLandmarkRefiner:
    """
    PRNet-based landmark refinement.
    Position map Regression Network for detailed face reconstruction.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        
        # Try to load PRNet
        try:
            from PRNet.api import PRN
            self.model = PRN(is_dlib=True)
        except ImportError:
            print("Warning: PRNet not available. Install from: https://github.com/YadiraF/PRNet")
    
    def refine_landmarks(self, image, initial_landmarks):
        """
        Refine landmarks using PRNet.
        
        Args:
            image: numpy array [H, W, 3]
            initial_landmarks: numpy array [68, 2]
        
        Returns:
            refined_landmarks: numpy array [68, 2]
            confidence: float
        """
        if self.model is None:
            # Fallback: Gaussian smoothing on initial landmarks
            return self._smooth_landmarks(initial_landmarks), 0.6
        
        # Use PRNet to get position map
        pos = self.model.process(image)
        
        if pos is None:
            return initial_landmarks, 0.5
        
        # Extract 68 landmarks from position map
        kpt = self.model.get_landmarks(pos)
        
        if kpt is not None and kpt.shape[0] >= 68:
            refined_landmarks = kpt[:68, :2]
            return refined_landmarks, 0.92
        else:
            return initial_landmarks, 0.5
    
    def _smooth_landmarks(self, landmarks):
        """
        Apply Gaussian smoothing to landmarks as fallback.
        """
        from scipy.ndimage import gaussian_filter1d
        
        smoothed = landmarks.copy()
        smoothed[:, 0] = gaussian_filter1d(landmarks[:, 0], sigma=1.0)
        smoothed[:, 1] = gaussian_filter1d(landmarks[:, 1], sigma=1.0)
        
        return smoothed


class TemporalLandmarkSmoother:
    """
    Temporal smoothing for landmark sequences.
    Reduces jitter in video sequences.
    """
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.landmark_buffer = []
    
    def smooth(self, landmarks):
        """
        Apply temporal smoothing.
        
        Args:
            landmarks: numpy array [68, 2]
        
        Returns:
            smoothed_landmarks: numpy array [68, 2]
        """
        self.landmark_buffer.append(landmarks)
        
        if len(self.landmark_buffer) > self.window_size:
            self.landmark_buffer.pop(0)
        
        # Average over buffer
        smoothed = np.mean(self.landmark_buffer, axis=0)
        
        return smoothed
    
    def reset(self):
        """Reset the buffer."""
        self.landmark_buffer = []


def refine_landmarks_sequence(images, initial_landmarks_list, method='mediapipe', device='cuda', temporal_smooth=True):
    """
    Refine a sequence of landmarks with temporal consistency.
    
    Args:
        images: list of images
        initial_landmarks_list: list of initial landmark arrays
        method: refinement method ('mediapipe', 'emoca', 'prnet')
        device: 'cuda' or 'cpu'
        temporal_smooth: whether to apply temporal smoothing
    
    Returns:
        refined_landmarks_list: list of refined landmark arrays
        confidences: list of confidence scores
    """
    refiner = LandmarkRefiner(method=method, device=device)
    smoother = TemporalLandmarkSmoother() if temporal_smooth else None
    
    refined_list = []
    confidence_list = []
    
    for img, initial_lms in zip(images, initial_landmarks_list):
        # Refine landmarks
        refined_lms, conf = refiner.refine_landmarks(img, initial_lms)
        
        # Apply temporal smoothing if enabled
        if smoother is not None:
            refined_lms = smoother.smooth(refined_lms)
        
        refined_list.append(refined_lms)
        confidence_list.append(conf)
    
    return refined_list, confidence_list


# Utility function for easy integration
def create_landmark_refiner(method='mediapipe', device='cuda'):
    """
    Factory function to create a landmark refiner.
    
    Args:
        method: 'mediapipe', 'emoca', or 'prnet'
        device: 'cuda' or 'cpu'
    
    Returns:
        refiner: LandmarkRefiner instance
    """
    return LandmarkRefiner(method=method, device=device)


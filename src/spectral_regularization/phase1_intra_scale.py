"""
Phase 1: Intra-scale Frequency Regularization

Implements progressive spectral truncation using DCT to guide each scale
to specialize in its target frequency band during early training.

Key components:
- DCTTransform: 2D DCT/IDCT operations
- SpectralTruncation: Progressive frequency cutoff scheduler

Reference: Section 3.2 of the paper
"""

import torch
import torch.nn.functional as F
import math
import warnings


class DCTTransform:
    """
    2D Discrete Cosine Transform (DCT-II) and its inverse.
    
    Implements orthonormal DCT basis for frequency domain analysis.
    Used to transform images to frequency domain for spectral truncation.
    """
    
    # Cache for DCT matrices to avoid recomputation
    _dct_cache = {}
    
    @staticmethod
    def dct_2d(x):
        """
        Apply 2D Discrete Cosine Transform.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            freq: Frequency domain representation [B, C, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        
        # Get or create DCT basis matrices
        PH = DCTTransform._get_dct_matrix(H, device, dtype)
        PW = DCTTransform._get_dct_matrix(W, device, dtype)
        
        # Reshape for batch matrix multiplication: [B*C, H, W]
        x_reshaped = x.reshape(B * C, H, W)
        
        # Apply 2D-DCT: F = P_H @ x @ P_W^T
        # First dimension: [B*C, H, W] @ [H, H]^T -> [B*C, H, W]
        freq = torch.matmul(PH, x_reshaped)
        
        # Second dimension: [B*C, H, W] @ [W, W]
        freq = torch.matmul(freq, PW.t())
        
        # Reshape back to [B, C, H, W]
        return freq.reshape(B, C, H, W)
    
    @staticmethod
    def idct_2d(freq):
        """
        Apply inverse 2D Discrete Cosine Transform.
        
        Args:
            freq: Frequency domain tensor [B, C, H, W]
            
        Returns:
            x: Spatial domain representation [B, C, H, W]
        """
        B, C, H, W = freq.shape
        device = freq.device
        dtype = freq.dtype
        
        # Get DCT basis matrices (same as forward)
        PH = DCTTransform._get_dct_matrix(H, device, dtype)
        PW = DCTTransform._get_dct_matrix(W, device, dtype)
        
        # Reshape for batch matrix multiplication
        freq_reshaped = freq.reshape(B * C, H, W)
        
        # Apply inverse: x = P_H^T @ F @ P_W
        # Since P is orthonormal, P^T is the inverse
        x = torch.matmul(PH.t(), freq_reshaped)
        x = torch.matmul(x, PW)
        
        # Reshape back
        return x.reshape(B, C, H, W)
    
    @staticmethod
    def _get_dct_matrix(N, device, dtype):
        """
        Generate orthonormal DCT-II basis matrix.
        
        Args:
            N: Matrix size (for N x N spatial dimension)
            device: torch device
            dtype: torch dtype
            
        Returns:
            dct_matrix: [N, N] orthonormal DCT basis
        """
        # Check cache
        cache_key = (N, device, dtype)
        if cache_key in DCTTransform._dct_cache:
            return DCTTransform._dct_cache[cache_key]
        
        # Create index arrays
        n = torch.arange(N, device=device, dtype=dtype)
        k = n.reshape(-1, 1)  # [N, 1]
        
        # DCT-II basis: cos(π * k * (2n + 1) / (2N))
        dct_matrix = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        
        # Orthonormal scaling factors
        # First row (DC component): sqrt(1/N)
        dct_matrix[0, :] *= math.sqrt(1.0 / N)
        # Other rows (AC components): sqrt(2/N)
        dct_matrix[1:, :] *= math.sqrt(2.0 / N)
        
        # Cache the result
        DCTTransform._dct_cache[cache_key] = dct_matrix
        
        return dct_matrix
    
    @staticmethod
    def clear_cache():
        """Clear the DCT matrix cache (useful for memory management)"""
        DCTTransform._dct_cache.clear()


class SpectralTruncation:
    """
    Progressive spectral truncation scheduler for intra-scale regularization.
    
    Gradually increases the frequency cutoff from low to high frequencies
    during early training (first 100 epochs), allowing higher scales to
    fully capture low-frequency information first.
    
    Reference: Section 3.2, Equation 4 of the paper
    """
    
    def __init__(self, tau_init=0.05, tau_final=1.0, num_epochs=100):
        """
        Initialize spectral truncation scheduler.
        
        Args:
            tau_init: Initial frequency cutoff (normalized, default: 0.05)
            tau_final: Final frequency cutoff (normalized, default: 1.0)
            num_epochs: Number of epochs to apply truncation (default: 100)
        """
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.num_epochs = num_epochs
        
        # Cache for radial masks
        self._mask_cache = {}
    
    def get_tau(self, epoch):
        """
        Get current frequency cutoff radius (τ) based on epoch.
        
        Implements linear schedule: τ(t) = τ_init + (τ_final - τ_init) * t / T
        
        Args:
            epoch: Current training epoch
            
        Returns:
            tau: Current cutoff radius [0, 1]
        """
        if epoch >= self.num_epochs:
            return self.tau_final
        
        # Linear interpolation
        progress = epoch / self.num_epochs
        tau = self.tau_init + (self.tau_final - self.tau_init) * progress
        
        return tau
    
    def create_radial_mask(self, H, W, tau, device, dtype):
        """
        Create soft radial mask for frequency truncation.
        
        Implements Equation 4 from the paper:
        M(u, v; t) = max(0, (τ(t) - r) / τ(t))
        where r = sqrt((u/H)^2 + (v/W)^2) is normalized frequency radius
        
        Args:
            H, W: Spatial dimensions
            tau: Current cutoff radius
            device: torch device
            dtype: torch dtype
            
        Returns:
            mask: [H, W] soft radial mask
        """
        # Check cache
        cache_key = (H, W, tau, device, dtype)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        
        # Create normalized frequency coordinates
        u = torch.arange(H, device=device, dtype=dtype) / H
        v = torch.arange(W, device=device, dtype=dtype) / W
        
        # Create 2D grid
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        
        # Compute normalized radial frequency
        # r = sqrt((u/H)^2 + (v/W)^2) - already normalized above
        radius = torch.sqrt(u_grid**2 + v_grid**2)
        
        # Apply soft truncation mask: max(0, (τ - r) / τ)
        # This creates a smooth transition from 1 (pass) to 0 (block)
        if tau > 0:
            mask = torch.clamp((tau - radius) / tau, 0.0, 1.0)
        else:
            mask = torch.zeros_like(radius)
        
        # Cache only if tau is final (to avoid cache explosion)
        if tau == self.tau_final:
            self._mask_cache[cache_key] = mask
        
        return mask
    
    def apply_truncation(self, images, epoch):
        """
        Apply progressive frequency truncation to input images.
        
        This is the main entry point for Phase 1 regularization.
        During early training, only low-frequency components are fed
        to the model, gradually adding higher frequencies.
        
        Args:
            images: Input images [B, C, H, W]
            epoch: Current training epoch
            
        Returns:
            truncated_images: Frequency-truncated images [B, C, H, W]
        """
        # Skip truncation after num_epochs
        if epoch >= self.num_epochs:
            return images
        
        # Get current tau
        tau = self.get_tau(epoch)
        
        B, C, H, W = images.shape
        device = images.device
        dtype = images.dtype
        
        # Transform to frequency domain
        freq = DCTTransform.dct_2d(images)
        
        # Create and apply radial mask
        mask = self.create_radial_mask(H, W, tau, device, dtype)
        
        # Broadcast mask to batch and channel dimensions: [1, 1, H, W]
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Apply mask in frequency domain
        freq_truncated = freq * mask
        
        # Transform back to spatial domain
        images_truncated = DCTTransform.idct_2d(freq_truncated)
        
        return images_truncated
    
    def clear_cache(self):
        """Clear the mask cache"""
        self._mask_cache.clear()
    
    def __repr__(self):
        return (f"SpectralTruncation(tau_init={self.tau_init}, "
                f"tau_final={self.tau_final}, num_epochs={self.num_epochs})")


# Utility functions for testing and validation

def visualize_dct_basis(N=8):
    """
    Visualize DCT basis functions (useful for debugging).
    
    Args:
        N: Size of DCT basis to visualize
        
    Returns:
        basis_images: [N, N, N, N] tensor of basis images
    """
    device = torch.device('cpu')
    dtype = torch.float32
    
    dct_matrix = DCTTransform._get_dct_matrix(N, device, dtype)
    
    # Create basis images by outer product
    basis_images = []
    for i in range(N):
        for j in range(N):
            basis = torch.outer(dct_matrix[i], dct_matrix[j])
            basis_images.append(basis)
    
    return torch.stack(basis_images).reshape(N, N, N, N)


def test_dct_reconstruction():
    """
    Test DCT/IDCT reconstruction accuracy.
    
    Returns:
        max_error: Maximum reconstruction error
    """
    # Create random test image
    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W)
    
    # Forward and inverse transform
    freq = DCTTransform.dct_2d(x)
    x_recon = DCTTransform.idct_2d(freq)
    
    # Compute reconstruction error
    error = torch.abs(x - x_recon)
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    print(f"DCT Reconstruction Test:")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Status: {'PASS' if max_error < 1e-4 else 'FAIL'}")
    
    return max_error


if __name__ == "__main__":
    # Run tests
    print("Testing Phase 1: Intra-scale Frequency Regularization\n")
    
    # Test 1: DCT reconstruction
    test_dct_reconstruction()
    
    # Test 2: Spectral truncation
    print("\nTesting Spectral Truncation:")
    truncator = SpectralTruncation(tau_init=0.05, tau_final=1.0, num_epochs=100)
    
    # Test at different epochs
    x = torch.randn(1, 3, 256, 256)
    for epoch in [0, 50, 100, 150]:
        x_trunc = truncator.apply_truncation(x, epoch)
        tau = truncator.get_tau(epoch)
        print(f"  Epoch {epoch:3d}: tau={tau:.3f}, output_shape={x_trunc.shape}")
    
    print("\nAll tests completed!")

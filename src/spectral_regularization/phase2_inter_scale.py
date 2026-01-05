"""
Phase 2: Inter-scale Latent Regularization

Implements inter-scale similarity regularization to suppress spectral aliasing
across scales by encouraging latent variables at different scales to remain
distinct from each other.

Key components:
- DWTDownsampler: Fast wavelet-based downsampling for scale alignment
- InterScaleRegularizer: L2 similarity penalty across adjacent scales

Reference: Section 3.3 of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class DWTDownsampler:
    """
    Fast Discrete Wavelet Transform downsampling using convolution.
    
    Uses Haar wavelet approximation for efficient 2x downsampling.
    This is faster than PyWavelets and fully differentiable.
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize DWT downsampler.
        
        Args:
            device: torch device for operations
        """
        self.device = device
        
        # Haar wavelet low-pass filter (approximation coefficients)
        # Normalized for energy preservation
        self.lowpass_filter = torch.tensor([
            [0.25, 0.25],
            [0.25, 0.25]
        ], dtype=torch.float32, device=device)
    
    def downsample(self, x):
        """
        Apply DWT downsampling with 2x reduction.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            downsampled: [B, C, H//2, W//2]
        """
        B, C, H, W = x.shape
        
        # Create depthwise convolution kernel [C, 1, 2, 2]
        kernel = self.lowpass_filter.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        
        # Apply depthwise convolution with stride=2
        # groups=C ensures each channel is processed independently
        downsampled = F.conv2d(
            x, kernel,
            stride=2,
            padding=0,
            groups=C
        )
        
        return downsampled
    
    def __repr__(self):
        return f"DWTDownsampler(device={self.device})"


class InterScaleRegularizer(nn.Module):
    """
    Inter-scale latent regularization to prevent spectral aliasing.
    
    Computes similarity penalty between latent variables at adjacent scales
    to encourage them to remain distinct and avoid redundant encoding.
    
    Reference: Section 3.3, Equation 6 of the paper
    """
    
    def __init__(self, delta=0.1, channels=[320, 320, 320], device='cuda'):
        """
        Initialize inter-scale regularizer.
        
        Args:
            delta: Regularization weight (default: 0.1)
            channels: List of channel numbers at each scale [s1, s2, s3]
            device: torch device
        """
        super().__init__()
        
        self.delta = delta
        self.device = device
        
        # DWT downsampler for scale alignment
        self.dwt = DWTDownsampler(device=device)
        
        # Channel alignment modules (1x1 convolutions)
        # These are only used during training to align channel dimensions
        self.channel_align = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_channels = channels[i + 1]  # Lower scale (finer)
            out_channels = channels[i]      # Upper scale (coarser)
            
            # 1x1 conv for channel alignment
            align_conv = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=1, stride=1, padding=0, bias=False
            )
            
            # Initialize with small weights to avoid disrupting training
            nn.init.kaiming_normal_(align_conv.weight, mode='fan_out', nonlinearity='linear')
            align_conv.weight.data *= 0.1  # Scale down initial weights
            
            self.channel_align.append(align_conv)
    
    def align_scales(self, z_lower, z_upper, align_idx):
        """
        Align lower scale latent to upper scale dimensions.
        
        Args:
            z_lower: Lower scale latent [B, C_lower, H, W]
            z_upper: Upper scale latent [B, C_upper, H//2, W//2]
            align_idx: Index for channel alignment module
            
        Returns:
            z_lower_aligned: [B, C_upper, H//2, W//2]
        """
        # Spatial downsampling via DWT
        z_lower_down = self.dwt.downsample(z_lower)
        
        # Channel alignment via 1x1 conv
        if align_idx < len(self.channel_align):
            z_lower_aligned = self.channel_align[align_idx](z_lower_down)
        else:
            z_lower_aligned = z_lower_down
        
        # Ensure spatial dimensions match (handle odd sizes)
        if z_lower_aligned.shape[2:] != z_upper.shape[2:]:
            z_lower_aligned = F.adaptive_avg_pool2d(
                z_lower_aligned, z_upper.shape[2:]
            )
        
        return z_lower_aligned
    
    def compute_similarity(self, z_upper, z_lower_aligned):
        """
        Compute L2 similarity between aligned latents.
        
        Higher similarity = more redundant encoding.
        We want to maximize distance (minimize similarity).
        
        Args:
            z_upper: Upper scale latent [B, C, H, W]
            z_lower_aligned: Aligned lower scale latent [B, C, H, W]
            
        Returns:
            similarity: Scalar L2 distance
        """
        # L2 mean squared error (lower = more similar)
        similarity = F.mse_loss(z_upper, z_lower_aligned)
        return similarity
    
    def forward(self, latents_hierarchy):
        """
        Compute inter-scale regularization loss.
        
        Args:
            latents_hierarchy: List of [z_s1, z_s2, z_s3]
                - z_s1: [B, C, H//4, W//4] (coarsest scale)
                - z_s2: [B, C, H//2, W//2]
                - z_s3: [B, C, H, W] (finest scale)
        
        Returns:
            reg_loss: Negative similarity loss (maximize distance)
        """
        if len(latents_hierarchy) < 2:
            return torch.tensor(0.0, device=self.device)
        
        total_similarity = 0.0
        num_pairs = 0
        
        # Iterate over adjacent scale pairs
        for i in range(len(latents_hierarchy) - 1):
            z_upper = latents_hierarchy[i]      # Coarser scale
            z_lower = latents_hierarchy[i + 1]  # Finer scale
            
            # Skip if either is None
            if z_upper is None or z_lower is None:
                continue
            
            # Align lower scale to upper scale
            try:
                z_lower_aligned = self.align_scales(z_lower, z_upper, i)
                
                # Compute similarity
                similarity = self.compute_similarity(z_upper, z_lower_aligned)
                total_similarity += similarity
                num_pairs += 1
                
            except Exception as e:
                warnings.warn(f"Error aligning scales {i}-{i+1}: {e}")
                continue
        
        # Average over all pairs
        if num_pairs > 0:
            avg_similarity = total_similarity / num_pairs
        else:
            avg_similarity = torch.tensor(0.0, device=self.device)
        
        # Negative sign: minimize negative similarity = maximize distance
        # This encourages scales to encode different information
        reg_loss = -self.delta * avg_similarity
        
        return reg_loss
    
    def __repr__(self):
        return (f"InterScaleRegularizer(delta={self.delta}, "
                f"num_align_modules={len(self.channel_align)})")


# Utility functions for testing

def test_dwt_downsampler():
    """Test DWT downsampling functionality"""
    print("Testing DWT Downsampler...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dwt = DWTDownsampler(device=device)
    
    # Test with various sizes
    test_cases = [
        (2, 320, 32, 32),
        (4, 320, 64, 64),
        (8, 320, 128, 128),
    ]
    
    for B, C, H, W in test_cases:
        x = torch.randn(B, C, H, W, device=device)
        y = dwt.downsample(x)
        
        expected_shape = (B, C, H//2, W//2)
        assert y.shape == expected_shape, f"Shape mismatch: {y.shape} != {expected_shape}"
        assert not torch.isnan(y).any(), "NaN detected in output"
        assert not torch.isinf(y).any(), "Inf detected in output"
        
        print(f"  [{B}, {C}, {H}, {W}] -> {tuple(y.shape)} ✓")
    
    print("DWT Downsampler test PASSED!\n")


def test_inter_scale_regularizer():
    """Test inter-scale regularization"""
    print("Testing Inter-Scale Regularizer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize regularizer
    regularizer = InterScaleRegularizer(
        delta=0.1,
        channels=[320, 320, 320],
        device=device
    ).to(device)
    
    # Create dummy hierarchical latents (mimicking HPCM scales)
    B = 4
    latents = [
        torch.randn(B, 320, 16, 16, device=device),  # s1: coarsest
        torch.randn(B, 320, 32, 32, device=device),  # s2: middle
        torch.randn(B, 320, 64, 64, device=device),  # s3: finest
    ]
    
    # Compute regularization loss
    reg_loss = regularizer(latents)
    
    print(f"  Input scales: {[tuple(x.shape) for x in latents]}")
    print(f"  Regularization loss: {reg_loss.item():.6f}")
    
    assert reg_loss.requires_grad, "Loss should be differentiable"
    assert not torch.isnan(reg_loss), "NaN detected in loss"
    assert not torch.isinf(reg_loss), "Inf detected in loss"
    
    # Test backward pass
    reg_loss.backward()
    
    for i, align_module in enumerate(regularizer.channel_align):
        assert align_module.weight.grad is not None, f"No gradient for align module {i}"
        print(f"  Align module {i} gradient norm: {align_module.weight.grad.norm().item():.6f} ✓")
    
    print("Inter-Scale Regularizer test PASSED!\n")


def test_gradient_flow():
    """Test gradient flow through regularizer"""
    print("Testing Gradient Flow...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    regularizer = InterScaleRegularizer(delta=0.1, device=device).to(device)
    
    # Create latents that require gradients
    latents = [
        torch.randn(2, 320, 16, 16, device=device, requires_grad=True),
        torch.randn(2, 320, 32, 32, device=device, requires_grad=True),
        torch.randn(2, 320, 64, 64, device=device, requires_grad=True),
    ]
    
    # Forward pass
    loss = regularizer(latents)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    for i, latent in enumerate(latents):
        assert latent.grad is not None, f"No gradient for latent {i}"
        grad_norm = latent.grad.norm().item()
        print(f"  Latent {i} gradient norm: {grad_norm:.6f} ✓")
    
    print("Gradient Flow test PASSED!\n")


if __name__ == "__main__":
    print("="*70)
    print("  Phase 2: Inter-scale Latent Regularization - Test Suite")
    print("="*70 + "\n")
    
    # Run tests
    test_dwt_downsampler()
    test_inter_scale_regularizer()
    test_gradient_flow()
    
    print("="*70)
    print("  ✓ All Phase 2 Tests PASSED!")
    print("="*70)
    print("\nPhase 2 is ready for integration with HPCM.")

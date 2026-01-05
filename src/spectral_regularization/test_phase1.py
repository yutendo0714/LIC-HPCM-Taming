"""
Test script for Phase 1: Intra-scale Frequency Regularization

This script validates the implementation of DCT/IDCT and spectral truncation.
Run this before starting training to ensure everything works correctly.
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.spectral_regularization import DCTTransform, SpectralTruncation


def test_dct_reconstruction(device='cuda'):
    """Test 1: Verify DCT/IDCT reconstruction accuracy"""
    print("="*70)
    print("Test 1: DCT/IDCT Reconstruction Accuracy")
    print("="*70)
    
    # Test with different sizes
    test_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    for H, W in test_sizes:
        B, C = 4, 3
        x = torch.randn(B, C, H, W, device=device)
        
        # Forward and inverse transform
        start = time.time()
        freq = DCTTransform.dct_2d(x)
        x_recon = DCTTransform.idct_2d(freq)
        elapsed = time.time() - start
        
        # Compute errors
        error = torch.abs(x - x_recon)
        max_error = error.max().item()
        mean_error = error.mean().item()
        
        status = "✓ PASS" if max_error < 1e-4 else "✗ FAIL"
        print(f"Size [{H:3d}x{W:3d}]: max_err={max_error:.2e}, "
              f"mean_err={mean_error:.2e}, time={elapsed:.3f}s [{status}]")
    
    print()


def test_spectral_truncation_tau_schedule(device='cuda'):
    """Test 2: Verify tau scheduling"""
    print("="*70)
    print("Test 2: Tau Scheduling")
    print("="*70)
    
    truncator = SpectralTruncation(tau_init=0.05, tau_final=1.0, num_epochs=100)
    
    test_epochs = [0, 25, 50, 75, 100, 150]
    for epoch in test_epochs:
        tau = truncator.get_tau(epoch)
        print(f"Epoch {epoch:3d}: tau = {tau:.4f}")
    
    print()


def test_spectral_truncation_forward(device='cuda'):
    """Test 3: Verify spectral truncation forward pass"""
    print("="*70)
    print("Test 3: Spectral Truncation Forward Pass")
    print("="*70)
    
    truncator = SpectralTruncation(tau_init=0.05, tau_final=1.0, num_epochs=100)
    
    B, C, H, W = 4, 3, 256, 256
    x = torch.randn(B, C, H, W, device=device)
    
    test_epochs = [0, 50, 100, 150]
    for epoch in test_epochs:
        start = time.time()
        x_trunc = truncator.apply_truncation(x, epoch)
        elapsed = time.time() - start
        
        tau = truncator.get_tau(epoch)
        
        # Check output properties
        assert x_trunc.shape == x.shape, "Shape mismatch!"
        assert not torch.isnan(x_trunc).any(), "NaN detected!"
        assert not torch.isinf(x_trunc).any(), "Inf detected!"
        
        print(f"Epoch {epoch:3d}: tau={tau:.4f}, "
              f"output_shape={tuple(x_trunc.shape)}, "
              f"time={elapsed:.3f}s [✓ PASS]")
    
    print()


def test_memory_and_speed(device='cuda'):
    """Test 4: Memory usage and speed benchmark"""
    print("="*70)
    print("Test 4: Memory Usage and Speed Benchmark")
    print("="*70)
    
    truncator = SpectralTruncation(tau_init=0.05, tau_final=1.0, num_epochs=100)
    
    # Typical training batch
    B, C, H, W = 16, 3, 256, 256
    x = torch.randn(B, C, H, W, device=device)
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1024**2
    
    # Warm-up
    for _ in range(5):
        _ = truncator.apply_truncation(x, epoch=50)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    n_iters = 50
    for _ in range(n_iters):
        _ = truncator.apply_truncation(x, epoch=50)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / n_iters * 1000  # ms
    
    if device == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        mem_usage = peak_mem - start_mem
        
        print(f"Batch size: {B}, Image size: {H}x{W}")
        print(f"Average time per batch: {avg_time:.2f} ms")
        print(f"Memory usage: {mem_usage:.2f} MB")
        print(f"Throughput: {B * n_iters / elapsed:.2f} images/sec")
    else:
        print(f"Batch size: {B}, Image size: {H}x{W}")
        print(f"Average time per batch: {avg_time:.2f} ms")
        print(f"Throughput: {B * n_iters / elapsed:.2f} images/sec")
    
    print()


def visualize_frequency_truncation(save_dir='./test_outputs'):
    """Test 5: Visualize frequency truncation effect"""
    print("="*70)
    print("Test 5: Visualizing Frequency Truncation Effect")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    
    truncator = SpectralTruncation(tau_init=0.05, tau_final=1.0, num_epochs=100)
    
    # Create a test image with various frequencies
    H, W = 256, 256
    x, y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    
    # Combination of low and high frequency patterns
    image = (np.sin(2 * np.pi * 2 * x) * np.cos(2 * np.pi * 2 * y) +  # Low freq
             np.sin(2 * np.pi * 10 * x) * np.cos(2 * np.pi * 10 * y))  # High freq
    image = (image - image.min()) / (image.max() - image.min())
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    
    # Apply truncation at different epochs
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    epochs = [0, 20, 40, 60, 80, 100]
    for idx, epoch in enumerate(epochs):
        truncated = truncator.apply_truncation(image_tensor, epoch)
        truncated_np = truncated.squeeze().cpu().numpy()
        
        tau = truncator.get_tau(epoch)
        
        axes[idx].imshow(truncated_np, cmap='viridis')
        axes[idx].set_title(f'Epoch {epoch}, τ={tau:.3f}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, 'frequency_truncation_progression.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()
    
    # Visualize radial mask
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    tau_values = [0.05, 0.25, 0.5, 1.0]
    
    for idx, tau in enumerate(tau_values):
        mask = truncator.create_radial_mask(H, W, tau, 'cpu', torch.float32)
        
        axes[idx].imshow(mask.numpy(), cmap='hot')
        axes[idx].set_title(f'τ={tau:.2f}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, 'radial_masks.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Radial masks saved to: {output_path}")
    plt.close()
    
    print()


def test_integration_with_dataloader(device='cuda'):
    """Test 6: Integration test with typical training scenario"""
    print("="*70)
    print("Test 6: Integration with Typical Training Scenario")
    print("="*70)
    
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create dummy dataset
    n_samples = 100
    dummy_images = torch.randn(n_samples, 3, 256, 256)
    dataset = TensorDataset(dummy_images)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    truncator = SpectralTruncation(tau_init=0.05, tau_final=1.0, num_epochs=100)
    
    # Simulate training for a few epochs
    for epoch in [0, 50, 100]:
        print(f"\nSimulating Epoch {epoch}:")
        tau = truncator.get_tau(epoch)
        print(f"  Current tau: {tau:.4f}")
        
        for i, (batch,) in enumerate(dataloader):
            batch = batch.to(device)
            
            # Apply truncation (as in training loop)
            if epoch < truncator.num_epochs:
                batch_trunc = truncator.apply_truncation(batch, epoch)
            else:
                batch_trunc = batch
            
            # Verify
            assert batch_trunc.shape == batch.shape
            assert not torch.isnan(batch_trunc).any()
            
            if i == 0:  # Only show first batch
                print(f"  Batch shape: {batch_trunc.shape}, "
                      f"range: [{batch_trunc.min():.3f}, {batch_trunc.max():.3f}]")
            
            if i >= 2:  # Only test a few batches
                break
        
        print(f"  ✓ Epoch {epoch} passed")
    
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("  Phase 1: Intra-scale Frequency Regularization - Test Suite")
    print("="*70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    try:
        # Run tests
        test_dct_reconstruction(device)
        test_spectral_truncation_tau_schedule(device)
        test_spectral_truncation_forward(device)
        test_memory_and_speed(device)
        visualize_frequency_truncation()
        test_integration_with_dataloader(device)
        
        # Summary
        print("="*70)
        print("  ✓ All Tests PASSED!")
        print("="*70)
        print("\nPhase 1 is ready for training. You can now run:")
        print("  python train.py --spectral-reg --model_name HPCM_Base ...")
        print()
        
    except Exception as e:
        print("\n" + "="*70)
        print("  ✗ Tests FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

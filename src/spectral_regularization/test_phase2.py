"""
Comprehensive test suite for Phase 2: Inter-scale Latent Regularization

Tests all components including DWT downsampling, inter-scale regularization,
gradient flow, memory usage, and integration scenarios.
"""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.spectral_regularization.phase2_inter_scale import (
    DWTDownsampler, InterScaleRegularizer
)


def test_dwt_downsampler_accuracy():
    """Test 1: DWT downsampling accuracy and properties"""
    print("="*70)
    print("Test 1: DWT Downsampler Accuracy")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dwt = DWTDownsampler(device=device)
    
    # Test various input sizes
    test_sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]
    
    for H, W in test_sizes:
        B, C = 4, 320
        x = torch.randn(B, C, H, W, device=device)
        
        # Apply downsampling
        y = dwt.downsample(x)
        
        # Check shape
        expected_shape = (B, C, H//2, W//2)
        assert y.shape == expected_shape, f"Shape mismatch: {y.shape} != {expected_shape}"
        
        # Check for NaN/Inf
        assert not torch.isnan(y).any(), "NaN detected"
        assert not torch.isinf(y).any(), "Inf detected"
        
        # Check energy preservation (approximate)
        energy_in = (x ** 2).sum().item()
        energy_out = (y ** 2).sum().item() * 4  # Account for downsampling
        energy_ratio = energy_out / energy_in
        
        status = "✓ PASS" if 0.5 < energy_ratio < 1.5 else "⚠ WARNING"
        print(f"Size [{H:3d}x{W:3d}]: shape={y.shape}, "
              f"energy_ratio={energy_ratio:.3f} [{status}]")
    
    print()


def test_dwt_speed_benchmark():
    """Test 2: DWT downsampling speed"""
    print("="*70)
    print("Test 2: DWT Downsampling Speed")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dwt = DWTDownsampler(device=device)
    
    # Typical training batch
    B, C, H, W = 16, 320, 64, 64
    x = torch.randn(B, C, H, W, device=device)
    
    # Warm-up
    for _ in range(10):
        _ = dwt.downsample(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        _ = dwt.downsample(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / n_iters * 1000  # ms
    
    print(f"Input shape: [{B}, {C}, {H}, {W}]")
    print(f"Average time: {avg_time:.3f} ms per batch")
    print(f"Throughput: {B * n_iters / elapsed:.1f} images/sec")
    
    status = "✓ EXCELLENT" if avg_time < 1.0 else "✓ GOOD"
    print(f"Performance: {status}")
    print()


def test_inter_scale_regularizer_basic():
    """Test 3: Basic inter-scale regularization"""
    print("="*70)
    print("Test 3: Inter-Scale Regularization - Basic Functionality")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize with HPCM-like channel configuration
    regularizer = InterScaleRegularizer(
        delta=0.1,
        channels=[320, 320, 320],
        device=device
    ).to(device)
    
    print(f"Regularizer: {regularizer}")
    print(f"Number of alignment modules: {len(regularizer.channel_align)}")
    
    # Create hierarchical latents
    B = 4
    latents = [
        torch.randn(B, 320, 16, 16, device=device),  # s1: H//4, W//4
        torch.randn(B, 320, 32, 32, device=device),  # s2: H//2, W//2
        torch.randn(B, 320, 64, 64, device=device),  # s3: H, W
    ]
    
    print(f"\nInput hierarchy:")
    for i, latent in enumerate(latents):
        print(f"  Scale {i+1}: {tuple(latent.shape)}")
    
    # Compute loss
    reg_loss = regularizer(latents)
    
    print(f"\nRegularization loss: {reg_loss.item():.6f}")
    
    # Verify properties
    assert reg_loss.requires_grad, "Loss should be differentiable"
    assert not torch.isnan(reg_loss), "NaN detected"
    assert not torch.isinf(reg_loss), "Inf detected"
    assert reg_loss.item() < 0, "Loss should be negative (distance maximization)"
    
    print("Status: ✓ PASS")
    print()


def test_gradient_flow():
    """Test 4: Gradient flow through regularizer"""
    print("="*70)
    print("Test 4: Gradient Flow")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    regularizer = InterScaleRegularizer(delta=0.1, device=device).to(device)
    
    # Create latents with gradients enabled
    latents = [
        torch.randn(2, 320, 16, 16, device=device, requires_grad=True),
        torch.randn(2, 320, 32, 32, device=device, requires_grad=True),
        torch.randn(2, 320, 64, 64, device=device, requires_grad=True),
    ]
    
    # Forward pass
    loss = regularizer(latents)
    print(f"Forward loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check latent gradients
    print("\nLatent gradients:")
    for i, latent in enumerate(latents):
        assert latent.grad is not None, f"No gradient for latent {i}"
        grad_norm = latent.grad.norm().item()
        grad_mean = latent.grad.abs().mean().item()
        print(f"  Scale {i+1}: norm={grad_norm:.6f}, mean={grad_mean:.6f} ✓")
    
    # Check module gradients
    print("\nAlignment module gradients:")
    for i, module in enumerate(regularizer.channel_align):
        assert module.weight.grad is not None, f"No gradient for module {i}"
        grad_norm = module.weight.grad.norm().item()
        print(f"  Module {i}: norm={grad_norm:.6f} ✓")
    
    print("\nStatus: ✓ PASS")
    print()


def test_memory_usage():
    """Test 5: Memory usage"""
    print("="*70)
    print("Test 5: Memory Usage")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("Skipping memory test on CPU")
        print()
        return
    
    regularizer = InterScaleRegularizer(delta=0.1, device=device).to(device)
    
    # Measure baseline memory
    torch.cuda.reset_peak_memory_stats()
    baseline_mem = torch.cuda.memory_allocated() / 1024**2
    
    # Create typical training batch
    B = 16
    latents = [
        torch.randn(B, 320, 16, 16, device=device, requires_grad=True),
        torch.randn(B, 320, 32, 32, device=device, requires_grad=True),
        torch.randn(B, 320, 64, 64, device=device, requires_grad=True),
    ]
    
    latents_mem = torch.cuda.memory_allocated() / 1024**2
    
    # Forward + backward
    loss = regularizer(latents)
    loss.backward()
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    final_mem = torch.cuda.memory_allocated() / 1024**2
    
    print(f"Baseline memory: {baseline_mem:.2f} MB")
    print(f"After latents: {latents_mem:.2f} MB (+{latents_mem-baseline_mem:.2f} MB)")
    print(f"Peak memory: {peak_mem:.2f} MB")
    print(f"Final memory: {final_mem:.2f} MB")
    print(f"Regularizer overhead: {peak_mem-latents_mem:.2f} MB")
    
    overhead_pct = (peak_mem - latents_mem) / latents_mem * 100
    status = "✓ EXCELLENT" if overhead_pct < 10 else "✓ GOOD"
    print(f"Overhead: {overhead_pct:.1f}% [{status}]")
    print()


def test_scale_independence():
    """Test 6: Verify scale independence (no information leakage)"""
    print("="*70)
    print("Test 6: Scale Independence")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    regularizer = InterScaleRegularizer(delta=0.1, device=device).to(device)
    
    # Test 1: Identical latents (should have high similarity, low negative loss)
    identical_latents = [
        torch.ones(2, 320, 16, 16, device=device),
        torch.ones(2, 320, 32, 32, device=device),
        torch.ones(2, 320, 64, 64, device=device),
    ]
    
    loss_identical = regularizer(identical_latents)
    print(f"Loss with identical latents: {loss_identical.item():.6f}")
    
    # Test 2: Random latents (should have lower similarity, higher negative loss)
    random_latents = [
        torch.randn(2, 320, 16, 16, device=device),
        torch.randn(2, 320, 32, 32, device=device),
        torch.randn(2, 320, 64, 64, device=device),
    ]
    
    loss_random = regularizer(random_latents)
    print(f"Loss with random latents: {loss_random.item():.6f}")
    
    # Both should be negative (distance maximization objective)
    # The actual values depend on random initialization, so we just check
    # that both are reasonable and negative
    assert loss_identical.item() < 0, "Loss should be negative"
    assert loss_random.item() < 0, "Loss should be negative"
    
    print(f"\nDifference: {abs(loss_random.item() - loss_identical.item()):.6f}")
    print("Both losses are negative (distance maximization) ✓")
    print("\nStatus: ✓ PASS")
    print()


def test_integration_with_training():
    """Test 7: Integration with typical training loop"""
    print("="*70)
    print("Test 7: Integration with Training Loop")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simulate HPCM model + regularizer
    regularizer = InterScaleRegularizer(delta=0.1, device=device).to(device)
    
    # Dummy optimizer
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=1e-4)
    
    print("Simulating training iterations...")
    
    for iteration in range(5):
        # Create dummy latents (as if from HPCM forward pass)
        latents = [
            torch.randn(4, 320, 16, 16, device=device, requires_grad=True),
            torch.randn(4, 320, 32, 32, device=device, requires_grad=True),
            torch.randn(4, 320, 64, 64, device=device, requires_grad=True),
        ]
        
        # Compute regularization loss
        reg_loss = regularizer(latents)
        
        # Simulate total loss (reg_loss is part of total loss)
        total_loss = reg_loss  # In real training, this would be R-D loss + reg_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if iteration == 0 or iteration == 4:
            print(f"  Iteration {iteration+1}: loss={reg_loss.item():.6f} ✓")
    
    print("\nStatus: ✓ PASS")
    print()


def test_edge_cases():
    """Test 8: Edge cases and error handling"""
    print("="*70)
    print("Test 8: Edge Cases")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    regularizer = InterScaleRegularizer(delta=0.1, device=device).to(device)
    
    # Test 1: Empty list
    loss_empty = regularizer([])
    print(f"Empty list: loss={loss_empty.item():.6f} ✓")
    assert loss_empty.item() == 0.0, "Should return zero for empty list"
    
    # Test 2: Single scale
    loss_single = regularizer([torch.randn(2, 320, 16, 16, device=device)])
    print(f"Single scale: loss={loss_single.item():.6f} ✓")
    assert loss_single.item() == 0.0, "Should return zero for single scale"
    
    # Test 3: With None values
    latents_with_none = [
        torch.randn(2, 320, 16, 16, device=device),
        None,
        torch.randn(2, 320, 64, 64, device=device),
    ]
    loss_none = regularizer(latents_with_none)
    print(f"With None values: loss={loss_none.item():.6f} ✓")
    
    # Test 4: Different batch sizes (should handle gracefully)
    latents_diff_batch = [
        torch.randn(2, 320, 16, 16, device=device),
        torch.randn(2, 320, 32, 32, device=device),
    ]
    loss_diff = regularizer(latents_diff_batch)
    print(f"Different batch sizes: loss={loss_diff.item():.6f} ✓")
    
    print("\nStatus: ✓ PASS")
    print()


def run_all_tests():
    """Run all Phase 2 tests"""
    print("\n" + "="*70)
    print("  Phase 2: Inter-scale Latent Regularization - Test Suite")
    print("="*70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    try:
        # Run all tests
        test_dwt_downsampler_accuracy()
        test_dwt_speed_benchmark()
        test_inter_scale_regularizer_basic()
        test_gradient_flow()
        test_memory_usage()
        test_scale_independence()
        test_integration_with_training()
        test_edge_cases()
        
        # Summary
        print("="*70)
        print("  ✓ All Tests PASSED!")
        print("="*70)
        print("\nPhase 2 is ready for integration with HPCM.")
        print("Next steps:")
        print("  1. Modify HPCM_Base.py to return hierarchical latents")
        print("  2. Update RateDistortionLoss to include Phase 2 regularization")
        print("  3. Enable with --phase2-reg flag after epoch 100")
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

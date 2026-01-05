"""
Spectral Regularization for Hierarchical Image Compression
Based on: Taming Hierarchical Image Coding Optimization: A Spectral Regularization Perspective

Modules:
- phase1_intra_scale: Intra-scale frequency regularization with DCT truncation
- phase2_inter_scale: Inter-scale latent regularization with DWT downsampling
"""

from .phase1_intra_scale import DCTTransform, SpectralTruncation
from .phase2_inter_scale import DWTDownsampler, InterScaleRegularizer

__all__ = [
    'DCTTransform',
    'SpectralTruncation',
    'DWTDownsampler',
    'InterScaleRegularizer',
]

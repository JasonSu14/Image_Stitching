# This file blends two images together using the specified blending mode

import numpy as np
from scipy.ndimage import distance_transform_edt

def blend_img(dest_img, mask, stitched_img, stitch_mask, blend_mode):
    assert dest_img.dtype == stitched_img.dtype, "Images types do not match"
    assert dest_img.dtype in [np.uint8, np.float32, np.float64], "Images types are not known"
    eps = 1e-10  # small constant
    # Get dimensions
    H, W, C = dest_img.shape

    # Initialize output image as zeros
    out_img = np.zeros((H, W, C), dtype=np.float64)
    input_type = dest_img.dtype

    # Convert inputs to float64 to avoid overflow/underflow issues
    dest_img = dest_img.astype(np.float64)
    stitched_img = stitched_img.astype(np.float64)

    # Create binary masks
    binary_mask_s = mask > 0
    binary_mask_d = stitch_mask > 0

    for c in range(C):
        channel_out = np.zeros((H, W), dtype=np.float64)
        S = dest_img[:, :, c]
        D = stitched_img[:, :, c]
        if blend_mode == 'overlay':
            # s first, then d overwrites s wherever there is overlap.
            channel_out[binary_mask_s] = S[binary_mask_s]
            channel_out[binary_mask_d] = D[binary_mask_d]
        elif blend_mode == 'blend':
            print(f"Mask_s shape: {mask.shape}")
            print(f"Mask_d shape: {stitch_mask.shape}")
            print(f"S shape: {dest_img.shape}")
            print(f"D shape: {stitched_img.shape}")
            # Use distance_transform_edt for Euclidean distance transformation
            weighted_mask_s = distance_transform_edt(np.logical_not(binary_mask_s))
            weighted_mask_d = distance_transform_edt(np.logical_not(binary_mask_d))
            mask_intersection = np.logical_or(binary_mask_s, binary_mask_d)
            channel_out[mask_intersection] = (S[mask_intersection] * weighted_mask_s[mask_intersection] + D[mask_intersection] * weighted_mask_d[mask_intersection]) / (weighted_mask_s[mask_intersection] + weighted_mask_d[mask_intersection] + eps)
        out_img[:, :, c] = channel_out

    # Convert back to the same type as input
    if input_type == np.uint8:
        # clip to 0-255 range after rounding
        out_img = np.round(out_img).clip(0, 255).astype(np.uint8)
    elif input_type == np.float32:
        out_img = out_img.astype(np.float32)
    
    return out_img
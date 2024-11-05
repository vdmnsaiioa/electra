import torch
import torch.nn as nn

def rsh_3(weights: torch.Tensor,
                xyz: torch.Tensor):

    """Computes all real spherical harmonics up to degree 3 with learnable weights."""
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z

    # Compute spherical harmonics without weights
    rsh = torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
            1.09254843059208 * xy,
            -1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            -1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            -0.590043589926644 * y * (3.0 * x2 - y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (1.5 - 7.5 * z2),
            1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
            0.304697199642977 * x * (1.5 - 7.5 * z2),
            1.44530572132028 * z * (x2 - y2),
            -0.590043589926644 * x * (x2 - 3.0 * y2),
        ],
        -1,
    )

    # Multiply the spherical harmonics by the learnable weights
    return rsh * weights

def rsh_2(weights: torch.Tensor,
                xyz: torch.Tensor):

    """Computes all real spherical harmonics up to degree 3 with learnable weights."""
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z

    # Compute spherical harmonics without weights
    rsh = torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
            1.09254843059208 * xy,
            -1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            -1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
        ],
        -1,
    )

    # Multiply the spherical harmonics by the learnable weights
    weighted_rsh = rsh * weights
    return weighted_rsh

import torch
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

import math
from torch.nn.functional import normalize


def compute_sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """
    Compute spherical harmonics basis functions up to a given degree.
    
    Args:
        degree (int): Maximum SH degree (e.g., 3).
        dirs (torch.Tensor): [N, 3] normalized directions.
    
    Returns:
        torch.Tensor: [N, (degree+1)^2] SH basis values.
    """
    x, y, z = dirs.unbind(-1)
    basis = []
    
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            if l == 0:
                basis.append(torch.ones_like(x))  # l=0, m=0
            elif l == 1:
                if m == -1: basis.append(y)       # l=1, m=-1
                elif m == 0: basis.append(z)      # l=1, m=0
                elif m == 1: basis.append(x)      # l=1, m=1
            elif l == 2:
                # Precomputed terms for l=2 (simplified)
                xx, yy, zz = x**2, y**2, z**2
                xy, yz, xz = x*y, y*z, x*z
                if m == -2: basis.append(xy)
                elif m == -1: basis.append(yz)
                elif m == 0: basis.append(3*zz - 1)  # Simplified form
                elif m == 1: basis.append(xz)
                elif m == 2: basis.append(xx - yy)
            # Extend to higher degrees (l=3) if needed...
    
    return torch.stack(basis, dim=-1)  # [N, (degree+1)^2]

def spherical_harmonics(degree: int, dirs: torch.Tensor, sh_coeffs: torch.Tensor) -> torch.Tensor:
    """
    Compute RGB colors from SH coefficients and view directions.
    
    Args:
        degree (int): SH degree (e.g., 3 for PLY files with 16 coefficients per channel).
        dirs (torch.Tensor): [N, 3] normalized view directions.
        sh_coeffs (torch.Tensor): [N, (degree+1)^2, 3] SH coefficients (including DC term).
    
    Returns:
        torch.Tensor: [N, 3] RGB colors.
    """
    basis = compute_sh_basis(degree, dirs)  # [N, (degree+1)^2]
    return torch.einsum('npc,nc->np', sh_coeffs, basis)  # [N, 3]


def render_gaussians(gaussians, camera):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussians = {k: v.to(device).to(torch.float32) for k, v in gaussians.items()}
    
    # Compute view direction from camera to Gaussians
    viewmat = camera["viewmat"].to(device)
    # camera_position = torch.inverse(viewmat[:3, :3]) @ (-viewmat[:3, 3])
    # view_dirs = normalize(camera_position - gaussians["means"])
    camera_position = torch.inverse(viewmat[:3, :3]) @ (-viewmat[:3, 3])
    view_dirs = normalize(camera_position - gaussians["means"])
    
    # Process SH coefficients (from PLY's f_dc and f_rest)
    # gaussians["colors"]  is Nx48
    f_dc = gaussians["colors"][:, :3]  # [N,3]
    f_rest = gaussians["colors"][:, 3:].reshape(-1, 15, 3)  # [N,15,3]
    # f_dc = gaussians["colors"]  # [N,3]
    # f_rest = gaussians["colorsh"].reshape(-1, 15, 3)  # [N,15,3]
    sh_coeffs = torch.cat([f_dc.unsqueeze(1), f_rest], dim=1)  # [N,16,3]
    
    # Compute colors with SH
    # colors = spherical_harmonics(3, view_dirs, sh_coeffs)
    # colors = torch.sigmoid(colors)  # Constrain to [0,1]
    colors = spherical_harmonics(3, view_dirs, sh_coeffs)
    colors = torch.sigmoid(colors)  # ✅

    # Normalize quaternions
    quats = gaussians["quats"] / gaussians["quats"].norm(dim=1, keepdim=True)
    
    # Rasterize
    rgb, alpha, _ = rasterization(
        means=gaussians["means"],
        quats=quats,
        scales=gaussians["scales"],
        opacities=gaussians["opacities"].squeeze(),
        colors=colors,
        viewmats=viewmat[None],
        Ks=camera["K"].to(device)[None],
        width=camera["width"],
        height=camera["height"],
    )
    
    return rgb.squeeze(0).cpu(), alpha.squeeze(0).cpu()

def render_gaussiansX(gaussians, camera):
    """
    Render Gaussians using gsplat.
    gaussians: Dict with keys ['means', 'quats', 'scales', 'opacities', 'colors']
    camera: A camera object with intrinsics & extrinsics
    Returns: tuple of (image, alpha) tensors
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move all tensors in gaussians to device and convert to float32
    gaussians = {k: v.to(device).to(torch.float32) for k, v in gaussians.items()}
    N = gaussians["means"].shape[0]  
    # Number of Gaussians
    sh_degree = 3
    num_sh_coeffs = (sh_degree + 1) ** 2
    
    gaussians["sh_degree"] = sh_degree
    gaussians["sh_coeffs"] = torch.zeros(N, num_sh_coeffs, 3).to(device).to(torch.float32)  # Reshape to (N, 16, 3)

    # Initialize other Gaussian parameters
    gaussians['means'] = torch.randn((N, 3), device=device)
    gaussians['quats']  = torch.randn((N, 4), device=device)
    gaussians['scales'] = torch.rand((N, 3), device=device) * 0.1
    gaussians['opacities'] = torch.rand((N,), device=device)
    gaussians["sh_coeffs"] = torch.randn(N, num_sh_coeffs, 3).to(device).to(torch.float32) 
    
   
    
  
    # Flatten quaternions if needed
    quats = gaussians["quats"] / gaussians["quats"].norm(dim=1, keepdim=True)
    
    # Get view directions from quaternions
    view_dirs = quat_to_direction(quats)

    # Convert SH colors to RGB if used
    if "sh_degree" in gaussians:
        from gsplat import spherical_harmonics
        colors = spherical_harmonics(gaussians["sh_degree"], view_dirs,gaussians["sh_coeffs"])
        # colors = spherical_harmonics(sh_degree, view_dirs, sh_coeffs)  # Ensure shapes match

    else:
        colors = gaussians["colors"]

    # Flatten opacities to match expected shape
    opacities = gaussians["opacities"].squeeze(-1)  # Convert from [N, 1] to [N]

    # Ensure view matrix and camera matrix are on the correct device and type
    viewmat = (camera["viewmat"].to(device).to(torch.float32) 
               if isinstance(camera["viewmat"], torch.Tensor) 
               else torch.tensor(camera["viewmat"], device=device, dtype=torch.float32))
    
    K = (camera["K"].to(device).to(torch.float32) 
         if isinstance(camera["K"], torch.Tensor) 
         else torch.tensor(camera["K"], device=device, dtype=torch.float32))

    # Rasterize
    # The rasterization function returns (rgb, alpha, depth)
    rgb, alpha, _ = rasterization(
        means=gaussians["means"],
        quats=quats,
        scales=gaussians["scales"],
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=camera["width"],
        height=camera["height"],
    )
    
    
    # Remove batch dimension and ensure proper shape for visualization
    rgb = rgb.squeeze(0)    # Remove batch dimension (1, H, W, C) -> (H, W, C)
    alpha = alpha.squeeze(0) # Remove batch dimension (1, H, W) -> (H, W)
    # Denormalize rgb to [0, 1] using min-max normalization
    rgb_min = rgb.amin()
    rgb_max = rgb.amax()
    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    else:
        rgb = torch.zeros_like(rgb)
    # rgb values are in [0, 1] range, convert to [0, 255] for visualization
    # [0.79297596..1.0482266].
    
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)  # Convert to uint8
    
    return rgb.cpu(), alpha.cpu()  # Return results on CPU for visualization


def quat_to_direction(quats):
    """
    Convert quaternions to direction vectors.
    quats: torch.Tensor of shape (N, 4) representing quaternions [w, x, y, z]
    Returns: torch.Tensor of shape (N, 3) representing direction vectors
    """
    # Ensure quaternions are normalized
    quats = quats / quats.norm(dim=1, keepdim=True)
    
    # Extract components
    w, x, y, z = quats.unbind(-1)
    
    # Convert quaternion to direction vector (forward direction = +z)
    # This transforms the vector [0, 0, 1] by the quaternion rotation
    direction = torch.stack([
        2 * (x*z + w*y),
        2 * (y*z - w*x),
        1 - 2 * (x*x + y*y)
    ], dim=-1)
    
    return direction



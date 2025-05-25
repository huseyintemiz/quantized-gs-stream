from plyfile import PlyData
import numpy as np
import torch
from io import BytesIO
from typing import Optional
from plyfile import PlyData, PlyElement
import math
import os
import torch.nn.functional as F


def read_gaussian_ckpt(ckpt_path: str, device: str = 'cuda') -> dict[str, torch.Tensor]:
    """
    Reads a Gaussian checkpoint file and returns a dictionary with Gaussian parameters as PyTorch tensors.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (str): Device to load the checkpoint on ('cuda' or 'cpu').

    Returns:
        dict[str, torch.Tensor]: Dictionary containing Gaussian parameters as PyTorch tensors.
    """
    # Load the checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)["splats"]
    
    # Extract Gaussian parameters
    gauss_data = {}
    gauss_data['means'] = ckpt["means"]
    gauss_data['quats'] = ckpt["quats"]
    gauss_data['scales'] = ckpt["scales"]
    gauss_data['opacities'] = ckpt["opacities"]
    gauss_data['sh0'] = ckpt["sh0"]  # nx1x3
    gauss_data['shN'] = ckpt["shN"]  # nxKx3
    
    print("Number of Gaussians:", len(gauss_data['means']))
  
    # Concatenate sh0 and shN along the second dimension
    gauss_data['colors'] = torch.cat([gauss_data['sh0'], gauss_data['shN']], dim=1)  # Resulting shape: [68739, 16, 3]
    
    return gauss_data
           
def save_gaussian_ckpt(ckpt_path: str, data: dict[str, torch.Tensor]) -> None:
    """
    Saves Gaussian splat details to a checkpoint file.

    Args:
        ckpt_path (str): Path to save the checkpoint file.
        data (dict[str, torch.Tensor]): Dictionary containing Gaussian parameters as PyTorch tensors.

    Returns:
        None
    """
    # Create a dictionary to save
    save_dict = {
        "means": data['means'],
        "quats": data['quats'],
        "scales": data['scales'],
        "opacities": data['opacities'],
        "sh0": data['colors'][:, 0:1, :],
        "shN": data['colors'][:, 1:, :],
    }
    
    data_to_save = {"step": 0, "splats": save_dict}

    # Save the data to the specified checkpoint path
    torch.save(data_to_save, ckpt_path)

def read_gaussian_ply(ply_path: str) -> dict[str, np.ndarray]:
    """
    Reads a PLY file storing Gaussian splat details and returns a dictionary with the data.

    Args:
        ply_path (str): Path to the PLY file.

    Returns:
        dict[str, np.ndarray]: Dictionary containing the PLY data with keys like 'x', 'y', 'z', etc.
    """
    # Read the PLY file
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex'].data

    # Convert to a dictionary of numpy arrays
    data = {}
    for name in vertex.dtype.names:
        data[name] = np.array(vertex[name])
    
    return data
  
def convert_gaussian_tensor_np2pt(ply_path: str, sh_degree: int = 3, device: str = 'cuda') -> dict[str, torch.Tensor]:
    """
    Converts Gaussian data from a PLY file to PyTorch tensors.

    Args:
        ply_path (str): Path to the PLY file.
        sh_degree (int): Degree of spherical harmonics. Default is 3.
        device (str): Device to load the tensors on ('cuda' or 'cpu'). Default is 'cuda'.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing Gaussian parameters as PyTorch tensors.
    """
    num_sh_coeffs = (sh_degree + 1) ** 2
    num_sh_rest = (num_sh_coeffs-1)*3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read Gaussian data from the PLY file
    given_gs = read_gaussian_ply(ply_path)
    
    gauss_data = {}
    gauss_data['means'] = np.stack([given_gs['x'], given_gs['y'], given_gs['z']], axis=1)
    gauss_data['quats'] = np.stack([given_gs['rot_0'], given_gs['rot_1'], given_gs['rot_2'], given_gs['rot_3']], axis=1)
    gauss_data['scales'] = np.stack([given_gs['scale_0'], given_gs['scale_1'], given_gs['scale_2']], axis=1)

    gauss_data['colors'] = np.concatenate([
        np.stack([given_gs['f_dc_0'], given_gs['f_dc_1'], given_gs['f_dc_2']], axis=1),  # DC terms (N, 3)
        np.stack([given_gs[f'f_rest_{i}'] for i in range(num_sh_rest)], axis=1)  # SH coefficients (N, K)
    ], axis=1)  # Resulting shape: (N, num_sh_coeffs, 3)

    gauss_data['opacities'] = given_gs['opacity'].reshape(-1, 1).squeeze(1)  # Opacity

    # Convert numpy arrays to PyTorch tensors
    for k in gauss_data:
        if isinstance(gauss_data[k], np.ndarray):
            gauss_data[k] = torch.tensor(gauss_data[k], dtype=torch.float32).to(device)

    # Reshape colors to match the expected shape
    gauss_data['colors'] = gauss_data['colors'].reshape(-1, num_sh_coeffs, 3)

    return gauss_data
            
def save_gaussian_ply(ply_path: str, data: dict[str, np.ndarray]) -> None:
    """
    Saves Gaussian splat details to a PLY file.

    Args:
        ply_path (str): Path to save the PLY file.
        data (dict[str, np.ndarray]): Dictionary containing Gaussian parameters as NumPy arrays.

    Returns:
        None
    """
    # Create a structured array for the PLY file
    vertex = np.zeros(data['x'].shape[0], dtype=[(name, 'f4') for name in data.keys()])
    for name in data.keys():
        vertex[name] = data[name]

    # Create a PlyData object
    plydata = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
    # plydata = PlyData([('vertex', vertex)], text=True)

    # Write to the specified PLY file path
    plydata.write(ply_path)
    
def splat2ply_bytes(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
) -> bytes:
    """Return the binary Ply file. Supported by almost all viewers.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 3)
        shN (torch.Tensor): Spherical harmonics. Shape (N, K*3)

    Returns:
        bytes: Binary Ply file representing the model.
    """

    num_splats = means.shape[0]
    buffer = BytesIO()

    # Write PLY header
    buffer.write(b"ply\n")
    buffer.write(b"format binary_little_endian 1.0\n")
    buffer.write(f"element vertex {num_splats}\n".encode())
    buffer.write(b"property float x\n")
    buffer.write(b"property float y\n")
    buffer.write(b"property float z\n")
    for i, data in enumerate([sh0, shN]):
        prefix = "f_dc" if i == 0 else "f_rest"
        for j in range(data.shape[1]):
            buffer.write(f"property float {prefix}_{j}\n".encode())
    buffer.write(b"property float opacity\n")
    for i in range(scales.shape[1]):
        buffer.write(f"property float scale_{i}\n".encode())
    for i in range(quats.shape[1]):
        buffer.write(f"property float rot_{i}\n".encode())
    buffer.write(b"end_header\n")

    # Concatenate all tensors in the correct order
    splat_data = torch.cat(
        [means, sh0, shN, opacities.unsqueeze(1), scales, quats], dim=1
    )
    # Ensure correct dtype
    splat_data = splat_data.to(torch.float32)

    # Write binary data
    float_dtype = np.dtype(np.float32).newbyteorder("<")
    buffer.write(splat_data.detach().cpu().numpy().astype(float_dtype).tobytes())

    return buffer.getvalue()

def export_splats(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    # format: Literal["ply", "splat", "ply_compressed"] = "ply",
    save_to: Optional[str] = None,
) -> bytes:
    """Export a Gaussian Splats model to bytes.
    The three supported formats are:
    - ply: A standard PLY file format. Supported by most viewers.
    - splat: A custom Splat file format. Supported by antimatter15 viewer.
    - ply_compressed: A compressed PLY file format. Used by Supersplat viewer.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 1, 3)
        shN (torch.Tensor): Spherical harmonics. Shape (N, K, 3)
        format (str): Export format. Options: "ply", "splat", "ply_compressed". Default: "ply"
        save_to (str): Output file path. If provided, the bytes will be written to file.
    """
    total_splats = means.shape[0]
    assert means.shape == (total_splats, 3), "Means must be of shape (N, 3)"
    assert scales.shape == (total_splats, 3), "Scales must be of shape (N, 3)"
    assert quats.shape == (total_splats, 4), "Quaternions must be of shape (N, 4)"
    assert opacities.shape == (total_splats,), "Opacities must be of shape (N,)"
    assert sh0.shape == (total_splats, 1, 3), "sh0 must be of shape (N, 1, 3)"
    assert (
        shN.ndim == 3 and shN.shape[0] == total_splats and shN.shape[2] == 3
    ), f"shN must be of shape (N, K, 3), got {shN.shape}"

    # Reshape spherical harmonics
    sh0 = sh0.squeeze(1)  # Shape (N, 3)
    shN = shN.permute(0, 2, 1).reshape(means.shape[0], -1)  # Shape (N, K * 3)

    # Check for NaN or Inf values
    invalid_mask = (
        torch.isnan(means).any(dim=1)
        | torch.isinf(means).any(dim=1)
        | torch.isnan(scales).any(dim=1)
        | torch.isinf(scales).any(dim=1)
        | torch.isnan(quats).any(dim=1)
        | torch.isinf(quats).any(dim=1)
        | torch.isnan(opacities).any(dim=0)
        | torch.isinf(opacities).any(dim=0)
        | torch.isnan(sh0).any(dim=1)
        | torch.isinf(sh0).any(dim=1)
        | torch.isnan(shN).any(dim=1)
        | torch.isinf(shN).any(dim=1)
    )

    # Filter out invalid entries
    valid_mask = ~invalid_mask
    means = means[valid_mask]
    scales = scales[valid_mask]
    quats = quats[valid_mask]
    opacities = opacities[valid_mask]
    sh0 = sh0[valid_mask]
    shN = shN[valid_mask]


    data = splat2ply_bytes(means, scales, quats, opacities, sh0, shN)
  

    if save_to:
        with open(save_to, "wb") as binary_file:
            binary_file.write(data)

    return data



# add main
if __name__ == "__main__":
    #######################################################
    # PLY FILE Operations check
    ply_path =  "/home/dipcik/avatar/gs-env/gsplat/examples/results/benchmark/man/ply/point_cloud_9999.ply"
    details = read_gaussian_ply(ply_path)
    
    print(details.keys())  # e.g. dict_keys(['x', 'y', 'z', 'nx', 'ny', 'nz', 'scale', 'r', 'g', 'b'])
    print(f"Number of Gaussians: {details['x'].shape[0]}")
    
    gauss_data = convert_gaussian_tensor_np2pt(ply_path)
    
    ply_export_path = "results/ply_export.ply"
    export_splats(
        means=gauss_data['means'],
        scales=gauss_data['scales'],
        quats=gauss_data['quats'],
        opacities=gauss_data['opacities'],
        sh0=gauss_data['colors'][:, 0:1, :],
        shN=gauss_data['colors'][:, 1:, :],
        save_to=ply_export_path,
    )
    if os.path.exists(ply_export_path):
        print("Successfully exported ply_export.ply")
    else:
        print("Export failed")
    
    #######################################################
    # PT FILE Operations check
    ckpt_path = "/home/dipcik/avatar/gs-env/gsplat/examples/results/benchmark/man/ckpts/ckpt_9999_rank0.pt"
    gaussians = read_gaussian_ckpt(ckpt_path)
    
    print(gaussians.keys())  # e.g. dict_keys(['means', 'quats', 'scales', 'opacities', 'sh0', 'shN'])
    save_gaussian_ckpt(
        ckpt_path="results/gaussian_ckpt.pt",
        data=gaussians
    )
    if os.path.exists("results/gaussian_ckpt.pt"):
        print("Successfully saved gaussian_ckpt.pt")
    else:
        print("Save failed")

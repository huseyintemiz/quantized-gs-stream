from plyfile import PlyData
import numpy as np
import torch
from io import BytesIO
from typing import Optional
from plyfile import PlyData, PlyElement
import math
import torch.nn.functional as F

def read_gaussian_ply(ply_path):
    """
    Reads a PLY file storing Gaussian splat details.
    Returns a dict with keys like 'x', 'y', 'z', 'nx', 'ny', 'nz', 'scale', 'r', 'g', 'b', etc.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex'].data

    # Convert to numpy structured array for easy access
    data = {}
    for name in vertex.dtype.names:
        data[name] = np.array(vertex[name])
    
    return data
  
def read_gaussian_ckpt(ckpt_path,device='cuda'):

   
    ckpt = torch.load(ckpt_path, map_location=device)["splats"]
    
    gauss_data = {}
    gauss_data['means'] = ckpt["means"]
    gauss_data['quats'] = ckpt["quats"]
    gauss_data['scales'] = ckpt["scales"]
    gauss_data['opacities'] = ckpt["opacities"]
    gauss_data['sh0'] = ckpt["sh0"] #nx1x3
    gauss_data['shN'] = ckpt["shN"] # nxKx3
    
    print("Number of Gaussians:", len( gauss_data['means']))
  

    # Concatenate sh0 and shN along the second dimension
    gauss_data['colors'] = torch.cat([gauss_data['sh0'], gauss_data['shN']], dim=1)  # Resulting shape: [68739, 16, 3]    # gauss_data['sh_degree'] = sh_degree
    
    gauss_data['scales'] = torch.sigmoid(gauss_data['scales'])
    
    return gauss_data
        
        
def save_gaussian_ckpt(ckpt_path, data):
    """
    Saves Gaussian splat details to a checkpoint file.
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
    # ckpt = {"splats": save_dict,
    #         "step": 0}
    # Save the dictionary to a file
    
    data = {"step": 0, "splats": save_dict}
    # torch.save(ckpt.state_dict(), ckpt_path)
    
    torch.save(
                    data, ckpt_path
                )
    
    
def convert_gaussian_tensor(ply_path):
    num_sh_coeffs = 16
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    given_gs = read_gaussian_ply(ply_path)
    
    gauss_data = {}
    
    gauss_data['means'] = np.stack([given_gs['x'], given_gs['y'], given_gs['z']], axis=1)
    #concat rot_0, rot_1, rot_2, rot_3 to r
    gauss_data['quats'] = np.stack([given_gs['rot_0'], given_gs['rot_1'], given_gs['rot_2'], given_gs['rot_3']], axis=1)
    gauss_data['scales'] = np.stack([given_gs['scale_0'], given_gs['scale_1'], given_gs['scale_2']], axis=1)

    gauss_data['colors'] = np.concatenate([
        np.stack([given_gs['f_dc_0'], given_gs['f_dc_1'], given_gs['f_dc_2']], axis=1),  # DC terms (N, 3)
        np.stack([given_gs[f'f_rest_{i}'] for i in range(45)], axis=1)  # SH coefficients (N, 45)
    ], axis=1)  # Resulting shape: (N, 48)

    gauss_data['opacities'] = given_gs['opacity'].reshape(-1, 1).squeeze(1)  # opacity
    # Stack f_rest_0 to f_rest_44 into gauss_data['f_rest'] 
    
    for k in gauss_data:
        if isinstance(gauss_data[k], np.ndarray):
            gauss_data[k] = torch.tensor(gauss_data[k], dtype=torch.float32).to(device)
    
    ##################################
    # randomly color  ### kendi renkleri sacmaliyor anlamadim  
    # N =  gauss_data['means'].shape[0]
    num_sh_coeffs = 16
    # sh_coeffs = torch.randn((N, num_sh_coeffs, 3))
    # # Normalize DC term (first coefficient) to be in [0,1] range
    # sh_coeffs[:, 0] = torch.sigmoid(sh_coeffs[:, 0])
    # sh_coeffs = sh_coeffs.to(device)
    
    # Disable random coloring
    # gauss_data['colors'] = sh_coeffs
    gauss_data['colors'] = gauss_data['colors'].reshape(-1, num_sh_coeffs, 3)
    # gauss_data['colors'][:, 0] = torch.sigmoid(gauss_data['colors'][:, 0])
    
    # Apply sigmoid to gauss_data['scales']
    gauss_data['scales'] = torch.sigmoid(gauss_data['scales'])
    ##################################

    return gauss_data
            
def save_gaussian_ply(ply_path, data):
    """
    Saves Gaussian splat details to a PLY file.
    """
    # Create a structured array
    vertex = np.zeros(data['x'].shape[0], dtype=[(name, 'f4') for name in data.keys()])
    for name in data.keys():
        vertex[name] = data[name]

    # Create a PlyData object
    plydata = PlyData([('vertex', vertex)], text=True)

    # Write to file
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
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python reader.py <path_to_ply>")
    #     sys.exit(1)
    # ply_path = sys.argv[1]   

    # Example usage:
    # details = read_gaussian_ply('gs-data/scannet_1.ply')
    # details = read_gaussian_ply('points3d.ply')

    # details = read_gaussian_ply('/home/dipcik/avatar/vq-stream/compressed_model/point_cloud.ply')
    # ply_path = 'gs-data/gaussians2v.ply'
    ply_path = "/home/dipcik/avatar/vq-stream/gs-data/gaussian_splatting/FO_dataset/drjohnson/point_cloud/iteration_30000/point_cloud.ply"
    ply_path ="/home/dipcik/avatar/gs-env/gsplat/examples/results/benchmark/room/ply/point_cloud_9999.ply"
    ply_path =  "/home/dipcik/avatar/gs-env/gsplat/examples/results/benchmark/man/ply/point_cloud_9999.ply"
    details = read_gaussian_ply(ply_path)
    
    print(details.keys())  # e.g. dict_keys(['x', 'y', 'z', 'nx', 'ny', 'nz', 'scale', 'r', 'g', 'b'])
    print(details['x'].shape)  # e.g. (N,)
    
    gauss_data = convert_gaussian_tensor(ply_path)
    
 
    # data = splat2ply_bytes(means, scales, quats, opacities, sh0, shN)
    
    export_splats(
        means=gauss_data['means'],
        scales=gauss_data['scales'],
        quats=gauss_data['quats'],
        opacities=gauss_data['opacities'],
        sh0=gauss_data['colors'][:, 0:1, :],
        shN=gauss_data['colors'][:, 1:, :],
        # sh0=sh0,
        # shN=shN,
        # format="ply",
        save_to=f"abcdd.ply",
    )
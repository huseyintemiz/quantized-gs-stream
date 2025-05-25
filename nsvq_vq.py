import torch
import numpy as np
from sklearn.cluster import KMeans

from gsplat import rasterization
from PIL import Image
import constriction

from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch import ResidualVQ
from reader import read_gaussian_ply,export_splats,convert_gaussian_tensor_np2pt,read_gaussian_ckpt,save_gaussian_ckpt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def nsvq_quantize(input_vec, codebook):
    """
    Apply NSVQ quantization.
    input_vec: Tensor of shape [B, D]
    codebook: Codebook object
    """
    with torch.no_grad():
        # indices = codebook.encode(input_vec)
        reconst, indices, _ = codebook(input_vec)

        quantized = codebook.get_output_from_indices(indices)

    # Compute residual error
    residual = input_vec - quantized

    # Add noise in the direction of residual
    noise = torch.randn_like(residual)
    noise = noise / noise.norm(dim=-1, keepdim=True) * residual.norm(dim=-1, keepdim=True)

    # Forward pass uses hard quantized vector, backward uses soft noised version
    output = quantized + noise
    return output, indices    

def render_image(gaussians, camera,sh_degree=3):
    
    # N = gaussians['means'].shape[0]  # Number of Gaussians
    color_sh_param = (sh_degree + 1) ** 2  # K = 16 for sh_degree=3
    
    # 
    gaussians['scales'] = torch.sigmoid(gaussians['scales']) #huseyin
    
    colors_rendered, alphas, meta = rasterization(gaussians['means'], gaussians['quats'], gaussians['scales'],gaussians['opacities'],gaussians['colors'], camera['viewmat'], camera["K"], camera['width'], camera['height'],sh_degree=sh_degree)
    # Convert rendered image to numpy and save
    image_np = (colors_rendered[0].cpu().numpy() * 255).astype(np.uint8)  # Convert to uint8
    image = Image.fromarray(image_np)  # Remove alpha channel
    image.save("results/rasterized_gaussians.png")
    
    return image_np,alphas, meta
    
def plot_gaussian_splats_3D(gaussians, camera,camera_extras):

    cam_pos = camera['viewmat'][0, :3, 3]
    
    # Compute the bounding box of the splats
    means = gaussians['means']
    if isinstance(means, torch.Tensor):
        min_coords, max_coords = means.min(dim=0).values, means.max(dim=0).values
    else:
        min_coords, max_coords = torch.tensor(means).min(dim=0).values, torch.tensor(means).max(dim=0).values

    # Compute the centroid and diagonal of the bounding box
    centroid = (min_coords + max_coords) / 2
    bounding_box_diagonal = torch.norm(max_coords - min_coords)

    # Place the camera at a distance proportional to the bounding box diagonal
    cam_distance = bounding_box_diagonal * 2.0  # Adjust multiplier for zoom level
    cam_pos = centroid + torch.tensor([0, 0, cam_distance], device=centroid.device, dtype=centroid.dtype)

    # Look-at matrix (camera at cam_pos, looking at centroid, up is +y)
        
        

    # Scatter all Gaussian splats
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter all points (means)
    means_np = gaussians['means'].cpu().numpy()  # Convert to numpy for plotting
    ax.scatter(means_np[:, 0], means_np[:, 1], means_np[:, 2], c='b', s=1, label='Gaussians')

    # Plot bounding box
    ax.scatter(min_coords[0].item(), min_coords[1].item(), min_coords[2].item(), c='r', label='Min')
    ax.scatter(max_coords[0].item(), max_coords[1].item(), max_coords[2].item(), c='g', label='Max')

    # Plot centroid
    ax.scatter(centroid[0].item(), centroid[1].item(), centroid[2].item(), c='orange', label='Centroid')

    # Plot camera position
    ax.scatter(cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item(), c='y', label='Camera')

    # Add camera plane
    def add_camera_plane(ax, cam_pos, viewmat, K, width, height, scale=1.0):
        # Compute camera plane corners in camera space
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        plane_corners = torch.tensor([
            [-cx / fx, -cy / fy, 1],  # Bottom-left
            [ (width - cx) / fx, -cy / fy, 1],  # Bottom-right
            [ (width - cx) / fx, (height - cy) / fy, 1],  # Top-right
            [-cx / fx, (height - cy) / fy, 1],  # Top-left
        ], device=cam_pos.device, dtype=cam_pos.dtype) * scale

        # Transform plane corners to world space
        plane_corners_world = (viewmat.inverse()[:3, :3] @ plane_corners.T).T + cam_pos

        # Plot the camera plane
        plane_corners_np = plane_corners_world.cpu().numpy()
        verts = [plane_corners_np]
        ax.add_collection3d(Poly3DCollection(verts, color='cyan', alpha=0.3))

    # Add the camera plane to the plot
    add_camera_plane(ax, cam_pos, viewmat[0], K[0], image_width, image_height)

    # Add camera view direction as an arrow
    # view_direction = -viewmat[0, :3, 2]  # Camera's forward direction (negative z-axis in view space)
    # Extract camera position and view direction
    view_direction = camera['viewmat'][0, :3, 2]  # Camera's forward direction (negative z-axis in view space)
    view_direction = -view_direction  # Invert the z-direction
        
    ax.quiver(
        cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item(),  # Camera position
        view_direction[0].item(), view_direction[1].item(), view_direction[2].item(),  # Direction vector
        color='red', length=bounding_box_diagonal.item() * 0.5, normalize=True, label='View Direction'
    )
    
    # Add labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Save or show the plot
    plt.savefig("results/scatter_with_camera_plane.png")
 
    print('Done 3d plot')

def plotly_splat_3D(gaussians, camera,camera_extras):
    # Convert Gaussian means to numpy
    means_np = gaussians['means'].cpu().numpy()
    
    # Place the camera at a distance proportional to the bounding box diagonal    
    cam_pos = camera_extras['cam_pos'].cpu().numpy()
    cam_distance = camera_extras['cam_distance'].item()
    viewmat = camera['viewmat'][0].cpu().numpy()
    
    # Extract camera position and view direction
    view_direction = camera['viewmat'][0, :3, 2]  # Camera's forward direction (negative z-axis in view space)
    print(view_direction)

    # Create a 3D scatter plot
    fig = go.Figure()

    # Add Gaussian splats as scatter points
    fig.add_trace(go.Scatter3d(
        x=means_np[:, 0],
        y=means_np[:, 1],
        z=means_np[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',  # Color of the points
            opacity=0.8
        ),
        name='Gaussians'
    ))

    # Add camera position as a point
    fig.add_trace(go.Scatter3d(
        x=[cam_pos[0].item()],
        y=[cam_pos[1].item()],
        z=[cam_pos[2].item()],
        mode='markers',
        marker=dict(
            size=5,
            color='yellow',
            opacity=1.0
        ),
        name='Camera'
    ))


    
    fig.add_trace(go.Scatter3d(
        x=[cam_pos[0].item(), cam_pos[0].item() + view_direction[0].item()],
        y=[cam_pos[1].item(), cam_pos[1].item() + view_direction[1].item()],
        z=[cam_pos[2].item(), cam_pos[2].item() + view_direction[2].item()],
        mode='lines',
        line=dict(
            color='red',
            width=3
        ),
        name='View Direction'
    ))

    # Set scene layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # Keep aspect ratio consistent
        ),
        title="Interactive 3D Scene",
        showlegend=True
    )
    # Add camera plane as a mesh
    # Compute camera plane corners in camera space
    K = camera['K'][0].cpu().numpy()  # Convert to numpy for easier indexing
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
  
    width, height = camera['width'], camera['height']
    scale = 1.0


    plane_corners = np.array([
        [-cx / fx, -cy / fy, 1],  # Bottom-left
        [(width - cx) / fx, -cy / fy, 1],  # Bottom-right
        [(width - cx) / fx, (height - cy) / fy, 1],  # Top-right
        [-cx / fx, (height - cy) / fy, 1],  # Top-left
    ]) * scale

    # Transform plane corners to world space
    R = np.linalg.inv(viewmat[:3, :3])
    t = cam_pos
    plane_corners_world = (R @ plane_corners.T).T + t

    # Add camera plane as a mesh
    x, y, z = plane_corners_world[:, 0], plane_corners_world[:, 1], plane_corners_world[:, 2]
    i = [0, 1, 2, 0]
    j = [1, 2, 3, 3]
    k = [2, 3, 0, 1]
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        color='cyan',
        opacity=0.3,
        alphahull=0,
        name='Camera Plane',
        i=[0, 1, 2, 0],
        j=[1, 2, 3, 3],
        k=[2, 3, 0, 1],
        showscale=False
    ))

    # Show the plot
    fig.show()

class NSVQCompressor:
    def __init__(self, config):
        self.config = config
        self.codebooks = {}

    def train_codebooks(self, gaussian_data):
        """
        Train codebooks using KMeans from training data.
        gaussian_data: dict of tensors {'s', 'r', 'c', 'csh'}
        """
        print("Training codebooks...")
        for attr in ['scales', 'quats', 'colors']:
      
            data = gaussian_data[attr]#.cpu().numpy()
            # codebook = Codebook(num_codewords=self.config[f'K_{attr}'], dim=data.shape[-1])
            # codebook = VectorQuantize(codebook_size=self.config[f'K_{attr}'],
            #                             dim=data.shape[-1],
            #                             kmeans_init = True,
            #                             kmeans_iters = 10)
            # Mean pixel-wise L1 difference: 5.434
            # Mean pixel-wise L1 difference: 4.983
            # Mean pixel-wise L1 difference: 11.206
            # Mean pixel-wise L1 difference: 7.449
            # Mean pixel-wise L1 difference: 3.141
            
            # codebook = VectorQuantize(
            #     dim=data.shape[-1],
            #     codebook_size=self.config[f'K_{attr}'],
            #     # codebook_dim=16,               # lower-dimensional codes
            #     # heads=8,                       # multiple heads
            #     separate_codebook_per_head=False,  # shared codebook
            #     use_cosine_sim=True,           # normalize vectors
            #     kmeans_init=True,
            #     threshold_ema_dead_code=2      # replace dead codes
            # )
            # Mean pixel-wise L1 difference: 18.179
            # Mean pixel-wise L1 difference: 47.290
            # Mean pixel-wise L1 difference: 25.972

            codebook = ResidualVQ(
                dim=data.shape[-1],              # size of each splat parameter vector
                codebook_size=self.config[f'K_{attr}'],    # number of entries in each codebook
                num_quantizers=4,      # more quantizers = finer approximation
                shared_codebook=False,  # share codebook across quantizers
                stochastic_sample_codes=False,
                kmeans_init=True       # better initialization
            ) 
            #Mean pixel-wise L1 difference: 1.578 
            # Mean pixel-wise L1 difference: 0.889
            # Mean pixel-wise L1 difference: 0.620
            # Mean pixel-wise L1 difference: 0.759
            # Mean pixel-wise L1 difference: 2.710
    
            self.codebooks[attr] = codebook.to(device=data.device)
            
       
        print("Codebooks trained.")

    def compress_frame(self, gaussians):
        """
        Compress one frame of Gaussians.
        gaussians: dict of tensors {'x', 'o', 's', 'r', 'c', 'csh'}
        Returns: dict of quantized indices
        """
        quantized_indices = {}
        with torch.no_grad():
            for attr in  ['scales', 'quats', 'colors']:
                _, indices, _ = self.codebooks[attr](gaussians[attr])

                quantized_indices[attr] = indices
                self.codebooks[attr].eval()
                
        return quantized_indices

    def decompress_frame(self, base_gaussians, compressed_data):
        """
        Reconstruct Gaussians from compressed indices.
        base_gaussians: dict containing unquantized attributes ('x', 'o')
        compressed_data: dict of quantized indices {'s', 'r', 'c', 'csh'}
        Returns: reconstructed Gaussians dict
        """
        recon = {k: v.clone() for k, v in base_gaussians.items()}
        for attr in ['scales', 'quats', 'colors']:
      
            # recon[attr] = self.codebooks[attr].decode(compressed_data[attr])
            recon[attr] = self.codebooks[attr].get_output_from_indices(compressed_data[attr])
        return recon
   
def entropy_encode(indices_tensor, symbol_min=0, symbol_max=10000):
    """
    Encode a tensor of indices into a bitstream using arithmetic coding.
    """
    # Convert to numpy and flatten
    indices = indices_tensor.long().cpu().numpy().flatten()
    
    # Create probability model (uniform distribution over the symbol range)
    num_symbols = symbol_max - symbol_min + 1
    pmf = np.ones(num_symbols) / num_symbols
    
    # Shift indices to be non-negative and ensure correct dtype
    shifted_indices = (indices - symbol_min).astype(np.int32)
    
    # Use range coder with numpy array
    encoder = constriction.stream.queue.RangeEncoder()
    categorical_model = constriction.stream.model.Categorical(pmf, perfect=False)
    
    # Encode indices as numpy array
    encoder.encode(shifted_indices, categorical_model)
    
    # Get compressed data
    compressed = encoder.get_compressed()
    return bytes(compressed)

def entropy_decode(compressed_bytes, shape, dtype=torch.long, symbol_min=0, symbol_max=10000):
    """
    Decode a compressed bitstream back into a tensor of indices.
    """
    # Create probability model
    num_symbols = symbol_max - symbol_min + 1
    pmf = np.ones(num_symbols) / num_symbols

    # Convert the bytes to a numpy array of dtype uint8
    compressed_array = np.frombuffer(compressed_bytes, dtype=np.uint32)

    # Setup decoder with numpy array (this is what constriction expects)
    decoder = constriction.stream.queue.RangeDecoder(compressed_array)
    num_elements = int(np.prod(shape))

    # Decode with categorical model
    categorical_model = constriction.stream.model.Categorical(pmf, perfect=False)
    decoded_indices = decoder.decode(categorical_model,num_elements)

    # Convert decoded indices to numpy array and shift back
    decoded_array = np.array(decoded_indices, dtype=np.int32) + symbol_min

    # Convert to tensor and reshape
    decoded_tensor = torch.tensor(decoded_array, dtype=dtype)
    return decoded_tensor.reshape(shape)
 
if __name__ == "__main__":
    
    DUMMY = False
    DUMMY_CAM = False
    FILTER_OUTLIERS = True
    DOWNSAMPLE = 5000 # Set to 0 to disable downsampling
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
    if DUMMY:
        ###################### DUMMY DATA ######################
        # Simulated Gaussian Avatar (smaller demo with fewer splats)
        N = 100  # Reduced number of splats for faster processing
        # Ensure all tensors are float32 and assigned to the CUDA device
    

        gaussians = {
            'means': torch.rand(N, 3, dtype=torch.float32, device=device),      # coordinates
            'opacities': torch.rand(N, dtype=torch.float32, device=device),  # opacity
            'scales': torch.rand(N, 3, dtype=torch.float32, device=device),     # scale
            'quats': torch.rand(N, 4, dtype=torch.float32, device=device),      # rotation
            'colors': torch.rand(N,16,3, dtype=torch.float32, device=device),    # color
        }
    else:
        ###################### READ REAL DATA ######################
        # LOAD PLY FILE
        # ply_path = "/home/dipcik/avatar/gs-env/gsplat/examples/results/benchmark/man/ply/point_cloud_9999.ply"
        # ply_path = "/home/dipcik/avatar/gs-env/gsplat/examples/results/benchmark/room/ply/point_cloud_9999.ply"
        # gaussians = convert_gaussian_tensor_np2pt(ply_path)
        
        # LOAD PT CKPT file Gaussians
        ckpt_path = "/home/dipcik/avatar/gs-env/gsplat/examples/results/benchmark/man/ckpts/ckpt_9999_rank0.pt"
        gaussians = read_gaussian_ckpt(ckpt_path)
    
    if DOWNSAMPLE:  # Downsample
        # Get random indices in range [0, N)
        N = gaussians['means'].shape[0]
        indices = np.random.choice(N, size=DOWNSAMPLE, replace=False)
    
        # Downsample the data
        for k in gaussians:
            if gaussians[k].shape[0] == N:
                if isinstance(gaussians[k], np.ndarray):
                    gaussians[k] = gaussians[k][indices]
                elif isinstance(gaussians[k], torch.Tensor):
                    gaussians[k] = gaussians[k][indices, ...]
              

    print("Opacities Min:", gaussians['opacities'].min().item())
    print("Opacities Max:", gaussians['opacities'].max().item())
    print("Colors Min:", gaussians['colors'].min().item())
    print("Colors Max:", gaussians['colors'].max().item())
    print("Means Shape:", gaussians['means'].shape)
    print("Quats Shape:", gaussians['quats'].shape)
    print("Scales Shape:", gaussians['scales'].shape)
    print("Opacities Shape:", gaussians['opacities'].shape)
    print("Colors Shape:", gaussians['colors'].shape)
    
    
    ###########################
    # Fit a statistically tight bounding box: use mean ± N*stddev (e.g., N=2 covers ~95% for normal dist)
    if FILTER_OUTLIERS:
        means = gaussians['means']
        if isinstance(means, torch.Tensor):
            mean_coords = means.mean(dim=0)
            std_coords = means.std(dim=0)
        else:
            mean_coords = torch.tensor(means).mean(dim=0)
            std_coords = torch.tensor(means).std(dim=0)

        N = 2.0  # Number of standard deviations to include (adjust as needed)
        new_min = mean_coords - N * std_coords
        new_max = mean_coords + N * std_coords

        # Find indices of points inside the statistically tight bounding box
        inside_mask = ((means >= new_min) & (means <= new_max)).all(dim=1)
        print(f"Keeping {inside_mask.sum().item()} / {means.shape[0]} points inside mean±{N}*std bounding box.")

        # Filter all attributes accordingly
        for k in gaussians:
            if gaussians[k].shape[0] == means.shape[0]:
                gaussians[k] = gaussians[k][inside_mask]
    ###########################
    
    # Configuration (smaller codebooks for faster training)
    config = {
        'K_scales': 256,     # 8 bits
        'K_quats': 256,     # 8 bits
        'K_colors': 128,     # 7 bits
        'K_csh': 128,   # 7 bits (added to fix the KeyError)
    }

    # Initialize compressor
    compressor = NSVQCompressor(config)

    # Train codebooks on initial data
    compressor.train_codebooks(gaussians)

    # Compress a frame
    compressed = compressor.compress_frame(gaussians)

    # Decompress
    base_attrs = {k: gaussians[k] for k in ['means', 'opacities']}
    gaussians_recon = compressor.decompress_frame(base_attrs, compressed)


    # Save the compressed data
    save_gaussian_ckpt( "results/gaussian_recon.pt",gaussians_recon)
    save_gaussian_ckpt( "results/gaussian.pt",gaussians)
    
    if DUMMY_CAM:
        viewmat = torch.eye(4, device=device, dtype=torch.float32)[None, :, :]  # Identity view matrix
        K = torch.tensor([[500, 0, 256], [0, 500, 256], [0, 0, 1]], device=device, dtype=torch.float32)[None, :, :]  # Intrinsics
        width, height = 512, 512  # Image resolution
        
        # Camera configuration
        camera = {
            "viewmat": viewmat,  # Identity matrix for simplicity
            "K": K,  # Intrinsics
            "width": width,
            "height": height,
        }
    else:
        
        # Compute the bounding box of the splats
        means = gaussians['means']
        if isinstance(means, torch.Tensor):
            min_coords, max_coords = means.min(dim=0).values, means.max(dim=0).values
        else:
            min_coords, max_coords = torch.tensor(means).min(dim=0).values, torch.tensor(means).max(dim=0).values

        # Compute the centroid and diagonal of the bounding box
        centroid = (min_coords + max_coords) / 2
        bounding_box_diagonal = torch.norm(max_coords - min_coords)

        # Place the camera at a distance proportional to the bounding box diagonal
        cam_distance = bounding_box_diagonal * 0.9  # Adjust multiplier for zoom level
        cam_pos = centroid + torch.tensor([0, 0, -1*cam_distance], device=centroid.device, dtype=centroid.dtype)

  
        def look_at(eye, target, up=torch.tensor([-1, 0, 0], dtype=torch.float32, device=centroid.device)):
            f = (target - eye)
            f = f / f.norm()
            u = up / up.norm()
            s = torch.cross(f, u)
            s = s / s.norm()
            u = torch.cross(s, f)
            m = torch.eye(4, device=eye.device, dtype=eye.dtype)
            m[0, :3] = s
            m[1, :3] = u
            m[2, :3] = -f  # Forward direction
            m[:3, 3] = eye
            m[2, :3] *= -1  # Invert z-direction globally
            return m
        viewmat = look_at(cam_pos, centroid).inverse()[None, :, :]  # Invert for world-to-camera

        # Adjust the intrinsic matrix (K) based on the bounding box size
        image_width, image_height = 512, 512
        focal_length = (image_width / 2) / torch.tan(torch.deg2rad(torch.tensor(30.0)))  # 30-degree FOV
        K = torch.tensor([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1]
        ], device=centroid.device, dtype=centroid.dtype)[None, :, :]

        # Camera configuration
        camera = {
            "viewmat": viewmat,
            "K": K,
            "width": image_width,
            "height": image_height,
        }
        camera_extras = {
            "cam_pos": cam_pos,
            "centroid": centroid,
            "min_coords": min_coords,
            "max_coords": max_coords,
            "bounding_box_diagonal": bounding_box_diagonal,
            "cam_distance": cam_distance,
        }
        
        print("Bounding Box Min:", min_coords)
        print("Bounding Box Max:", max_coords)
        print("Bounding Box Centroid:", centroid)
        print("Camera Position:", cam_pos)


        # plot_gaussian_splats_3D(gaussians, camera,camera_extras)
        plotly_splat_3D(gaussians_recon, camera,camera_extras)

  
    # Reshape gaussians['colors'] to (N, 16, 3)
    sh_degree = 3
    color_sh_param = (sh_degree + 1) ** 2  # K = 16 for sh_degree=3
    

    image,_,_ = render_image(gaussians, camera,sh_degree=sh_degree)
    image_recon,_,_ = render_image(gaussians_recon, camera,sh_degree=sh_degree)
   
   
    # Calculate pixel-wise difference (L1 norm per pixel)
    pixel_diff = np.abs(image.astype(np.int32) - image_recon.astype(np.int32))
    mean_pixel_diff = pixel_diff.mean()
    print(f"Mean pixel-wise L1 difference: {mean_pixel_diff:.3f}")
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(image_recon)
    axs[1].set_title("Reconstructed")
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig("results/original_vs_reconstructed.png")


    # Print size comparison
    original_size = sum(v.numel() * 4 for v in gaussians.values()) / (1024 ** 1)  # MB
    
    compressed_size = sum(
        v.numel() * ((config[f'K_{k}'] - 1).bit_length() if f"K_{k}" in config else 32) / 8
        for k, v in compressed.items()
    )/ (1024 ** 1)
    
    uncompressed_size = sum(
        gaussians[k].numel() * gaussians[k].element_size() for k in ['means', 'opacities']
    )/ (1024 ** 1)  # MB
    print(f"Uncompressed Size: {uncompressed_size:.2f} MB,compressed_size: {compressed_size:.2f} MB")
    
    compressed_size += uncompressed_size
    

    print(f"Original Size: {original_size:.2f} KB")
    print(f"Compressed Size: {compressed_size:.2f} KB")
    print(f"Compression Ratio: {original_size / compressed_size:.1f}x")
    
  
    # After compressing the frame with NSVQ
    # exit()
    print("\n=== Entropy Coding Demo ===")
    
    # Apply entropy coding to the compressed indices
    compressed_bits = {}
    total_bytes = 0
    
    print("\nCompressing each attribute...")
    for attr in ['scales', 'quats', 'colors']:


        # Encode the indices
        encoded = entropy_encode(compressed[attr])
        compressed_bits[attr] = encoded
        
        # Calculate sizes
        original_bytes = compressed[attr].numel() * compressed[attr].element_size()
        compressed_bytes = len(encoded)
        total_bytes += compressed_bytes
        
        print(f"{attr}: {original_bytes} bytes -> {compressed_bytes} bytes "
              f"(ratio: {original_bytes/compressed_bytes:.2f}x)")

    print(f"\nTotal compressed size: {total_bytes/1024:.2f} KB")

    # Decompress the data
    print("\nDecompressing data...")
    decompressed = {}
    for attr in ['scales', 'quats', 'colors']:
        # Get original shape
        original_shape = compressed[attr].shape
        
        # Decode
        decoded = entropy_decode(compressed_bits[attr], shape=original_shape)
        decompressed[attr] = decoded
        
        # Verify reconstruction
        is_equal = torch.all(decoded == compressed[attr].to(decoded.device))
        print(f"{attr}: Reconstruction successful: {is_equal}")

    # Compare a few random values
    print("\nSample value comparison (first 10 elements):")
    attr = 'scales'  # example attribute
    print(f"Original : {compressed[attr][:10]}")
    print(f"Decoded  : {decompressed[attr][:10]}")

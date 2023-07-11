import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pickle
import sys
import torch
import tqdm
import trimesh
from torchvision.transforms.functional import resize
from pytorch3d.transforms import rotation_conversions as transforms

from banmo_utils import banmo
from data_utils import write_img
from geom_utils import get_vertex_colors, warp_fw
from render_utils import pyrender_render_mesh, softras_render_mesh, softras_render_points

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Densepose Feature Visualization

def load_dpfeat_mesh_vertex_embeddings_colors(is_human):
    """Loads Densepose mesh vertex embeddings and colors

    Args
        is_human [bool]: Whether to use sheep or human

    Returns
        mesh_vertex_embeddings [Nembed, 16]: Densepose mesh vertex embeddings
        mesh_vertex_colors [Nembed, 3]: Densepose mesh vertex colors
    """
    if is_human:
        canonical_mesh_name = "smpl_27554"
    else:
        canonical_mesh_name = "sheep_5004"
    canonical_mesh_path = f"banmo_deps/mesh_material/{canonical_mesh_name}_sph.pkl"
    
    os.makedirs("cache", exist_ok=True)
    cachedir = f"cache/densepose_{canonical_mesh_name}_vertex_embed.npy"
    if os.path.exists(cachedir):
        mesh_vertex_embeddings = np.load(cachedir) # Nembed, 16
    else:
        # Initialize Densepose
        detectron_base = "third_party/detectron2"
        sys.path.insert(0, detectron_base)
        sys.path.insert(0, f"{detectron_base}/projects/DensePose")
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        from densepose import add_densepose_config
        from densepose.modeling import build_densepose_embedder

        if is_human:
            model_key = "densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl"
        else:
            model_key = "densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl"
        config_path = f"third_party/detectron2/projects/DensePose/configs/cse/{model_key.split('/')[0]}.yaml"
        weight_path = f"https://dl.fbaipublicfiles.com/densepose/cse/{model_key}"

        # Load Densepose vertex embeddings and colors
        config = get_cfg()
        add_densepose_config(config)
        config.merge_from_file(config_path)
        config.MODEL.WEIGHTS = weight_path
        model = build_model(config)
        DetectionCheckpointer(model).load(config.MODEL.WEIGHTS)

        embedder = build_densepose_embedder(config)
        mesh_vertex_embeddings = embedder(canonical_mesh_name) # Nembed, 16
        mesh_vertex_embeddings = torch.nn.functional.normalize(mesh_vertex_embeddings, p=2, dim=-1) # Nembed, 16
        mesh_vertex_embeddings = mesh_vertex_embeddings.detach().cpu().numpy() # Nembed, 16
        np.save(cachedir, mesh_vertex_embeddings)

    cachedir = f"cache/densepose_{canonical_mesh_name}_vertex_colors.npy"
    if os.path.exists(cachedir):
        mesh_vertex_colors = np.load(cachedir) # Nembed, 3
    else:
        with open(canonical_mesh_path, "rb") as f:
            mesh_vertex_colors = pickle.load(f)["vertices"].astype(np.float32) # Nembed, 3
        cmin = np.amin(mesh_vertex_colors, axis=0, keepdims=True) # 1, 3
        cmax = np.amax(mesh_vertex_colors, axis=0, keepdims=True) # 1, 3
        mesh_vertex_colors = (mesh_vertex_colors - cmin) / (cmax - cmin) # Nembed, 3
        np.save(cachedir, mesh_vertex_colors)

    mesh_vertex_embeddings = torch.from_numpy(mesh_vertex_embeddings)
    mesh_vertex_colors = torch.from_numpy(mesh_vertex_colors)
    return mesh_vertex_embeddings, mesh_vertex_colors


def vis_dpfeat(dp_feats, is_human):
    """Visualize Densepose features by projecting it into 3D canonical space

    Args
        dp_feats [..., 16, H, W]: Input Densepose features

    Returns
        dp_feats_vis [..., 3, H, W]: RGB visualization of dp_feats
    """
    from densepose.modeling.cse.utils import squared_euclidean_distance_matrix

    mesh_vertex_embeddings, mesh_vertex_colors = load_dpfeat_mesh_vertex_embeddings_colors(is_human)
    mesh_vertex_embeddings = mesh_vertex_embeddings.to(device)
    mesh_vertex_colors = mesh_vertex_colors.to(device)

    dp_feats_shape = dp_feats.shape
    (C, H, W) = dp_feats.shape[-3:]
    assert C == 16
    dp_feats = dp_feats.view(-1, C, H, W)

    # Compute Densepose visualization
    all_dpfeat_vis = []
    for i in range(dp_feats.shape[0]):
        dp_feats_i = dp_feats[i].to(device)
        dists = squared_euclidean_distance_matrix(dp_feats_i.view(16, -1).T, mesh_vertex_embeddings) # H*W, N_embed
        dpfeat_idxs = dists.argmin(dim=1) # H*W
        dpfeat_vis = mesh_vertex_colors[dpfeat_idxs].view(dp_feats.shape[-2:] + (3,)) # H, W, 3

        # Set background as black
        dpfeat_vis = torch.where(
            torch.sum(torch.abs(dp_feats_i), dim=0)[:, :, None] == 0, torch.zeros_like(dpfeat_vis), dpfeat_vis
        ).cpu() # H, W, 3
        all_dpfeat_vis.append(dpfeat_vis)

    all_dpfeat_vis = torch.stack(all_dpfeat_vis, dim=0) # T, H, W, 3
    all_dpfeat_vis = torch.movedim(all_dpfeat_vis, -1, -3) # T, 3, H, W
    all_dpfeat_vis = all_dpfeat_vis.view(dp_feats_shape[:-3] + all_dpfeat_vis.shape[-3:]) # T, 3, H, W
    return all_dpfeat_vis
    

# ===== 2D Pose Code

def vis_latent_code_2d_init(xlim, ylim, zlim):
    """Worker process initialization function for visualizing a single latent code using 6D XYZRGB PCA
    
    Args
        xlim [Tuple(int, int)]: X-axis limits
        ylim [Tuple(int, int)]: Y-axis limits
        zlim [Tuple(int, int)]: Z-axis limits
    """
    if len(multiprocessing.current_process()._identity) == 0:
        fig = plt.figure(0)
    else:
        fig = plt.figure(multiprocessing.current_process()._identity[0])
    fig.clear()

    # Setup axes for plotting 3D XYZ
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)


def vis_latent_code_2d(frame, dp_feat, idxs, codes_vis_pred, codes_vis_actual):
    """Visualize a single latent code using 6D XYZRGB PCA.

    Args
        frame [H, W, 3]: RGB frame to show
        dp_feat [H, W, 3]: Densepose features to show
        idxs [T]: List of frame indices
        codes_vis_pred [T, 6]: Predicted codes to show, projected to 6D with PCA
        codes_vis_actual [T, 6]: Ground-truth codes to show, projected to 6D with PCA

    Returns
        vis [H_, W_, 3]: Visualization with Densepose features on left, RGB frame
            in middle, and latent code plotted using 6D XYZRGB PCA on right
    """
    if len(multiprocessing.current_process()._identity) == 0:
        fig = plt.figure(0)
    else:
        fig = plt.figure(multiprocessing.current_process()._identity[0])

    ax = fig.axes[0]
    ax.clear()
    
    # Plot predicted and actual codes
    ax.scatter(
        codes_vis_pred[:, 0], codes_vis_pred[:, 1], codes_vis_pred[:, 2],
        color=codes_vis_pred[:, 3:], marker="o"
    )
    ax.scatter(
        codes_vis_actual[:, 0], codes_vis_actual[:, 1], codes_vis_actual[:, 2],
        color=codes_vis_actual[:, 3:], marker="o"
    )

    for idx, code_vis_pred, code_vis_actual in zip(idxs, codes_vis_pred, codes_vis_actual):
        ax.text(*code_vis_pred[:3], f"{idx}p")
        ax.text(*code_vis_actual[:3], f"{idx}a")

    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    frame = (255.0 * np.clip(frame[:3], 0, 1)).transpose(1, 2, 0).astype(np.uint8) # H, W, 3
    frame = cv2.resize(frame, (frame.shape[1] * plot.shape[0] // frame.shape[0], plot.shape[0])) # H', W', 3
    dp_feat = (255.0 * np.clip(dp_feat[:3], 0, 1)).transpose(1, 2, 0).astype(np.uint8) # H, W, 3
    dp_feat = cv2.resize(dp_feat, (dp_feat.shape[1] * plot.shape[0] // dp_feat.shape[0], plot.shape[0])) # H', W', 3
    plot = np.hstack([dp_feat, frame, plot])
    return plot


def vis_latent_codes_2d(
    frames, dp_feats, codes_pred, codes_actual, *,
    path_prefix=None, n_workers=16, memory_limit=None, eps=1e-12, window_dim=1,
):
    """Save a video visualizing latent codes using 6D XYZRGB PCA.

    Args
        frames [T, 3, H, W]: RGB frame to show
        dp_feats [T, 3, H, W]: Densepose features visualization to show
        codes_pred [T, C]: Predicted codes to show
        codes_actual [T, C]: Ground-truth codes to show
        path_prefix [str]: Prefix for output paths
        n_workers [int]: Number of multiprocessing workers
        memory_limit [int or None]: If provided, defines chunk size for warp_fw
        eps [float]: Small value used to avoid division by zero
        window_dim [int]: Number of codes to show on screen at a given time
    """
    # Concatenate codes into same space
    T, C = codes_pred.shape
    codes_pred = codes_pred.to(device)
    codes_actual = codes_actual.to(device)

    codes_pred = codes_pred.view(T, C) # T, C
    codes_actual = codes_actual.view(T, C) # T, C
    codes = torch.cat([codes_pred, codes_actual], dim=0) # 2*T, C

    # Center codes before PCA
    codes_mean = torch.mean(codes, dim=0) # 1, C
    codes -= codes_mean # 2*T, C

    # Use PCA to get six principal axes of both codes
    U, S, V = torch.svd(codes) # U: 2*T, C | s: C | V: C, C
    codes_vis = torch.matmul(codes, V[:, :6]) # 2*T, 6

    # Normalize PCA output
    codes_vis_min = torch.min(codes_vis, dim=0, keepdim=True).values # 1, 6
    codes_vis_max = torch.max(codes_vis, dim=0, keepdim=True).values # 1, 6
    codes_vis = (codes_vis - codes_vis_min) / (codes_vis_max - codes_vis_min + eps) # 2*T, 6
    codes_vis_pred = codes_vis[:T] # T, 6
    codes_vis_actual = codes_vis[T:] # T, 6
    assert codes_vis_pred.shape[0] == T and codes_vis_actual.shape[0] == T, \
        f"Expected pred and actual codes to have length {T}, " \
        f"but found {codes_vis_pred.shape[0]} and {codes_vis_actual.shape[0]}"
    
    codes_vis_pred = codes_vis_pred.detach().cpu().numpy() # T, 6
    codes_vis_actual = codes_vis_actual.detach().cpu().numpy() # T, 6
    
    # Export video, using multiprocessing to parallelize across frames
    idxs = np.arange(T, dtype=np.int64)

    matplotlib.rcParams["figure.max_open_warning"] = 10000 * n_workers + 1
    args = (
        (
            frames[i], dp_feats[i], idxs[max(0, i - window_dim) : i + 1],
            codes_vis_pred[max(0, i - window_dim) : i + 1], codes_vis_actual[max(0, i - window_dim) : i + 1],
        )
        for i in range(T)
    )
    if n_workers == 0:
        vis_latent_code_2d_init()
        plot_stack = [vis_latent_code_2d(*arg) for arg in tqdm.tqdm(args, total=T)]
    else:
        mp = multiprocessing.get_context("spawn")
        with mp.Pool(n_workers, vis_latent_code_2d_init) as p:
            plot_stack = p.starmap(vis_latent_code_2d, tqdm.tqdm(args, total=T))

    imageio.mimwrite(f"{path_prefix}.mp4", plot_stack, fps=10)


def vis_bone(vidname, frameid, root_pose, angles, camera_ks, path_prefix, img_dim=480):
    """Visualize bone structure given joint angles

    Args:
        vidname [str]: Index of source video
        frameid [int]: Frame index within source video
        root_pose [12,]: Predicted root body pose
        angles [njoints, 3]: Predicted joint angles
        camera_ks [6,]: Camera intrinsics + width and height
        path_prefix [str]: Prefix for output paths
        img_dim [int]: Output size of image
    """
    robot = banmo().robot.urdf
    angles = angles.cpu().numpy().reshape(-1)  # J*3,
    
    cfg = {}
    for idx, name in enumerate(robot.angle_names):
        cfg[name] = angles[idx]
    
    fk = robot.visual_trimesh_fk(cfg=cfg)

    robot_meshes = []
    for tm in fk:
        pose = fk[tm]
        faces = tm.faces
        faces -= faces.min()
        tm = trimesh.Trimesh(tm.vertices, tm.faces) # ugly workaround for uv = []
        tm = tm.copy()
        tm.vertices = tm.vertices.dot(pose[:3, :3].T) + pose[:3, 3][None]
        robot_meshes.append(tm)
    robot_mesh = trimesh.util.concatenate(robot_meshes)

    pts_robot = torch.tensor(robot_mesh.vertices, dtype=torch.float32)  # npts, 3
    faces_robot = torch.tensor(robot_mesh.faces, dtype=torch.int64)  # npts, 3
    rgb_robot = torch.tensor(robot_mesh.visual.vertex_colors[:, :3] / 255.0, dtype=torch.float32)

    rmat = root_pose[:9].cpu().view(3, 3)  # 3, 3
    tmat = root_pose[9:].cpu()  # 3,
    pts_robot = torch.sum(rmat * pts_robot[:, None, :] / 10, dim=-1) + tmat  # npts, 3

    vis_robot = pyrender_render_mesh(
        pts_robot, faces_robot, rgb_robot, camera_ks, img_dim=img_dim
    )[:3].moveaxis(0, -1).cpu().numpy()

    os.makedirs(f"{path_prefix}/{vidname}", exist_ok=True)
    write_img(f"{path_prefix}/{vidname}/{vidname}-bone-{frameid:05d}.jpg", vis_robot)


def vis_bone_rts_3d(
    video_impaths, frame_videoid, frame_offsets, all_pred_poses, all_pred_angles, all_camera_ks,
    path_prefix=None, n_workers=0
):
    n_workers = 0
    """Given predicted vertex locations and colors, save per-frame results to meshes

    Args
        video_impaths [List(List(str))]: A list mapping video i to a list of image paths for that video
        frame_videoid [List(int)]: A list mapping frame i to its parent video
        frame_offsets [List(int)]: A list mapping frame i to its offset within parent video
        all_pred_poses [T, 12]: Predicted root body poses
        all_pred_angles [T, J, 3]: Predicted joint angles
        all_camera_ks [T, 6]: Camera intrinsics + width and height
        path_prefix [str]: Prefix for output paths
        n_workers [int]: Number of multiprocessing workers
    """
    T = all_pred_angles.shape[0]
    video_names = [] # nvid,
    for impaths in video_impaths:
        jpeg_prefix = "database/DAVIS/JPEGImages/Full-Resolution/"
        if jpeg_prefix in impaths[0]:
            video_name = impaths[0].split(jpeg_prefix)[1].split("/")[0]
            video_names.append(video_name)
        else:
            raise ValueError(f"Cannot parse video name in save_pose_dfms_mesh")

    vidnames = [video_names[i] for i in frame_videoid]

    args = zip(vidnames, frame_offsets, all_pred_poses, all_pred_angles, all_camera_ks)
    args = (
        (vidname, frameid, root_body_pose, joint_angle, camera_ks, path_prefix)
        for vidname, frameid, root_body_pose, joint_angle, camera_ks in args
    )
    if n_workers == 0:
        [vis_bone(*arg) for arg in tqdm.tqdm(args, total=T)]
    else:
        mp = multiprocessing.get_context("fork")
        with mp.Pool(n_workers) as p:
            p.starmap(vis_bone, tqdm.tqdm(args, total=T))


# ===== Mesh output for evaluation

def save_pose_dfm_mesh(vidname, frameid, mesh_rest, pts_pred, rgb_pred, camera_ks, path_prefix):
    """Given predicted vertex locations and colors, update rest mesh and save to .obj

    Args
        vidname [str]: Index of source video
        frameid [int]: Frame index within source video
        mesh_rest [Trimesh]: Rest mesh
        pts_pred [npts, 3]: Predicted deformed vertex positions
        rgb_pred [npts, 3]: Predicted vertex colors
        camera_ks [6,]: Camera intrinsics + width and height
        path_prefix [str]: Prefix for output paths
    """
    mesh_rest = mesh_rest.copy()
    mesh_rest.vertices = pts_pred.detach().numpy() # npts, 3
    mesh_rest.visual.vertex_colors[:, :3] = 255.0 * rgb_pred.detach().numpy() # npts, 3
    os.makedirs(f"{path_prefix}/{vidname}", exist_ok=True)
    mesh_rest.export(f"{path_prefix}/{vidname}/{vidname}-mesh-{frameid:05d}.obj")

    # Save camera params in banmo format:
    # [R00 R01 R02  T0]
    # [R10 R11 R12  T1]
    # [R20 R21 R22  T2]
    # [ fx  fy  px  py]
    cam_matrix = np.eye(4)
    cam_matrix[3, :4] = camera_ks[:4]
    np.savetxt(f"{path_prefix}/{vidname}/{vidname}-cam-{frameid:05d}.txt", cam_matrix)


def save_pose_dfms_mesh(
    video_impaths, frame_videoid, frame_offsets, mesh_rest, all_pts_pred, all_rgb_pred, all_camera_ks,
    path_prefix=None, n_workers=0
):
    """Given predicted vertex locations and colors, save per-frame results to meshes

    Args
        video_impaths [List(List(str))]: A list mapping video i to a list of image paths for that video
        frame_videoid [List(int)]: A list mapping frame i to its parent video
        frame_offsets [List(int)]: A list mapping frame i to its offset within parent video
        mesh_rest [Trimesh]: Rest mesh
        all_pts_pred [T, npts, 3]: Predicted deformed vertex positions
        all_rgb_pred [T, npts, 3]: Predicted vertex colors
        all_camera_ks [T, 6]: Camera intrinsics + width and height
        path_prefix [str]: Prefix for output paths
        n_workers [int]: Number of multiprocessing workers
    """
    T = all_pts_pred.shape[0]
    video_names = [] # nvid,
    for impaths in video_impaths:
        jpeg_prefix = "database/DAVIS/JPEGImages/Full-Resolution/"
        if jpeg_prefix in impaths[0]:
            video_name = impaths[0].split(jpeg_prefix)[1].split("/")[0]
            video_names.append(video_name)
        else:
            raise ValueError(f"Cannot parse video name in save_pose_dfms_mesh")

    vidnames = [video_names[i] for i in frame_videoid]

    args = zip(vidnames, frame_offsets, all_pts_pred, all_rgb_pred, all_camera_ks)
    args = (
        (vidname, frameid, mesh_rest, pts_pred, rgb_pred, camera_ks, path_prefix)
        for vidname, frameid, pts_pred, rgb_pred, camera_ks in args
    )
    if n_workers == 0:
        [save_pose_dfm_mesh(*arg) for arg in tqdm.tqdm(args, total=T)]
    else:
        mp = multiprocessing.get_context("fork")
        with mp.Pool(n_workers) as p:
            p.starmap(save_pose_dfm_mesh, tqdm.tqdm(args, total=T))


def save_pose_dfms_img(
    video_impaths, frame_videoid, frame_offsets, frames, faces_bone, pts_bone, rgb_bone,
    faces_can, pts_pred, rgb_pred, root_pose, camera_ks, *,
    pts_actual=None, rgb_actual=None, path_prefix=None, img_dim=640, n_workers=16, include_actual=True,
    save_predictions=False, memory_limit=None, use_ddp=True, ddp_rank=0, ddp_size=1,
):
    """Output a video where rest mesh is deformed and colorized according to
    given root body pose, bone transforms, and texture codes

    Args
        frames [T, 3, H, W]: RGB frames to show
        faces_bones [nfaces_bones, 3]: Faces of bone rest mesh
        pts_bones [T, npts_bones, 3]: Predicted deformed bone vertex positions
        rgb_bones [T, npts_bones, 3]: Predicted bone vertex colors
        faces_can [nfaces, 3]: Faces of canonical rest mesh
        pts_pred [T, npts, 3]: Predicted deformed vertex positions
        rgb_pred [T, npts, 3]: Predicted vertex colors
        path_prefix [str]: Prefix for output paths
        img_dim [int]: Dimension of output frames
        n_workers [int]: Number of multiprocessing workers
        include_actual [bool]: Whether to include actual banmo outputs for comparison
        save_predictions [bool]: Whether to save predictions as npz
        memory_limit [int or None]: If provided, defines chunk size for rendering
        use_ddp [bool]: Whether to parallelize rendering over GPUs
        ddp_rank [int]: Distributed data parallel rank
        ddp_size [int]: Distributed data parallel size
    """
    include_actual = False

    T, npts = pts_pred.shape[:2]
    T, npts_bones = pts_bone.shape[:2]
    video_names = [] # nvid,
    for impaths in video_impaths:
        jpeg_prefix = "database/DAVIS/JPEGImages/Full-Resolution/"
        if jpeg_prefix in impaths[0]:
            video_name = impaths[0].split(jpeg_prefix)[1].split("/")[0]
            video_names.append(video_name)
        else:
            raise ValueError(f"Cannot parse video name in save_pose_dfms_mesh")

    vidnames = [video_names[i] for i in frame_videoid]

    # Render predicted and actual results
    if memory_limit is None:
        chunk_size = T
    else:
        memory_per_chunk = img_dim * img_dim * 4 * 4
        chunk_size = (memory_limit + memory_per_chunk - 1) // memory_per_chunk

    if use_ddp and ddp_rank == 0:
        gather_list = [
            torch.empty(5, chunk_size, img_dim, img_dim, 4, dtype=torch.float32) for rank in range(ddp_size)
        ] # ddp_size; 5, chunk_size, H, W, 4
    else:
        gather_list = None

    for i_ in tqdm.trange(0, T, chunk_size * ddp_size):
        i = min(T, i_ + chunk_size * ddp_rank)
        Tch = min(T - i, chunk_size)
        faces_can_ch = faces_can.expand(Tch, -1, -1) # Tch, nfaces, 3
        faces_bone_ch = faces_bone.expand(Tch, -1, -1) # Tch, nfaces_bones, 3
        camera_ks_ch = camera_ks[i:i+chunk_size].detach().to(device) # Tch, 6
        root_pose_ch = root_pose[i:i+chunk_size].detach().to(device) # Tch, 12
        rmat_ch = root_pose_ch[:, :9].view(-1, 1, 3, 3) # Tch, 1, 3, 3
        tmat_ch = root_pose_ch[:, 9:].view(-1, 1, 3) # Tch, 1, 3

        # Render predicted rgb view1
        pts_pred_ch = pts_pred[i:i+chunk_size].detach().to(device) # Tch, npts, 3
        rgb_pred_ch = rgb_pred[i:i+chunk_size].detach().to(device) # Tch, npts, 3

        vis_pred_rgb1_ch = pyrender_render_mesh(
            pts_pred_ch, faces_can_ch, rgb_pred_ch, camera_ks_ch, img_dim=img_dim
        ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4

        # Render actual rgb view1
        pts_actual_ch = pts_actual[i:i+chunk_size].detach().to(device) # Tch, npts, 3
        rgb_actual_ch = rgb_actual[i:i+chunk_size].detach().to(device) # Tch, npts, 3

        vis_actual_rgb1_ch = pyrender_render_mesh(
            pts_actual_ch, faces_can_ch, rgb_actual_ch, camera_ks_ch, img_dim=img_dim
        ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4

        # Render predicted gray view1
        cmin = torch.amin(pts_pred_ch, dim=1, keepdims=True) # Tch, 1, 3
        cmax = torch.amax(pts_pred_ch, dim=1, keepdims=True) # Tch, 1, 3
        rgb_pred_ch = torch.full_like(rgb_pred_ch, 0.5) # Tch, npts, 3
        
        vis_pred_gray1_ch = pyrender_render_mesh(
            pts_pred_ch, faces_can_ch, rgb_pred_ch, camera_ks_ch, directional=True, img_dim=img_dim
        ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4

        # Render bones view1
        pts_bone_ch = pts_bone[i:i+chunk_size].detach().to(device) # Tch, npts_bone, 3
        rgb_bone_ch = rgb_bone[i:i+chunk_size].detach().to(device) # Tch, npts_bone, 3

        vis_pred_bone1_ch = pyrender_render_mesh(
            pts_bone_ch, faces_bone_ch, rgb_bone_ch, camera_ks_ch, img_dim=img_dim
        ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4

        # Render predicted gray view2
        rot90 = torch.tensor([0, 0, 1, 0, 1, 0, -1, 0, 0], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        # Untranslate, unrotate, rot90, rotate, translate
        pts_pred_ch_ = pts_pred_ch # Tch, npts, 3
        pts_pred_ch_ = pts_pred_ch_ - tmat_ch # Tch, npts, 3
        pts_pred_ch_ = torch.sum(rmat_ch.swapaxes(-2, -1) * pts_pred_ch_[:, :, None, :], dim=-1) # Tch, npts, 3
        center = torch.mean(pts_pred_ch_, dim=1, keepdims=True) # Tch, 1, 3
        pts_pred_ch_ = torch.sum(rot90 * (pts_pred_ch_ - center)[:, :, None, :], dim=-1) + center # Tch, npts, 3
        pts_pred_ch_ = torch.sum(rmat_ch * pts_pred_ch_[:, :, None, :], dim=-1)+ tmat_ch # Tch, npts, 3

        vis_pred_gray2_ch = pyrender_render_mesh(
            pts_pred_ch_, faces_can_ch, rgb_pred_ch, camera_ks_ch, directional=True, img_dim=img_dim
        ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4
        
        # Render bones view2
        pts_bone_ch_ = pts_bone_ch # Tch, npts, 3
        pts_bone_ch_ = pts_bone_ch_ - tmat_ch # Tch, npts, 3
        pts_bone_ch_ = torch.sum(rmat_ch.swapaxes(-2, -1) * pts_bone_ch_[:, :, None, :], dim=-1) # Tch, npts, 3
        center = torch.mean(pts_bone_ch_, dim=1, keepdims=True) # Tch, 1, 3
        pts_bone_ch_ = torch.sum(rot90 * (pts_bone_ch_ - center)[:, :, None, :], dim=-1) + center # Tch, npts, 3
        pts_bone_ch_ = torch.sum(rmat_ch * pts_bone_ch_[:, :, None, :], dim=-1)+ tmat_ch # Tch, npts, 3

        vis_pred_bone2_ch = pyrender_render_mesh(
            pts_bone_ch_, faces_bone_ch, rgb_bone_ch, camera_ks_ch, directional=True, img_dim=img_dim
        ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4

        # Render predicted gray view3
        rot90 = torch.tensor([0, 0, -1, 0, 1, 0, 1, 0, 0], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        # Untranslate, unrotate, rot90, rotate, translate
        pts_pred_ch_ = pts_pred_ch # Tch, npts, 3
        pts_pred_ch_ = pts_pred_ch_ - tmat_ch # Tch, npts, 3
        pts_pred_ch_ = torch.sum(rmat_ch.swapaxes(-2, -1) * pts_pred_ch_[:, :, None, :], dim=-1) # Tch, npts, 3
        center = torch.mean(pts_pred_ch_, dim=1, keepdims=True) # Tch, 1, 3
        pts_pred_ch_ = torch.sum(rot90 * (pts_pred_ch_ - center)[:, :, None, :], dim=-1) + center # Tch, npts, 3
        pts_pred_ch_ = torch.sum(rmat_ch * pts_pred_ch_[:, :, None, :], dim=-1) + tmat_ch # Tch, npts, 3

        vis_pred_gray3_ch = pyrender_render_mesh(
            pts_pred_ch_, faces_can_ch, rgb_pred_ch, camera_ks_ch, directional=True, img_dim=img_dim
        ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4

        # Render bones view3
        pts_bone_ch_ = pts_bone_ch # Tch, npts, 3
        pts_bone_ch_ = pts_bone_ch_ - tmat_ch # Tch, npts, 3
        pts_bone_ch_ = torch.sum(rmat_ch.swapaxes(-2, -1) * pts_bone_ch_[:, :, None, :], dim=-1) # Tch, npts, 3
        center = torch.mean(pts_bone_ch_, dim=1, keepdims=True) # Tch, 1, 3
        pts_bone_ch_ = torch.sum(rot90 * (pts_bone_ch_ - center)[:, :, None, :], dim=-1) + center # Tch, npts, 3
        pts_bone_ch_ = torch.sum(rmat_ch * pts_bone_ch_[:, :, None, :], dim=-1) + tmat_ch # Tch, npts, 3

        vis_pred_bone3_ch = pyrender_render_mesh(
            pts_bone_ch_, faces_bone_ch, rgb_bone_ch, camera_ks_ch, directional=True, img_dim=img_dim
        ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4

        # Distribute results
        vis_pred_ch = torch.stack([
            vis_pred_rgb1_ch, vis_actual_rgb1_ch,
            vis_pred_gray1_ch, vis_pred_gray2_ch, vis_pred_gray3_ch,
            vis_pred_bone1_ch, vis_pred_bone2_ch, vis_pred_bone3_ch,
        ], dim=0) # 8, Tch, H, W, 4, 3

        if use_ddp:
            vis_pred_padded_ch = torch.cat([
                vis_pred_ch, torch.zeros(8, chunk_size - Tch, img_dim, img_dim, 4, dtype=torch.float32)
            ], dim=1) # 5, chunk_size, H, W, 4
            torch.distributed.gather(vis_pred_padded_ch, gather_list, dst=0)

            if ddp_rank == 0:
                i_list = [min(T, i_ + chunk_size * rank) for rank in range(ddp_size)]
                Tch_list = [min(T - idx, chunk_size) for idx in i_list]
                vis_pred_gather_ch_list = [
                    gathered[:, :Tch] for gathered, Tch in zip(gather_list, Tch_list)
                ] # ddp_size; 8, Tch, H, W, 4
                vis_pred_gather_ch = torch.cat(vis_pred_gather_ch_list, dim=1) # 5, Tch_all, H, W, 4
        else:
            vis_pred_gather_ch = vis_pred_ch # 5, Tch, H, W, 4
        del vis_pred_ch

        # Save images
        if ddp_rank == 0:
            (
                vis_pred_rgb1_ch, vis_actual_rgb1_ch,
                vis_pred_gray1_ch, vis_pred_gray2_ch, vis_pred_gray3_ch,
                vis_pred_bone1_ch, vis_pred_bone2_ch, vis_pred_bone3_ch,
            ) = vis_pred_gather_ch
            for (
                vidname, frameid, frame, vis_pred_rgb1, vis_actual_rgb1,
                vis_pred_gray1, vis_pred_gray2, vis_pred_gray3,
                vis_pred_bone1, vis_pred_bone2, vis_pred_bone3,
            ) in zip(
                vidnames[i:i+chunk_size*ddp_size], frame_offsets[i:i+chunk_size*ddp_size],
                frames[i:i+chunk_size*ddp_size], vis_pred_rgb1_ch.numpy(), vis_actual_rgb1_ch.numpy(),
                vis_pred_gray1_ch.numpy(), vis_pred_gray2_ch.numpy(), vis_pred_gray3_ch.numpy(),
                vis_pred_bone1_ch.numpy(), vis_pred_bone2_ch.numpy(), vis_pred_bone3_ch.numpy(),
            ):
                frame = cv2.resize(frame[:3].transpose(1, 2, 0), (img_dim, img_dim)) # H, W, 3
                mask_pred_rgb1 = vis_pred_rgb1[:, :, 3:] # H, W, 1
                vis_pred_rgb1 = vis_pred_rgb1[:, :, :3] # H, W, 3
                mask_actual_rgb1 = vis_actual_rgb1[:, :, 3:] # H, W, 1
                vis_actual_rgb1 = vis_actual_rgb1[:, :, :3] # H, W, 3

                mask_pred_gray1 = vis_pred_gray1[:, :, 3:] # H, W, 1
                vis_pred_gray1 = vis_pred_gray1[:, :, :3] # H, W, 3
                mask_pred_gray2 = vis_pred_gray2[:, :, 3:] # H, W, 1
                vis_pred_gray2 = vis_pred_gray2[:, :, :3] # H, W, 3
                mask_pred_gray3 = vis_pred_gray3[:, :, 3:] # H, W, 1
                vis_pred_gray3 = vis_pred_gray3[:, :, :3] # H, W, 3
                
                mask_pred_bone1 = vis_pred_bone1[:, :, 3:] # H, W, 1
                vis_pred_bone1 = vis_pred_bone1[:, :, :3] # H, W, 3
                mask_pred_bone2 = vis_pred_bone2[:, :, 3:] # H, W, 1
                vis_pred_bone2 = vis_pred_bone2[:, :, :3] # H, W, 3
                mask_pred_bone3 = vis_pred_bone3[:, :, 3:] # H, W, 1
                vis_pred_bone3 = vis_pred_bone3[:, :, :3] # H, W, 3
                
                # Set background
                vis_pred_rgb1 = np.where(mask_pred_rgb1 == 0, (1 + frame) / 2, vis_pred_rgb1) # H, W, 3
                vis_actual_rgb1 = np.where(mask_actual_rgb1 == 0, (1 + frame) / 2, vis_actual_rgb1) # H, W, 3
                vis_pred_gray1 = np.where(mask_pred_gray1 == 0, 1, vis_pred_gray1) # H, W, 3
                vis_pred_gray2 = np.where(mask_pred_gray2 == 0, 1, vis_pred_gray2) # H, W, 3
                vis_pred_gray3 = np.where(mask_pred_gray3 == 0, 1, vis_pred_gray3) # H, W, 3
                vis_pred_bone1 = np.where(mask_pred_bone1 == 0, 1, vis_pred_bone1) # H, W, 3
                vis_pred_bone2 = np.where(mask_pred_bone2 == 0, 1, vis_pred_bone2) # H, W, 3
                vis_pred_bone3 = np.where(mask_pred_bone3 == 0, 1, vis_pred_bone3) # H, W, 3

                vis_pred_gray_bone1 = 0.5 * vis_pred_gray1 + 0.5 * vis_pred_bone1 # H, W, 3
                vis_pred_gray_bone2 = 0.5 * vis_pred_gray2 + 0.5 * vis_pred_bone2 # H, W, 3
                vis_pred_gray_bone3 = 0.5 * vis_pred_gray3 + 0.5 * vis_pred_bone3 # H, W, 3

                plot = np.hstack((
                    frame, vis_actual_rgb1, vis_pred_rgb1,
                    vis_pred_gray_bone1, vis_pred_gray_bone2, vis_pred_gray_bone3,
                )) # H, 5*W, 3
                os.makedirs(f"{path_prefix}/{vidname}", exist_ok=True)
                write_img(f"{path_prefix}/{vidname}/{vidname}-img-{frameid:05d}.jpg", plot)


def save_pose_dfms_mplex_img(
    freq, temporal_radius, frames, pts_can, faces_can, bones_rst, bone_rts_rst, centroid,
    poses_pred, bones_pred, envs_pred, camera_ks, poses_pred_mplex, mplex_wt, *,
    path_prefix=None, img_dim=640, n_workers=16, memory_limit=None, use_ddp=False, ddp_rank=0, ddp_size=1,
):
    T, M = mplex_wt.shape
    for frameid in tqdm.trange(0, T, freq):
        idx0 = max(0, frameid - temporal_radius)
        idx1 = min(T, frameid + temporal_radius + 1)

        frames_mplex = resize(
            torch.from_numpy(np.array(frames[idx0:idx1])).detach().to(device), (img_dim, img_dim)
        ) # 2r+1, 3, H, W
        poses_pred_mplex_ = poses_pred_mplex[idx0:idx1].detach().to(device) # 2r+1, M, 12
        bones_pred_mplex = bones_pred[idx0:idx1, None, :].expand(-1, M, -1).detach().to(device) # 2r+1, M, B*12
        camera_ks_mplex = camera_ks[idx0:idx1, None, :].expand(-1, M, -1).detach().to(device) # 2r+1, 6
        mplex_wt_ = torch.nn.functional.softmax(mplex_wt[idx0:idx1], dim=-1) # 2r+1, M

        rr, M = mplex_wt_.shape
        faces_can_mplex = faces_can[None, None, :, :].expand(rr, M, -1, -1) # 2r+1, M, nfaces, 3

        # Render results
        pts_pred_mplex = warp_fw(
            banmo(), pts_can, bones_rst, bone_rts_rst, centroid,
            poses_pred_mplex_, bones_pred_mplex, memory_limit=memory_limit
        ) # 2r+1, M, npts, 3
        rgb_pred_mplex = torch.full_like(pts_pred_mplex, 0.5) # 2r+1, M, npts, 3

        vis_pred = pyrender_render_mesh(
            pts_pred_mplex, faces_can_mplex, rgb_pred_mplex, camera_ks_mplex, directional=True, img_dim=img_dim
        ).permute(0, 1, 3, 4, 2) # 2r+1, M, H, W, 4

        # Sort vis_pred by mplex_wt and set background
        rr, M, H, W, C = vis_pred.shape
        #sort_idxs = torch.argsort(mplex_wt_, dim=1, descending=True)[:, :, None, None, None] # 2r+1, M, 1, 1, 1
        #vis_pred = torch.gather(vis_pred, 1, sort_idxs.expand(-1, -1, H, W, C)) # 2r+1, M, H, W, 4

        mask_pred = vis_pred[..., 3:] # 2r+1, M, H, W, 1
        vis_pred = vis_pred[..., :3] # 2r+1, M, H, W, 3
        vis_pred = torch.where(mask_pred == 0, 1, vis_pred) # 2r+1, M, H, W, 3
        vis_pred = torch.cat([frames_mplex[:, None].permute(0, 1, 3, 4, 2), vis_pred], dim=1) # 2r+1, M+1, H, W, 3

        # Save image
        os.makedirs(f"{path_prefix}", exist_ok=True)
        vis_pred = torch.moveaxis(vis_pred, 0, 2).reshape((M + 1) * H, rr * W, 3) # (M+1)*H, (2r+1)*W, 3
        write_img(f"{path_prefix}/pose_dfms_mplex_{frameid:05d}.jpg", vis_pred.cpu().numpy())


# ===== 3D root body pose and bone transforms

def vis_pose_dfms_3d(
    frames, dp_feats, faces_can, pts_pred, rgb_pred, pts_actual, rgb_actual, camera_ks, *,
    path_prefix=None, img_dim=224, n_workers=16, save_predictions=False, memory_limit=None,
    use_ddp=True, ddp_rank=0, ddp_size=1,
):
    """Output a video where rest mesh is deformed and colorized according to
    given root body pose, bone transforms, and texture codes

    Args
        frames [T, 3, H, W]: RGB frames to show
        dp_feats [T, 3, H, W]: Densepose features visualizations to show
        faces_can [nfaces, 3]: Faces of canonical rest mesh
        pts_pred [T, npts, 3]: Predicted deformed vertex positions
        rgb_pred [T, npts, 3]: Predicted vertex colors
        pts_actual [T, npts, 3]: Ground-truth deformed vertex positions
        rgb_actual [T, npts, 3]: Ground-truth vertex colors
        path_prefix [str]: Prefix for output paths
        img_dim [int]: Dimension of output frames
        n_workers [int]: Number of multiprocessing workers
        save_predictions [bool]: Whether to save predictions as npz
        memory_limit [int or None]: If provided, defines chunk size for rendering
        use_ddp [bool]: Whether to parallelize rendering over GPUs
        ddp_rank [int]: Distributed data parallel rank
        ddp_size [int]: Distributed data parallel size
    """
    T, npts = pts_pred.shape[:2]

    # Save predicted results
    if save_predictions and ddp_rank == 0:
        np.savez_compressed(f"{path_prefix}_pts_pred.npz", pts_pred)
        np.savez_compressed(f"{path_prefix}_rgb_pred.npz", rgb_pred)

    # Render predicted and actual results
    if memory_limit is None:
        chunk_size = T
    else:
        memory_per_chunk = img_dim * img_dim * 4 * 4
        chunk_size = (memory_limit + memory_per_chunk - 1) // memory_per_chunk

    if use_ddp and ddp_rank == 0:
        gather_list = [
            torch.empty(chunk_size, img_dim, img_dim, 4, dtype=torch.float32) for rank in range(ddp_size)
        ] # ddp_size; chunk_size, H, W, 4
    else:
        gather_list = None

    with imageio.get_writer(f"{path_prefix}.mp4", fps=10) as writer:
        for i_ in tqdm.trange(0, T, chunk_size * ddp_size):
            i = min(T, i_ + chunk_size * ddp_rank)
            Tch = min(T - i, chunk_size)
            faces_can_ch = faces_can.expand(Tch, -1, -1) # Tch, nfaces, 3
            camera_ks_ch = camera_ks[i:i+chunk_size].detach().to(device) # Tch, 6

            # Render predicted
            pts_pred_ch = pts_pred[i:i+chunk_size].detach().to(device) # Tch, npts, 3
            rgb_pred_ch = rgb_pred[i:i+chunk_size].detach().to(device) # Tch, npts, 3

            vis_pred_ch = pyrender_render_mesh(
                pts_pred_ch, faces_can_ch, rgb_pred_ch, camera_ks_ch, img_dim=img_dim
            ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4
            del pts_pred_ch, rgb_pred_ch

            # Distribute predicted
            if use_ddp:
                vis_pred_padded_ch = torch.cat([
                    vis_pred_ch, torch.zeros(chunk_size - Tch, img_dim, img_dim, 4, dtype=torch.float32)
                ], dim=0) # chunk_size, H, W, 4
                torch.distributed.gather(vis_pred_padded_ch, gather_list, dst=0)

                if ddp_rank == 0:
                    i_list = [min(T, i_ + chunk_size * rank) for rank in range(ddp_size)]
                    Tch_list = [min(T - idx, chunk_size) for idx in i_list]
                    vis_pred_gather_ch_list = [
                        gathered[:Tch] for gathered, Tch in zip(gather_list, Tch_list)
                    ] # ddp_size; Tch, H, W, 4
                    vis_pred_gather_ch = torch.cat(vis_pred_gather_ch_list, dim=0) # Tch_all, H, W, 4
            else:
                vis_pred_gather_ch = vis_pred_ch # Tch, H, W, 4
            del vis_pred_ch
            
            # Render actual
            pts_actual_ch = pts_actual[i:i+chunk_size].detach().to(device) # Tch, npts, 3
            rgb_actual_ch = rgb_actual[i:i+chunk_size].detach().to(device) # Tch, npts, 3

            vis_actual_ch = pyrender_render_mesh(
                pts_actual_ch, faces_can_ch, rgb_actual_ch, camera_ks_ch, img_dim=img_dim
            ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4
            del pts_actual_ch, faces_can_ch, rgb_actual_ch, camera_ks_ch

            # Distribute actual
            if use_ddp:
                vis_actual_padded_ch = torch.cat([
                    vis_actual_ch, torch.zeros(chunk_size - Tch, img_dim, img_dim, 4, dtype=torch.float32)
                ], dim=0) # chunk_size, H, W, 4
                torch.distributed.gather(vis_actual_padded_ch, gather_list, dst=0)

                if ddp_rank == 0:
                    i_list = [min(T, i_ + chunk_size * rank) for rank in range(ddp_size)]
                    Tch_list = [min(T - idx, chunk_size) for idx in i_list]
                    vis_actual_gather_ch_list = [
                        gathered[:Tch] for gathered, Tch in zip(gather_list, Tch_list)
                    ] # ddp_size; Tch, H, W, 4
                    vis_actual_gather_ch = torch.cat(vis_actual_gather_ch_list, dim=0) # Tch_all, H, W, 4
            else:
                vis_actual_gather_ch = vis_actual_ch # Tch, H, W, 4
            del vis_actual_ch

            # Save to image writer
            if ddp_rank == 0:
                for frame, dp_feat, vis_pred, vis_actual in zip(
                    frames[i:i+chunk_size*ddp_size], dp_feats[i:i+chunk_size*ddp_size],
                    vis_pred_gather_ch.numpy(), vis_actual_gather_ch.numpy(),
                ):
                    frame = cv2.resize(frame[:3].transpose(1, 2, 0), (img_dim, img_dim)) # H, W, 3
                    dp_feat = cv2.resize(dp_feat[:3].transpose(1, 2, 0), (img_dim, img_dim)) # H, W, 3
                    mask_pred = vis_pred[:, :, 3:] # H, W, 1
                    mask_actual = vis_actual[:, :, 3:] # H, W, 1
                    vis_pred = vis_pred[:, :, :3] # H, W, 3
                    vis_actual = vis_actual[:, :, :3] # H, W, 3
                    
                    # Blend frame with white to form background
                    vis_pred = np.where(mask_pred == 0, (1 + frame) / 2, vis_pred) # H, W, 3
                    vis_actual = np.where(mask_actual == 0, (1 + frame) / 2, vis_actual) # H, W, 3
                    
                    frame = (255.0 * np.clip(frame, 0, 1)).astype(np.uint8) # H, W, 3
                    dp_feat = (255.0 * np.clip(dp_feat, 0, 1)).astype(np.uint8) # H, W, 3
                    vis_pred = (255.0 * np.clip(vis_pred, 0, 1)).astype(np.uint8) # H, W, 3
                    vis_actual = (255.0 * np.clip(vis_actual, 0, 1)).astype(np.uint8) # H, W, 3

                    plot = np.vstack((np.hstack((frame, vis_actual)), np.hstack((dp_feat, vis_pred)))) # H_, W_, 3
                    writer.append_data(plot)


def vis_pose_dfms_mplex_3d(
    model, frames, dp_feats, pts_can, faces_can, bones_rst, bone_rts_rst, centroid,
    pts_pred, rgb_pred, pts_actual, rgb_actual,
    poses_pred, bones_pred, envs_pred, camera_ks, poses_pred_mplex, mplex_wt, *,
    path_prefix=None, img_dim=224, n_workers=16, memory_limit=None, use_ddp=False, ddp_rank=0, ddp_size=1,
):
    """Output a video where rest mesh is deformed and colorized according to
    given root body pose, bone transforms, and texture codes

    Args
        model [BanmoCategoryModel]: Model for texture stealing network
        frames [T, 3, H, W]: RGB frames to show
        dp_feats [T, 3, H, W]: Densepose features visualizations to show
        pts_can [npts, 3]: Vertices in canonical rest mesh
        faces_can [nfaces, 3]: Faces in canonical rest mesh
        bones_rst [B, 10]: Rest bones
        bone_rts_rst [B, 12]: Rest bone transforms
        centroid [3,]: Mesh centroid before centering
        pts_pred [T, npts, 3]: Predicted deformed vertex positions
        rgb_pred [T, npts, 3]: Predicted vertex colors
        pts_actual [T, npts, 3]: Ground-truth deformed vertex positions
        rgb_actual [T, npts, 3]: Ground-truth vertex colors
        poses_pred [T, 12]: Predicted root body poses
        bones_pred [T, B*12]: Predicted bone transforms
        envs_pred [T, Ce]: Predicted texture codes
        camera_ks [T, 6]: Camera intrinsics (px, py, cx, cy) per frame
        poses_pred_mplex [T, M, 12]: Predicted multiplexed root body poses
        mplex_wt [T, M]: Predicted probabilities for pose multiplex
        path_prefix [str]: Prefix for output paths
        n_workers [int]: Number of multiprocessing workers
        memory_limit [int or None]: If provided, defines chunk size for warp_fw
        use_ddp [bool]: Whether to parallelize rendering over GPUs
        ddp_rank [int]: Distributed data parallel rank
        ddp_size [int]: Distributed data parallel size
    """
    T, M = mplex_wt.shape
    
    poses_pred_mplex = poses_pred_mplex.detach().to(device) # T, M, 12
    bones_pred_mplex = bones_pred[:, None, :].expand(-1, M, -1).detach().to(device).clone() # T, M, B*12
    envs_pred = envs_pred.detach().to(device) # T, Ce
    camera_ks_mplex = camera_ks[:, None, :].expand(-1, M, -1).detach().to(device).clone() # T, M, 6

    mplex_wt = torch.nn.functional.softmax(mplex_wt, dim=-1) # T, M

    # Render predicted and actual results
    npts = pts_can.shape[0]
    nfaces = faces_can.shape[0]
    n_vid = int(np.ceil(np.sqrt(M)))
    img_dim = 224 * 3 // n_vid

    if memory_limit is None:
        chunk_size = T
    else:
        memory_per_chunk = M * (npts * 3 + img_dim * img_dim * 4) * 4
        chunk_size = (memory_limit + memory_per_chunk - 1) // memory_per_chunk

    # Allocate distributed buffers
    if use_ddp and ddp_rank == 0:
        gather_pred_list = [
            torch.empty(chunk_size, n_vid * img_dim, n_vid * img_dim, 3, dtype=torch.float32)
            for rank in range(ddp_size)
        ] # ddp_size; chunk_size, Nv*H, Nv*W, 3
        gather_actual_list = [
            torch.empty(chunk_size, 224, 224, 4, dtype=torch.float32) for rank in range(ddp_size)
        ] # ddp_size; chunk_size, H, W, 4
    else:
        gather_pred_list = None
        gather_actual_list = None

    with imageio.get_writer(f"{path_prefix}.mp4", fps=10) as writer:
        for i_ in tqdm.trange(0, T, chunk_size * ddp_size):
            i = min(T, i_ + chunk_size * ddp_rank)
            Tch = min(T, i + chunk_size) - i
            poses_pred_mplex_ch = poses_pred_mplex[i:i+chunk_size].detach().to(device) # Tch, M, 12 
            bones_pred_mplex_ch = bones_pred_mplex[i:i+chunk_size].detach().to(device) # Tch, M, B*12
            envs_pred_ch = envs_pred[i:i+chunk_size].detach().to(device) # Tch, Ce
            camera_ks_mplex_ch = camera_ks_mplex[i:i+chunk_size].detach().to(device) # Tch, M, 6
            faces_can_mplex_ch = faces_can.expand(Tch * M, -1, -1) # Tch*M, nfaces, 3
            faces_can_ch = faces_can.expand(Tch, -1, -1) # Tch, nfaces, 3
            mplex_wt_ch = mplex_wt[i:i+chunk_size].detach().cpu().numpy() # Tch, M

            # Render predicted
            if Tch > 0:
                pts_pred_mplex_ch = warp_fw(
                    banmo(), pts_can, bones_rst, bone_rts_rst, centroid,
                    poses_pred_mplex_ch, bones_pred_mplex_ch, memory_limit=memory_limit
                ).view(Tch * M, npts, 3) # Tch*M, npts, 3
                del poses_pred_mplex_ch, bones_pred_mplex_ch
                rgb_pred_ch = get_vertex_colors(
                    banmo(), pts_can, centroid, env_codes=envs_pred_ch, memory_limit=memory_limit
                )[:, None, :, :].expand(-1, M ,-1, -1).reshape(Tch * M, npts, 3) # Tch*M, npts, 3
                del envs_pred_ch
            else:
                pts_pred_mplex_ch = torch.empty(0, npts, 3, dtype=torch.float32, device=device)
                rgb_pred_ch = torch.empty(0, npts, 3, dtype=torch.float32, device=device)

            vis_pred_ch = pyrender_render_mesh(
                pts_pred_mplex_ch, faces_can_mplex_ch, rgb_pred_ch, camera_ks_mplex_ch, img_dim=img_dim,
            ).view(-1, M, 4, img_dim, img_dim).permute(0, 1, 3, 4, 2).cpu().numpy() # Tch, M, H, W, 4
            del pts_pred_mplex_ch, rgb_pred_ch

            # Set background to mplex_wt and add text label
            vis_pred_out_ch = np.empty((Tch, M, img_dim, img_dim, 3), dtype=np.float32)
            for t in range(Tch):
                for m in range(M):
                    mask_pred_img = vis_pred_ch[t, m, :, :, 3:] # H, W, 1
                    vis_pred_img = vis_pred_ch[t, m, :, :, :3] # H, W, 3
                    vis_pred_img = np.where(mask_pred_img == 0, mplex_wt_ch[t, m], vis_pred_img) # H, W, 3
                    del mask_pred_img
                    vis_pred_out_ch[t, m, :, :, :] = cv2.putText(
                        np.ascontiguousarray(vis_pred_img), f"{100 * mplex_wt_ch[t, m]:.2f}%", org=(0, img_dim),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 1), thickness=2,
                    )
                    del vis_pred_img

            # Reshape into square
            vis_pred_square_ch = np.zeros(
                (Tch, n_vid * img_dim, n_vid * img_dim, 3), dtype=np.float32
            ) # Tch, Nv*H, Nv*W, 3
            for vy in range(n_vid):
                for vx in range(n_vid):
                    m = vy * n_vid + vx
                    if m >= M:
                        continue
                    vis_pred_square_ch[:, img_dim*vy : img_dim*(vy+1), img_dim*vx : img_dim*(vx+1), :] = (
                        vis_pred_out_ch[:, m, :, :, :]
                    )
            del vis_pred_out_ch
            vis_pred_square_ch = torch.from_numpy(vis_pred_square_ch) # Tch, Nv*H, Nv*W, 3

            # Distribute predicted
            if use_ddp:
                vis_pred_square_padded = torch.cat([
                    vis_pred_square_ch,
                    torch.zeros(chunk_size - Tch, n_vid * img_dim, n_vid * img_dim, 3, dtype=torch.float32)
                ], dim=0) # chunk_size, Nv*H, Nv*W, 3
                torch.distributed.gather(vis_pred_square_padded, gather_pred_list, dst=0)

                if ddp_rank == 0:
                    i_list = [min(T, i_ + chunk_size * rank) for rank in range(ddp_size)]
                    Tch_list = [min(T - idx, chunk_size) for idx in i_list]
                    vis_pred_gather_ch_list = [
                        gathered[:Tch] for gathered, Tch in zip(gather_pred_list, Tch_list)
                    ] # ddp_size; Tch, Nv*H, Nv*W, 3
                    vis_pred_square_gather_ch = torch.cat(vis_pred_gather_ch_list, dim=0) # Tch_all, Nv*H, Nv*W, 3
            else:
                vis_pred_square_gather_ch = vis_pred_square_ch # Tch, Nv*H, Nv*W, 3
            del vis_pred_square_ch

            # Render actual
            pts_actual_ch = pts_actual[i:i+chunk_size].detach().to(device) # Tch, npts, 3
            rgb_actual_ch = rgb_actual[i:i+chunk_size].detach().to(device) # Tch, npts, 3
            camera_ks_ch = camera_ks[i:i+chunk_size].detach().to(device) # Tch, 6

            vis_actual_ch = pyrender_render_mesh(
                pts_actual_ch, faces_can_ch, rgb_actual_ch, camera_ks_ch, img_dim=224,
            ).permute(0, 2, 3, 1).cpu() # Tch, H, W, 4
            del pts_actual_ch, faces_can_ch, rgb_actual_ch, camera_ks_ch

            # Distribute actual
            if use_ddp:
                vis_actual_padded = torch.cat([
                    vis_actual_ch, torch.zeros(chunk_size - Tch, 224, 224, 4, dtype=torch.float32)
                ], dim=0) # chunk_size, H, W, 4
                torch.distributed.gather(vis_actual_padded, gather_actual_list, dst=0)

                if ddp_rank == 0:
                    i_list = [min(T, i_ + chunk_size * rank) for rank in range(ddp_size)]
                    Tch_list = [min(T - idx, chunk_size) for idx in i_list]
                    vis_actual_gather_ch_list = [
                        gathered[:Tch] for gathered, Tch in zip(gather_actual_list, Tch_list)
                    ] # ddp_size; Tch, H, W, 4
                    vis_actual_gather_ch = torch.cat(vis_actual_gather_ch_list, dim=0) # Tch_all, H, W, 4
            else:
                vis_actual_gather_ch = vis_actual_ch # Tch, H, W, 4
            del vis_actual_ch

            # Save to image writer
            if ddp_rank == 0:
                for frame, dp_feat, vis_pred_sq, vis_actual in zip(
                    frames[i:i+chunk_size*ddp_size], dp_feats[i:i+chunk_size*ddp_size],
                    vis_pred_square_gather_ch.numpy(), vis_actual_gather_ch.numpy()
                ):
                    frame = cv2.resize(frame[:3].transpose(1, 2, 0), (224, 224)) # H, W, 3
                    dp_feat = cv2.resize(dp_feat[:3].transpose(1, 2, 0), (224, 224)) # H, W, 3
                    vis_pred_sq = cv2.resize(vis_pred_sq, (224 * 3, 224 * 3)) # 3*H, 3*W, 3; float32 within [0, 1]
                    mask_actual = vis_actual[:, :, 3:] # H, W, 1
                    vis_actual = vis_actual[:, :, :3] # H, W, 3

                    vis_actual = np.where(mask_actual == 0, (1 + frame) / 2, vis_actual) # H, W, 3

                    frame = (255.0 * np.clip(frame, 0, 1)).astype(np.uint8) # H, W, 3
                    dp_feat = (255.0 * np.clip(dp_feat, 0, 1)).astype(np.uint8) # H, W, 3
                    vis_actual = (255.0 * np.clip(vis_actual, 0, 1)).astype(np.uint8) # H, W, 3
                    vis_pred_sq = (255.0 * np.clip(vis_pred_sq, 0, 1)).astype(np.uint8) # 3*H, 3*W, 3

                    plot = np.vstack((np.hstack((frame, dp_feat, vis_actual)), vis_pred_sq)) # 4*H, 3*W, 3
                    writer.append_data(plot)


# ===== 2D implicit PDF pose distribution visualization

def vis_pose_distr_2d_init():
    """Worker process initialization function for visualizing a single pose distribution"""
    if len(multiprocessing.current_process()._identity) == 0:
        fig = plt.figure(0)
    else:
        fig = plt.figure(multiprocessing.current_process()._identity[0])
    fig.clear()

    # Setup axes for plotting rotation distribution
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")

    # Add a color wheel showing the tilt angle to color conversion
    ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection="polar")
    theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
    radii = np.linspace(0.4, 0.5, 2)
    _, theta_grid = np.meshgrid(radii, theta)
    cmap_val = 0.5 + theta_grid / (2 * np.pi)
    ax.pcolormesh(theta, radii, cmap_val.T, cmap=plt.cm.hsv)
    ax.set_yticklabels([])
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ax.get_xticks().tolist()))
    ax.set_xticklabels([
        r"90$\degree$", None, r"180$\degree$", None, r"270$\degree$", None, r"0$\degree$", None
    ], fontsize=14)
    ax.spines["polar"].set_visible(False)
    plt.text(
        0.5, 0.5, "Tilt", fontsize=14, horizontalalignment="center",
        verticalalignment="center", transform=ax.transAxes
    )


def vis_pose_distr_2d(
    frame, dp_feat, tilt_pred, lon_pred, lat_pred, probs_pred, tilt_actual, lon_actual, lat_actual
):
    """Visualize a single pose distribution using Implicit PDF's method

    Args
        frame [H, W, 3]: RGB frame to show
        dp_feat [H, W, 3]: Densepose features to show
        tilt_pred [M]: Predicted pose tilt
        lon_pred [M]: Predicted pose longitude
        lat_pred [M]: Predicted pose latitude
        probs_pred [M]: Predicted pose probabilities to show
        tilt_actual: Actual pose tilt
        lon_actual: Actual pose longitude
        lat_actual: Actual pose latitude

    Returns
        vis [H_, W_, 3]: Visualization with Densepose features on left, RGB frame
            in middle, and pose distribution plotted using Implicit PDF's method on right
    """
    if len(multiprocessing.current_process()._identity) == 0:
        fig = plt.figure(0)
    else:
        fig = plt.figure(multiprocessing.current_process()._identity[0])
    assert len(fig.axes) == 2 and "Mollweide" in str(type(fig.axes[0])) and "Polar" in str(type(fig.axes[1])), \
        f"Invalid figure axes: {fig.axes}"

    ax = fig.axes[0]
    ax.clear()

    # Show ground-truth rotations behind the output with white filling interior
    color = plt.cm.hsv(0.5 + tilt_actual / (2 * np.pi)) # 3,
    ax.scatter(lon_actual, lat_actual, s=2500, edgecolors=color, facecolors="none", marker="o", linewidth=4)
    ax.scatter(lon_actual, lat_actual, s=2500, edgecolors="none", facecolors="#ffffff", marker="o", linewidth=4)

    # Show predicted rotation distribution
    ax.scatter(lon_pred, lat_pred, s=2000 * probs_pred, c=plt.cm.hsv(0.5 + tilt_pred / (2 * np.pi)))
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    frame = (255.0 * np.clip(frame[:3], 0, 1)).transpose(1, 2, 0).astype(np.uint8) # H, W, 3
    frame = cv2.resize(frame, (frame.shape[1] * plot.shape[0] // frame.shape[0], plot.shape[0])) # H', W', 3
    dp_feat = (255.0 * np.clip(dp_feat[:3], 0, 1)).transpose(1, 2, 0).astype(np.uint8) # H, W, 3
    dp_feat = cv2.resize(dp_feat, (dp_feat.shape[1] * plot.shape[0] // dp_feat.shape[0], plot.shape[0])) # H', W', 3
    plot = np.hstack([dp_feat, frame, plot])
    return plot


def vis_pose_distrs_2d(
    frames, dp_feats, all_poses_pred, all_probs_pred, poses_actual, *, path_prefix=None, n_workers=16,
):
    """Save a video visualizing pose distributions using Implicit PDF's method

    Args
        frames [T, H, W, 3]: RGB frames to show
        dp_feats [T, H, W, 3]: Densepose features to show
        all_poses_pred [T, M, 12]: Predicted pose distribution to show
        all_probs_pred [T, M]: Predicted pose probabilities to show
        poses_actual [T, 12]: Actual pose
        path_prefix [str]: Prefix for output paths
        n_workers [int]: Number of multiprocessing workers
    """
    T, M = all_probs_pred.shape

    all_poses_pred = all_poses_pred[..., :9].view(T, M, 3, 3) # T, M, 3, 3
    all_probs_pred = torch.softmax(all_probs_pred, dim=-1).detach().cpu().numpy() # T, M
    poses_actual = poses_actual[..., :9].view(T, 3, 3) # T, 3, 3

    # Convert poses to tilt, longitude, and latitude
    eulers_actual = transforms.matrix_to_euler_angles(poses_actual, convention="XYZ") # T, 3
    xyz_actual = poses_actual[:, :, 0] # T, 3
    tilt_actual = eulers_actual[:, 0].detach().cpu().numpy() # T,
    lon_actual = torch.arctan2(xyz_actual[:, 0], -xyz_actual[:, 1]).detach().cpu().numpy() # T,
    lat_actual = torch.arcsin(xyz_actual[:, 2]).detach().cpu().numpy() # T,

    eulers_pred = transforms.matrix_to_euler_angles(all_poses_pred, convention="XYZ") # T, M, 3
    xyz_pred = all_poses_pred[:, :, :, 0] # T, M, 3
    tilt_pred = eulers_pred[:, :, 0].detach().cpu().numpy() # T, M
    lon_pred = torch.arctan2(xyz_pred[:, :, 0], -xyz_pred[:, :, 1]).detach().cpu().numpy() # T, M
    lat_pred = torch.arcsin(xyz_pred[:, :, 2]).detach().cpu().numpy() # T, M

    # Export video, using multiprocessing to parallelize across frames
    matplotlib.rcParams["figure.max_open_warning"] = 10000 * n_workers + 1
    args = (
        (
            frames[i], dp_feats[i], tilt_pred[i], lon_pred[i], lat_pred[i],
            all_probs_pred[i], tilt_actual[i], lon_actual[i], lat_actual[i],
        )
        for i in range(T)
    )
    if n_workers == 0:
        vis_pose_distr_2d_init()
        plot_stack = [vis_pose_distr_2d(*arg) for arg in tqdm.tqdm(args, total=T)]
    else:
        mp = multiprocessing.get_context("spawn")
        with mp.Pool(n_workers, vis_pose_distr_2d_init) as p:
            plot_stack = p.starmap(vis_pose_distr_2d, tqdm.tqdm(args, total=T))

    imageio.mimwrite(f"{path_prefix}.mp4", plot_stack, fps=10)

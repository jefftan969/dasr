import argparse
import cv2
import imageio
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pytorch3d
import pytorch3d.ops
import pytorch3d.transforms
import sys
import torch
import tqdm
import trimesh
from pytorch3d.ops.knn import _KNN
from pytorch3d.ops.points_alignment import (
    ICPSolution, SimilarityTransform, corresponding_points_alignment, _apply_similarity_transform
)
from pytorch3d.ops.utils import wmean

sys.path.append("third_party/chamfer3D")
from dist_chamfer_3D import chamfer_3DDist

from data_utils import read_img, write_img
from geom_utils import load_mesh
from render_utils import pyrender_render_mesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opts):
    cd_avg, f010_avg, f005_avg, f002_avg = ama_eval(
        opts.load_dir, opts.seqname, opts.vidid, verbose=opts.verbose, render_vid=opts.load_dir
    )
    print(f"Finished evaluating {opts.load_dir}::{opts.seqname}{opts.vidid}")
    print(f"  Avg chamfer dist: {100 * cd_avg:.2f}cm")
    print(f"  Avg f-score at d=10cm: {100 * f010_avg:.1f}%")
    print(f"  Avg f-score at d=5cm:  {100 * f005_avg:.1f}%")
    print(f"  Avg f-score at d=2cm:  {100 * f002_avg:.1f}%")


def ama_eval(load_dir, seqname, vidid, verbose=False, render_vid=None, correct_tra=False):
    """Evaluate a sequence of AMA videos

    Args
        load_dir [str]: Directory to load predicted meshes from
        seqname [str]: Name of sequence (e.g. T_samba)
        vidid [int]: Video identifier (e.g. 1)
        verbose [bool]: Whether to print eval metrics
        render_vid [str]: If provided, output an error video to this path
        correct_tra [bool]: If True, use ground-truth translations to evaluate

    Returns:
        cd_avg [float]: Chamfer distance (cm), averaged across all frames
        f010_avg [float]: F-score at 10cm threshold, averaged across all frames
        f005_avg [float]: F-score at 5cm threshold, averaged across all frames
        f002_avg [float]: F-score at 2cm threshold, averaged across all frames
    """
    nframes = len(os.listdir(f"database/ama_eval/{seqname}/meshes"))
    vidname = f"{seqname}{vidid}"

    # Load meshes
    all_verts_pred = []
    all_faces_pred = []
    all_cam_pred = []
    all_verts_gt = []
    all_faces_gt = []
    all_cam_gt = []

    for i in tqdm.trange(nframes, desc=f"Loading {vidname}"):
        mesh_path_pred = f"{load_dir}/{vidname}/{vidname}-mesh-{i:05d}.obj"
        cam_path_pred = f"{load_dir}/{vidname}/{vidname}-cam-{i:05d}.txt"
        mesh_path_gt = f"database/ama_eval/{seqname}/meshes/mesh_{i:04d}.obj"
        cam_path_gt = f"database/ama_eval/{seqname}/calibration/Camera{vidid}.Pmat.cal"
        
        verts_pred, faces_pred, cam_pred, verts_gt, faces_gt, cam_gt = ama_load(
            mesh_path_pred, cam_path_pred, mesh_path_gt, cam_path_gt
        )
        all_verts_pred.append(verts_pred)
        all_faces_pred.append(faces_pred)
        all_cam_pred.append(cam_pred)
        all_verts_gt.append(verts_gt)
        all_faces_gt.append(faces_gt)
        all_cam_gt.append(cam_gt)

    all_verts_pred = align_seqs(all_verts_pred, all_verts_gt, correct_tra=correct_tra, verbose=verbose)

    # Evaluate metrics: chamfer distance and f-score (@10cm, @5cm, @2cm)
    bbox_max = 1
    metrics = torch.zeros(nframes, 4, dtype=torch.float32, device=device) # nframes, 4
    chamLoss = chamfer_3DDist()
    if render_vid is not None:
        writer = imageio.get_writer(f"{render_vid}/eval_error_{vidname}.mp4", fps=10)

    for i in tqdm.trange(nframes, desc=f"Evaluating {vidname}"):
        verts_pred = all_verts_pred[i]  # 1, npts_pred, 3
        verts_gt = all_verts_gt[i]  # 1, npts_gt, 3

        raw_cd_fw, raw_cd_bw, _, _ = chamLoss(verts_gt, verts_pred)  # 1, npts_gt | 1, npts_pred

        cd = torch.mean(torch.sqrt(raw_cd_fw)) + torch.mean(torch.sqrt(raw_cd_bw))
        f010, _, _ = fscore(raw_cd_fw, raw_cd_bw, threshold=(bbox_max * 0.10) ** 2)
        f005, _, _ = fscore(raw_cd_fw, raw_cd_bw, threshold=(bbox_max * 0.05) ** 2)
        f002, _, _ = fscore(raw_cd_fw, raw_cd_bw, threshold=(bbox_max * 0.02) ** 2)

        metrics[i, 0] = cd
        metrics[i, 1] = f010
        metrics[i, 2] = f005
        metrics[i, 3] = f002

        if render_vid is not None:
            # Get focal and ppoint, and infer width/height from ppoint
            camera_ks = all_cam_pred[i]
            camera_ks = torch.cat([camera_ks, 2 * camera_ks[2:]], dim=0) # 6,
            cmap_vis = plt.get_cmap("plasma")
            Rmat_vis = torch.matmul(
                torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32, device=device),
                pytorch3d.transforms.RotateAxisAngle(30, "X").get_matrix()[0, :3, :3].to(device),
            ) # 3, 3
            Tmat_vis = torch.tensor([-0.02, 0.05, -0.02], dtype=torch.float32, device=device)[None, None] # 1, 1, 3

            # Correct vertices by swapping X/Y axes and adding an offset
            # Then, render predicted and gt chamfer distance colormap visualizations
            verts_vis_pred = torch.matmul(verts_pred, Rmat_vis.swapaxes(-1, -2)) + Tmat_vis
            colors_pred = torch.tensor(cmap_vis((5 * raw_cd_bw).cpu().numpy()), dtype=torch.float32)[..., :3]
            error_pred = pyrender_render_mesh(
                verts_vis_pred.cpu(), all_faces_pred[i], colors_pred, camera_ks, directional=True, img_dim=480,
            )[0, :3].swapaxes(0, -1) # 480, 480, 3
            
            verts_vis_gt = torch.matmul(verts_gt, Rmat_vis.swapaxes(-1, -2)) + Tmat_vis
            colors_gt = torch.tensor(cmap_vis((5 * raw_cd_fw).cpu().numpy()), dtype=torch.float32)[..., :3]
            error_gt = pyrender_render_mesh(
                verts_vis_gt.cpu(), all_faces_gt[i], colors_gt, camera_ks, directional=True, img_dim=480,
            )[0, :3].swapaxes(0, -1) # 480, 480, 3

            frame = read_img(f"database/DAVIS/JPEGImages/Full-Resolution/{vidname}/{i:05d}.jpg", resize=(480, 480))
            error_img = np.concatenate([frame, error_pred, error_gt], axis=1) # 480, 480*3, 3
            writer.append_data((255.0 * error_img).astype(np.uint8))

        if verbose:
            print(f"Frame {i}: CD={100 * cd:.2f}cm, f@10cm={100 * f010:.1f}%, "
                  f"f@5cm={100 * f005:.1f}%, f@2cm={100 * f002:.1f}%")

    if render_vid is not None:
        writer.close()

    metrics = torch.mean(metrics, dim=0) # 4,
    cd_avg, f010_avg, f005_avg, f002_avg = tuple(float(x) for x in metrics)

    if verbose:
        print(f"Finished evaluating {load_dir}::{vidname}")
        print(f"  Avg chamfer dist: {100 * cd_avg:.2f}cm")
        print(f"  Avg f-score at d=10cm: {100 * f010_avg:.1f}%")
        print(f"  Avg f-score at d=5cm:  {100 * f005_avg:.1f}%")
        print(f"  Avg f-score at d=2cm:  {100 * f002_avg:.1f}%")

    return cd_avg, f010_avg, f005_avg, f002_avg


def ama_load(mesh_path_pred, cam_path_pred, mesh_path_gt, cam_path_gt, device="cuda"):
    """Load a pair of ground-truth and predicted meshes/cameras for AMA sequences

    Args
        mesh_path_pred [str]: Path to predicted mesh
        cam_path_pred [str]: Path to predicted camera extrinsics
        mesh_path_gt [str]: Path to ground-truth mesh
        cam_path_gt [str]: Path to ground-truth camera extrinsics
        device (torch.device): Device to load values onto

    Returns
        verts_pred [1, npts_pred, 3]: Predicted vertices
        faces_pred [1, nfaces, 3]: Predicted faces
        K_pred [4,]: Predicted camera intrinsics
        verts_gt [1, npts_gt, 3]: Ground-truth vertices
        faces_gt [1, nfaces, 3]: Ground-truth faces
        K_gt [4,]: Ground-truth camera intrinsics
    """
    # Load predicted meshes and cameras
    mesh_pred = trimesh.load(mesh_path_pred, process=False)
    verts_pred = torch.tensor(mesh_pred.vertices[None], dtype=torch.float32, device=device) # 1, npts_pred, 3
    faces_pred = torch.tensor(mesh_pred.faces[None], dtype=torch.int64, device=device) # 1, nfaces_pred, 3
    cam_pred = np.loadtxt(cam_path_pred) # 4, 4
    
    R_pred = np.matmul(cv2.Rodrigues(np.array([0., 0., 0.]))[0], cam_pred[:3, :3])
    T_pred = cam_pred[:3, 3]
    K_pred = cam_pred[3, :]
    focal_pred = K_pred[:2]  # 2,
    K_pred = torch.tensor(K_pred, dtype=torch.float32, device=device)  # 4,

    R_pred = torch.tensor(R_pred, dtype=torch.float32, device=device)
    T_pred = torch.tensor(T_pred, dtype=torch.float32, device=device)
    verts_pred = obj_to_cam(verts_pred, R_pred, T_pred) # 1, npts_pred, 3

    # Load ground-truth meshes and cameras
    mesh_gt = trimesh.load(mesh_path_gt, process=False)
    verts_gt = torch.tensor(mesh_gt.vertices[None], dtype=torch.float32, device=device) # 1, npts_gt, 3
    faces_gt = torch.tensor(mesh_gt.faces[None], dtype=torch.int64, device=device) # 1, nfaces_gt, 3
    cam_gt = np.loadtxt(cam_path_gt)
    
    K_gt, R_gt, T_gt, _, _, _, _ = cv2.decomposeProjectionMatrix(cam_gt)
    T_gt = T_gt[:3, 0] / T_gt[3, 0]
    T_gt = np.dot(R_gt, -T_gt[..., None])[..., 0]
    K_gt = K_gt / K_gt[-1, -1]
    K_gt = np.array([K_gt[0, 0], K_gt[1, 1], K_gt[0, 2], K_gt[1, 2]])
    focal_gt = K_gt[:2]  # 2,
    K_gt = torch.tensor(K_gt, dtype=torch.float32, device=device)  # 4,

    R_gt = torch.tensor(R_gt, dtype=torch.float32, device=device)
    T_gt = torch.tensor(T_gt, dtype=torch.float32, device=device)
    verts_gt = obj_to_cam(verts_gt, R_gt, T_gt) # 1, npts_gt, 3

    # Center depth and correct for focal length
    focal_correction_ratio = np.sqrt((focal_pred[0] * focal_pred[1]) / (focal_gt[0] * focal_gt[1]))
    verts_pred[..., -1] -= torch.mean(verts_pred[..., -1]) * (1 - 1 / focal_correction_ratio)

    return verts_pred, faces_pred, K_pred, verts_gt, faces_gt, K_gt


def obj_to_cam(verts, R_mat, T_mat):
    """Apply rotation and translation matrices to vertices

    Args
        verts [bs, npts, 3]: Input vertices
        R_mat [bs, 3, 3]: Rotation to apply
        T_mat [bs, 3]: Translation to apply

    Returns
        verts [bs, npts, 3]: Transformed vertices
    """
    return torch.matmul(verts, R_mat.swapaxes(-1, -2)) + T_mat


def fscore(dist1, dist2, threshold=0.001):
    """Calculates F-score between two point clouds with corresponding threshold

    Args
        dist1 [bs, npts1]: Point cloud distances 1
        dist2 [bs, npts2]: Point cloud distances 2
        threshold [float]: Threshold, in squared pointcloud euclidean distances

    Returns (fscore, precision, recall)
    """
    precision1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision1 * precision2 / (precision1 + precision2)
    fscore = torch.where(torch.isnan(fscore), 0., fscore)
    return fscore, precision1, precision2


def timeseries_knn_points(
    X, Y, lengths_X=None, lengths_Y=None, norm=2, K=1, version=-1, return_nn=False, return_sorted=True
):
    """K-nearest neighbors on two time series of point clouds.

    Args:
        X [bs, T, size_X, dim]: A batch of `bs` time series, each with `T`
            point clouds containing `size_X` points of dimension `dim`
        Y [bs, T, size_Y, dim]: A batch of `bs` time series, each with `T`
            point clouds containing `size_Y` points of dimension `dim`
        lengths_X [bs, T]: Length of each point cloud in X, in range [0, size_X]
        lengths_Y [bs, T]: Length of each point cloud in Y, in range [0, size_Y]
        norm (int): Which norm to use, either 1 for L1-norm or 2 for L2-norm
        K (int): Number of nearest neighbors to return
        version (int): Which KNN implementation to use in the backend
        return_nn (bool): If True, returns K nearest neighbors in p2 for each point
        return_sorted (bool0: If True, return nearest neighbors sorted in
            ascending order of distance

    Returns:
        dists [bs, T*size_X, K]: Squared distances to nearest neighbors
        idx [bs, T*size_X, K]: Indices of K nearest neighbors from X to Y.
            If `X_idx[n, t, i, k] = j` then `Y[n, j]` is the k-th nearest
            neighbor to `X_idx[n, t, i]` in `Y[n]`.
        nn [bs, T*size_X, K, dim]: Coords of the K-nearest neighbors from X to Y.
    """
    if (X.shape[3] != Y.shape[3]) or (X.shape[1] != Y.shape[1]) or (X.shape[0] != Y.shape[0]):
        raise ValueError("X and Y should have same number of batch, time, and data dimensions")
    bs, T, size_X, dim = X.shape
    bs, T, size_Y, dim = Y.shape
    
    # Call knn_points, treating time as a batch dimension
    dists, idx, nn = pytorch3d.ops.knn_points(
        X.reshape(bs * T, size_X, dim), Y.reshape(bs * T, size_Y, dim),
        lengths1=lengths_X.reshape(bs * T), lengths2=lengths_Y.reshape(bs * T),
        norm=norm, K=K, version=version, return_nn=return_nn, return_sorted=return_sorted
    )  # bs*T, size_X, K  |  bs*T, size_X, K  |  bs*T, size_X, K, dim

    # Reshape into batched time-series of points, and offset points along T-dimension
    dists = dists.reshape(bs, T * size_X, K)  # bs, T*size_X, K
    nn = nn.reshape(bs, T * size_X, K, dim) if return_nn else None  # bs, T*size_X, K, dim

    idx = idx.reshape(bs, T, size_X, K)  # bs, T, size_X, K
    offsets = torch.cumsum(lengths_Y, dim=-1) - lengths_Y  # bs, T
    idx += offsets[:, :, None, None].repeat(1, 1, size_X, K)  # bs, T, size_X, K
    idx = idx.reshape(bs, T * size_X, K)  # bs, T*size_X, K

    return _KNN(dists=dists, idx=idx, knn=nn)


def timeseries_pointclouds_to_tensor(X_pts):
    """Convert a time series of variable-length point clouds to a padded tensor

    Args:
        X_pts [List(bs, npts[t], dim)]: List of length T, containing a
            time-series of variable-length point cloud batches

    Returns:
        X [bs, T, npts, dim]: Padded pointcloud tensor
        num_points_X [bs, T]: Number of points in each point cloud
    """
    bs, _, dim = X_pts[0].shape
    T = len(X_pts)
    device = X_pts[0].device

    num_points_X = torch.tensor([X_pts[t].shape[1] for t in range(T)], dtype=torch.int64)  # T,
    num_points_X = num_points_X.to(device)[None, :].repeat(bs, 1)  # bs, T

    npts = torch.max(num_points_X)
    X = X_pts[0].new_zeros(bs, T, npts, dim)  # bs, T, npts, dim
    for t in range(T):
        npts_t = X_pts[t].shape[1]
        X[:, t, :npts_t] = X_pts[t]  # bs, T, npts[t], dim

    return X, num_points_X


def timeseries_iterative_closest_point(
    X_pts, Y_pts, init_transform=None, max_iterations=100, relative_rmse_thr=1e-6,
    estimate_scale=False, allow_reflection=False, verbose=False
):
    """Execute the ICP algorithm to find a similarity transform (R, T, s)
    between two time series of differently-sized point clouds
    
    Args:
        X_pts [List(bs, npts[t], dim)]: Time-series of variable-length
            point cloud batches
        Y_pts [List(bs, npts[t], dim)]: Time-series of variable-length
            point cloud batches
        init_transform [SimilarityTransform]: If provided, initialization for
            the similarity transform, containing orthonormal matrices
            R [bs, dim, dim], translations T [bs, dim], and scaling s[bs,]
        max_iterations (int): Maximum number of ICP iterations
        relative_rmse_thr (float): Threshold on relative root mean square error
            used to terminate the algorithm
        estimate_scale (bool): If True, estimate a scaling component of the
            transformation, otherwise assume identity scale
        allow_reflection (bool): If True, allow algorithm to return `R`
            which is orthonormal but has determinant -1
        verbose: If True, print status messages during each ICP iteration

    Returns: ICPSolution with the following fields
        converged (bool): Boolean flag denoting whether the algorithm converged
        rmse (float): Attained root mean squared error after termination
        Xt [bs, T, size_X, dim]: Point cloud X transformed with final similarity
            transformation (R, T, s)
        RTs (SimilarityTransform): Named tuple containing a batch of similarity transforms:
            R [bs, dim, dim] Orthonormal matrices
            T [bs, dim]: Translations
            s [bs,]: Scaling factors
        t_history (list(SimilarityTransform)): List of similarity transform
            parameters after each ICP iteration
    """
    # Convert input Pointclouds structures to padded tensors
    X, num_points_X = timeseries_pointclouds_to_tensor(X_pts)  # bs, T, size_X, dim  |  bs, T
    Y, num_points_Y = timeseries_pointclouds_to_tensor(Y_pts)  # bs, T, size_Y, dim  |  bs, T

    if (X.shape[3] != Y.shape[3]) or (X.shape[1] != Y.shape[1]) or (X.shape[0] != Y.shape[0]):
        raise ValueError("X and Y should have same number of batch, time, and data dimensions")
    bs, T, size_X, dim = X.shape
    bs, T, size_Y, dim = Y.shape

    # Handle heterogeneous input
    if ((num_points_Y < size_Y).any() or (num_points_X < size_X).any()) and (num_points_Y != num_points_X).any():
        mask_X = (
            torch.arange(size_X, dtype=torch.int64, device=X.device)[None, None, :]
            < num_points_X[:, :, None]
        ).type_as(X)  # bs, T, size_X
    else:
        mask_X = X.new_ones(bs, T, size_X)  # bs, T, size_X
    
    X = X.reshape(bs, T * size_X, dim)  # bs, T*size_X, dim
    Y = Y.reshape(bs, T * size_Y, dim)  # bs, T*size_Y, dim
    mask_X = mask_X.reshape(bs, T * size_X)  # bs, T*size_X

    # Clone the initial point cloud
    X_init = X.clone()  # bs, T*size_X, dim

    # Initialize transformation with identity
    sim_R = torch.eye(dim, device=X.device, dtype=X.dtype)[None].repeat(bs, 1, 1)  # bs, 3, 3
    sim_T = X.new_zeros((bs, dim))  # bs, dim
    sim_s = X.new_ones(bs)  # bs,

    prev_rmse = None
    rmse = None
    iteration = -1
    converged = False
    t_history = []

    # Main loop over ICP iterations
    for iteration in range(max_iterations):
        X_nn_points = timeseries_knn_points(
            X.reshape(bs, T, size_X, dim), Y.reshape(bs, T, size_Y, dim),
            lengths_X=num_points_X, lengths_Y=num_points_Y, K=1, return_nn=True
        ).knn[:, :, 0, :]

        # Get alignment of nearest neighbors from Y with X_init
        sim_R, sim_T, sim_s = corresponding_points_alignment(
            X_init, X_nn_points, weights=mask_X, estimate_scale=estimate_scale,
            allow_reflection=allow_reflection
        )

        # Apply the estimated similarity transform to X_init
        X = _apply_similarity_transform(X_init, sim_R, sim_T, sim_s)

        # Add current transformation to history
        t_history.append(SimilarityTransform(sim_R, sim_T, sim_s))

        # Compute root mean squared error
        X_sq_diff = torch.sum((X - X_nn_points) ** 2, dim=2)
        rmse = wmean(X_sq_diff[:, :, None], mask_X).sqrt()[:, 0, 0]

        # Compute relative rmse change
        if prev_rmse is None:
            relative_rmse = rmse.new_ones(bs)
        else:
            relative_rmse = (prev_rmse - rmse) / prev_rmse

        if verbose:
            print(
                f"ICP iteration {iteration}: mean/max rmse={rmse.mean():1.2e}/{rmse.max():1.2e}; "
                f"mean relative rmse={relative_rmse.mean():1.2e}"
            )

        # Check for convergence
        if (relative_rmse <= relative_rmse_thr).all():
            converged = True
            break

        # Update the previous rmse
        prev_rmse = rmse

    X = X.reshape(bs, T, size_X, dim)  # bs, T, size_X, dim
    return ICPSolution(converged, rmse, X, SimilarityTransform(sim_R, sim_T, sim_s), t_history)


def align_seqs(all_verts_pred, all_verts_gt, correct_tra=False, verbose=False):
    """Align predicted mesh sequence to the ground-truths

    Args:
        all_verts_pred (List(bs, npts[t], 3)): Time-series of predicted mesh batches
        all_verts_gt (List(bs, npts[t], 3)): Time-series of ground-truth mesh batches
        correct_tra (bool): If True, use ground-truth translations to evaluate
        verbose (bool): Whether to print ICP results

    Returns:
        out_verts_pred (List(bs, npts[t], 3)): Time-series of aligned predicted mesh batches
    """
    nframes = len(all_verts_pred)

    # Compute coarse scale estimate (in the correct order of magnitude)
    fitted_scale = torch.zeros(nframes, dtype=torch.float32, device=device)  # nframes,
    for i in range(nframes):
        verts_pred = all_verts_pred[i]  # 1, npts_pred, 3
        verts_gt = all_verts_gt[i]  # 1, npts_gt, 3
        fitted_scale[i] = (
            (torch.max(verts_gt[..., -1]) + torch.min(verts_gt[..., -1])) /
            (torch.max(verts_pred[..., -1]) + torch.min(verts_pred[..., -1]))
        )
    fitted_scale = torch.mean(fitted_scale)

    out_verts_pred = [verts_pred * fitted_scale for verts_pred in all_verts_pred]

    # Use ICP to align the first frame and fine-tune the scale estimate
    frts0 = timeseries_iterative_closest_point(
        out_verts_pred[:1], all_verts_gt[:1], estimate_scale=True, max_iterations=100, verbose=verbose
    )
    R_icp0, T_icp0, s_icp0 = frts0.RTs  # 1, 3, 3  |  1, 3  |  1, 1

    for i in range(nframes):
        out_verts_pred[i] = _apply_similarity_transform(out_verts_pred[i], R_icp0, T_icp0, s_icp0)

        if correct_tra:
            # Optionally, replace predicted translations with ground-truth
            center_pred = torch.mean(out_verts_pred[i], dim=1, keepdims=True)  # 1, 1, 3
            center_gt = torch.mean(all_verts_gt[i], dim=1, keepdims=True)  # 1, 1, 3
            out_verts_pred[i] += center_gt - center_pred

    # Run global ICP across the point cloud time-series
    frts = timeseries_iterative_closest_point(
        out_verts_pred, all_verts_gt, estimate_scale=False, max_iterations=100, verbose=verbose
    )
    R_icp, T_icp, s_icp = frts.RTs  # 1, 3, 3  |  1, 3  |  1, 1

    for i in range(nframes):
        out_verts_pred[i] = _apply_similarity_transform(out_verts_pred[i], R_icp, T_icp, s_icp)

    return out_verts_pred


def format_latex_table(latex_table, format_spec):
    """Given a Latex table of results, write LaTeX code with best results bolded

    Args:
        latex_table (List(List(str or float)): List of table rows
        format_spec (List): A list of (spec, scale, precision) for each column:
            spec (bool or None): None if column is non-numeric, True if lower
                is better, and False if higher is better
            scale (float): Scale factor for this parameter
            precision (int): Number of decimal points to render

    Returns:
        latex_table_str (str): Formatted table of results
    """
    # Compute best entry per column
    latex_table_best = list(latex_table[0])
    for row in latex_table[1:]:
        for col_idx, (spec, scale, precision) in enumerate(format_spec):
            elt = row[col_idx]
            if spec is None:
                continue
            elif spec:
                latex_table_best[col_idx] = min(latex_table_best[col_idx], elt)
            else:
                latex_table_best[col_idx] = max(latex_table_best[col_idx], elt)

    # Format latex table, bolding the best results
    latex_table_str = []
    for row in latex_table:
        row_str = []

        for col_idx, (spec, scale, precision) in enumerate(format_spec):
            elt = row[col_idx]

            if spec is None:
                elt_str = elt
            elif elt == latex_table_best[col_idx]:
                elt_str = "\\" + f"textbf{{{scale * elt:.{precision}f}}}"
            else:
                elt_str = f"{scale * elt:.{precision}f}"

            row_str.append(elt_str)

        row_str = " & ".join(row_str)
        latex_table_str.append(row_str)

    latex_table_str = " \\\\\n".join(latex_table_str)
    return latex_table_str


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seqname", type=str, default="T_samba")
    parser.add_argument("--vidid", type=int, default=1)
    parser.add_argument("--load_dir", type=str,
            default="output/20230516-194100_6_human_networkx/eval_pose_dfms_mesh_000")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--n_workers", type=int, default=8)

    opts = parser.parse_args()
    
    for attr in opts.__dict__:
        if getattr(opts, attr) == "True":
            setattr(opts, attr, True)
        elif getattr(opts, attr) == "False":
            setattr(opts, attr, False)

    return opts


if __name__ == "__main__":
    opts = parse_args()
    main(opts)

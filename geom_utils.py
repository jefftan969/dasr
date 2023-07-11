import glob
import math
import multiprocessing
import numpy as np
import time
import torch
import tqdm
import trimesh
from pytorch3d.transforms import rotation_conversions as transforms
from scipy.spatial.transform import Rotation as R

from banmo_utils import banmo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rtk_invert(rtk_in):
    """Invert a rigid transform.

    Args
        rtk_in [..., 12]: Input rigid transforms, represented as 12-dim
            vectors [rot 1..9, trans 1..3].

    Returns
        rtk_i [..., 12]: Inverted rigid transform
    """
    prefix_shape = rtk_in.shape[:-1]
    rmat = rtk_in[..., :9].view(prefix_shape + (3, 3)) # ..., 3, 3
    tmat = rtk_in[..., 9:] # ..., 3
    rmat_i = rmat.swapaxes(-2, -1); del rmat # ..., 3, 3
    tmat_i = -torch.sum(rmat_i * tmat[..., None, :], dim=-1); del tmat # ..., 3
    rmat_i = rmat_i.reshape(prefix_shape + (9,)) # ..., 9
    rtk_i = torch.cat([rmat_i, tmat_i], dim=-1); del rmat_i, tmat_i # ..., 12
    return rtk_i


def rts_invert(rts_in):
    """Invert a rigid transform.

    Args
        rts_in [..., 3, 4]: Input rigid transforms, represented as 3x4
            matrices [rot 3x3, trans 1x3].

    Returns
        rts_i [..., 3, 4]: Inverted rigid transform
    """
    rmat = rts_in[..., :3, :3] # ..., 3, 3
    tmat = rts_in[..., :3, 3]; del rts_in # ..., 3
    rmat_i = rmat.swapaxes(-2, -1); del rmat # ..., 3, 3
    tmat_i = -torch.sum(rmat_i * tmat[..., None, :], dim=-1, keepdims=True); del tmat # ..., 3, 1
    rts_i = torch.cat([rmat_i, tmat_i], dim=-1); del rmat_i, tmat_i # ..., 3, 4
    return rts_i


def rtk_to_4x4(rtk_in):
    """Convert a rigid transform to a 4x4 homogeneous matrix.

    Args
        rtk_in [..., 12]: Input rigid transforms, represented as 12-dim
            vectors [rot 1..9, trans 1..3].

    Returns
        rts [..., 4, 4]: Rigid transform as homogeneous matrix
    """
    prefix_shape = rtk_in.shape[:-1]
    rtk_in = rtk_in.view(-1, 12) # -1, 12
    bs = rtk_in.shape[0]

    rts = torch.eye(4, dtype=torch.float32, device=rtk_in.device)[None].expand(bs, -1, -1).clone() # -1, 4, 4
    rts[:, :3, :3] = rtk_in[:, :9].view(-1, 3, 3)
    rts[:, :3, 3:] = rtk_in[:, 9:].view(-1, 3, 1); del rtk_in

    rts = rts.view(prefix_shape + (4, 4)) # ..., 4, 4
    return rts


def rtk_compose(rtk1, rtk2):
    """Compose two rigid transforms.

    Args
        rtk1 [..., 12]: Input rigid transform, represented as 12-dim
            vectors [rot 1..9, trans 1..3].
        rtk2 [..., 12]: Input rigid transform, represented as 12-dim
            vectors [rot 1..9, trans 1..3].

    Returns
        rtk [..., 12]: Composed rigid transform
    """
    assert rtk1.shape[:-1] == rtk2.shape[:-1], \
        f"Non-matching prefix shapes {rtk1.shape[:-1]}, {rtk2.shape[:-1]}"

    rts1 = rtk_to_4x4(rtk1); del rtk1 # ..., 4, 4
    rts2 = rtk_to_4x4(rtk2); del rtk2 # ..., 4, 4
    rts = torch.matmul(rts1, rts2); del rts1, rts2 # ..., 4, 4
    rvec = rts[..., :3, :3].reshape(rts.shape[:-2] + (9,)) # ..., 9
    tvec = rts[..., :3, 3]; del rts # ..., 3
    rtk = torch.cat([rvec, tvec], dim=-1); del rvec, tvec # ..., 12
    return rtk


def vec_to_sim3(vec):
    """Converts a 10-dim bone vector to xyz center, SO(3) orientation, and xyz scale

    Args
        bone_vec [..., 10]: 10-dim bone vector, represented as Gaussian ellipsoids
            [center 0..3, orient (real-first quaternion 0..4, scale 0..3]

    Returns: (center, orient, scale) where
        center [..., 3]: 3D center point
        orient [..., 3, 3]: 3x3 orientation matrix
        scale [..., 3]: 3D scale
    """
    center = vec[..., :3] # ..., 3
    orient = vec[..., 3:7] # ..., 4; real first
    orient = torch.nn.functional.normalize(orient, p=2, dim=-1) # ..., 4
    orient = transforms.quaternion_to_matrix(orient) # ..., 3, 3
    scale = torch.exp(vec[..., 7:]) # ..., 3
    return center, orient, scale


def gauss_skinning(bones, pts, truncate_softmax=None, skin_aux=None, memory_limit=None, device=None):
    """Computes skinning weights for a set of 3D points. For each point, bones
    are assigned a weight proportional to the Mahalanobis distance from
    points to bones.

    Args
        bones [num_bones, 10]: Bones represented as Gaussian ellipsoids
            [center 0..3, orient (real-first quaternion 0..4), scale 0..3]
        pts [..., 3]: Input 3D points
        truncate_softmax [int]: Max number of bones to consider contribution from
        skin_aux [2,]: Additional skinning parameters `log_scale` and `w_const`
        memory_limit [int]: If passed, maximum amount of memory to use per chunk,
            specified in number of bytes
        device [torch.device]: Target device
    
    Returns:
        skin [..., num_bones]: Skinning weights for each point
    """
    assert bones.ndim == 2 and bones.shape[-1] == 10, \
        f"`bones` should have shape [num_bones, 10] instead of {bones.shape}"
    
    device = pts.device if device is None else device
    prefix_shape = pts.shape[:-1]
    pts = pts.view(-1, 3) # -1, 3
    T = pts.shape[0]
    B = bones.shape[-2]

    log_scale, w_const = skin_aux
    center, orient, scale = vec_to_sim3(bones); del bones # B, 3 | B, 3, 3 | B, 3
    orient = orient.swapaxes(-2, -1) # B, 3, 3; transpose R

    if memory_limit is None:
        chunk_size = T
    else:
        memory_per_chunk = B * (3 + 3 + 3 + 3 + 1) * 4
        chunk_size = (memory_limit + memory_per_chunk - 1) // memory_per_chunk

    skin = []
    for i in range(0, T, chunk_size):
        pts_ch = pts[i:i+chunk_size] # Tch, 3

        # Mahalanobis distance [(p-v)^TR^T]S[R(p-v)]
        # Transform a vector to the local coordinate
        # Larger mdis (e.g. center farther from pts_ch, smaller scale)
        # becomes more negative after negative sum, and therefore has low skin weight
        if log_scale == 0 and w_const == 0:
            # New, dog80 SkelHead impl, using no log_scale and truncated softmax
            mdis_ch = center[None, :, :] - pts_ch[:, None, :]; del pts_ch # Tch, B, 3
            mdis_ch = torch.sum(orient[None, :, :, :] * mdis_ch[:, :, None, :], dim=-1) # Tch, B, 3
            mdis_ch = mdis_ch / scale[None, :, :] # Tch, B, 3
            mdis_ch = -torch.sum(mdis_ch ** 2, dim=-1) # Tch, B

        else:
            # Old, cat70 RTHead impl, using log_scale and softmax
            mdis_ch = center[None, :, :] - pts_ch[:, None, :]; del pts_ch # Tch, B, 3
            mdis_ch = torch.sum(orient[None, :, :, :] * mdis_ch[:, :, None, :], dim=-1) # Tch, B, 3
            mdis_ch = mdis_ch ** 2 * (100 * torch.exp(log_scale) * scale[None, :, :]) # Tch, B, 3
            mdis_ch = -10 * torch.sum(mdis_ch, dim=-1) # Tch, B

        # Apply truncated softmax
        if truncate_softmax is not None:
            B_trunc = min(B, truncate_softmax)
            mdis_topk_ch, mdis_idx_ch = torch.topk(mdis_ch, B_trunc, dim=-1, largest=True) # Tch, B_trunc
            mdis_ch = torch.full_like(mdis_ch, -np.inf) # Tch, B
            mdis_ch = torch.scatter(mdis_ch, -1, mdis_idx_ch, mdis_topk_ch); del mdis_topk_ch, mdis_idx_ch # Tch, B

        skin_ch = torch.softmax(mdis_ch, dim=-1).to(device); del mdis_ch # Tch, B
        skin.append(skin_ch); del skin_ch

    del pts
    skin = torch.cat(skin, dim=0).view(prefix_shape + (B,)) # ..., B
    return skin


def dual_quaternion_apply(dq, pts):
    """Apply dual quaternions to a tensor of points

    Args
        dq [Tuple(Tensor(..., 4), Tensor(..., 4))]: Dual quaternion to apply
        pts [Tensor(..., 3)]: Points
    """
    assert dq[0].shape[:-1] == dq[1].shape[:-1], \
        f"Non-matching prefix shapes {dq[0].shape[:-1]}, {dq[1].shape[:-1]}"

    qr, qd = dq; del dq # ..., 4 | ..., 4
    qr_conj = torch.cat([qr[..., :1], -qr[..., 1:]], dim=-1) # ..., 4
    t = 2 * transforms.quaternion_raw_multiply(qd, qr_conj)[..., 1:]; del qd, qr_conj # ..., 3
    out = transforms.quaternion_apply(qr, pts) + t; del qr, pts, t # ..., 3
    return out


def blend_skinning(bone_rts, skin, xyz_in, blend_method="dual_quat", memory_limit=None, device=None):
    """Given per-frame rigid transforms that each bone will apply, and a vector of
    bone weights, deform a set of 3D points by the weighted bone transform

    Args
        bone_rts [..., num_bones, 3, 4]: Rigid transforms that each bone will apply,
            expressed as 3x4 matrices
        skin [..., num_points, num_bones]: Skinning weights for each 3D point
        xyz_in [..., num_points, 3]: 3D points to deform
        blend_method [str]: Method for blending bones by skinning weights (axis_angle or dual_quat)
        memory_limit [int]: If passed, maximum amount of memory to use per chunk,
            specified in number of bytes
        device [torch.device]: Target device

    Returns: xyz [..., num_points, 3]
    """
    assert bone_rts.shape[:-3] == skin.shape[:-2] == xyz_in.shape[:-2], \
        f"Non-matching prefix shapes {bone_rts.shape[:-3]}, {skin.shape[:-2]}, {xyz_in.shape[:-2]}."

    device = bone_rts.device if device is None else device
    prefix_shape = bone_rts.shape[:-3]
    B = bone_rts.shape[-3]
    N = xyz_in.shape[-2]

    bone_rts = bone_rts.reshape(-1, B, 3, 4) # -1, B, 3, 4
    skin = skin.reshape(-1, N, B) # -1, N, B
    xyz_in = xyz_in.reshape(-1, N, 3) # -1, N, 3
    T = xyz_in.shape[0]

    rmat = bone_rts[:, :, :3, :3] # -1, B, 3, 3
    tmat = bone_rts[:, :, :3, 3]; del bone_rts # -1, B, 3

    if memory_limit is None:
        chunk_size = T
    else:
        memory_per_chunk = N * B * 4
        chunk_size = (memory_limit + memory_per_chunk - 1) // memory_per_chunk

    xyz = []
    for i in range(0, T, chunk_size):
        xyz_ch = xyz_in[i:i+chunk_size] # -1, N, 3
        skin_ch = skin[i:i+chunk_size] # -1, N, B
        rmat_ch = rmat[i:i+chunk_size] # -1, B, 3, 3
        tmat_ch = tmat[i:i+chunk_size] # -1, B, 3

        if blend_method == "axis_angle":
            # Averaging on the rotation manifold
            rot_ch = transforms.matrix_to_axis_angle(rmat_ch); del rmat_ch # -1, B, 3
            rot_wt_ch = torch.sum(skin_ch[:, :, :, None] * rot_ch[:, None, :, :], dim=-2); del rot_ch # -1, N, 3
            rmat_wt_ch = transforms.axis_angle_to_matrix(rot_wt_ch); del rot_wt_ch # -1, N, 3, 3
            tmat_wt_ch = torch.sum(skin_ch[:, :, :, None] * tmat_ch[:, None, :, :], dim=-3); del tmat_ch # -1, N, 3

            xyz_ch = (torch.sum(rmat_wt_ch * xyz_ch[:, :, None, :], dim=-1) + tmat_wt_ch).to(device) # -1, N, 3
            del rmat_wt_ch, tmat_wt_ch

        elif blend_method == "dual_quat":
            # Make sure blending quaternions on the same hemisphere by computing sign
            qr_ch = transforms.matrix_to_quaternion(rmat_ch) # -1, B, 4
            qr_ch = qr_ch[:, None, :, :].expand(-1, N, -1, -1); del rmat_ch # -1, N, B, 4
            pivot_ch = skin_ch.argmax(dim=-1)[:, :, None, None].expand(-1, -1, -1, 4) # -1, N, 1, 4
            sign_ch = torch.where(
                torch.sum(torch.gather(qr_ch, -2, pivot_ch) * qr_ch, dim=-1, keepdims=True) > 0, 1, -1
            ); del pivot_ch # -1, N, B, 1
            qr_ch = sign_ch * qr_ch; del sign_ch # -1, N, B, 4

            qt_ch = torch.cat([torch.zeros_like(tmat_ch[:, :, :1]), tmat_ch], dim=-1); del tmat_ch # -1, B, 4
            qt_ch = qt_ch[:, None, :, :].expand(-1, N, -1, -1) # -1, N, B, 3
            qd_ch = 0.5 * transforms.quaternion_raw_multiply(qt_ch, qr_ch); del qt_ch # -1, N, B, 4
            qr_wt_ch = torch.sum(skin_ch[:, :, :, None] * qr_ch, dim=-2); del qr_ch # -1, N, B, 4
            qd_wt_ch = torch.sum(skin_ch[:, :, :, None] * qd_ch, dim=-2); del qd_ch # -1, N, B, 4

            qr_norm_inv_ch = 1 / torch.norm(qr_wt_ch, p=2, dim=-1, keepdim=True) # -1, N, B, 1
            qr_wt_ch = qr_wt_ch * qr_norm_inv_ch # -1, N, B, 4
            qd_wt_ch = qd_wt_ch * qr_norm_inv_ch; del qr_norm_inv_ch # -1, N, B, 4

            xyz_ch = dual_quaternion_apply((qr_wt_ch, qd_wt_ch), xyz_ch).to(device) # -1, N, 3
            del qr_wt_ch, qd_wt_ch # -1, N, 3
            
        else:
            raise ValueError(f"Invalid blend_method '{blend_method}'")

        xyz.append(xyz_ch); del xyz_ch

    del xyz_in, skin, rmat, tmat
    xyz = torch.cat(xyz, dim=0).view(prefix_shape + (N, 3)) # ..., N, 3
    return xyz


def lbs_fw(
    bone_rts_fw, skin, xyz_in, *, find_bone_dfm=False, bones=None, blend_method="dual_quat",
    memory_limit=None, device=None
):
    """Perform forward linear blend skinning from canonical space to frame space,
    using axis angle formulation. (Leads to some artifacts)
    Given per-frame rigid transforms that each bone will apply, and a vector of
    bone weights, deform a set of 3D points by the weighted bone transform.
    Return the deformed points in frame coords and forward-deformed bones.

    Args
        bone_rts_fw [..., num_bones, 12]: Per-frame rigid transforms that
            each bone will apply, expressed as 12-dim vectors [rot 1..9, trans 1..3]
        skin [..., num_points, num_bones]: Skinning weights for each point
        xyz_in [..., num_points, 3]: 3D points to deform
        find_bone_dfm [bool]: Whether to perform bone deformation or not
        bones [num_bones, 10]: Bones represented as Gaussian ellipsoids
            [center 0..3, orient (real-first quaternion 0..4), scale 0..3],
            only required if `fine_bone_dfm` is True
        blend_method [str]: Method for blending bones by skinning weights (axis_angle or dual_quat)
        memory_limit [int]: If provided, maximum amount of memory to use per chunk,
            specified in number of bytes
        device [torch.device]: Target device

    Returns: (xyz, bones_dfm) if find_bone_dfm is True else xyz, where:
        xyz [..., num_points, 3]: Deformed points
        bones_dfm [..., num_bones, 10]: Deformed bones
    """
    assert bones is None or bones.ndim == 2 and bones.shape[-1] == 10, \
        f"`bones` should have shape [num_bones, 10] instead of {bones.shape}"
    assert bone_rts_fw.shape[:-2] == skin.shape[:-2] == xyz_in.shape[:-2], \
        f"Non-matching prefix shapes {bone_rts_fw.shape[:-2]}, {skin.shape[:-2]}, {xyz_in.shape[:-2]}"

    prefix_shape = xyz_in.shape[:-2]
    B = bone_rts_fw.shape[-2]
    bs = xyz_in.shape[0]

    rmat = bone_rts_fw[..., :, :9].view(prefix_shape + (B, 3, 3)) # ..., B, 3, 3
    tmat = bone_rts_fw[..., :, 9:].view(prefix_shape + (B, 3, 1)) # ..., B, 3, 1
    bone_rts_fw = torch.cat([rmat, tmat], dim=-1); del rmat, tmat # ..., B, 3, 4

    xyz = blend_skinning(
        bone_rts_fw, skin, xyz_in, blend_method=blend_method, memory_limit=memory_limit, device=device
    ); del xyz_in # ..., N, 3

    if find_bone_dfm:
        bones_in = bones[None].expand(bone_rts_fw.shape[:-3] + (-1, -1)) # ..., B, 10
        bones_dfm = bone_transform(bones_in, bone_rts_fw) # ..., B, 10; bone coordinates after deform
        return xyz, bones_dfm
    else:
        return xyz


def bone_transform(bones_in, bone_rts, is_vec=False):
    """Given bones represented as Gaussian ellipsoids, and rigid transforms
    associated with each bone, apply the rigid transforms to the bones

    Args
        bones_in [..., num_bones, 10]: Bones represented as Gaussian ellipsoids
            [center 0..3, orient (real-first quaternion 0..4), scale 0..3]
        bone_rts [..., num_bones, 12] or [..., num_bones, 3, 4]: Rigid transforms
            associated with each bone, applied to bone coordinate transforms (left-multiply).
            Expressed as 12-dim vectors [rot 1..9, trans 1..3] if `is_vec` is
            True, or 3x4 matrices if `is_vec` is False.
        is_vec [bool]: Whether `bone_rts` are expressed as 12-dim vectors
            or 3x4 matrices.

    Returns
        bones_dfm [..., num_bones, 10]: Deformed bones
    """
    if is_vec:
        assert bones_in.shape[:-2] == bone_rts.shape[:-2], \
            f"Non-matching prefix shapes {bones_in.shape[:-2]}, {bone_rts.shape[:-2]}."
    else:
        assert bones_in.shape[:-2] == bone_rts.shape[:-3], \
            f"Non-matching prefix shapes {bones_in.shape[:-2]}, {bone_rts.shape[:-3]}."

    prefix_shape = bones_in.shape[:-2]
    B = bones_in.shape[-2]
    bones = bones_in.view(-1, B, 10)
    if is_vec:
        bone_rts = bone_rts.view(-1, B, 12)
    else:
        bone_rts = bone_rts.view(-1, B, 3, 4)

    center = bones[:, :, :3] # -1, B, 3
    orient = bones[:, :, 3:7] # -1, B, 4; real first
    scale = bones[:, :, 7:] # -1, B, 3

    if is_vec:
        rmat = bone_rts[:, :, :9].view(-1, B, 3, 3) # -1, B, 3, 3
        tmat = bone_rts[:, :, 9:] # -1, B, 3
    else:
        rmat = bone_rts[:, :, :3, :3] # -1, B, 3, 3
        tmat = bone_rts[:, :, :3, 3] # -1, B, 3

    # Move bone coordinates
    center = torch.sum(rmat * center[:, :, None, :], dim=-1) + tmat # -1, B, 3
    rquat = transforms.matrix_to_quaternion(rmat) # -1, B, 4
    orient = transforms.quaternion_raw_multiply(rquat, orient) # -1, B, 4

    bones = torch.cat([center, orient, scale], dim=-1) # -1, B, 10
    bones = bones.view(prefix_shape + (B, 10)) # ..., B, 10
    return bones


def compute_bone_from_joint(model, init_scale=False):
    """Compute center, orient, and scale from URDF definition

    Args
        model: Banmo model containing `robot` and `nerf_body_rts` networks

    Returns
        bones_rst [num_bones, 10]: Corrected bone locations
    """
    # Compute bones as center of links
    urdf = model.robot.urdf

    # Get canonical sim3 and joint centers
    sim3 = model.robot.sim3 # 10,
    joints, _ = model.nerf_body_rts.forward_abs() # 1, 1, n_joints*12 | 1, 75
    joints = joints.view(-1, 12).to(device) # n_joints, 12
    rmat = joints[:, :9].view(-1, 3, 3) # n_joints, 3, 3
    tmat = joints[:, 9:].view(-1, 3, 1) # n_joints, 3, 1
    fk = torch.cat([rmat, tmat], dim=-1) # n_joints, 3, 4

    # Update joint to link centers
    center = []
    scale = []
    orient = []
    idx = 0
    for link in urdf._reverse_topo:
        path = urdf._paths_to_base[link] # urdfpy.Link object
        if len(path) > 1:
            joint = urdf._G.get_edge_data(path[0], path[1])["joint"]
            if joint.name not in urdf.name2query_idx:
                continue

        if len(link.visuals) > 0:
            link_bounds = link.visuals[0].geometry.meshes[0].bounds # 2, 3

            # Scale factor
            link_scale = torch.tensor(link_bounds[1] - link_bounds[0], dtype=torch.float32, device=device) # 3,
            link_scale = link_scale * 5 * torch.exp(sim3[7:]) # 3,

            # Bone center
            fk_rot = fk[None, idx, :3, :3] # 1, 3, 3
            fk_tra = fk[None, idx, :3, 3] # 1, 3
            link_corners = trimesh.bounds.corners(link_bounds) # 8, 3
            link_corners += link.visuals[0].origin[None, :3, 3]
            link_corners = torch.tensor(link_corners, dtype=torch.float32, device=device) # 8, 3
            link_corners = link_corners * torch.exp(sim3[7:])[None, :] # 8, 3
            link_corners = torch.sum(fk_rot * link_corners[:, None, :], dim=-1) + fk_tra # 8, 3
            link_center = link_corners.mean(dim=0) # 3,

        else:
            link_scale = torch.tensor([1, 1, 1], dtype=torch.float32, device=device) * torch.exp(-3.5) # 3,
            link_center = fk[idx, :3, 3] # 3,

        link_orient = fk[idx, :3, :3] # 3, 3
        link_orient = transforms.matrix_to_quaternion(link_orient) # 4,
        link_scale = torch.log(link_scale) # 3,

        idx += 1

        center.append(link_center)
        orient.append(link_orient)
        scale.append(link_scale)

    center = torch.stack(center, dim=0) # B, 3
    orient = torch.stack(orient, dim=0) # B, 4
    if init_scale:
        scale = torch.stack(scale, dim=0) # B, 3
    else:
        scale = model.bones[:, 7:] # B, 3

    bones = torch.cat([center, orient, scale], dim=-1) # B, 10
    return bones


def zero_to_rest_bone(model, bones_rst):
    """Correct bone locations by applying the per-bone rigid transforms derived from
    an object's rest pose. Depending on model.opts.pre_skel, may be initialized from URDF

    Args
        model: Banmo model containing `rest_pose_code` and `nerf_body_rts` networks
        bones_rst [..., num_bones, 10]: Bones represented as Gaussian ellipsoids
            [center 0..3, orient (real-first quaternion 0..4, scale 0..3]

    Returns:
        bones_rst [..., num_bones, 10]: Corrected bone locations
        bone_rts_rst [num_bones, 12]: Rigid transforms associated with each bone,
            derived from an object's rest pose
    """
    B = bones_rst.shape[-2]
    rest_pose_code = model.rest_pose_code(torch.tensor([0], dtype=torch.int64, device=device))

    if isinstance(model.nerf_body_rts, torch.nn.Sequential):
        bone_rts_rst = model.nerf_body_rts[1](rest_pose_code)[0].view(B, 12) # B, 12
    else:
        bone_rts_rst = model.nerf_body_rts.forward_decode(rest_pose_code, None)[0].view(B, 12) # B, 12
        if model.opts.pre_skel != "":
            # If skeleton model is defined, overwrite bones with URDF definition
            bones_rst = compute_bone_from_joint(model).to(device) # B, 10
            return bones_rst, bone_rts_rst

    bones_rst = bone_transform(bones_rst, bone_rts_rst, is_vec=True) # ..., B, 10
    return bones_rst, bone_rts_rst


def zero_to_rest_dpose(bone_rts_fw, bone_rts_rst):
    """Correct a set of per-bone rigid transforms by applying the inverse of the rigid
    transforms associated with an object's rest pose.

    Args
        bone_rts_fw [..., num_bones, 12]: Per-bone rigid transforms to correct,
            expressed as 12-dim vectors [rot 1..9, trans 1..3].
        bone_rts_rst [..., num_bones, 12]: Rigid transforms associated with each bone,
            derived from an object's rest pose

    Returns:
        bone_rts_fw [..., num_bones, 12]: Corrected per-bone rigid transforms
    """
    assert bone_rts_fw.shape[:-2] == bone_rts_rst.shape[:-2], \
        f"Non-matching prefix shapes {bone_rts_fw.shape[:-2]}, {bone_rts_rst.shape[:-2]}."
    bone_rts_rst_inv = rtk_invert(bone_rts_rst) # ..., B, 12
    bone_rts_fw = rtk_compose(bone_rts_fw, bone_rts_rst_inv) # ..., B, 12
    return bone_rts_fw


def K2mat(K):
    """Convert a 4-tuple of camera intrinsics to matrix

    Args
        K [..., 4]: Camera intrinsics (fx, fy, px, py)

    Returns
        Kmat [..., 3, 3]: Camera intrinsics matrix
    """
    prefix_shape = K.shape[:-1]
    K = K.view(-1, 4) # -1, 4
    bs = K.shape[0]

    Kmat = torch.zeros(bs, 3, 3, dtype=torch.float32, device=device) # -1, 3, 3
    Kmat[:, 0, 0] = K[:, 0]
    Kmat[:, 1, 1] = K[:, 1]
    Kmat[:, 0, 2] = K[:, 2]
    Kmat[:, 1, 2] = K[:, 3]
    Kmat[:, 2, 2] = 1

    Kmat = Kmat.view(prefix_shape + (3, 3)) # ..., 3, 3
    return Kmat


def warp_fw(
    model, pts_can, bones_rst, bone_rts_rst, centroid, root_poses, bone_rts_fw, *,
    blend_method="dual_quat", memory_limit=None, device=None,
):
    """Use linear blend skinning to apply deformation to a set of canonical points.

    Args
        model: Banmo model containing `rest_pose_code` and `nerf_body_rts` networks
        pts_can [npts, 3] or [..., npts, 3]: 3D points to deform
        bones_rst [num_bones, 10]: Rest bones
        bone_rts_rst [num_bones, 12]: Rest bone transforms
        centroid [3,] or [..., 3]: Mesh centroid before centering
        root_poses [..., 12]: Root body poses for deforming the points
        bone_rts_fw [..., B*12]: Bone transforms for deforming the points
        blend_method [str]: Method for blending bones by skinning weights (axis_angle or dual_quat)
        memory_limit [int]: If provided, maximum amount of memory to use per chunk,
            specified in number of bytes
        device [torch.device]: Target device

    Returns
        pts_dfm [..., num_points, 3]: Forward-deformed points
    """
    assert root_poses.shape[:-1] == bone_rts_fw.shape[:-1], \
        f"Non-matching prefix shapes {root_poses.shape[:-1]}, {bone_rts_fw.shape[:-1]}"
    assert pts_can.ndim == 2 or pts_can.shape[:-2] == root_poses.shape[:-1], \
        f"Non-matching pts_can shape: expected [npts, 3] or prefix shape {root_poses.shape[:-1]} " \
        f"but found shape {pts_can.shape}"

    B = bones_rst.shape[0]
    N = pts_can.shape[-2]
    
    prefix_shape = root_poses.shape[:-1]
    root_poses = root_poses.reshape(-1, 12) # T, 12
    bone_rts_fw = bone_rts_fw.reshape(-1, B, 12) # T, B, 12
    if pts_can.ndim != 2:
        pts_can = pts_can.reshape(-1, N, 3) # T, N, 3
        centroid = centroid.reshape(-1, 1, 3) # T, 1, 3
    T = root_poses.shape[0]
    
    if memory_limit is None:
        chunk_size = T
    else:
        memory_per_chunk = N * B * 4
        chunk_size = (memory_limit + memory_per_chunk - 1) // memory_per_chunk

    pts_dfm_out = []
    for i in range(0, T, chunk_size):
        Tch = min(T - i, chunk_size)
        root_poses_ch = root_poses[i:i+chunk_size] # Tch, 12
        bone_rts_fw_ch = bone_rts_fw[i:i+chunk_size] # Tch, B, 12

        # Compute forward skinning weights: for each point, what bones is it controlled by
        if pts_can.ndim == 2:
            skin_forward_ch = gauss_skinning(
                bones_rst, centroid + pts_can, skin_aux=model.skin_aux, memory_limit=memory_limit
            )[None, :, :].expand(Tch, -1, -1) # Tch, N, B
            pts_can_ch = pts_can[None].expand(Tch, -1, -1) # Tch, N, 3
            centroid_ch = centroid[None, None, :].expand(Tch, 1, -1) # Tch, 1, 3
        else:
            pts_can_ch = pts_can[i:i+chunk_size] # Tch, N, 3
            centroid_ch = centroid[i:i+chunk_size] # Tch, 1, 3
            skin_forward_ch = gauss_skinning(
                bones_rst, centroid_ch + pts_can_ch, skin_aux=model.skin_aux, memory_limit=memory_limit
            ) # T, N, B
    
        # Perform linear blend skinning in original, non-centered coords to compute deformed points and bones
        pts_dfm_ch = lbs_fw(
            bone_rts_fw_ch, skin_forward_ch, centroid_ch + pts_can_ch,
            blend_method=blend_method, memory_limit=memory_limit
        ) - centroid_ch; del bone_rts_fw_ch, skin_forward_ch, pts_can_ch, centroid_ch # T, N, 3
 
        # Apply root body pose transform (pre-multiply) to compute final deformed points
        rot_ch = root_poses_ch[:, None, :9].view(Tch, 1, 3, 3) # Tch, 1, 3, 3
        tra_ch = root_poses_ch[:, None, 9:]; del root_poses_ch # Tch, 1, 3
        pts_dfm_ch = (torch.sum(rot_ch * pts_dfm_ch[:, :, None, :], dim=-1) + tra_ch).to(device) # Tch, N, 3
        del rot_ch, tra_ch

        pts_dfm_out.append(pts_dfm_ch); del pts_dfm_ch

    del root_poses, bone_rts_fw
    pts_dfm_out = torch.cat(pts_dfm_out, dim=0).view(prefix_shape + (N, 3)) # T, N, 3
    return pts_dfm_out


def compute_face_normals(vertices, faces, eps=1e-13, memory_limit=None, device=None):
    """Compute face normal vectors for a batch of meshes with identical topology.
    Performs trimesh's normal computation algorithm on GPU.
    Reference: https://github.com/mikedh/trimesh/blob/master/trimesh/primitives.py#L71

    Args
        vertices [..., num_points, 3]: Mesh vertices
        faces [num_faces, 3]: Mesh faces
        eps [float]: Floating point threshold, equal to 100x the resolution of a float
        memory_limit [int]: If passed, maximum amount of memory to use per chunk,
            specified in number of bytes
        device [torch.device]: Target device

    Returns
        normals_pred [..., num_faces, 3]: Predicted face normals
    """
    assert faces.ndim == 2 and faces.shape[-1] == 3, \
        f"Faces should have shape [num_faces, 3] but found {faces.shape}"

    prefix_shape = vertices.shape[:-2]
    N = vertices.shape[-2]
    F = faces.shape[-2]
    vertices = vertices.reshape(-1, N, 3) # T, N, 3
    T = vertices.shape[0]

    if memory_limit is None:
        chunk_size = T
    else:
        memory_per_chunk = F * (9 + 3 + 1 + 3 + 3) * 4
        chunk_size = (memory_limit + memory_per_chunk - 1) // memory_per_chunk

    normals_pred = []
    for i in range(0, T, chunk_size):
        Tch = min(T - i, chunk_size)
        vertices_ch = vertices[i:i+chunk_size, :, None, :].expand(-1, -1, 3, -1) # Tch, N, 3, 3
        faces_ch = faces[None, :, :, None].expand(Tch, -1, -1, 3) # Tch, F, 3, 3

        tris_ch = torch.gather(vertices_ch, 1, faces_ch); del vertices_ch, faces_ch # Tch, F, 3, 3

        cross_ch = torch.cross(
            tris_ch[:, :, 1, :] - tris_ch[:, :, 0, :], tris_ch[:, :, 2, :] - tris_ch[:, :, 1, :]
        ); del tris_ch # Tch, F, 3
        norm_ch = torch.norm(cross_ch, p=2, dim=-1, keepdim=True) # Tch, F, 1
        normals_pred_ch = torch.where(norm_ch <= eps, 0, cross_ch / norm_ch).to(device) # Tch, F, 3
        normals_pred.append(normals_pred_ch); del cross_ch, norm_ch, normals_pred_ch

    del vertices, faces
    normals_pred = torch.cat(normals_pred, dim=0).view(prefix_shape + (F, 3)) # ..., F, 3
    return normals_pred


# ===== Mesh loading and visualization

def get_vertex_colors(model, xyz_query, centroid, env_codes, memory_limit=None, device=None):
    """Compute color for mesh vertices per frame by evaluating nerf_coarse with environment code

    Args
        model [Banmo]: Banmo model containing `embedding_xyz` and `nerf_coarse`
        xyz_query [npts, 3] or [..., npts, 3]: Per-frame mesh vertex locations to assign colors to
        centroid [3,] or [..., 3]: Mesh centroid prior to centering
        env_codes [..., Ce]: Per-frame environment codes
        memory_limit [int]: If passed, maximum amount of memory to use per chunk,
            specified in number of bytes
        device [torch.device]: Target device

    Returns
        colors [..., npts, 3]: Per-frame RGB colors for each vertex
    """
    prefix_shape = env_codes.shape[:-1]
    Ce = env_codes.shape[-1]
    env_codes = env_codes.reshape(-1, Ce) # T, Ce
    T = env_codes.shape[0]
    N = xyz_query.shape[-2]
    
    if memory_limit is None:
        chunk_size = T
    else:
        memory_per_chunk = N * (64 + 63 + 3 + 127) * 8 * 4
        chunk_size = (memory_limit + memory_per_chunk - 1) // memory_per_chunk

    colors = []
    for i in range(0, T, chunk_size):
        Tch = min(T - i, chunk_size)
        env_codes_ch = env_codes[i:i+chunk_size, None, :].expand(-1, N, -1) # Tch, N, Ce
        
        if xyz_query.ndim == 2:
            xyz_emb_ch = model.embedding_xyz(centroid + xyz_query)[None, :, :].expand(Tch, -1, -1) # Tch, N, 63

        else:
            xyz_query_ch = xyz_query[i:i+chunk_size] # Tch, N, 3
            centroid_ch = centroid[i:i+chunk_size, None, :].expand(Tch, 1, -1) # Tch, 1, 3

            xyz_query_ch = centroid_ch + xyz_query_ch; del centroid_ch # Tch, N, 3
            xyz_emb_ch = model.embedding_xyz(xyz_query_ch.view(Tch * N, -1)).view(Tch, N, -1) # Tch, N, 63
            del xyz_query_ch

        xyz_emb_ch = torch.cat([xyz_emb_ch, env_codes_ch], -1); del env_codes_ch # Tch, N, 127
        colors_ch = model.nerf_coarse(xyz_emb_ch.view(Tch * N, 1, -1)).view(Tch, N, -1)[:, :, :3] # Tch, N, 3
        colors_ch = torch.clamp(colors_ch, 0, 1).to(device); del xyz_emb_ch # Tch, N, 3

        colors.append(colors_ch); del colors_ch
   
    del env_codes
    colors = torch.cat(colors, dim=0) # T, N, 3
    return colors


# ===== Mesh I/O

def load_mesh(mesh_path, center_mesh=True):
    """Load a mesh from the given mesh path, and perform postprocessing if needed

    Args
        mesh_path [str]: Path to rest mesh .obj file
        center_mesh [bool]: If True, return a centered mesh by subtracting the
            centroid. If False, do not center the mesh

    Returns
        mesh [Trimesh]: Rest mesh, possibly centered
        centroid [3,]: Equal to 0 if `center_mesh` is False, or `mesh.centroid`
            prior to centering if `center_mesh` is True.
    """
    mesh = trimesh.exchange.load.load(mesh_path)
    
    # Mesh post-processing
    if len(mesh.vertices) > 0:
        # Keep only the largest connected component
        mesh = [x for x in mesh.split(only_watertight=False)]
        mesh = sorted(mesh, key=lambda x: x.vertices.shape[0], reverse=True)[0]

        # Assign color based on canonical location
        colors = mesh.vertices # npts, 3
        colors_min = np.min(colors, axis=0, keepdims=True) # 1, 3
        colors_max = np.max(colors, axis=0, keepdims=True) # 1, 3
        colors = (colors - colors_min) / (colors_max - colors_min) # npts, 3

        mesh.visual.vertex_colors[:, :3] = colors * 255

    # Compute mesh centroid and optionally center
    if center_mesh:
        centroid = mesh.centroid
        mesh.vertices -= centroid
    else:
        centroid = 0 * mesh.centroid
    centroid = torch.tensor(centroid, dtype=torch.float32, device=device)

    return mesh, centroid


def label_colormap():
    return np.array([
        [155, 122, 157],
        [ 45, 245,  50],
        [ 71,  25,  64],
        [231, 176,  35],
        [125, 249, 245],
        [ 32,  75, 253],
        [241,  31, 111],
        [218,  71, 252],
        [248, 220, 197],
        [ 34, 194, 198],
        [108, 178,  96],
        [ 33, 101, 119],
        [125, 100,  26],
        [209, 235, 102],
        [116, 105, 241],
        [100,  50, 147],
        [193, 159, 222],
        [ 95, 254, 138],
        [197, 130,  75],
        [144,  31, 211],
        [ 46, 150,  26],
        [242,  90, 174],
        [179,  41,  38],
        [118, 204, 174],
        [145, 209,  38],
        [188,  74, 125],
        [ 95, 158, 210],
        [237, 152, 130],
        [ 53, 151, 157],
        [ 69,  86, 193],
        [ 60, 204, 122],
        [251,  77,  58],
        [174, 248, 170],
        [ 28,  81,  36],
        [252, 134, 243],
        [ 62, 254, 193],
        [ 68, 209, 254],
        [ 44,  25, 184],
        [131,  58,  80],
        [188, 251,  27],
        [156,  25, 132],
        [248,  36, 225],
        [ 95, 130,  63],
        [222, 204, 244],
        [185, 186, 134],
        [160, 146,  44],
        [244, 196,  89],
        [ 39,  60,  87],
        [134, 239,  87],
        [ 25, 166,  97],
        [ 79,  36, 229],
        [ 45, 130, 216],
        [177,  90, 200],
        [ 86, 218,  30],
        [ 97, 115, 165],
        [159, 104,  99],
        [168, 220, 219],
        [134,  76, 180],
        [ 31, 238, 157],
        [ 79, 140, 253],
        [124,  23,  27],
        [245, 234,  46],
        [188,  30, 174],
        [253, 246, 148],
        [228,  94,  92],
    ])


def bones_to_mesh(bones, len_max=0.1):
    """Save bones to a skeleton mesh

    Args:
        bones [n_bones, 10]: Gaussian bones

    Returns:
        bone_mesh [Trimesh]: Skeleton mesh
    """
    bones = bones.detach().cpu().numpy()
    B = len(bones)
    elips_list = []
    elips = trimesh.creation.uv_sphere(radius=0.1, count=[16, 16])
    # remove identical vertices
    elips = trimesh.Trimesh(vertices=elips.vertices, faces=elips.faces)
    N_elips = len(elips.vertices)
    parent = banmo().robot.urdf.parent_idx

    for idx, bone in enumerate(bones):
        center = bone[0:3]  # 3,
        orient = bone[3:7]  # 4, real first
        orient = orient / np.linalg.norm(orient, 2, axis=-1)  # 4,
        orient = orient[[1, 2, 3, 0]]
        orient = R.from_quat(orient).as_matrix()  # 3, 3
        orient = orient.T  # transpose R
        scale = np.exp(bone[7:10])  # 3,
        # bone coord to root coord
        elips_verts = elips.vertices
        elips_verts = elips_verts * scale[None]
        elips_verts = elips_verts.dot(orient)
        elips_verts = elips_verts + center[None]
        elips_sub = trimesh.Trimesh(vertices=elips_verts, faces=elips.faces)

        if parent[idx] > -1:
            center_parent = bones[parent[idx]][:3]
        else:
            center_parent = center - len_max / 200
        link = np.stack([center, center_parent], axis=0)
        link = trimesh.creation.cylinder(len_max / 100, segment=link, sections=5)
        N_link = link.vertices.shape[0]
        elips_sub = trimesh.util.concatenate([elips_sub, link])

        elips_list.append(elips_sub)

    elips = trimesh.util.concatenate(elips_list)

    colormap = label_colormap()[:B]
    colormap = np.tile(colormap[:, None], (1, N_elips + N_link, 1))
    colormap[:, N_elips:] = 128
    colormap = colormap.reshape((-1, 3))
    elips.visual.vertex_colors[:len(colormap), :3] = colormap
    tmp = np.sum(elips.visual.vertex_colors[:, :3])
    return elips


def gl_projection(pts_dfms, near=1, far=100, img_dim=224, viewing_angle=30):
    """Apply OpenGL's projection matrix from 3D object space to 2D image space
    Reference: www.songho.ca/opengl/gl_projectionmatrix.html

    Args
        pts_dfms [..., 3]: Points in 3D object space
        near [float]: Near bound
        far [float]: Far bound
        img_dim [int]: Window size
        viewing_angle [float]: Field of view, for determining eye_z

    Retrns
        pts_dfms_window [..., 3]: Points in 2D image space
    """
    pts_dfms_obj = torch.cat([pts_dfms, torch.ones_like(pts_dfms[..., :1])], dim=-1) # ..., 4
    del pts_dfms

    # [Obj => Eye] Apply ModelView matrix
    # View component derived from softras transforms.py defaults
    # Model component derived from what we apply in render_utils.py/softras_render_mesh
    eye_z = -(1 / math.tan(math.radians(viewing_angle)) + 1)
    m_model_view = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, eye_z],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device) # 4, 4

    # [Eye => Clip] Apply Projection matrix (orthographic projection)
    n, f, l, r, b, t = near, far, -1, 1, -1, 1
    m_proj = torch.tensor([
        [2 / (r - l), 0, 0, -(r + l) / (r - l)],
        [0, 2 / (t - b), 0, -(t + b) / (t - b)],
        [0, 0, -2 / (f - n), -(f + n) / (f - n)],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device) # 4, 4

    m_model_view_proj = torch.matmul(m_proj, m_model_view) # 4, 4
    pts_dfms_clip = torch.sum(m_model_view_proj * pts_dfms_obj[..., None, :], dim=-1); del pts_dfms_obj # ..., 4

    # [Clip => NDC] Divide by w
    pts_dfms_ndc = pts_dfms_clip[..., :3] / pts_dfms_clip[..., 3:]; del pts_dfms_clip # ..., 3

    # [NDC => Window] Apply viewport transform
    x, y, w, h = 0, 0, img_dim, img_dim
    pts_dfms_window = torch.stack([
        x + w / 2 * (pts_dfms_ndc[..., 0] + 1),
        y + h / 2 * (pts_dfms_ndc[..., 1] + 1),
        n + (f - n) / 2 * (pts_dfms_ndc[..., 2] + 1),
    ], dim=-1); del pts_dfms_ndc # ..., 3

    return pts_dfms_window

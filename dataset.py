import glob
import json
import multiprocessing
import numpy as np
import os
import time
import tqdm
import torch
from datetime import date

from banmo_utils import banmo
from geom_utils import warp_fw, get_vertex_colors, zero_to_rest_dpose, vec_to_sim3
from data_utils import (
    GLOB_TEMPLATES, read_img, read_pfm, write_img, camera_ks_from_banmo_config, seqnames_from_banmo_config
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL_CACHE = {}

# ===== Template Classes

class BaseDataset(torch.utils.data.Dataset):
    """Computes filepaths and other info from a list of videos to use.
    Frames will be loaded according to their order in the full banmo dataset,
    but can be locally ordered according to an arbitrary sequence of videos.
    
    Args
        banmo_seqname [string]: Name of the original banmo .config file for this dataset.
            Needed to establish correspondences between our dataset indices and banmo's,
            since banmo uses raw frame indices to index into FrameCode and other embeddings
        videos [List(string)]: A list of videos to include in this dataset.
        dataset_glob_type [string]: Type of glob template used to load dataset.
        n_data_workers [int]: Number of worker processes to use for dataloading
    """
    def __init__(self, banmo_seqname, videos, dataset_glob_type, *, n_data_workers=0):
        super().__init__()
        self.banmo_seqname = banmo_seqname
        self.videos = videos
        self.dataset_glob_type = dataset_glob_type
        self.n_data_workers = n_data_workers

        # Whether dataset is enabled. This attribute can be set from externally
        # to avoid unnecessary dataloading when the result won't be used
        self.enabled = True

        # Compute all seqnames available in full banmo dataset
        banmo_seqnames = seqnames_from_banmo_config(banmo_seqname)
        
        # Based on the dataset type, compute globs for each video. Use %s as a placeholder for seqname
        if dataset_glob_type in GLOB_TEMPLATES:
            banmo_globs = [GLOB_TEMPLATES[dataset_glob_type] % seq for seq in banmo_seqnames]
        else:
            raise RuntimeError(f"Unsupported dataset type '{dataset_glob_type}'")

        # Compute video and frame offsets in the full banmo dataset
        # Each glob represents a video, each globbed item represents a frame
        self.banmo_video_offsets = [0] # A list mapping video i to the number of frames appearing before it
        self.banmo_video_impaths = []  # A list mapping video i to a list of image paths for that video
        self.banmo_frame_videoid = []  # A list mapping video i to a list of parent videos per frame
        self.banmo_frame_offsets = []  # A list mapping video i to a list of parent video offsets per frame
        self.banmo_nvideos = 0 # Number of videos
        self.banmo_nframes = 0 # Number of frames

        for g in banmo_globs:
            video_paths = [os.path.abspath(filename) for filename in sorted(glob.glob(g))]
            n_frames = len(video_paths)
            if n_frames == 0:
                print(f"Warning: Empty video '{g}'")

            self.banmo_video_offsets.append(self.banmo_video_offsets[-1] + n_frames)
            self.banmo_video_impaths.append(video_paths)
            self.banmo_frame_videoid += [self.banmo_nvideos] * n_frames
            self.banmo_frame_offsets += list(range(n_frames))
            self.banmo_nvideos += 1
            self.banmo_nframes += n_frames

        # Collect video and frame offsets for the specific videos in this dataset.
        # These offsets are in the same order as videos
        self.video_offsets = [0] # A list mapping video i to the number of frames appearing before it
        self.video_impaths = []  # A list mapping video i to a list of image paths for that video
        self.frame_videoid = []  # A list mapping frame i to its parent video
        self.frame_offsets = []  # A list mapping frame i to its offset within parent video
        self.nvideos = 0 # Number of videos
        self.nframes = 0 # Number of frames

        self.banmo_glob_idxs = {g: i for i, g in enumerate(banmo_globs)}
        self.idx_to_banmo = [] # Index mapping from this dataset to full banmo dataset
        if isinstance(videos, str):
            video_path = f"configs/{banmo_seqname}/{videos}"
            if os.path.exists(video_path) and video_path.endswith(".json"):
                with open(video_path, "r") as videos_file:
                    videos = json.load(videos_file)
            else:
                videos = video_path.split(",")

        if isinstance(videos, list):
            if len(videos) == 0:
                raise ValueError(f"Videos is an empty list")
        else:
            raise ValueError(
                f"Expected `videos` to be a JSON file or a list of videos, but found '{videos}'"
            )
        globs = [GLOB_TEMPLATES[dataset_glob_type] % seq for seq in videos]

        for g in globs:
            i = self.banmo_glob_idxs[g]
            self.idx_to_banmo.append(i)
            video_paths = self.banmo_video_impaths[i]
            n_frames = len(video_paths)

            self.video_offsets.append(self.video_offsets[-1] + n_frames)
            self.video_impaths.append(video_paths)
            self.frame_videoid += [self.nvideos] * n_frames
            self.frame_offsets += list(range(n_frames))
            self.nvideos += 1
            self.nframes += n_frames

    def __len__(self):
        return self.nframes

    def __getitem__(self, idx):
        # Should be overridden by subclass
        raise NotImplementedError


class GroundTruthDataset(BaseDataset):
    """Template class for a dataset that returns ground-truth values from saved banmo model.

    Args
        dataset_label [str]: The type of ground-truth data returned by this dataset.
            Used in debug output, and to name the cachedir if applicable.
        use_cache [bool]: Whether to speed up dataloading by caching all data as .npy file
        temporal_radius [int]: Size of temporal window in each direction
    """
    def __init__(
        self, banmo_seqname, videos, dataset_label, *, pad_mode="constant_nan", use_cache=True, temporal_radius=0
    ):
        super().__init__(banmo_seqname, videos, "img")
        self.dataset_label = dataset_label
        self.pad_mode = pad_mode
        self.use_cache = use_cache
        self.temporal_radius = temporal_radius

        cachedir = f"cache/{banmo_seqname}/{dataset_label}_cache.npy"

        if use_cache:
            os.makedirs(f"cache/{banmo_seqname}", exist_ok=True)
            if os.path.exists(cachedir):
                # Preload values from global numpy cache
                banmo_data = np.load(cachedir, mmap_mode="r")
                assert banmo_data.shape[0] == self.banmo_nframes, \
                    f"Expected {self.banmo_nframes} frames in {cachedir} but found {banmo_data.shape[0]}"
            else:
                # Compute banmo ground-truths and store in global cache
                start_time = time.time()
                print(f"Computing ground-truth {dataset_label}...")

                with torch.no_grad():
                    banmo_data = self.compute_banmo_data().cpu().numpy()
                    np.save(cachedir, banmo_data)    

                print(f"Computed ground-truth {dataset_label} in {time.time() - start_time:.3f}s")
        else:
            global LOCAL_CACHE
            if cachedir in LOCAL_CACHE:
                # Preload values from local cache
                banmo_data = LOCAL_CACHE[cachedir]
                assert banmo_data.shape[0] == self.banmo_nframes, \
                    f"Expected {self.banmo_nframes} frames in {cachedir} but found {banmo_data.shape[0]}"
            else:
                # Compute banmo ground-truths and store in local cache
                start_time = time.time()
                print(f"Computing ground-truth {dataset_label}...")

                with torch.no_grad():
                    banmo_data = self.compute_banmo_data().cpu().numpy()
                    LOCAL_CACHE[cachedir] = banmo_data

                print(f"Computed ground-truth {dataset_label} in {time.time() - start_time:.3f}s")
                

        # Compute per-video data by indexing into banmo dataset
        self.video_data = []
        for i in range(self.nvideos):
            banmo_idx = self.idx_to_banmo[i]
            offset0 = self.banmo_video_offsets[banmo_idx]
            offset1 = self.banmo_video_offsets[banmo_idx + 1]
            self.video_data.append(banmo_data[offset0:offset1])

        self.all_data = torch.from_numpy(np.concatenate(self.video_data, axis=0))
        
        # Pad ground-truth values from each video
        if self.temporal_radius > 0:
            pad_dim = 1 * [(self.temporal_radius, self.temporal_radius)] + (self.all_data.ndim - 1) * [(0, 0)]
            if self.pad_mode == "constant_nan":
                self.video_data = [
                    np.pad(seq, pad_dim, mode="constant", constant_values=np.nan) for seq in self.video_data
                ]
            elif self.pad_mode == "edge":
                self.video_data = [np.pad(seq, pad_dim, mode="edge") for seq in self.video_data]
            else:
                raise ValueError(f"Invalid pad mode '{self.pad_mode}'")

    def compute_banmo_data(self):
        """Compute ground-truth data by calling into banmo"""
        # Should be overridden by subclass
        raise NotImplementedError

    def __getitem__(self, idx):
        if not self.enabled:
            return 0

        videoid = self.frame_videoid[idx]
        offset0 = self.frame_offsets[idx]
        offset1 = offset0 + 2 * self.temporal_radius + 1
        data = self.video_data[videoid][offset0:offset1] # T, C
        data = torch.from_numpy(data) # T, C
        return data


class FrameDataset(BaseDataset):
    """Template class for a dataset that returns frames from banmo DAVIS dataset

    Args
        dataset_label [str]: The type of frames returned by this dataset. Used in debug
            output, and to name the cachedir if applicable.
        img_dim [int]: Resize all loaded images to this size for storage
        use_cache [bool]: Whether to speed up dataloading by caching each glob as a single .npy file
        temporal_radius [int]: Size of temporal window in each direction
        invalid_date [float]: Invalidate any possibly stored cache
            if its last modified Unix timestamp is before this value
    """
    def __init__(
        self, banmo_seqname, videos, dataset_glob_type, dataset_label, *, img_dim=224, use_cache=True,
        n_data_workers=0, temporal_radius=0, invalid_date=0,
    ):
        super().__init__(banmo_seqname, videos, dataset_glob_type, n_data_workers=n_data_workers)
        self.dataset_label = dataset_label
        self.img_dim = img_dim
        self.use_cache = use_cache
        self.temporal_radius = temporal_radius

        # Load videos
        if n_data_workers == 0:
            self.all_videos = [None for i in range(self.nframes)] # nframes; C, H, W
            for i, paths in enumerate(self.video_impaths):
                offset0 = self.video_offsets[i]
                offset1 = self.video_offsets[i + 1]
                video = self.load_video(
                    paths, img_dim, use_cache, dataset_label, invalid_date
                )
                for j in range(offset0, offset1):
                    self.all_videos[j] = video[j - offset0]
        else:
            mp = multiprocessing.get_context("spawn")
            with mp.Pool(n_data_workers) as p:
                args = [
                    (paths, img_dim, use_cache, dataset_label, invalid_date)
                    for paths in self.video_impaths
                ]
                self.all_videos = p.starmap(self.load_video, tqdm.tqdm(args))

        self.all_videos = self.postprocess_all_videos(self.all_videos) # nframes; C, H, W

        # Split all frames by video
        self.videos = []
        for i in range(self.nvideos):
            offset0 = self.video_offsets[i]
            offset1 = self.video_offsets[i + 1]
            self.videos.append(self.all_videos[offset0:offset1])

        # Pad frames from each video
        if self.temporal_radius > 0:
            for i in range(len(self.videos)):
                left_vids = self.videos[i][:self.temporal_radius]
                right_vids = self.videos[i][-self.temporal_radius:][::-1]
                self.videos[i] = left_vids + self.videos[i] + right_vids

    @classmethod
    def load_video(cls, paths, img_dim, use_cache, dataset_label, invalid_date):
        rootdir = os.path.dirname(paths[0])
        cachedir = os.path.join(rootdir, f"{dataset_label}_{img_dim}_cache.npy")

        if use_cache:
            if os.path.exists(cachedir) and os.path.getmtime(cachedir) > invalid_date:
                # Preload images in directory from global numpy cache. Use mmap for faster load
                images = np.load(cachedir, mmap_mode="r")
                assert images.shape[0] == len(paths), \
                    f"Expected {len(paths)} images in {cachedir} but found {images.shape[0]}"
            else:
                # Load images from directory and cache to numpy
                print(f"Loading frames from {rootdir}")
                images = None
                for i, path in enumerate(paths):
                    frame = np.ascontiguousarray(cls.load_frame(path, img_dim))
                    if images is None:
                        images = np.empty((len(paths),) + frame.shape, dtype=np.float32) # N, C, H, W
                    images[i] = frame
                np.save(cachedir, images)
        else:
            global LOCAL_CACHE
            if cachedir in LOCAL_CACHE:
                # Preload images from directory to local cache
                images = LOCAL_CACHE[cachedir]
                assert images.shape[0] == len(paths), \
                    f"Expected {len(paths)} images in {cachedir} but found {images.shape[0]}"
            else:
                # Load images from directory and store in local cache
                print(f"Loading frames from {rootdir}")
                images = None
                for i, path in enumerate(paths):
                    frame = np.ascontiguousarray(cls.load_frame(path, img_dim))
                    if images is None:
                        images = np.empty((len(paths),) + frame.shape, dtype=np.float32) # N, C, H, W
                    images[i] = frame
                LOCAL_CACHE[cachedir] = images
         
        return images

    @classmethod
    def load_frame(cls, path, img_dim):
        # Should be overridden by subclass
        raise NotImplementedError

    @classmethod
    def postprocess_all_videos(cls, all_videos):
        # Optionally overridden by subclass
        return all_videos

    @property
    def video_data(self):
        return self.videos

    @property
    def all_data(self):
        return self.all_videos

    def __getitem__(self, idx):
        if not self.enabled:
            return 0

        videoid = self.frame_videoid[idx]
        offset0 = self.frame_offsets[idx]
        offset1 = offset0 + 2 * self.temporal_radius + 1
        frames = self.videos[videoid][offset0:offset1] # T; C, H, W
        frames = torch.from_numpy(np.array(frames)) # T, C, H, W
        return frames


# ===== Dataset Utilities

class TupleDataset(torch.utils.data.Dataset):
    """Returns a tuple of items sampled from multiple datasets.

    Args
        datasets [List(Dataset)]: List of datasets
        return_idx [bool]: Whether to return indices along with outputs.
            This dataset returns (idx, *datasets) if True and (*datasets) if False
    """
    def __init__(self, *datasets, return_idx=False):
        super().__init__()
        self.datasets = datasets
        self.return_idx = return_idx

        for i in range(1, len(datasets)):
            assert len(self.datasets[0]) == len(self.datasets[i]), \
                f"Dataset 0 and Dataset {i} have non-corresponding lengths. " \
                f"Dataset 0: {len(self.datasets[0])}, " \
                f"Dataset {i}: {len(self.datasets[i])}"

    @property
    def all_data(self):
        return tuple(dataset.all_data for dataset in self.datasets)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        if self.return_idx:
            return (idx,) + tuple(dataset.__getitem__(idx) for dataset in self.datasets)
        else:
            return tuple(dataset.__getitem__(idx) for dataset in self.datasets)


# ===== Ground Truth Datasets

class RootBodyPoseDataset(GroundTruthDataset):
    """Returns ground-truth SE(3) root body poses, evaluated using nerf_root_rts
    from saved banmo model

    Args
        centroid [banmo_nvid, 3]: Mesh centroid offset
    """
    def __init__(self, banmo_seqname, videos, centroid, *, use_cache=True, temporal_radius=0):
        self.centroid = centroid
        super().__init__(
            banmo_seqname, videos, "root_body_poses", pad_mode="constant_nan",
            use_cache=use_cache, temporal_radius=temporal_radius,
        )

    def compute_banmo_data(self):
        query_times = torch.arange(self.banmo_nframes, dtype=torch.int64, device=device) # nframes,
        vid = torch.tensor(self.banmo_frame_videoid, dtype=torch.int64, device=device) # nframes,
        
        centroid_vid = self.centroid[None].expand(self.banmo_nvideos, -1) # banmo_nvid, 3

        # Root body pose
        root_rts = banmo().nerf_root_rts(query_times)[:, 0] # nframes, 12
        root_rot = root_rts[:, :9].view(-1, 3, 3) # nframes, 3, 3
        root_tra = root_rts[:, 9:] # nframes, 3

        # Add per-video sim3_can transformation from SkelHead
        # Previously, P_view = G_root @ (G_se3 @ G_fk) @ P_canonical
        # We want P_view = (G_root @ G_se3) @ G_fk @ P_canonical
        if not isinstance(banmo().nerf_body_rts, torch.nn.Sequential):
            # sim3: [center 3, orient 4 (real-first quat), log_scale 3]
            # sim3_can: [urdf -> zero]; sim3_vid: [urdf -> time t]
            sim3_can = banmo().nerf_body_rts.sim3[None] # 1, 10
            sim3_vid = sim3_can + banmo().nerf_body_rts.sim3_vid # nvid, 10
            center_can, orient_can, scale_can = vec_to_sim3(sim3_can) # 1, 3 | 1, 3, 3 | 1, 3
            center_vid, orient_vid, scale_vid = vec_to_sim3(sim3_vid) # nvid, 3 | nvid, 3, 3 | nvid, 3

            # From banmo() nnutils/nerf.py, fk = G_sim3_t @ fk', and fk_z = G_sim3_z @ fk_z'
            # forward_decode() computes fk @ fk_z^{-1} = G_sim3_t @ fk' @ fk_z^{-1} @ G_sim3_z^{-1}
            # So, in forward_decode we can replace G_sim3_t with G_sim3_z, and here
            # we can apply G_sim3_t @ G_sim3_z^{-1} to root body pose
            # [rot_z, tra_z]: urdf-space to zero-space
            sim3_rot_z = orient_can # 1, 3, 3
            sim3_tra_z = center_can # 1, 3
            # [rot_zi, tra_zi]: zero-space to urdf-space
            sim3_rot_zi = torch.swapaxes(sim3_rot_z, -2, -1) # 1, 3, 3
            sim3_tra_zi = -torch.sum(sim3_rot_zi * sim3_tra_z[:, None, :], dim=-1) # 1, 3
            # [rot_t, tra_t]: urdf-space to video-space
            sim3_rot_t = orient_vid[vid] # nframes, 3, 3
            sim3_tra_t = center_vid[vid] # nframes, 3
            # [rot, tra]: zero-space to video-space
            sim3_rot = torch.matmul(sim3_rot_t, sim3_rot_zi) # nframes, 3, 3
            sim3_tra = torch.sum(sim3_rot_t * sim3_tra_zi[:, None, :], dim=-1) + sim3_tra_t # nframes, 3

            root_rot_ = torch.matmul(root_rot, sim3_rot) # nframes, 3, 3
            root_tra_ = torch.sum(root_rot * sim3_tra[:, None, :], dim=-1) + root_tra # nframes, 3
            root_rot, root_tra = root_rot_, root_tra_
        
        # Add centroid offset to counteract mesh centering
        root_tra = torch.sum(root_rot * centroid_vid[vid, None, :], dim=-1) + root_tra # nframes, 3

        # Add offset to Z-coordinate based on near-far plane
        # From BANMo's nnutis/banmo.py::create_base_se3()
        root_tra[:, -1] += 0.3
        
        # Final root body pose computation
        root_rts = torch.cat([root_rot.view(-1, 9), root_tra], dim=-1) # nframes, 12
        return root_rts


class BoneTransformDataset(GroundTruthDataset):
    """Returns ground-truth B*12-dim bone transforms, evaluated from saved banmo model"""
    def __init__(self, banmo_seqname, videos, bones_rst, bone_rts_rst, *, use_cache=True, temporal_radius=0):
        self.bones_rst = bones_rst
        self.bone_rts_rst = bone_rts_rst
        super().__init__(
            banmo_seqname, videos, "bone_rts", pad_mode="constant_nan",
            use_cache=use_cache, temporal_radius=temporal_radius,
        )

    def compute_banmo_data(self):
        query_times = torch.arange(self.banmo_nframes, dtype=torch.int64, device=device) # nframes,
        vid = torch.tensor(self.banmo_frame_videoid, dtype=torch.int64, device=device) # nframes,

        # Per-frame bone transforms without rest pose correction
        pose_codes = banmo().pose_code(query_times) # nframes, Cp
        T = pose_codes.shape[0]
        B = self.bones_rst.shape[0]

        if isinstance(banmo().nerf_body_rts, torch.nn.Sequential):
            # Old cat70 RTHead impl: nn.Sequential(idx -> pose_code, pose_code -> bone_rts)
            bone_rts_fw = banmo().nerf_body_rts[1](pose_codes).view(T, B, 12) # nframes, B, 12    
        else:
            # New dog80 SkelHead impl: .pose_code() and .forward_decode()
            bone_rts_fw, _ = banmo().nerf_body_rts.forward_decode(pose_codes, vid) # nframes, B*12
            bone_rts_fw = bone_rts_fw.view(T, B, 12) # nframes, B, 12

        # Per-frame bone transforms, with rest pose correction
        bone_rts_rst = self.bone_rts_rst[None, :, :].expand(T, -1, -1) # nframes, B, 12
        bone_rts_fw = zero_to_rest_dpose(bone_rts_fw, bone_rts_rst) # nframes, B, 12
        bone_rts_fw = bone_rts_fw.view(-1, B * 12) # nframes, B*12
        return bone_rts_fw


class JointAngleDataset(GroundTruthDataset):
    """Returns ground-truth (B-1)*3 bone transforms, evaluated from saved banmo model"""
    def __init__(self, banmo_seqname, videos, bones_rst, bone_rts_rst, *, use_cache=True, temporal_radius=0):
        self.bones_rst = bones_rst
        self.bone_rts_rst = bone_rts_rst
        super().__init__(
            banmo_seqname, videos, "joint_angle", pad_mode="constant_nan",
            use_cache=use_cache, temporal_radius=temporal_radius,
        )

    def compute_banmo_data(self):
        query_times = torch.arange(self.banmo_nframes, dtype=torch.int64, device=device) # nframes,
        vid = torch.tensor(self.banmo_frame_videoid, dtype=torch.int64, device=device) # nframes,

        pose_codes = banmo().pose_code(query_times) # nframes, Cp
        T = pose_codes.shape[0]
        B = self.bones_rst.shape[0]
        J = B - 1

        if isinstance(banmo().nerf_body_rts, torch.nn.Sequential):
            # Old cat70 RTHead impl: nn.Sequential(idx -> pose_code, pose_code -> bone_rts)
            joint_angles = torch.zeros(T, J * 3, dtype=torch.float32, device=device) # nframes, J*3
        else:
            # New dog80 SkelHead impl: .pose_code() and .forward_decode()
            _, joint_angles = banmo().nerf_body_rts.forward_decode(pose_codes, vid) # nframes, J*3

        joint_angles = joint_angles.view(-1, J, 3) # nframes, J, 3
        return joint_angles


class EnvCodeDataset(GroundTruthDataset):
    """Returns ground-truth environment codes, evaluated using EnvCode
    from saved banmo model
    """
    def __init__(self, banmo_seqname, videos, *, use_cache=True, temporal_radius=0):
        super().__init__(
            banmo_seqname, videos, "env_codes", pad_mode="constant_nan",
            use_cache=use_cache, temporal_radius=temporal_radius,
        )

    def compute_banmo_data(self):
        query_times = torch.arange(self.banmo_nframes, dtype=torch.int64, device=device) # nframes
        env_codes = banmo().env_code(query_times) # nframes, Ce
        return env_codes


class GroundTruthMeshDataset(GroundTruthDataset):
    """Returns deformed canonical mesh point clouds where we first apply the deformation
    specified by banmo ground-truth pose codes, then the ground-truth root body pose transformation

    Args
        bones_rst [B, 10]: Rest bones
        bone_rts_rst [B, 12]: Rest bone transforms
        pts_can [npts, 3]: Ground-truth mesh vertices to deform using warp_fw
        centroid [3,]: Mesh centroid offset
        memory_limit [int]: Maximum amount of memory to use for warp_fw
    """
    def __init__(
        self, banmo_seqname, videos, bones_rst, bone_rts_rst, pts_can, centroid, *,
        use_cache=True, temporal_radius=0, memory_limit=None
    ):
        self.bones_rst = bones_rst
        self.bone_rts_rst = bone_rts_rst
        self.pts_can = pts_can
        self.centroid = centroid
        self.memory_limit = memory_limit
        super().__init__(
            banmo_seqname, videos, "ground_truth_meshes", pad_mode="constant_nan",
            use_cache=use_cache, temporal_radius=temporal_radius,
        )
        
    def compute_banmo_data(self):
        query_times = torch.arange(self.banmo_nframes, dtype=torch.int64, device=device) # nframes
        vid = torch.tensor(self.banmo_frame_videoid, dtype=torch.int64, device=device) # nframes,
        
        pts_can_vid = self.pts_can[None].expand(self.banmo_nvideos, -1, -1) # banmo_nvid, npts, 3
        centroid_vid = self.centroid[None].expand(self.banmo_nvideos, -1) # banmo_nvid, 3
        
        # Root body pose
        root_rts = banmo().nerf_root_rts(query_times)[:, 0] # nframes, 12
        root_rot = root_rts[:, :9].view(-1, 3, 3) # nframes, 3, 3
        root_tra = root_rts[:, 9:] # nframes, 3

        # Add per-video sim3_can transformation from SkelHead
        # Previously, P_view = G_root @ (G_se3 @ G_fk) @ P_canonical
        # We want P_view = (G_root @ G_se3) @ G_fk @ P_canonical
        if not isinstance(banmo().nerf_body_rts, torch.nn.Sequential):
            # sim3: [center 3, orient 4 (real-first quat), log_scale 3]
            # sim3_can: [urdf -> zero]; sim3_vid: [urdf -> time t]
            sim3_can = banmo().nerf_body_rts.sim3[None] # 1, 10
            sim3_vid = sim3_can + banmo().nerf_body_rts.sim3_vid # nvid, 10
            center_can, orient_can, scale_can = vec_to_sim3(sim3_can) # 1, 3 | 1, 3, 3 | 1, 3
            center_vid, orient_vid, scale_vid = vec_to_sim3(sim3_vid) # nvid, 3 | nvid, 3, 3 | nvid, 3

            # From banmo() nnutils/nerf.py, fk = G_sim3_t @ fk', and fk_z = G_sim3_z @ fk_z'
            # forward_decode() computes fk @ fk_z^{-1} = G_sim3_t @ fk' @ fk_z^{-1} @ G_sim3_z^{-1}
            # So, in forward_decode we can replace G_sim3_t with G_sim3_z, and here
            # we can apply G_sim3_t @ G_sim3_z^{-1} to root body pose
            # [rot_z, tra_z]: urdf-space to zero-space
            sim3_rot_z = orient_can # 1, 3, 3
            sim3_tra_z = center_can # 1, 3
            # [rot_zi, tra_zi]: zero-space to urdf-space
            sim3_rot_zi = torch.swapaxes(sim3_rot_z, -2, -1) # 1, 3, 3
            sim3_tra_zi = -torch.sum(sim3_rot_zi * sim3_tra_z[:, None, :], dim=-1) # 1, 3
            # [rot_t, tra_t]: urdf-space to video-space
            sim3_rot_t = orient_vid[vid] # nframes, 3, 3
            sim3_tra_t = center_vid[vid] # nframes, 3
            # [rot, tra]: zero-space to video-space
            sim3_rot = torch.matmul(sim3_rot_t, sim3_rot_zi) # nframes, 3, 3
            sim3_tra = torch.sum(sim3_rot_t * sim3_tra_zi[:, None, :], dim=-1) + sim3_tra_t # nframes, 3

            root_rot_ = torch.matmul(root_rot, sim3_rot) # nframes, 3, 3
            root_tra_ = torch.sum(root_rot * sim3_tra[:, None, :], dim=-1) + root_tra # nframes, 3
            root_rot, root_tra = root_rot_, root_tra_
        
        # Add centroid offset to counteract mesh centering
        root_tra = torch.sum(root_rot * centroid_vid[vid, None, :], dim=-1) + root_tra # nframes, 3

        # Add offset to Z-coordinate based on near-far plane
        # From BANMo's nnutis/banmo.py::create_base_se3()
        root_tra[:, -1] += 0.3

        # Final root body pose
        root_rts = torch.cat([root_rot.view(-1, 9), root_tra], dim=-1) # nframes, 12

        # Per-frame bone transforms without rest pose correction
        pose_codes = banmo().pose_code(query_times) # nframes, Cp
        T = pose_codes.shape[0]
        B = self.bones_rst.shape[0]

        if isinstance(banmo().nerf_body_rts, torch.nn.Sequential):
            # Old cat70 RTHead impl: nn.Sequential(idx -> pose_code, pose_code -> bone_rts)
            bone_rts_fw = banmo().nerf_body_rts[1](pose_codes).view(T, B, 12) # nframes, B, 12
            blend_method = "axis_angle"
        else:
            # New dog80 SkelHead impl: .pose_code() and .forward_decode()
            bone_rts_fw, _ = banmo().nerf_body_rts.forward_decode(pose_codes, vid) # nframes, B*12
            bone_rts_fw = bone_rts_fw.view(T, B, 12) # nframes, B, 12
            blend_method = "dual_quat"

        # Per-frame bone transforms, with rest pose correction
        bone_rts_rst = self.bone_rts_rst[None, :, :].expand(T, -1, -1) # nframes, B, 12
        bone_rts_fw = zero_to_rest_dpose(bone_rts_fw, bone_rts_rst) # nframes, B, 12
        bone_rts_fw = bone_rts_fw.view(-1, B * 12) # nframes, B*12

        pts_dfm = warp_fw(
            banmo(), pts_can_vid[vid], self.bones_rst, self.bone_rts_rst, centroid_vid[vid],
            root_poses=root_rts, bone_rts_fw=bone_rts_fw,
            blend_method=blend_method, memory_limit=self.memory_limit, device="cpu"
        ) # nframes, N, 3
        return pts_dfm


class GroundTruthColorDataset(GroundTruthDataset):
    """Returns vertex colors for each point in deformed canonical mesh point cloud,
    computed using NeRF coarse model and environment codes. Color is computed by
    querying the original rest mesh, not per-video deformed mesh.

    Args
        pts_can [npts, 3]: Ground-truth mesh vertices to colorize
        centroid [3,]: Mesh centroid offset per video
        memory_limit [int]: Maximum amount of memory to use for get_vertex_colors
    """
    def __init__(
        self, banmo_seqname, videos, pts_can, centroid, *,
        use_cache=True, temporal_radius=0, memory_limit=None
    ):
        self.pts_can = pts_can
        self.centroid = centroid
        self.memory_limit = memory_limit
        super().__init__(
            banmo_seqname, videos, "ground_truth_colors", pad_mode="constant_nan",
            use_cache=use_cache, temporal_radius=temporal_radius,
        )

    def compute_banmo_data(self):
        query_times = torch.arange(0, self.banmo_nframes, dtype=torch.int64, device=device) # nframes,
        vid = torch.tensor(self.banmo_frame_videoid, dtype=torch.int64, device=device) # nframes,
        
        pts_can_vid = self.pts_can[None].expand(self.banmo_nvideos, -1, -1) # banmo_nvid, npts, 3
        centroid_vid = self.centroid[None].expand(self.banmo_nvideos, -1) # banmo_nvid, 3

        env_codes = banmo().env_code(query_times) # nframes, Ce
        colors = get_vertex_colors(
            banmo(), pts_can_vid[vid], centroid_vid[vid], env_codes,
            memory_limit=self.memory_limit, device="cpu"
        ) # nframes, N, 3
        return colors


class CameraIntrinsicsDataset(GroundTruthDataset):
    """Returns camera intrinsics loaded from banmo config file"""
    def __init__(self, banmo_seqname, videos, *, use_cache=True, temporal_radius=0):
        super().__init__(
            banmo_seqname, videos, "camera_ks", pad_mode="edge",
            use_cache=use_cache, temporal_radius=temporal_radius,
        )

    def compute_banmo_data(self):
        config_video_ks = camera_ks_from_banmo_config(self.banmo_seqname) # nvideos, 4
        banmo_video_ks = banmo().ks_param.cpu().numpy() # nvideos, 4
        video_ks = np.concatenate((banmo_video_ks, config_video_ks[:, 2:]), axis=-1) # nvideos, 6 [px py cx cy w h]
        camera_ks = torch.tensor(video_ks[self.banmo_frame_videoid], dtype=torch.float32) # nframes, 6
        return camera_ks


# ===== Frame Datasets

class DpfeatDataset(FrameDataset):
    """Returns pretrained cropped Densepose features loaded from .pfm, originally from banmo Densepose"""
    def __init__(
        self, banmo_seqname, videos, *, use_cache=True, n_data_workers=0, temporal_radius=0,
        invalid_date=0,
    ):
        super().__init__(
            banmo_seqname, videos, "dp_feat", "dp_feat",
            img_dim=112, use_cache=use_cache, n_data_workers=n_data_workers,
            temporal_radius=temporal_radius, invalid_date=invalid_date,
        )

    @classmethod
    def load_frame(cls, path, img_dim):
        dpfeat, _ = read_pfm(path) # 16*H, W
        dpfeat = dpfeat.reshape((16, img_dim, img_dim)) # 16, H, W

        # Normalize each 16-dim Densepose feature vector.
        # Each 16-dim feature should lie on the unit sphere
        eps = 1e-12 # Small value to avoid division by zero
        dpfeat /= np.maximum(np.linalg.norm(dpfeat, axis=0, keepdims=True), eps) # 16, H, W
        return dpfeat


class DpfeatQuadVisDataset(FrameDataset):
    """Returns visualizations of cropped Densepose features loaded from .pfm, for quadrupeds"""
    def __init__(
        self, banmo_seqname, videos, *, img_dim=112, use_cache=True, n_data_workers=0, temporal_radius=0,
        invalid_date=0,
    ):
        super().__init__(
            banmo_seqname, videos, "dp_feat", "dp_feat_quad_vis",
            img_dim=img_dim, use_cache=use_cache, n_data_workers=n_data_workers,
            temporal_radius=temporal_radius, invalid_date=invalid_date,
        )

    @classmethod
    def load_frame(cls, path, img_dim):
        dpfeat, _ = read_pfm(path)
        dpfeat = np.ascontiguousarray(dpfeat.reshape((16, img_dim, img_dim))) # 16, H, W

        # Normalize each 16-dim Densepose feature vector.
        # Each 16-dim feature should lie on the unit sphere
        eps = 1e-12 # Small value to avoid division by zero
        dpfeat /= np.maximum(np.linalg.norm(dpfeat, axis=0, keepdims=True), eps) # 16, H, W

        from vis_utils import vis_dpfeat
        dpfeat = vis_dpfeat(torch.from_numpy(dpfeat), is_human=False).numpy() # 3, H, W
        return dpfeat


class DpfeatHumanVisDataset(FrameDataset):
    """Returns visualizations of cropped Densepose features loaded from .pfm, for humans"""
    def __init__(
        self, banmo_seqname, videos, *, img_dim=112, use_cache=True, n_data_workers=0, temporal_radius=0,
        invalid_date=0,
    ):
        super().__init__(
            banmo_seqname, videos, "dp_feat", "dp_feat_human_vis",
            img_dim=img_dim, use_cache=use_cache, n_data_workers=n_data_workers,
            temporal_radius=temporal_radius, invalid_date=invalid_date,
        )

    @classmethod
    def load_frame(cls, path, img_dim):
        dpfeat, _ = read_pfm(path)
        dpfeat = dpfeat.reshape((16, img_dim, img_dim)) # 16, H, W

        # Normalize each 16-dim Densepose feature vector.
        # Each 16-dim feature should lie on the unit sphere
        eps = 1e-12 # Small value to avoid division by zero
        dpfeat /= np.maximum(np.linalg.norm(dpfeat, axis=0, keepdims=True), eps) # 16, H, W

        from vis_utils import vis_dpfeat
        dpfeat = vis_dpfeat(torch.from_numpy(dpfeat), is_human=True).numpy() # 3, H, W
        return dpfeat


class FullRGBDataset(FrameDataset):
    """Returns full RGB frames loaded from .jpg, originally from banmo dataset"""
    def __init__(
        self, banmo_seqname, videos, *, img_dim=224, use_cache=True, n_data_workers=0, temporal_radius=0,
        invalid_date=0,
    ):
        super().__init__(
            banmo_seqname, videos, "img", "rgb_full",
            img_dim=img_dim, use_cache=use_cache, n_data_workers=n_data_workers,
            temporal_radius=temporal_radius, invalid_date=invalid_date,
        )

    @classmethod
    def load_frame(cls, path, img_dim):
        image = read_img(path, resize=(img_dim, img_dim)) # H, W, 3
        image = np.transpose(image, (2, 0, 1)) # 3, H, W
        return image


class CropMaskRGBDataset(FrameDataset):
    """Returns cropped and masked RGB frames loaded from .jpg, originally from banmo dataset"""
    def __init__(
        self, banmo_seqname, videos, *, img_dim=224, use_cache=True, n_data_workers=0, temporal_radius=0,
        invalid_date=0,
    ):
        super().__init__(
            banmo_seqname, videos, "img", "rgb_crop_mask",
            img_dim=img_dim, use_cache=use_cache, n_data_workers=n_data_workers,
            temporal_radius=temporal_radius, invalid_date=invalid_date,
        )

    @classmethod
    def load_frame(cls, path, img_dim):
        bbox_root = os.path.dirname(path).replace("JPEGImages", "Densepose")
        bbox_path = os.path.join(bbox_root, f"bbox-{os.path.basename(path)}".replace(".jpg", ".txt"))
        # Convert to float to deal with scientific notation, then convert to int for indexing
        bbox = tuple(int(float(n)) for n in open(bbox_path, "r").readlines())
        
        mask_path = path.replace("JPEGImages", "Annotations").replace(".jpg", ".png")
        mask = read_img(mask_path, crop=bbox, resize=(img_dim, img_dim)) # H, W, 3
        image = read_img(path, crop=bbox, resize=(img_dim, img_dim)) # H, W, 3

        # Interpret RGB segmentation image as boolean mask
        mask = np.amax(mask, axis=-1, keepdims=True) # H, W, 1
        image = np.where(mask == 0, np.zeros_like(image), image) # H, W, 3
        image = np.transpose(image, (2, 0, 1)) # 3, H, W
        return image

import glob
import json
import numpy as np
import os
import random
import time
import torch
import tqdm
from collections import OrderedDict
from pytorch3d import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import resize
from sklearn.metrics import mean_squared_error

from banmo_utils import banmo
from data_utils import build_aug2d_pipeline, aug2d, alloc_empty, all_reduce_list
from dataset import (
    TupleDataset, CameraIntrinsicsDataset, RootBodyPoseDataset, BoneTransformDataset, JointAngleDataset,
    EnvCodeDataset, GroundTruthMeshDataset, GroundTruthColorDataset, DpfeatQuadVisDataset, DpfeatHumanVisDataset,
    DpfeatDataset, FullRGBDataset, CropMaskRGBDataset,
)
from geom_utils import bones_to_mesh, load_mesh, warp_fw, zero_to_rest_bone, get_vertex_colors
from models import BanmoFeedForward
from vis_utils import (
    vis_latent_codes_2d, vis_pose_distrs_2d, vis_pose_dfms_3d, vis_pose_dfms_mplex_3d, vis_bone_rts_3d,
    save_pose_dfms_mesh, save_pose_dfms_img, save_pose_dfms_mplex_img,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, opts):
        self.opts = opts

        # Fix random seeds
        os.environ["PYTHONHASHSEED"] = str(opts.random_seed)
        torch.manual_seed(opts.random_seed)
        torch.cuda.manual_seed(opts.random_seed)
        torch.cuda.manual_seed_all(opts.random_seed)
        np.random.seed(opts.random_seed)
        random.seed(opts.random_seed)

        # Dump parameters
        if opts.ddp_rank == 0:
            print(f"Logging results to '{opts.logdir}'")
        with open(f"{opts.logdir}/opts.json", "w") as f:
            f.write(json.dumps(opts.__dict__, default=repr, indent=4))
        
        # Model
        self.model = BanmoFeedForward(opts).to(device)

        if opts.summarize:
            from torchsummary import summary
            forward_func = self.model.forward
            T = 2 * opts.temporal_radius + 1
            M = opts.pose_multiplex_count

            self.model.forward = self.model.regressor_forward
            summary(self.model, [(19, 224, 224), (19, 224, 224)])
            self.model.forward = self.model.temporal_forward
            summary(self.model, [(T, M * opts.pose_feat_dim), (T, opts.pose_code_dim), (T, opts.env_code_dim)])
            self.model.forward = forward_func

        # Distributed data parallel
        if opts.use_ddp:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Dataset
        mesh_path = f"{opts.banmo_path}/mesh_rest_{opts.mesh_rest_res:03d}.obj"
        self.mesh_rest, self.centroid = load_mesh(mesh_path) # Trimesh | 3,
        self.pts_can = torch.tensor(self.mesh_rest.vertices, dtype=torch.float32, device=device) # npts, 3
        self.faces_can = torch.tensor(self.mesh_rest.faces, dtype=torch.int64, device=device) # nfaces, 3
        self.bones_rst, self.bone_rts_rst = zero_to_rest_bone(banmo(), banmo().bones) # B, 10 | B, 12

        self.mesh_bone = bones_to_mesh(self.bones_rst)  # Trimesh
        self.mesh_bone.vertices -= self.centroid.cpu().numpy()
        #self.mesh_bone = trimesh.util.concatenate([self.mesh_rest, self.mesh_bone])
        self.pts_bone = torch.tensor(self.mesh_bone.vertices, dtype=torch.float32, device=device) # npts_bone, 3
        self.rgb_bone = torch.tensor(
            self.mesh_bone.visual.vertex_colors[:, :3], dtype=torch.float32, device=device
        ) # npts_bone, 3
        self.faces_bone = torch.tensor(self.mesh_bone.faces, dtype=torch.int64, device=device) # nfaces_bone, 3
        
        if opts.train_videos.endswith(".json"):
            with open(f"configs/{opts.seqname}/{opts.train_videos}", "r") as train_videos_file:
                self.train_videos = json.load(train_videos_file)
        else:
            self.train_videos = opts.train_videos.split(",")

        if opts.eval_videos.endswith(".json"):
            with open(f"configs/{opts.seqname}/{opts.eval_videos}", "r") as eval_videos_file:
                self.eval_videos = json.load(eval_videos_file)
        else:
            self.eval_videos = opts.eval_videos.split(",")

        if opts.ddp_rank == 0:
            print("Train videos: ", self.train_videos)
            print("Eval videos: ", self.eval_videos)

        videos_in_both = set(self.train_videos).intersection(set(self.eval_videos))
        if opts.ddp_rank == 0 and len(videos_in_both) != 0:
            print("Warning: Duplicated videos between train and eval set: ", list(videos_in_both))

        if opts.use_aug2d:
            self.aug2d_pipeline_rgb_dpfeat = build_aug2d_pipeline(opts.aug2d_pipeline_rgb_dpfeat)
            self.aug2d_pipeline_rgb = build_aug2d_pipeline(opts.aug2d_pipeline_rgb)
            self.aug2d_pipeline_dpfeat = build_aug2d_pipeline(opts.aug2d_pipeline_dpfeat)
        
        def make_dataset(banmo_seqname, videos):
            # RGB and Densepose feature datasets
            common_opts = {
                "banmo_seqname": banmo_seqname,
                "videos": videos,
                "n_data_workers": opts.n_data_workers,
                "temporal_radius": 0,
                "use_cache": opts.use_cache_img,
            }

            rgb_dataset = CropMaskRGBDataset(**common_opts)
            dpfeat_dataset = DpfeatDataset(**common_opts)
            
            # RGB and Densepose visualization datasets
            common_opts = {
                "banmo_seqname": banmo_seqname,
                "videos": videos,
                "n_data_workers": opts.n_data_workers,
                "temporal_radius": opts.temporal_radius,
                "use_cache": opts.use_cache_img,
            }
            rgb_vis_dataset = FullRGBDataset(**common_opts)
            if "ama" in banmo_seqname:
                dpfeat_vis_dataset = DpfeatHumanVisDataset(**common_opts)
            else:
                dpfeat_vis_dataset = DpfeatQuadVisDataset(**common_opts)

            # Return individual dp_feats as CNN inputs and meshes, vertex colors,
            # root body poses, bone transforms, and environment codes as labels
            common_opts = {
                "banmo_seqname": banmo_seqname,
                "videos": videos,
                "temporal_radius": opts.temporal_radius,
                "use_cache": opts.use_cache_gt,
            }
            mesh_dataset = GroundTruthMeshDataset(
                bones_rst=self.bones_rst, bone_rts_rst=self.bone_rts_rst,
                pts_can=self.pts_can, centroid=self.centroid,
                memory_limit=opts.memory_limit, **common_opts
            )
            color_dataset = GroundTruthColorDataset(
                pts_can=self.pts_can, centroid=self.centroid,
                memory_limit=opts.memory_limit, **common_opts
            )
            pose_dataset = RootBodyPoseDataset(centroid=self.centroid, **common_opts)
            bone_dataset = BoneTransformDataset(
                bones_rst=self.bones_rst, bone_rts_rst=self.bone_rts_rst, **common_opts
            )
            angle_dataset = JointAngleDataset(
                bones_rst=self.bones_rst, bone_rts_rst=self.bone_rts_rst, **common_opts
            )
            env_dataset = EnvCodeDataset(**common_opts)
            ks_dataset = CameraIntrinsicsDataset(**common_opts)
            
            dataset = TupleDataset(
                rgb_dataset, dpfeat_dataset, ks_dataset, mesh_dataset, color_dataset,
                pose_dataset, bone_dataset, angle_dataset, env_dataset,
                return_idx=True,
            )
            dataset.rgb_vis_dataset = rgb_vis_dataset
            dataset.dpfeat_vis_dataset = dpfeat_vis_dataset

            return dataset

        self.train_dataset = make_dataset(opts.seqname, self.train_videos)
        self.eval_dataset = make_dataset(opts.seqname, self.eval_videos)

        self.n_train_frames = len(self.train_dataset)
        self.n_eval_frames = len(self.eval_dataset)

        # Sampler
        def make_sampler(dataset):
            if opts.use_ddp:
                return torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=opts.ddp_size,
                    rank=opts.ddp_rank,
                    shuffle=True,
                    seed=opts.random_seed,
                    drop_last=False,
                )
            else:
                return None

        self.train_sampler = make_sampler(self.train_dataset)
        self.eval_sampler = make_sampler(self.eval_dataset)

        # Dataloader
        def make_dataloader(dataset, sampler):
            return torch.utils.data.DataLoader(
                dataset,
                shuffle=(sampler is None),
                batch_size=opts.batch_size,
                drop_last=False,
                pin_memory=False,
                num_workers=0,
                sampler=sampler,
            )

        self.train_loader = make_dataloader(self.train_dataset, self.train_sampler)
        self.eval_loader = make_dataloader(self.eval_dataset, self.eval_sampler)

        # Optimizer
        def make_optimizer(params):
            if opts.optimizer == "SGD":
                return torch.optim.SGD(params, lr=opts.learning_rate, weight_decay=opts.weight_decay)
            elif opts.optimizer == "Adam":
                return torch.optim.Adam(params, lr=opts.learning_rate, weight_decay=opts.weight_decay)
            elif opts.optimizer == "AdamW":
                return torch.optim.AdamW(params, lr=opts.learning_rate, weight_decay=opts.weight_decay)
            else:
                raise RuntimeError(f"Invalid optimizer specification '{opts.optimizer}'")

        self.regressor_optimizer = make_optimizer(self.model.regressor_params)
        self.temporal_optimizer = make_optimizer(self.model.temporal_params)

        # Scheduler
        def make_scheduler(optimizer, scheduler_kwargs):
            if opts.scheduler is None or opts.scheduler == "ConstantLR":
                return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=0)
            elif opts.scheduler == "OneCycleLR":
                return torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=opts.learning_rate,
                    epochs=opts.n_epochs,
                    steps_per_epoch=(self.n_train_frames + opts.batch_size - 1) // opts.batch_size,
                    **scheduler_kwargs,
                )
            else:
                raise RuntimeError(f"Invalid scheduler specification '{opts.scheduler}'")

        scheduler_kwargs = json.loads(opts.scheduler_kwargs)
        self.regressor_scheduler = make_scheduler(self.regressor_optimizer, scheduler_kwargs)
        self.temporal_scheduler = make_scheduler(self.temporal_optimizer, scheduler_kwargs)

        # Load checkpoint
        if opts.load_params is not None:
            print(f"Loading saved model params from '{opts.load_params}'")
            # Store original OneCycleLR scheduler's last epoch and total steps
            if not opts.load_lr:
                if isinstance(self.regressor_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    regressor_scheduler_last_epoch = self.regressor_scheduler.last_epoch
                    regressor_scheduler_total_steps = self.regressor_scheduler.total_steps
                if isinstance(self.temporal_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    temporal_scheduler_last_epoch = self.temporal_scheduler.last_epoch
                    temporal_scheduler_total_steps = self.temporal_scheduler.total_steps

            if opts.load_params.endswith(".pth"):
                checkpoint = torch.load(opts.load_params, map_location=device)
            else:
                checkpoint = torch.load(f"{opts.load_params}/params/params_latest.pth", map_location=device)

            # Load checkpoint
            self.model.load_state_dict(checkpoint["model"])
            self.regressor_optimizer.load_state_dict(checkpoint["regressor_optimizer"])
            self.temporal_optimizer.load_state_dict(checkpoint["temporal_optimizer"])
            self.regressor_scheduler.load_state_dict(checkpoint["regressor_scheduler"])
            self.temporal_scheduler.load_state_dict(checkpoint["temporal_scheduler"])
            
            # Reset original OneCycleLR scheduler steps
            if not opts.load_lr:
                if isinstance(self.regressor_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.regressor_scheduler.last_epoch = regressor_scheduler_last_epoch
                    self.regressor_scheduler.total_steps = regressor_scheduler_total_steps
                if isinstance(self.temporal_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.temporal_scheduler.last_epoch = temporal_scheduler_last_epoch
                    self.temporal_scheduler.total_steps = temporal_scheduler_total_steps
           
        # Losses and logger
        def so3_log_loss(so3_log_pred, so3_log_target, eps=1e-4):
            """Compute geodesic loss between rotation matrices represented as SO3 log

            Args
                so3_log_pred [..., J, 3]: Predicted SO3 pose
                so3_log_target [..., J, 3]: Ground-truth SO3 pose
                eps [float]: Small value to prevent Infinity in acos

            Returns:
                rot_loss [...,]: Rotation geodesic loss
            """
            assert so3_log_pred.shape == so3_log_target.shape, \
                f"Expected shapes to match, but found {so3_log_pred.shape} and {so3_log_target.shape}"
            prefix_shape = so3_log_pred.shape[:-2]
            J = so3_log_pred.shape[-2]
            so3_log_pred = so3_log_pred.reshape(-1, J, 3) # -1, J, 3
            so3_log_target = so3_log_target.reshape(-1, J, 3) # -1, J, 3

            # Set loss to zero over padded values (denoted by nan in `so3_log_target`)
            so3_log_target = torch.where(torch.isnan(so3_log_target), so3_log_pred, so3_log_target) # -1, J, 3

            se3_log_pred = torch.cat([torch.zeros_like(so3_log_pred), so3_log_pred], dim=-1) # -1, J, 6
            se3_log_target = torch.cat([torch.zeros_like(so3_log_target), so3_log_target], dim=-1) # -1, J, 6
            rt_pred = transforms.se3_exp_map(se3_log_pred.view(-1, 6)).view(-1, J, 4, 4) # -1, J, 4, 4
            rt_target = transforms.se3_exp_map(se3_log_target.view(-1, 6)).view(-1, J, 4, 4) # -1, J, 4, 4

            rot_pred = rt_pred[:, :, :3, :3] # -1, J, 3, 3
            rot_target = rt_target[:, :, :3, :3] # -1, J, 3, 3
            rot = torch.matmul(rot_pred, rot_target.swapaxes(-1, -2)) # -1, J, 3, 3

            # Find rotation angle of rotation matrix
            cos = (rot[:, :, 0, 0] + rot[:, :, 1, 1] + rot[:, :, 2, 2] - 1) / 2 # -1, J
            cos = torch.clamp(cos, -1 + eps, 1 - eps) # -1, J
            rot_loss = torch.acos(cos) # -1, J

            tra_pred = rt_pred[:, :, :3, 3] # -1, J, 3
            tra_target = rt_target[:, :, :3, 3] # -1, J, 3
            tra_loss = torch.sum((tra_pred - tra_target) ** 2, dim=-1) # -1, J

            loss = torch.mean(rot_loss + tra_loss, dim=-1) # -1,
            loss = loss.view(prefix_shape) # ...,
            return loss


        def mse_loss(x_pred, x_target):
            """Compute pointwise L2 distances between two pointwise arrays.
            
            Args
                x_pred [..., N, 3]: Predicted points
                x_target [..., N, 3]: Ground-truth points

            Returns:
                mse_loss [...]: L2 loss between points
            """
            assert x_pred.shape == x_target.shape, \
                f"Expected shapes to match, but found {x_pred.shape} and {x_target.shape}"
            # Set target to prediction over padded values (denoted by NaN) for zero loss
            x_target = torch.where(torch.isnan(x_target), x_pred, x_target) # ..., N, 3

            dist_sq = torch.sum((x_pred - x_target) ** 2, axis=-1) # ..., N
            dist_sq = torch.mean(dist_sq, axis=-1) # ...
            return dist_sq

            
        self.pose_dfm_loss = mse_loss
        self.texture_loss = mse_loss
        self.angle_loss = so3_log_loss

        # Logging and output directory on rank 0
        if opts.ddp_rank == 0:
            os.makedirs(f"{opts.logdir}/tensorboard", exist_ok=True)
            self.log = SummaryWriter(f"{opts.logdir}/tensorboard")

            os.makedirs(f"{opts.logdir}/params", exist_ok=True)

            # Metrics - for easy flush to disk per epoch, we use a mmap-ed file per metric
            def open_mmap(filename, shape):
                """Create an mmap'ed array as a .npy file and return it"""
                np.save(filename, np.full(shape, np.nan))
                return np.load(filename, mmap_mode="r+")

            n_train_iters = (self.n_train_frames + opts.batch_size - 1) // opts.batch_size
            n_train_iters = ((n_train_iters + opts.ddp_size - 1) // opts.ddp_size) * opts.ddp_size
            n_eval_iters = (self.n_eval_frames + opts.batch_size - 1) // opts.batch_size
            n_eval_iters = ((n_eval_iters + opts.ddp_size - 1) // opts.ddp_size) * opts.ddp_size
            
            # Per-epoch metrics
            self.epoch_metrics = [
                "loss", "pose_dfm_loss", "texture_loss",
                "regr_loss", "regr_pose_dfm_loss", "regr_texture_loss",
                "var", "pose_acc", "bone_acc", "angle_acc", "env_acc",
                "pts_pos_err", "pts_vel_err", "pts_acc_err",
                "rgb_pos_err", "rgb_vel_err", "rgb_acc_err",
            ]
            self.train_epoch_metrics = {k: float("inf") for k in self.epoch_metrics}
            self.eval_epoch_metrics = {k: float("inf") for k in self.epoch_metrics}
            self.train_epoch_metric_files = {
                k: open_mmap(os.path.join(opts.logdir, f"train_epoch_{k}.npy"), (opts.n_epochs,))
                for k in self.epoch_metrics
            }
            self.eval_epoch_metric_files = {
                k: open_mmap(os.path.join(opts.logdir, f"eval_epoch_{k}.npy"), (opts.n_epochs,))
                for k in self.epoch_metrics
            }

            # Per-iter metrics
            self.iter_metrics = [
                "loss", "pose_dfm_loss", "texture_loss",
                "regr_loss", "regr_pose_dfm_loss", "regr_texture_loss",
            ]
            self.train_iter_metrics = {k: float("inf") for k in self.iter_metrics}
            self.eval_iter_metrics = {k: float("inf") for k in self.iter_metrics}
            self.train_iter_metric_files = {
                k: open_mmap(os.path.join(opts.logdir, f"train_iter_{k}.npy"), (opts.n_epochs * n_train_iters,))
                for k in self.iter_metrics
            }
            self.eval_iter_metric_files = {
                k: open_mmap(os.path.join(opts.logdir, f"eval_iter_{k}.npy"), (opts.n_epochs * n_eval_iters,))
                for k in self.iter_metrics
            }
        

    def run_one_epoch(self, epoch, is_train=False):
        opts = self.opts

        if is_train:
            self.model.train()
            dataloader = self.train_loader
            gradient_context_manager = torch.enable_grad
        else:
            self.model.eval()
            dataloader = self.eval_loader
            gradient_context_manager = torch.no_grad
                    
        # Gather input tensors
        all_rgb_vis = dataloader.dataset.rgb_vis_dataset.all_data
        all_dpfeat_vis = dataloader.dataset.dpfeat_vis_dataset.all_data
        (
            all_rgb_imgs, all_dp_feats, all_camera_ks, all_actual_pts, all_actual_rgb,
            all_gt_poses, all_gt_bones, all_gt_angles, all_gt_envs,
        ) = dataloader.dataset.all_data

        # Allocate output tensors
        N = len(dataloader.dataset)
        M = opts.pose_multiplex_count
        B = opts.n_bones
        J = opts.n_bones - 1
        n_iters_ddp = (N + opts.ddp_size - 1) // opts.ddp_size # split by device, then split by batch
        n_iters_ddp = (n_iters_ddp + opts.batch_size - 1) // opts.batch_size # equal to ceil(N / opts.batch_size)
        n_iters = n_iters_ddp * opts.ddp_size # round up to multiple of ddp_size
        npts = self.pts_can.shape[0]
        npts_bone = self.pts_bone.shape[0]
        npts_samp = min(npts, opts.mesh_loss_npts_samp)

        item_loss = alloc_empty((N,))
        iter_loss = alloc_empty((opts.ddp_size, n_iters_ddp))
        pose_dfm_iter_loss = alloc_empty((opts.ddp_size, n_iters_ddp))
        texture_iter_loss = alloc_empty((opts.ddp_size, n_iters_ddp))
        angle_iter_loss = alloc_empty((opts.ddp_size, n_iters_ddp))
        regr_iter_loss = alloc_empty((opts.ddp_size, n_iters_ddp))
        regr_pose_dfm_iter_loss = alloc_empty((opts.ddp_size, n_iters_ddp))
        regr_texture_iter_loss = alloc_empty((opts.ddp_size, n_iters_ddp))
        regr_angle_iter_loss = alloc_empty((opts.ddp_size, n_iters_ddp))
        
        # Regressor output: Root body pose feats, pose codes, and texture codes
        all_raw_pose_feats = alloc_empty((N, M * opts.pose_feat_dim))
        all_raw_code_feats = alloc_empty(
            (N, (opts.n_bones - 1) * 3 if opts.pose_code_dim == 0 else opts.pose_code_dim)
        )
        all_raw_env_feats = alloc_empty((N, opts.env_code_dim))

        # Regressor and temporal encoder output: Root body poses, bone transforms, and texture codes
        all_pred_poses_mplex = alloc_empty((N, M, 12))
        all_pred_probs_mplex = alloc_empty((N, M))
        all_pred_poses = alloc_empty((N, 12))
        all_pred_bones = alloc_empty((N, B * 12))
        all_pred_angles = alloc_empty((N, J, 3))
        all_pred_envs = alloc_empty((N, opts.env_code_dim))

        # Predicted vertex locations and colors
        all_pred_pts = alloc_empty((N, npts, 3), "cpu")
        all_pred_rgb = alloc_empty((N, npts, 3), "cpu")
        all_bone_pts = alloc_empty((N, npts_bone, 3), "cpu")
        all_bone_rgb = alloc_empty((N, npts_bone, 3), "cpu")

        start_time = time.time()
        with gradient_context_manager():
            # ===== Stage 1: Train the CNN regressor backbone frame-by-frame
            dataloader.dataset.datasets[0].enabled = True # Enable rgb_imgs dataset
            dataloader.dataset.datasets[1].enabled = True # Enable dp_feats dataset
            if opts.use_ddp:
                dataloader.sampler.set_epoch(2 * epoch)
            for i, batch in enumerate(dataloader):
                (
                    idx_, rgb_imgs_, dp_feats_, camera_ks_, pts_dfms_, pts_rgbs_,
                    gt_poses_, gt_bones_, gt_angles_, gt_envs_,
                ) = batch
                
                # Only consider the current timestep
                idx = idx_.to(device) # bs
                T = pts_dfms_.shape[1]
                rgb_imgs = rgb_imgs_[:, 0].to(device) # bs, 3, H_, W_
                dp_feats = dp_feats_[:, 0].to(device) # bs, 16, H_, W_
                camera_ks = camera_ks_[:, T//2].to(device) # bs, 6
                pts_dfms = pts_dfms_[:, T//2].to(device) # bs, npts, 3
                pts_rgbs = pts_rgbs_[:, T//2].to(device) # bs, npts, 3
                gt_poses = gt_poses_[:, T//2].to(device) # bs, 12
                gt_bones = gt_bones_[:, T//2].to(device) # bs, B*12
                gt_angles = gt_angles_[:, T//2].to(device) # bs, J, 3
                gt_envs = gt_envs_[:, T//2].to(device) # bs, Ce
                del rgb_imgs_, dp_feats_, camera_ks_, pts_dfms_, pts_rgbs_
                del gt_poses_, gt_bones_, gt_angles_, gt_envs_

                if opts.log_data:
                    print(f"Logging data before augmentations...")
                    from data_utils import write_img
                    from vis_utils import vis_dpfeat
                    dpfeat_vis = vis_dpfeat(dp_feats, "ama" in opts.banmo_path) # bs, 3, H, W
                    os.makedirs(f"{opts.logdir}/data_preaug", exist_ok=True)

                    for bb in range(len(idx)):
                        rgb = rgb_imgs[bb].moveaxis(0, -1).cpu().numpy() # H, W, 3
                        write_img(f"{opts.logdir}/data/rgb_preaug_img{bb:03d}.png", rgb); del rgb

                        dpf = dpfeat_vis[bb].moveaxis(0, -1).cpu().numpy() # H, W, 3
                        write_img(f"{opts.logdir}/data/dpf_preaug_img{bb:03d}.png", dpf); del dpf

                    del dpfeat_vis

                # Perform 2D augmentation on entire batch (faster than one at a time within dataloader)
                if opts.use_aug2d:
                    rgb_imgs = resize(rgb_imgs, (224, 224)) # bs, 3, H, W
                    dp_feats = resize(dp_feats, (224, 224)) # bs, 16, H, W
                    rgb_dpfeat = torch.cat([rgb_imgs, dp_feats], dim=1) # bs, 19, H, W

                    rgb_dpfeat = aug2d(rgb_dpfeat, self.aug2d_pipeline_rgb_dpfeat) # bs, 19, H, W
                    rgb_imgs = rgb_dpfeat[:, :3, :, :] # bs, 3, H, W
                    dp_feats = rgb_dpfeat[:, 3:, :, :] # bs, 16, H, W
                    del rgb_dpfeat

                    rgb_imgs = aug2d(rgb_imgs, self.aug2d_pipeline_rgb) # bs, 3, H, W
                    dp_feats = aug2d(dp_feats, self.aug2d_pipeline_dpfeat) # bs, 16, H, W

                if opts.log_data:
                    print(f"Logging data after augmentations...")
                    from data_utils import write_img
                    from vis_utils import vis_dpfeat
                    dpfeat_vis = vis_dpfeat(dp_feats, "ama" in opts.banmo_path) # bs, 3, H, W
                    os.makedirs(f"{opts.logdir}/data_postaug", exist_ok=True)

                    for bb in range(len(idx)):
                        rgb = rgb_imgs[bb].moveaxis(0, -1).cpu().numpy() # H, W, 3
                        write_img(f"{opts.logdir}/data/rgb_postaug_img{bb:03d}.png", rgb); del rgb

                        dpf = dpfeat_vis[bb].moveaxis(0, -1).cpu().numpy() # H, W, 3
                        write_img(f"{opts.logdir}/data/dpf_postaug_img{bb:03d}.png", dpf); del dpf

                    del dpfeat_vis

                # Extract features
                (out_pose_mplex, out_prob_mplex, out_pose_feat, out_pose_code, out_env) = (
                    self.model.regressor_forward(rgb_imgs, dp_feats)
                ) # bs, M, 12 | bs, M | bs, Cf | bs, Cp | bs, Ce
                bs, Cf = out_pose_feat.shape
                bs, Cp = out_pose_code.shape
                del rgb_imgs, dp_feats

                # Decode pose feats into bone transforms
                out_bone, out_angle = self.model.bone_transform_decoder(out_pose_code) # bs, B*12
                out_bone_mplex = out_bone[:, None].expand(-1, M, -1) # bs, M, B*12
                out_angle_mplex = out_angle[:, None].expand(-1, M, -1, -1) # bs, M, J, 3
                camera_ks_mplex = camera_ks[:, None].expand(-1, M, -1) # bs, M, 6

                gt_poses_mplex_tra = gt_poses[:, None, 9:].expand(-1, M, -1) # bs, M, 3
                out_pose_mplex[:, :, 9:] = torch.where(
                    gt_poses_mplex_tra.isnan(),
                    out_pose_mplex[:, :, 9:],
                    1e-6 * out_pose_mplex[:, :, 9:] + (1 - 1e-6) * gt_poses_mplex_tra,
                ); del gt_poses_mplex_tra # bs, M, 3
                del gt_poses, gt_bones, gt_envs
                
                bs, M, Cr = out_pose_mplex.shape
                bs, M, Cb = out_bone_mplex.shape
                bs, M, Ca, _ = out_angle_mplex.shape
                bs, M, Ck = camera_ks_mplex.shape
                bs, Ce = out_env.shape

                # Use predicted multiplexed root poses and bone transforms to warp mesh forward,
                # outputting M deformed mesh hypotheses.
                mesh_idxs = torch.multinomial(
                    torch.ones(npts, dtype=torch.float32, device=device), npts_samp, replacement=False
                ) # npts_samp

                # Perform forward warping and compute textures
                pts_pred_mplex = warp_fw(
                    banmo(), self.pts_can[mesh_idxs], self.bones_rst, self.bone_rts_rst, self.centroid,
                    out_pose_mplex, out_bone_mplex,
                    blend_method=opts.blend_method, memory_limit=opts.memory_limit,
                ) # bs, M, npts_samp, 3
                pts_target_mplex = pts_dfms[:, None, mesh_idxs, :].expand(-1, M, -1, -1) # bs, M, npts_samp, 3
                del pts_dfms

                rgb_pred_mplex = get_vertex_colors(
                    banmo(), self.pts_can[mesh_idxs], self.centroid, out_env, memory_limit=opts.memory_limit
                )[:, None, :, :].expand(-1, M, -1, -1) # bs, M, npts_samp, 3
                rgb_target_mplex = pts_rgbs[:, None, mesh_idxs, :].expand(-1, M, -1, -1) # bs, M, npts_samp, 3
                del pts_rgbs

                # Compute angles
                angle_pred_mplex = out_angle_mplex # bs, M, J, 3
                angle_target_mplex = gt_angles[:, None, :, :].expand(-1, M, -1, -1) # bs, M, J, 3

                # Use multiple choice loss for poses
                if opts.pose_multiple_choice_loss:
                    mplex_max_mask = (
                        out_prob_mplex == torch.amax(out_prob_mplex, dim=-1, keepdim=True)
                    )[:, :, None, None] # bs, M, 1, 1
                    pts_pred_mplex = (
                        mplex_max_mask * pts_pred_mplex + (~mplex_max_mask) * pts_pred_mplex.detach()
                    ) # bs, M, npts_samp, 3
                    rgb_pred_mplex = (
                        mplex_max_mask * rgb_pred_mplex + (~mplex_max_mask) * rgb_pred_mplex.detach()
                    ) # bs, M, npts_samp, 3

                # Apply loss
                pose_dfm_loss_mplex = self.pose_dfm_loss(pts_pred_mplex, pts_target_mplex) # bs, M
                pose_dfm_loss = self.model.weight_mplex(pose_dfm_loss_mplex, out_prob_mplex) # bs,
                pose_dfm_loss = pose_dfm_loss * opts.pose_dfm_loss_scale # bs,

                texture_loss_mplex = self.texture_loss(rgb_pred_mplex, rgb_target_mplex) # bs, M
                texture_loss = self.model.weight_mplex(texture_loss_mplex, out_prob_mplex) # bs,
                texture_loss = texture_loss * opts.texture_loss_scale # bs,

                angle_loss_mplex = self.angle_loss(angle_pred_mplex, angle_target_mplex) # bs, M
                angle_loss = self.model.weight_mplex(angle_loss_mplex, out_prob_mplex) # bs,
                angle_loss = angle_loss * opts.angle_loss_scale # bs,
                
                loss = pose_dfm_loss + texture_loss + angle_loss # bs,
                del pts_target_mplex, rgb_target_mplex, angle_target_mplex

                if is_train:
                    (opts.loss_scale * loss.mean()).backward()
                    if opts.use_ddp and opts.ddp_size > 1:
                        for k, v in self.model.named_parameters():
                            if opts.sync_grad and isinstance(v.grad, torch.Tensor):
                                torch.distributed.all_reduce(v.grad)
                                v.grad /= opts.ddp_size
                    self.regressor_optimizer.step()
                    self.regressor_scheduler.step()
                    self.regressor_optimizer.zero_grad(set_to_none=True)
            
                loss = loss.detach()
                pose_dfm_loss = pose_dfm_loss.detach()
                texture_loss = texture_loss.detach()
                angle_loss = angle_loss.detach()

                regr_iter_loss[opts.ddp_rank, i] = loss.mean()
                regr_pose_dfm_iter_loss[opts.ddp_rank, i] = pose_dfm_loss.mean()
                regr_texture_iter_loss[opts.ddp_rank, i] = texture_loss.mean()
                regr_angle_iter_loss[opts.ddp_rank, i] = angle_loss.mean()
                del loss, pose_dfm_loss, texture_loss, angle_loss

                # Save outputs, enforcing that outputs are ordered in time
                out_pose_feat = out_pose_feat.detach()
                out_pose_code = out_pose_code.detach()
                out_env = out_env.detach()

                all_pred_poses_mplex.scatter_(0, idx[:, None, None].expand(-1, M, 12), out_pose_mplex)
                all_pred_probs_mplex.scatter_(0, idx[:, None].expand(-1, M), out_prob_mplex)
                all_raw_pose_feats.scatter_(0, idx[:, None].expand(-1, Cf), out_pose_feat)
                all_raw_code_feats.scatter_(0, idx[:, None].expand(-1, Cp), out_pose_code)
                all_raw_env_feats.scatter_(0, idx[:, None].expand(-1, Ce), out_env)
                del idx, out_pose_feat, out_pose_code, out_env, out_prob_mplex, pts_pred_mplex, rgb_pred_mplex

            if opts.use_ddp:
                all_reduce_list(
                    all_pred_poses_mplex, all_pred_probs_mplex,
                    all_raw_pose_feats, all_raw_code_feats, all_raw_env_feats
                )
                

            # ===== Stage 2: Compute root body pose feats, pose codes, texture codes per video with padding
            raw_pose_feats = []
            raw_code_feats = []
            raw_env_feats = []

            nvideos = dataloader.dataset.datasets[0].nvideos
            for i in range(nvideos):
                offset0 = dataloader.dataset.datasets[0].video_offsets[i]
                offset1 = dataloader.dataset.datasets[0].video_offsets[i + 1]
                raw_pose_feats.append(all_raw_pose_feats[offset0:offset1])
                raw_code_feats.append(all_raw_code_feats[offset0:offset1])
                raw_env_feats.append(all_raw_env_feats[offset0:offset1])

            # Feats at the end from 3D augmented data
            offset_end = dataloader.dataset.datasets[0].video_offsets[-1]
            raw_pose_feats.append(all_raw_pose_feats[offset_end:])
            raw_code_feats.append(all_raw_code_feats[offset_end:])
            raw_env_feats.append(all_raw_env_feats[offset_end:])

            del all_raw_pose_feats, all_raw_code_feats, all_raw_env_feats

            R = opts.temporal_radius
            if R > 0:
                pad_dim = (raw_pose_feats[0].ndim - 1) * (0, 0) + 1 * (R, R)
                raw_pose_feats = [torch.nn.functional.pad(seq, pad_dim) for seq in raw_pose_feats]

                pad_dim = (raw_code_feats[0].ndim - 1) * (0, 0) + 1 * (R, R)
                raw_code_feats = [torch.nn.functional.pad(seq, pad_dim) for seq in raw_code_feats]
                
                pad_dim = (raw_env_feats[0].ndim - 1) * (0, 0) + 1 * (R, R)
                raw_env_feats = [torch.nn.functional.pad(seq, pad_dim) for seq in raw_env_feats]


            # ===== Stage 3: Pass precomputed per-image CNN features into temporal encoder
            dataloader.dataset.datasets[0].enabled = False # Disable rgb_imgs dataset
            dataloader.dataset.datasets[1].enabled = False # Disable dp_feats dataset
            if opts.use_ddp:
                dataloader.sampler.set_epoch(2 * epoch + 1)
            for i, batch in enumerate(dataloader):
                idx_, _, _, camera_ks_, pts_dfms_, pts_rgbs_, gt_poses_, gt_bones_, gt_angles_, gt_envs_ = batch
                
                idx = idx_.to(device) # bs
                camera_ks = camera_ks_.to(device) # bs, T, 6
                pts_dfms = pts_dfms_.to(device) # bs, T, npts, 3
                pts_rgbs = pts_rgbs_.to(device) # bs, T, npts, 3
                gt_poses = gt_poses_.to(device) # bs, T, 12
                gt_bones = gt_bones_.to(device) # bs, T, B*12
                gt_angles = gt_angles_.to(device) # bs, T, J, 3
                gt_envs = gt_envs_.to(device) # bs, T, Ce
                del camera_ks_, pts_dfms_, pts_rgbs_, gt_poses_, gt_bones_, gt_angles_, gt_envs_

                # Collect a batch of precomputed pose/code features, and rgb_vis frames
                raw_pose_feats_batch = []
                raw_code_feats_batch = []
                raw_env_feats_batch = []

                for getitem_idx in idx_:
                    nframes = dataloader.dataset.datasets[0].nframes

                    if getitem_idx >= nframes:
                        videoid = -1
                        offset0 = getitem_idx - nframes
                    else:
                        videoid = dataloader.dataset.datasets[0].frame_videoid[getitem_idx]
                        offset0 = dataloader.dataset.datasets[0].frame_offsets[getitem_idx]
                    offset1 = offset0 + 2 * opts.temporal_radius + 1

                    raw_pose_feat_elt = raw_pose_feats[videoid][offset0:offset1] # T, M, Cf
                    raw_code_feat_elt = raw_code_feats[videoid][offset0:offset1] # T, Cp
                    raw_env_feat_elt = raw_env_feats[videoid][offset0:offset1] # T, Ce
                    rgb_vis_elt = dataloader.dataset.rgb_vis_dataset.videos[videoid][offset0:offset1] # T; 3, H, W
                    
                    raw_pose_feats_batch.append(raw_pose_feat_elt)
                    raw_code_feats_batch.append(raw_code_feat_elt)
                    raw_env_feats_batch.append(raw_env_feat_elt)

                del raw_pose_feat_elt, raw_code_feat_elt, raw_env_feat_elt
                raw_pose_feats_batch = torch.stack(raw_pose_feats_batch, dim=0).detach().to(device) # bs, T, M, Cf
                raw_code_feats_batch = torch.stack(raw_code_feats_batch, dim=0).detach().to(device) # bs, T, Cp
                raw_env_feats_batch = torch.stack(raw_env_feats_batch, dim=0).detach().to(device) # bs, T, Ce

                # Apply temporal encoder
                out_pose, out_bone, out_angle, out_env = self.model.temporal_forward(
                    raw_pose_feats_batch, raw_code_feats_batch, raw_env_feats_batch
                ) # bs, T, 12 | bs, T, B*12 | bs, T, J, 3 | bs, T, Ce

                out_pose[..., 9:] = torch.where(
                    gt_poses[..., 9:].isnan(),
                    out_pose[..., 9:],
                    1e-6 * out_pose[..., 9:] + (1 - 1e-6) * gt_poses[..., 9:]
                ) # bs, T, 3

                bs, T, Cr = out_pose.shape
                bs, T, Cb = out_bone.shape
                bs, T, Ca, _ = out_angle.shape
                bs, T, Ce = out_env.shape
                bs, T, Ck = camera_ks.shape
                del raw_pose_feats_batch, raw_code_feats_batch, raw_env_feats_batch

                # Use predicted codes to warp mesh forward and predict vertex colors
                # To reduce computation, we randomly subsample a portion of the mesh
                mesh_idxs = torch.multinomial(
                    torch.ones(npts, dtype=torch.float32, device=device), npts_samp, replacement=False
                ) # npts_samp

                # Perform forward warping and compute texture
                pts_pred = warp_fw(
                    banmo(), self.pts_can[mesh_idxs], self.bones_rst, self.bone_rts_rst, self.centroid,
                    out_pose, out_bone,
                    blend_method=opts.blend_method, memory_limit=opts.memory_limit,
                ) # bs, T, npts_samp, 3
                pts_target = pts_dfms[:, :, mesh_idxs, :] # bs, T, npts_samp, 3
                del pts_dfms
            
                rgb_pred = get_vertex_colors(
                    banmo(), self.pts_can[mesh_idxs], self.centroid, out_env.reshape(bs * T, Ce),
                    memory_limit=opts.memory_limit
                ).view(bs, T, npts_samp, 3) # bs, T, npts_samp, 3
                rgb_target = pts_rgbs[:, :, mesh_idxs, :] # bs, T, npts_samp, 3
                del pts_rgbs

                angle_pred = out_angle # bs, T, J, 3
                angle_target = gt_angles # bs, T, J, 3

                # Apply loss
                pose_dfm_loss = opts.pose_dfm_loss_scale * self.pose_dfm_loss(pts_pred, pts_target) # bs[, T]
                texture_loss = opts.texture_loss_scale * self.texture_loss(rgb_pred, rgb_target) # bs[, T]
                angle_loss = opts.angle_loss_scale * self.angle_loss(angle_pred, angle_target) # bs[, T]

                loss = pose_dfm_loss + texture_loss + angle_loss # bs[, T]

                if is_train:
                    (opts.loss_scale * loss.mean()).backward()
                    if opts.use_ddp and opts.ddp_size > 1:
                        for k, v in self.model.named_parameters():
                            if opts.sync_grad and isinstance(v.grad, torch.Tensor):
                                torch.distributed.all_reduce(v.grad)
                                v.grad /= opts.ddp_size
                    self.temporal_optimizer.step()
                    self.temporal_scheduler.step()
                    self.temporal_optimizer.zero_grad(set_to_none=True)

                loss = loss.detach()
                pose_dfm_loss = pose_dfm_loss.detach()
                texture_loss = texture_loss.detach()
                angle_loss = angle_loss.detach()

                loss = loss[:, T//2].detach() # bs,
                item_loss.scatter_(0, idx, loss)

                iter_loss[opts.ddp_rank, i] = loss.mean()
                pose_dfm_iter_loss[opts.ddp_rank, i] = pose_dfm_loss.mean()
                texture_iter_loss[opts.ddp_rank, i] = texture_loss.mean()
                angle_iter_loss[opts.ddp_rank, i] = angle_loss.mean()
                del loss, pose_dfm_loss, texture_loss

                # Save outputs, enforcing that outputs are ordered in time
                out_pose = out_pose[:, T//2] # bs, 12
                out_bone = out_bone[:, T//2] # bs, B*12
                out_angle = out_angle[:, T//2] # bs, J, 3
                out_env = out_env[:, T//2] # bs, Ce
                camera_ks = camera_ks[:, T//2] # bs, 6

                out_pose = out_pose.detach()
                out_bone = out_bone.detach()
                out_angle = out_angle.detach()
                out_env = out_env.detach()
                camera_ks = camera_ks.detach()

                with torch.no_grad():
                    pts_pred_all = warp_fw(
                        banmo(), self.pts_can, self.bones_rst, self.bone_rts_rst, self.centroid,
                        out_pose, out_bone,
                        blend_method=opts.blend_method, memory_limit=opts.memory_limit,
                    ).detach().cpu() # bs, npts, 3
                    pts_bone_all = warp_fw(
                        banmo(), self.pts_bone, self.bones_rst, self.bone_rts_rst, self.centroid,
                        out_pose, out_bone,
                        blend_method=opts.blend_method, memory_limit=opts.memory_limit,
                    ).detach().cpu() # bs, npts, 3

                    rgb_pred_all = get_vertex_colors(
                        banmo(), self.pts_can, self.centroid, out_env, memory_limit=opts.memory_limit
                    ).detach().cpu() # bs, npts, 3
                    rgb_bone_all = self.rgb_bone[None].expand(bs, -1, -1).detach().cpu() # bs, npts, 3

                all_pred_poses.scatter_(0, idx[:, None].expand(-1, Cr), out_pose)
                all_pred_bones.scatter_(0, idx[:, None].expand(-1, Cb), out_bone)
                all_pred_angles.scatter_(0, idx[:, None, None].expand(-1, Ca, 3), out_angle)
                all_pred_envs.scatter_(0, idx[:, None].expand(-1, Ce), out_env)
                all_pred_pts.scatter_(0, idx_[:, None, None].expand(-1, npts, 3), pts_pred_all)
                all_pred_rgb.scatter_(0, idx_[:, None, None].expand(-1, npts, 3), rgb_pred_all)
                all_bone_pts.scatter_(0, idx_[:, None, None].expand(-1, npts_bone, 3), pts_bone_all)
                all_bone_rgb.scatter_(0, idx_[:, None, None].expand(-1, npts_bone, 3), rgb_bone_all)
                del idx, out_pose, out_bone, out_env
                del pts_pred_all, rgb_pred_all, pts_bone_all, rgb_bone_all
                del pts_pred, rgb_pred, pts_target, rgb_target

            if opts.use_ddp:
                all_reduce_list(
                    all_pred_poses, all_pred_bones, all_pred_angles, all_pred_envs,
                    all_pred_pts, all_pred_rgb, all_bone_pts, all_bone_rgb,
                    item_loss, iter_loss, pose_dfm_iter_loss, texture_iter_loss, angle_iter_loss,
                    regr_iter_loss, regr_pose_dfm_iter_loss, regr_texture_iter_loss, regr_angle_iter_loss
                )
            
       
        # Metrics, logs, and model save on rank 0
        mode = "train" if is_train else "eval"
        if opts.ddp_rank == 0:
            # Output metrics
            with torch.no_grad():
                if opts.mesh_rest_res != 256:
                    # Vertex position error
                    all_pred_pts_pos = all_pred_pts.to(device)
                    all_actual_pts_pos = all_actual_pts.to(device)

                    pts_pos_err = 100 * torch.nn.functional.mse_loss(all_pred_pts_pos, all_actual_pts_pos).item()
                    all_pred_pts_vel = all_pred_pts_pos[1:] - all_pred_pts_pos[:-1]
                    all_actual_pts_vel = all_actual_pts_pos[1:] - all_actual_pts_pos[:-1]
                    del all_pred_pts_pos, all_actual_pts_pos
                    
                    pts_vel_err = 100 * torch.nn.functional.mse_loss(all_pred_pts_vel, all_actual_pts_vel).item()
                    all_pred_pts_acc = all_pred_pts_vel[1:] - all_pred_pts_vel[:-1]
                    all_actual_pts_acc = all_actual_pts_vel[1:] - all_actual_pts_vel[:-1]
                    del all_pred_pts_vel, all_actual_pts_vel

                    pts_acc_err = 100 * torch.nn.functional.mse_loss(all_pred_pts_acc, all_actual_pts_acc).item()
                    del all_pred_pts_acc, all_actual_pts_acc
                else:
                    pts_pos_err = 0.0
                    pts_vel_err = 0.0
                    pts_acc_err = 0.0

                if opts.mesh_rest_res != 256:
                    # Vertex color error
                    all_pred_rgb_pos = all_pred_rgb.to(device)
                    all_actual_rgb_pos = all_actual_rgb.to(device)

                    rgb_pos_err = torch.nn.functional.mse_loss(all_pred_rgb_pos, all_actual_rgb_pos).item()
                    all_pred_rgb_vel = all_pred_rgb_pos[1:] - all_pred_rgb_pos[:-1]
                    all_actual_rgb_vel = all_actual_rgb_pos[1:] - all_actual_rgb_pos[:-1]
                    del all_pred_rgb_pos, all_actual_rgb_pos
                    
                    rgb_vel_err = torch.nn.functional.mse_loss(all_pred_rgb_vel, all_actual_rgb_vel).item()
                    all_pred_rgb_acc = all_pred_rgb_vel[1:] - all_pred_rgb_vel[:-1]
                    all_actual_rgb_acc = all_actual_rgb_vel[1:] - all_actual_rgb_vel[:-1]
                    del all_pred_rgb_vel, all_actual_rgb_vel

                    rgb_acc_err = torch.nn.functional.mse_loss(all_pred_rgb_acc, all_actual_rgb_acc).item()
                    del all_pred_rgb_acc, all_actual_rgb_acc
                else:
                    rgb_pos_err = 0.0
                    rgb_vel_err = 0.0
                    rgb_acc_err = 0.0
                
                # Don't include padding values in mean_squared_error calculation
                gt_poses = all_gt_poses.to(device) # T, 9
                target_poses = torch.where(torch.isnan(gt_poses), all_pred_poses, gt_poses) # T, 9
                pose_acc = torch.nn.functional.mse_loss(all_pred_poses, target_poses).item()
                del gt_poses, target_poses

                gt_bones = all_gt_bones.to(device) # T, B*12
                target_bones = torch.where(torch.isnan(gt_bones), all_pred_bones, gt_bones) # T, B*12
                bone_acc = torch.nn.functional.mse_loss(all_pred_bones, target_bones).item()
                del gt_bones, target_bones
                 
                gt_angles = all_gt_angles.to(device) # T, J, 3
                target_angles = torch.where(torch.isnan(gt_angles), all_pred_angles, gt_angles) # T, J, 3
                angle_acc = torch.nn.functional.mse_loss(all_pred_angles, target_angles).item()
                del gt_angles, target_angles

                gt_envs = all_gt_envs.to(device)
                target_envs = torch.where(torch.isnan(gt_envs), all_pred_envs, gt_envs)
                env_acc = torch.nn.functional.mse_loss(all_pred_envs, target_envs).item()
                del gt_envs, target_envs

                epoch_metrics = {
                    "loss": iter_loss.mean().item(),
                    "pose_dfm_loss": pose_dfm_iter_loss.mean().item(),
                    "texture_loss": texture_iter_loss.mean().item(),
                    "angle_loss": angle_iter_loss.mean().item(),
                    "regr_loss": regr_iter_loss.mean().item(),
                    "regr_pose_dfm_loss": regr_pose_dfm_iter_loss.mean().item(),
                    "regr_texture_loss": regr_texture_iter_loss.mean().item(),
                    "regr_angle_loss": regr_angle_iter_loss.mean().item(),
                    "var": torch.var(item_loss, dim=0, unbiased=True).item(),
                    "pose_acc": pose_acc,
                    "bone_acc": bone_acc,
                    "angle_acc": angle_acc,
                    "env_acc": env_acc,
                    "pts_pos_err": pts_pos_err,
                    "pts_vel_err": pts_vel_err,
                    "pts_acc_err": pts_acc_err,
                    "rgb_pos_err": rgb_pos_err,
                    "rgb_vel_err": rgb_vel_err,
                    "rgb_acc_err": rgb_acc_err,
                }

                iter_metrics = {
                    "loss": iter_loss.view(-1).detach().cpu().numpy(),
                    "pose_dfm_loss": pose_dfm_iter_loss.view(-1).detach().cpu().numpy(),
                    "texture_loss": texture_iter_loss.view(-1).detach().cpu().numpy(),
                    "angle_loss": angle_iter_loss.view(-1).detach().cpu().numpy(),
                    "regr_loss": regr_iter_loss.view(-1).detach().cpu().numpy(),
                    "regr_pose_dfm_loss": regr_pose_dfm_iter_loss.view(-1).detach().cpu().numpy(),
                    "regr_texture_loss": regr_texture_iter_loss.view(-1).detach().cpu().numpy(),
                    "regr_angle_loss": regr_angle_iter_loss.view(-1).detach().cpu().numpy(),
                }

                all_metrics = {}
                for k, v in epoch_metrics.items():
                    all_metrics[f"{k}"] = v
                for k, v in iter_metrics.items():
                    all_metrics[f"iter_{k}"] = v.min()

                if is_train:
                    for k in self.epoch_metrics:
                        self.train_epoch_metrics[k] = min(self.train_epoch_metrics[k], epoch_metrics[k])
                        self.train_epoch_metric_files[k][epoch] = epoch_metrics[k]
                        self.train_epoch_metric_files[k].flush()
                    for k in self.iter_metrics:
                        self.train_iter_metrics[k] = min(self.train_iter_metrics[k], iter_metrics[k].min())
                        self.train_iter_metric_files[k][epoch*n_iters : (epoch+1)*n_iters] = iter_metrics[k]
                        self.train_iter_metric_files[k].flush()
                else:
                    for k in self.epoch_metrics:
                        self.eval_epoch_metrics[k] = min(self.eval_epoch_metrics[k], epoch_metrics[k])
                        self.eval_epoch_metric_files[k][epoch] = epoch_metrics[k]
                        self.eval_epoch_metric_files[k].flush()
                    for k in self.iter_metrics:
                        self.eval_iter_metrics[k] = min(self.eval_iter_metrics[k], iter_metrics[k].min())
                        self.eval_iter_metric_files[k][epoch*n_iters : (epoch+1)*n_iters] = iter_metrics[k]
                        self.eval_iter_metric_files[k].flush()

                print(
                    (f"{opts.exp_key}\n" if opts.exp_key is not None else "") + 
                    f"{mode.title()} epoch {epoch}: " +
                    ", ".join(f"{k.replace('_', ' ').title()} {v:.5f}" for k, v in epoch_metrics.items()) +
                    f", Time {time.time() - start_time:.3f}s"
                )

            # Update logs
            self.log.add_scalars(mode, all_metrics, epoch)
        
            # Save model
            if is_train and epoch % opts.save_freq == 0 and (opts.save_at_epoch_zero or epoch > 0):
                print("Saving model...")
                start_time = time.time()
                checkpoint = {
                    "model": self.model.state_dict(),
                    "regressor_optimizer": self.regressor_optimizer.state_dict(),
                    "regressor_scheduler": self.regressor_scheduler.state_dict(),
                    "temporal_optimizer": self.temporal_optimizer.state_dict(),
                    "temporal_scheduler": self.temporal_scheduler.state_dict(),
                }
                torch.save(checkpoint, f"{opts.logdir}/params/params_latest.pth")
                if opts.keep_saved_models:
                    torch.save(checkpoint, f"{opts.logdir}/params/params_{epoch:04d}.pth")
                print(f"Time {time.time() - start_time:.3f}s")

        # Perform visualization
        if epoch % opts.vis_freq == 0 and (opts.vis_at_epoch_zero or epoch > 0):
            with torch.no_grad():
                # Visualizations on rank 0
                if opts.ddp_rank == 0:
                    if opts.vis_bone_rts_2d:
                        print("Visualizing predicted bone transforms in 2D...")
                        start_time = time.time()
                        vis_latent_codes_2d(
                            all_rgb_vis, all_dpfeat_vis,
                            all_pred_bones, all_gt_bones,
                            path_prefix=f"{opts.logdir}/{mode}_bone_rts_2d_{epoch:03d}",
                            n_workers=opts.n_vis_workers * opts.ddp_size, memory_limit=opts.memory_limit,
                        )
                        print(f"Time {time.time() - start_time:.3f}s")

                    if opts.vis_env_codes_2d:
                        print("Visualizing predicted env codes in 2D...")
                        start_time = time.time()
                        vis_latent_codes_2d(
                            all_rgb_vis, all_dpfeat_vis,
                            all_pred_envs, all_gt_envs,
                            path_prefix=f"{opts.logdir}/{mode}_env_codes_2d_{epoch:03d}",
                            n_workers=opts.n_vis_workers * opts.ddp_size, memory_limit=opts.memory_limit,
                        )
                        print(f"Time {time.time() - start_time:.3f}s")

                    if opts.vis_pose_distr_mplex_2d:
                        print("Visualizing predicted multiplex pose distribution in 2D...")
                        start_time = time.time()
                        vis_pose_distrs_2d(
                            all_rgb_vis, all_dpfeat_vis,
                            all_pred_poses_mplex, all_pred_probs_mplex, all_gt_poses,
                            path_prefix=f"{opts.logdir}/{mode}_pose_distr_mplex_2d_{epoch:03d}",
                            n_workers=opts.n_vis_workers * opts.ddp_size,
                        )
                        print(f"Time {time.time() - start_time:.3f}s")

                    if opts.vis_bone_rts_3d:
                        print("Visualizing predicted bone transforms in 3D...")
                        start_time = time.time()
                        mode_dataset = self.train_dataset if mode == "train" else self.eval_dataset
                        video_impaths = mode_dataset.datasets[0].video_impaths
                        frame_videoid = mode_dataset.datasets[0].frame_videoid # nframes,
                        frame_offsets = mode_dataset.datasets[0].frame_offsets # nframes,
                        vis_bone_rts_3d(
                            video_impaths, frame_videoid, frame_offsets,
                            all_pred_poses, all_pred_angles, all_camera_ks,
                            path_prefix=f"{opts.logdir}/{mode}_vis_bones_{epoch:03d}",
                            n_workers=opts.n_vis_workers * opts.ddp_size,
                        )
                        print(f"Time {time.time() - start_time:.3f}s")

                    if opts.save_pose_dfms_mesh:
                        print("Saving per-frame deformed meshes to obj...")
                        start_time = time.time()
                        mode_dataset = self.train_dataset if mode == "train" else self.eval_dataset
                        video_impaths = mode_dataset.datasets[0].video_impaths
                        frame_videoid = mode_dataset.datasets[0].frame_videoid # nframes,
                        frame_offsets = mode_dataset.datasets[0].frame_offsets # nframes,
                        save_pose_dfms_mesh(
                            video_impaths, frame_videoid, frame_offsets, self.mesh_rest,
                            all_pred_pts, all_pred_rgb, all_camera_ks,
                            path_prefix=f"{opts.logdir}/{mode}_pose_dfms_mesh_{epoch:03d}",
                            n_workers=opts.n_vis_workers * opts.ddp_size,
                        )
                        print(f"Time {time.time() - start_time:.3f}s")
                
                # Visualizations on all ranks in parallel
                if opts.save_pose_dfms_img:
                    if opts.ddp_rank == 0:
                        print("Saving per-frame deformed meshes to img...")
                        start_time = time.time()
                    mode_dataset = self.train_dataset if mode == "train" else self.eval_dataset
                    video_impaths = mode_dataset.datasets[0].video_impaths
                    frame_videoid = mode_dataset.datasets[0].frame_videoid # nframes,
                    frame_offsets = mode_dataset.datasets[0].frame_offsets # nframes,
                    save_pose_dfms_img(
                        video_impaths, frame_videoid, frame_offsets, all_rgb_vis,
                        self.faces_bone, all_bone_pts, all_bone_rgb,
                        self.faces_can, all_pred_pts, all_pred_rgb,
                        all_pred_poses, all_camera_ks,
                        pts_actual=all_actual_pts, rgb_actual=all_actual_rgb,
                        path_prefix=f"{opts.logdir}/{mode}_pose_dfms_img_{epoch:03d}",
                        n_workers=opts.n_vis_workers, memory_limit=opts.memory_limit,
                        use_ddp=opts.use_ddp, ddp_rank=opts.ddp_rank, ddp_size=opts.ddp_size,
                    )
                    if opts.ddp_rank == 0:
                        print(f"Time {time.time() - start_time:.3f}s")

                if opts.save_pose_dfms_mplex_img:
                    if opts.ddp_rank == 0:
                        print("Saving per-frame deformed mplex meshes to img...")
                        save_pose_dfms_mplex_img(
                            25, opts.temporal_radius, all_rgb_vis, self.pts_bone, self.faces_bone,
                            self.bones_rst, self.bone_rts_rst, self.centroid,
                            all_pred_poses, all_pred_bones, all_pred_envs, all_camera_ks,
                            all_pred_poses_mplex, all_pred_probs_mplex,
                            path_prefix=f"{opts.logdir}/{mode}_pose_dfms_mplex_img_{epoch:03d}",
                            n_workers=opts.n_vis_workers, memory_limit=opts.memory_limit,
                            use_ddp=opts.use_ddp, ddp_rank=opts.ddp_rank, ddp_size=opts.ddp_size,
                        )
                        print(f"Time {time.time() - start_time:.3f}s")

                if opts.vis_pose_dfms_3d:
                    if opts.ddp_rank == 0:
                        print("Visualizing predicted poses and deformations in 3D...")
                        start_time = time.time()
                    vis_pose_dfms_3d(
                        all_rgb_vis, all_dpfeat_vis, self.faces_can,
                        all_pred_pts, all_pred_rgb, all_actual_pts, all_actual_rgb, all_camera_ks,
                        path_prefix=f"{opts.logdir}/{mode}_pose_dfms_3d_{epoch:03d}",
                        n_workers=opts.n_vis_workers, memory_limit=opts.memory_limit,
                        use_ddp=opts.use_ddp, ddp_rank=opts.ddp_rank, ddp_size=opts.ddp_size,
                    )
                    if opts.ddp_rank == 0:
                        print(f"Time {time.time() - start_time:.3f}s")

                if opts.vis_pose_dfms_mplex_3d:
                    if opts.ddp_rank == 0:
                        print("Visualizing predicted multiplexed poses and deformations in 3D...")
                        start_time = time.time()
                    vis_pose_dfms_mplex_3d(
                        self.model, all_rgb_vis, all_dpfeat_vis, self.pts_bone, self.faces_bone,
                        self.bones_rst, self.bone_rts_rst, self.centroid,
                        all_pred_pts, all_pred_rgb, all_actual_pts, all_actual_rgb,
                        all_pred_poses, all_pred_bones, all_pred_envs, all_camera_ks,
                        all_pred_poses_mplex, all_pred_probs_mplex, 
                        path_prefix=f"{opts.logdir}/{mode}_pose_dfms_mplex_3d_{epoch:03d}",
                        n_workers=opts.n_vis_workers, memory_limit=opts.memory_limit,
                        use_ddp=opts.use_ddp, ddp_rank=opts.ddp_rank, ddp_size=opts.ddp_size,
                    )
                    if opts.ddp_rank == 0:
                        print(f"Time {time.time() - start_time:.3f}s")

        if opts.ddp_rank == 0:
            return epoch_metrics["loss"]


    def train(self):
        opts = self.opts
        if opts.ddp_rank == 0:
            print(f"Training {opts.logdir} for {opts.n_epochs} epochs...")
        for epoch in range(opts.n_epochs):
            if not opts.eval:
                self.run_one_epoch(epoch, is_train=True)

            if opts.eval or epoch % opts.eval_freq == 0:
                self.run_one_epoch(epoch, is_train=False)

        # Early exit if using DDP
        if opts.ddp_rank != 0:
            return 0

        # float(v) casts from numpy float32 to Python float type, for proper JSON serialization
        out = {"logdir": opts.logdir}
        for k, v in self.train_epoch_metrics.items():
            out[f"train_epoch_{k}"] = float(v)
        for k, v in self.train_iter_metrics.items():
            out[f"train_iter_{k}"] = float(v)
        for k, v in self.eval_epoch_metrics.items():
            out[f"eval_epoch_{k}"] = float(v)
        for k, v in self.eval_iter_metrics.items():
            out[f"eval_iter_{k}"] = float(v)
        return out

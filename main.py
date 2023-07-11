import argparse
import copy
import json
import multiprocessing
import os
import socket
import torch
import traceback
from datetime import datetime

from banmo_utils import register_banmo
from train_utils import Trainer

def main(opts):
    """Train a model with the given command-line options returned by parse_args.
    If successful, dump the results to `result.json` in the output directory.
    If an exception occurs, dump the traceback and exception message to
    `result.json` and re-raise.
    """
    out = None
    try:
        # Load banmo
        register_banmo(opts.banmo_path, opts.seqname)

        # Set OMP workers
        torch.set_num_threads(opts.n_omp_workers)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Anomaly detection
        if opts.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        # Set up distributed data parallel
        if opts.use_ddp:
            os.environ["MASTER_ADDR"] = opts.ddp_addr
            os.environ["MASTER_PORT"] = str(opts.ddp_port)
            torch.distributed.init_process_group("gloo", rank=opts.ddp_rank, world_size=opts.ddp_size)

        if opts.ddp_rank == 0:
            print(f"Initializing trainer with logdir '{opts.logdir}'")
        trainer = Trainer(opts)
        out = trainer.train()
        return out

    except Exception as e:
        out = "".join(traceback.format_exception(None, e, e.__traceback__))
        raise

    finally:
        if opts.ddp_rank == 0:
            print(f"Finished {opts.logdir}")
        with open(f"{opts.logdir}/result.json", "w") as f:
            if isinstance(out, str):
                f.write(out)
            else:
                f.write(json.dumps(out, default=repr, indent=4))

        if opts.use_ddp:
            torch.distributed.destroy_process_group()


def main_ddp(opts):
    """Distributed data parallel wrapper for main function"""
    # Create unique output directory
    opts.logdir = make_unique_logdir(opts.logdir_prefix, opts.exp_key)

    if opts.use_ddp:
        # Allow CUDA from within multiprocessing
        mp = multiprocessing.get_context("spawn")

        if os.getenv("CUDA_VISIBLE_DEVICES") is None:
            num_gpus = int(os.popen("nvidia-smi -L | wc -l").read())
            gpus = list(range(num_gpus))
        else:
            gpus = [int(n) for n in os.getenv("CUDA_VISIBLE_DEVICES").split(",")]
        
        # Set OMP workers
        if opts.n_omp_workers is None:
            opts.n_omp_workers = min(4, multiprocessing.cpu_count() // len(gpus))
            
        # Increment port until it is unused
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            while s.connect_ex((opts.ddp_addr, opts.ddp_port)) == 0:
                opts.ddp_port += 1

        # Spawn processes
        print(f"Using DDP at {opts.ddp_addr}:{opts.ddp_port} with GPUs {gpus}")
        spawned_procs = []
        for rank, gpu_id in enumerate(gpus):
            opts = copy.deepcopy(opts)
            opts.ddp_rank = rank
            opts.ddp_size = len(gpus)

            # Set env vars here since they get copied on process creation
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            os.environ["OMP_NUM_THREADS"] = str(opts.n_omp_workers)
            proc = mp.Process(target=main, args=(opts,))
            proc.start()
            spawned_procs.append(proc)

        # Wait for all processes to finish
        for proc in spawned_procs:
            proc.join()

    else:
        # Set OMP workers
        if opts.n_omp_workers is None:
            opts.n_omp_workers = 1

        main(opts)


def make_unique_logdir(logdir_prefix, exp_key):
    """Given an output prefix and experiment key, build a unique logdir

    Args
        logdir_prefix [string]: Prefix for output directory
        exp_key [string]: A key describing the current experiment, if applicable

    Returns
        logdir[string]: Path to unique logdir
    """
    if exp_key is not None:
        logdir = f"{logdir_prefix}_{exp_key}"
    else:
        logdir = f"{logdir_prefix}"

    if os.path.exists(logdir):
        idx = 1
        while True:
            if exp_key is not None:
                logdir = f"{logdir_prefix}_{exp_key}_{idx}"
            else:
                logdir = f"{logdir_prefix}_{idx}"

            if os.path.exists(logdir):
                idx += 1
            else:
                break
    
    os.makedirs(logdir, exist_ok=True)
    return logdir


def parse_args():
    """Parse command-line args
    
    Returns
        opts [Namespace]: Command-line options
    """
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_aug2d_rgb_dpfeat = json.dumps([
        ["Pad", {"padding": 48}],
        ["RandomResizedCrop", {
            "size": [224, 224],
            "scale": [(1 - 48 / 224) ** 2, 1],
            "ratio": [3 / 4, 4 / 3],
        }],
    ])
        

    default_aug2d_dpfeat = json.dumps([
        ["ColorJitter", {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
        }],
        ["RandomErasing", {
            "scale": [0, 0.15],
            "value": 0,
            "inplace": True,
        }], 
    ])

    default_aug2d_rgb = json.dumps([
        ["RandomErasing", {
            "scale": [0, 0.15],
            "value": 0,
            "inplace": True,
        }], 
    ])
    
    # Inputs and outputs
    group = parser.add_argument_group("Preset option groups")
    group.add_argument("--eval", default=False, action="store_true",
            help="Run the model in evaluation mode, without training")
    group.add_argument("--lowmem", default=False, action="store_true",
            help="Run the model with reduced batch size for low-memory GPUs")
    group.add_argument("--highmem", default=False, action="store_true",
            help="Run the model with increased batch size for high-memory GPUs")

    group.add_argument("--detect_anomaly", default=False, action="store_true",
            help="Run the model with PyTorch anomaly detection")
    group.add_argument("--summarize", default=False, action="store_true",
            help="Summarize model using torchsummary")
    group.add_argument("--log_data", default=False, action="store_true",
            help="Whether to log dataloader outputs to logdir")

    group.add_argument("--cat76", default=False, action="store_true",
            help="Run optimization on cat76 dataset")
    group.add_argument("--dog87", default=False, action="store_true",
            help="Run optimization on dog87 dataset")

    group.add_argument("--human", default=False, action="store_true",
            help="Run optimization on human dataset")
    
    group = parser.add_argument_group("Distributed Data Parallel")
    group.add_argument("--use_ddp", type=str, choices=["True", "False"], default="False",
            help="Whether to use distributed data parallel")
    group.add_argument("--ddp_addr", type=str, default="localhost",
            help="Master address for distributed data parallel")
    group.add_argument("--ddp_port", type=int, default=12000,
            help="Master port for distributed data parallel")
    group.add_argument("--ddp_rank", type=int, default=0,
            help="Rank for distributed data parallel")
    group.add_argument("--ddp_size", type=int, default=1,
            help="Size for distributed data parallel")
    group.add_argument("--sync_grad", type=str, choices=["True", "False"], default="True",
            help="Whether to sync gradients manually")

    group = parser.add_argument_group("Inputs")
    group.add_argument("--seqname", type=str, default=None,
            help="Banmo seqname of data to use for underlying banmo model")
    group.add_argument("--train_videos", type=str, default="trainA.json",
            help="Videos to use during training")
    group.add_argument("--eval_videos", type=str, default="evalA.json",
            help="Videos to use during evaluation")
    group.add_argument("--use_cache_img", type=str, choices=["True", "False"], default="True",
            help="Whether to use global cache for dataloading images")
    group.add_argument("--use_cache_gt", type=str, choices=["True", "False"], default="False",
            help="Whether to use global cache for dataloading ground-truths")
    group.add_argument("--use_cache_gtvid", type=str, choices=["True", "False"], default="True",
            help="Whether to use global cache for dataloading per-video ground-truths")

    group.add_argument("--pose_and_bone_input", type=str, default="rgb_dpfeat",
            help="Type of input to use for shape/deformation predictor")
    group.add_argument("--env_code_input", type=str, default="rgb_dpfeat",
            help="Type of input to use for texture predictor")

    group = parser.add_argument_group("Input paths")
    group.add_argument("--data_path", type=str, default="database/DAVIS",
            help="Path to training/eval image dataset")
    group.add_argument("--banmo_path", type=str, default=None,
            help="Path to trained banmo dependencies, with banmo_opts.json, params.pth, vars.npy, mesh_rest.obj")
    group.add_argument("--mesh_rest_res", type=int, default=128,
            help="Rest mesh marching cubes resolution")
    
    group = parser.add_argument_group("Outputs")
    group.add_argument("--logdir_prefix", type=str, default=f"output/{time_str}",
            help="Prefix for output files. Defaults to `output/{time_str}` if not provided")
    group.add_argument("--exp_key", type=str, default=None,
            help="A key describing this experiment. If provided, will be appended after the `logdir_prefix`.")

    # Dataset and model parameters
    group = parser.add_argument_group("Task/model selection")
    group.add_argument("--predict", type=str, default="pose_dfm",
            help="Whether the model should predict poses only (pose), or poses and deformations (pose_dfm)")
    group.add_argument("--mesh_loss_npts_samp", type=int, default=1000,
            help="Number of vertices to sample for mesh loss computation")
    
    group = parser.add_argument_group("CNN regressor")
    group.add_argument("--regressor_type", type=str, default="resnet18",
            help="Which architecture to use for CNN regressor from frames")
    group.add_argument("--prefix_type", type=str, default="conv",
            help="Which preprocessing architecture to use for CNN regressor from frames")

    group = parser.add_argument_group("Rotation decoder")
    group.add_argument("--rotation_decoder_type", type=str, default="continuous_6d",
            help="Which rotation decoder to apply on root body pose codes")
    group.add_argument("--rotation_decoder_kwargs", type=str, default="{}",
            help="Rotation decoder additional args")
    group.add_argument("--pose_multiplex_count", type=int, default=6,
            help="Number of instances for root body pose multiplexing")
    group.add_argument("--pose_multiple_choice_loss", type=str, choices=["True", "False"], default="False",
            help="Whether to use multiple choice loss for pose multiplex")
    
    group = parser.add_argument_group("Bone transform decoder")
    group.add_argument("--bone_transform_decoder_type", type=str, default="banmo",
            help="Which bone transform decoder to apply on pose codes")
    group.add_argument("--bone_transform_decoder_kwargs", type=str, default="{}",
            help="Bone transform decoder additional args")

    group = parser.add_argument_group("Data augmentation")
    group.add_argument("--use_aug2d", type=str, choices=["True", "False"], default="True",
            help="Whether to use 2D data augmentation on the input images")
    group.add_argument("--aug2d_pipeline_rgb_dpfeat", type=str, default=default_aug2d_rgb_dpfeat,
            help="JSON encoded specification of data augmentation pipeline for both rgb and dpfeat")
    group.add_argument("--aug2d_pipeline_rgb", type=str, default=default_aug2d_rgb,
            help="JSON encoded specification of data augmentation pipeline for rgb")
    group.add_argument("--aug2d_pipeline_dpfeat", type=str, default=default_aug2d_dpfeat,
            help="JSON encoded specification of data augmentation pipeline for dpfeat")

    group = parser.add_argument_group("Temporal Encoder")
    group.add_argument("--temporal_encoder_type", type=str, default="conv",
            help="Temporal encoder type")
    group.add_argument("--temporal_encoder_kwargs", type=str, default="{}",
            help="Temporal encoder additional args")
    group.add_argument("--temporal_radius", type=int, default=6,
            help="Radius of frames available in both directions to temporal encoder")

    group.add_argument("--root_pose_dim", type=int, default=20,
            help="Dimension of root body pose latent vector, equivalent to 3x4 SE3")
    group.add_argument("--pose_feat_dim", type=int, default=30,
            help="Dimension of root body pose feature vector, including extra features for temporal encoder")
    group.add_argument("--pose_code_dim", type=int, default=16,
            help="Dimension of banmo pose code latent vector")
    group.add_argument("--env_code_dim", type=int, default=64,
            help="Dimension of banmo environment code latent vector")
    group.add_argument("--n_bones", type=int, default=26,
            help="Number of bones to use")
    group.add_argument("--blend_method", type=str, default="dual_quat",
            help="Which method to use for blending bones by skinning weights")
    group.add_argument("--use_dense_supervision", type=str, choices=["True", "False"], default="True",
            help="Whether to use dense or sparse temporal supervision on output codes")

    # Training parameters
    group = parser.add_argument_group("Training parameters")
    group.add_argument("--n_epochs", type=int, default=1000,
            help="Number of epochs")
    group.add_argument("--batch_size", type=int, default=96,
            help="Batch size of dataloader")
    group.add_argument("--memory_limit", type=int, default=int(512e6),
            help="Memory limit of chunked computations, in bytes")
    group.add_argument("--random_seed", type=int, default=0,
            help="Random number generator seed")

    group = parser.add_argument_group("Optimization parameters")
    group.add_argument("--optimizer", type=str, default="AdamW",
            help="Which optimizer to use")
    group.add_argument("--learning_rate", type=float, default=5e-4,
            help="Learning rate for optimizer")
    group.add_argument("--weight_decay", type=float, default=0,
            help="Weight decay for optimizer")

    group.add_argument("--loss_scale", type=float, default=1.0,
            help="Scale factor for loss before optimization")
    group.add_argument("--pose_dfm_loss_scale", type=float, default=1000.0,
            help="Scale factor for pose_dfm loss")
    group.add_argument("--texture_loss_scale", type=float, default=0.1,
            help="Scale factor for texture loss")
    group.add_argument("--angle_loss_scale", type=float, default=1.0,
            help="Scale factor for angle loss")
    
    group.add_argument("--scheduler", type=str, default="OneCycleLR",
            help="Which learning rate scheduler to use")
    group.add_argument("--scheduler_kwargs", type=str,
            default='{"pct_start": 0.3, "div_factor": 25, "final_div_factor": 10000}',
            help="Additional arguments for learning rate scheduler")

    group = parser.add_argument_group("Model save/load")
    group.add_argument("--load_params", type=str, default=None,
            help="If provided, load saved model params from the given .pth file")
    group.add_argument("--load_lr", type=str, choices=["True", "False"], default="True",
            help="Whether to load the learning rate from params")
    group.add_argument("--save_freq", type=int, default=80,
            help="Number of epochs between each model save")
    group.add_argument("--save_at_epoch_zero", type=str, choices=["True", "False"], default="False",
            help="Whether to save model at epoch zero")
    group.add_argument("--keep_saved_models", type=str, choices=["True", "False"], default="False",
            help="Whether to keep previous saved models")
    
    group = parser.add_argument_group("Model visualization")
    group.add_argument("--eval_freq", type=int, default=20,
            help="Number of epochs between each evaluation run")
    group.add_argument("--vis_freq", type=int, default=80,
            help="Number of epochs between each visualization run")
    group.add_argument("--vis_at_epoch_zero", type=str, choices=["True", "False"], default="False",
            help="Whether to perform visualization at epoch zero")
    group.add_argument("--out_freq", type=int, default=20,
            help="Number of epochs between each model output save")
    group.add_argument("--out_at_epoch_zero", type=str, choices=["True", "False"], default="False",
            help="Whether to output at epoch zero")

    group.add_argument("--vis_bone_rts_2d", type=str, choices=["True", "False"], default="False",
            help="Whether to visualize predicted bone transforms in 2D")
    group.add_argument("--vis_env_codes_2d", type=str, choices=["True", "False"], default="False",
            help="Whether to visualize predicted environment codes in 2D")
    group.add_argument("--vis_root_body_poses_3d", type=str, choices=["True", "False"], default="False",
            help="Whether to visualize predicted root body poses in 3D")
    group.add_argument("--vis_bone_rts_3d", type=str, choices=["True", "False"], default="False",
            help="Whether to visualize predicted bone transforms in 3D")
    group.add_argument("--vis_pose_dfms_3d", type=str, choices=["True", "False"], default="True",
            help="Whether to visualize predicted root body poses and pose codes in 3D")
    group.add_argument("--vis_pose_dfms_mplex_3d", type=str, choices=["True", "False"], default="False",
            help="Whether to visualize multiplexed predicted root body poses and pose codes in 3D")
    group.add_argument("--vis_pose_distr_mplex_2d", type=str, choices=["True", "False"], default="False",
            help="Whether to visualize predicted multiplex pose distribution in 2D")

    # Evaluation parameters
    group = parser.add_argument_group("Evaluation")
    group.add_argument("--save_pose_dfms_mesh", type=str, choices=["True", "False"], default="False",
            help="Whether to save per-frame deformed meshes to obj")
    group.add_argument("--save_pose_dfms_img", type=str, choices=["True", "False"], default="False",
            help="Whether to save per-frame deformed meshes to img")
    group.add_argument("--save_pose_dfms_mplex_img", type=str, choices=["True", "False"], default="False",
            help="Whether to save per-frame multiplexed deformed meshes to img")

    group = parser.add_argument_group("Worker threads")
    group.add_argument("--n_omp_workers", type=int, default=None,
            help="Number of workers to use within OpenMP. By default, use min(4, NCPU // NGPU) if DDP else 1")
    group.add_argument("--n_data_workers", type=int, default=0,
            help="Number of workers to use for initial dataset loading. Use 0 when loading from cache")
    group.add_argument("--n_vis_workers", type=int, default=0,
            help="Number of workers to use for visualization")
    group.add_argument("--n_aug3d_workers", type=int, default=16,
            help="Number of workers to use for 3D augmentation")

    parser.add_argument("positional_args", nargs="*")
    opts = parser.parse_args()

    # Coerce 'True' and 'False' attributes to bool
    for attr in opts.__dict__:
        if getattr(opts, attr) == "True":
            setattr(opts, attr, True)
        elif getattr(opts, attr) == "False":
            setattr(opts, attr, False)
    
    # Visualize at epoch 0 if loading saved params
    if opts.load_params is not None:
        opts.vis_at_epoch_zero = True

    # Optimization targets
    if opts.cat76:
        opts.seqname = "cat76"
        opts.banmo_path = "banmo_deps/checkpoints/hmnerf-cate-pretrain-cat-pikachu-init-cat76-ft2_120"
        opts.n_bones = 26
    elif opts.dog87:
        opts.seqname = "dog87"
        opts.banmo_path = "banmo_deps/checkpoints/hmnerf-cate-pretrain-shiba-haru-init-dog87-ft2_120"
        opts.n_bones = 26
    elif opts.human:
        opts.seqname = "human47"
        opts.banmo_path = "banmo_deps/checkpoints/hmnerf-cate-ama-ft2_120"
        opts.n_bones = 19

    # Load model params
    params_replace = {
        "seqname", "banmo_path", "posenet_path", "logdir_prefix", "exp_key", "predict", "texture",
        "use_morphology_code", "use_pretrained_regressor", "regressor_type", "prefix_type",
        "rotation_decoder_type", "rotation_decoder_kwargs", "pose_multiplex_count", "pose_multiple_choice_loss",
        "bone_transform_decoder_type", "bone_transform_decoder_kwargs",
        "temporal_encoder_type", "temporal_encoder_kwargs", "temporal_radius",
        "root_pose_dim", "pose_feat_dim", "pose_code_dim", "env_code_dim", "n_bones",
        "optimizer", "scheduler", "scheduler_kwargs",
    }
    if opts.load_params is not None:
        load_path = opts.load_params.split("/params")[0] + "/opts.json"
        with open(load_path, "r") as load_file:
            load_opts = json.load(load_file)
        print(f"Loading opts from '{load_path}'...")
        for k, load_v in load_opts.items():
            if hasattr(opts, k) and k in params_replace:
                curr_v = getattr(opts, k)
                if curr_v != load_v:
                    print(f" - Replacing opts.{k}={curr_v} with {load_v}")
                    setattr(opts, k, load_v)
            elif k in params_replace:
                setattr(opts, k, load_v)

    # Evaluation mode
    if opts.eval:
        opts.n_epochs = 1
        opts.vis_at_epoch_zero = True
        opts.save_at_epoch_zero = False

    # Reduce batch size for low-memory GPUs
    if opts.lowmem:
        opts.batch_size //= 2

    # Increase batch size for high-memory GPUs
    if opts.highmem:
        opts.batch_size *= 2

    print(f"Using dataset '{opts.seqname}' with banmo path '{opts.banmo_path}'")
    return opts

if __name__ == '__main__':
    opts = parse_args()
    main_ddp(opts)

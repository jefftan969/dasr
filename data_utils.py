import configparser
import cv2
import glob
import json
import numpy as np
import os
import re
import sys
import time
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOB_TEMPLATES = {
    "img": f"database/DAVIS/JPEGImages/Full-Resolution/%s/{5*'[0-9]'}.jpg",
    "mask": f"database/DAVIS/Annotations/Full-Resolution/%s/{5*'[0-9]'}.png",
    "mask_vis": f"database/DAVIS/Annotations/Full-Resolution/%s/vis-{5*'[0-9]'}.jpg",
    "dp_sil": f"database/DAVIS/Densepose/Full-Resolution/%s/{5*'[0-9]'}.pfm",
    "dp_feat": f"database/DAVIS/Densepose/Full-Resolution/%s/feat-{5*'[0-9]'}.pfm",
    "dp_bbox": f"database/DAVIS/Densepose/Full-Resolution/%s/bbox-{5*'[0-9]'}.txt",
    "dp_vis": f"database/DAVIS/Densepose/Full-Resolution/%s/vis-{5*'[0-9]'}.jpg",
}


# ===== Image I/O

def read_img(filename, *, crop=None, resize=None):
    """Reads an image from file.

    Args
        filename [string]: Input filename
        crop [Tuple(int) or None]: If provided, crop image to bounding box before
            resize. Specified as tuple (x0, y0, x1, y1)
        resize [Tuple(int) or None]: If provided, resize image to this size

    Returns
        image [np.ndarray]: An image stored as an HxWx3 array
    """
    if not os.path.exists(filename):
        raise ValueError(f"Image at '{filename}' does not exist")
    if not os.path.isfile(filename):
        raise ValueError(f"Image at '{filename}' is not a file")

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop is not None:
        x0, y0, x1, y1 = crop
        image = image[y0:y1, x0:x1]
    if resize is not None:
        image_shape = image.shape[:len(resize)]
        resize = tuple(image_shape[i] if resize[i] is None else resize[i] for i in range(len(resize)))
        image = cv2.resize(image, resize[::-1])
    image = image.astype(np.float32) / 255.0
    return image


def write_img(filename, image, *, normalize=False):
    """Writes an image to file.

    Args
        filename [string]: Output filename
        image [np.ndarray]: An image stored as an HxWx3 array
        normalize [bool]: Whether to normalize the image to between 0 and 1
    """
    if normalize:
        image = np.copy(image)
        for c in range(3):
            cmin = image[:, :, c].min()
            cmax = image[:, :, c].max()
            image[:, :, c] = (image[:, :, c] - cmin) / (cmax - cmin)
    image = (255.0 * np.clip(image, 0, 1)).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)


def read_pfm(filename):
    """Reads a PFM image from file.

    Args
        filename [string]: Input filename

    Returns
        image [np.ndarray]: An image stored as an HxW array
    """
    with open(filename, "rb") as f:
        header = f.readline().rstrip().decode("utf-8")
        color = None
        width = None
        height = None
        scale = None
        endian = None

        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise Exception(f"Not a PFM file: '{filename}'.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(f.readline().rstrip().decode("utf-8"))

        if scale < 0:
            endian = '<' # litle-endian
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.ascontiguousarray(np.flipud(data))
        return data, scale


def write_pfm(filename, image, *, scale=1):
    """Writes an image to file.

    Args
        filename [string]: Output filename
        image [np.ndarray]: An image stored as an HxW or HxWx3 array
        scale [int]: Scale for image, defaults to 1
    """
    # Case to HxWx3
    if image.ndim == 3 and image.shape[-1] == 2:
        image = np.concatenate((image, np.zeros(image.shape[:2] + (1,))), axis=-1)
        image = image.astype(np.float32)

    with open(filename, "wb") as f:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3: # color image
            color = True
        elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1): # greyscale
            color = False
        else:
            raise Exception("Image must have HxWx3, HxWx1, or HxW dimensionality")

        f.write("PF\n".encode() if color else "Pf\n".encode())
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())

        endian = image.dtype.byteorder

        if endian == "<" or (endian == "=" and sys.byteorder == "little"):
            scale = -scale

        f.write(f"{scale}\n".encode())

        image.tofile(f)


# ===== Config Parsing

def seqnames_from_banmo_config(seqname):
    """Parse a banmo dataset .config file and return a list of videos
    
    Args
        seqname [str]: Name of banmo config file

    Returns
        videos [List(str)]: List of video names
    """
    config = configparser.RawConfigParser()
    config_path = f"configs/{seqname}.config"
    if not os.path.exists(config_path):
        raise RuntimeError(f"Couldn't find banmo dataset config file '{config_path}'")
    config.read(config_path)

    # Seqnames in the order specified by banmo .config file
    nvideos = len(config.sections()) - 1 # Subtract 1 because we don't include default section
    videos = []
    for i in range(nvideos):
        dataname = f"data_{i}"
        datapath = config.get(dataname, "datapath")
        video_name = datapath.split("/")[-2]
        videos.append(video_name)

    return videos


def camera_ks_from_banmo_config(seqname):
    """Parse a banmo dataset .config file and return a list of camera intrinsics per video
    
    Args
        seqname [str]: Name of banmo config file

    Returns
        camera_ks [nvideos, 4]: Matrix of (fx, fy, cx, cy) values per video
    """
    config = configparser.RawConfigParser()
    config_path = f"configs/{seqname}.config"
    if not os.path.exists(config_path):
        raise RuntimeError(f"Couldn't find banmo dataset config file '{config_path}'")
    config.read(config_path)

    # Camera intrinsics in the order specified by banmo .config file
    nvideos = len(config.sections()) - 1 # Subtract 1 because we don't include default section
    camera_ks = np.zeros((nvideos, 4), dtype=np.float32) # nvideos, 4
    for i in range(nvideos):
        dataname = f"data_{i}"
        ks = config.get(dataname, "ks")
        camera_ks[i] = tuple(float(elt) for elt in ks.split(" "))

    return camera_ks


def data_info_from_banmo_config(seqname):
    """Parse a banmo dataset .config file and return a data_info dict
    
    Args
        seqname [str]: Name of banmo config file

    Returns
        data_info [Dict]: Data info dict with "offset", "impath", and "len_evalloader" keys
    """
    banmo_seqnames = seqnames_from_banmo_config(seqname)

    # Video info dict required by banmo
    # Each glob represents a video, each globbed item represents a frame
    jpeg_globs = [f"database/DAVIS/JPEGImages/Full-Resolution/{seq}/{5*'[0-9]'}.jpg" for seq in banmo_seqnames]
    video_offsets = [0] # A list mapping video i to the number of frames appearing before it
    video_impaths = []  # A list mapping video i to a list of image paths for that video

    for g in jpeg_globs:
        num_frames = 0
        video_paths = []
        for filename in sorted(glob.glob(g)):
            video_paths.append(os.path.abspath(filename))
            num_frames += 1

        video_offsets.append(video_offsets[-1] + num_frames)
        video_impaths.append(video_paths)

    if video_offsets[-1] == 0:
        raise RuntimeError("No data available at 'database/DAVIS'")

    data_info = {
        "offset": np.array(video_offsets),
        "impath": video_impaths,
        # Last image of each video is not included in evalloader length
        "len_evalloader": video_offsets[-1] - len(video_offsets) + 1,
    }
    return data_info


# ===== 2D Data Augmentation

def build_aug2d_pipeline(config_in, print_timing_details=False):
    """Build a data augmentation pipeline from a dict or JSON configuration string.
    The string should be specified as a list of pairs, where the first item is the name
    of the augmentation class within torchvision.transforms and the second item is
    a dictionary of arguments to pass to the augmentation class.
    
    Example: The string "[['Pad', {'padding': [5, 2]}], ['RandomAffine', {'degrees': 10}]]"
        produces the following data augmentation pipeline:

        transforms.Compose([
            transforms.Pad(padding=[5, 2]),
            transforms.RandomAffine(degrees=10),
        ])

    Args
        config_in [dict or str]: Input aug2d data pipeline config
        print_timing_details [bool]: Whether to print debug timing details during pipeline

    Returns
        aug2d_pipeline [transforms.Transform]: Composition of transforms
    """
    # Parse input config dict or string
    if config_in is None:
        return None
    elif isinstance(config_in, str):
        config = json.loads(config_in)
    elif isinstance(config_in, dict):
        config = config_in
    else:
        raise RuntimeError(f"Invalid config_in, cannot build data aug pipeline: '{config_in}'")

    if print_timing_details:
        class TimingClass(torch.nn.Module):
            def __init__(self, label, prev):
                super().__init__()
                self.label = label
                self.prev = prev
                self.time = time.time()

            def forward(self, x):
                torch.cuda.synchronize()
                print(f"Time for {self.label}: ", time.time() - self.prev.time)
                self.time = time.time()
                return x

    # Build pipeline
    aug_pipeline = []
    timing_prev = None
    for transform_class, transform_args in config:
        assert hasattr(transforms, transform_class), \
            f"Invalid data augmentation transform '{transform_class}'"
        global_vars = {"transforms": transforms}
        local_vars = {"transform_args": transform_args}
        # Initialize an instance of `transform_class` with args provided by `transform_args`
        aug = eval(f"transforms.{transform_class}(**transform_args)", global_vars, local_vars)
        aug_pipeline.append(aug)

        if print_timing_details:
            timing_class = TimingClass(transform_class, timing_prev)
            timing_prev = timing_class
            aug_pipeline.append(timing_class)

    if print_timing_details:
        aug_pipeline[1].prev = aug_pipeline[-1]

    # Convert transform to an nn.Sequential
    aug_pipeline = transforms.Compose(aug_pipeline)
    return aug_pipeline
    

def aug2d(frames, aug2d_pipeline):
    """Perform 2D data augmentation

    Args
        frames [..., C, H, W]: Frames to data-augment
        aug2d_pipeline [transforms.Transform]: 2D augmentation pipeline to apply

    Returns
        frames [..., C, H_, W_]: Augmented frames
    """
    prefix_shape = frames.shape[:-3]
    C, H, W = frames.shape[-3:]
    if C == 3:
        frames = frames.reshape(-1, 3, H, W) # -1, C, H, W
    else:
        frames = frames.reshape(-1, 1, H, W) # -1*C, 1, H, W
    frames = aug2d_pipeline(frames) # if C == 3 then -1, C, H, W else -1*C, 1, H, W

    H_, W_ = frames.shape[-2:]
    frames = frames.view(prefix_shape + (C, H_, W_))
    return frames


# ===== Training outputs

def alloc_empty(shape, device=device):
    """Helper function for allocating output tensors. Tensors are initialized
    with FLOAT_MIN and can be all-reduced using reduce_op=MAX for DDP. Async
    handle is stored in `dist_handle` attribute of tensor
    
    Args
        shape [Tuple]: Desired shape

    Returns
        out [shape]: Output tensor
    """
    fmin = torch.finfo(torch.float32).min
    out = torch.full(shape, fmin, dtype=torch.float32, device=device, requires_grad=False)
    return out


def all_reduce_list(*tensor_list, async_op=True):
    """Helper function for async-reducing a list of output tensors
    
    Args
        *tensor_list [List(Tensor)]: Tensors to reduce, with FLOAT_MIN for values not on this rank
    """
    handles = []
    for tensor in tensor_list:
        handles.append(torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX, async_op=async_op))
    for handle in handles:
        if handle is not None:
            handle.wait()

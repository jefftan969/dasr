import cv2
import glob
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import tqdm

from data_utils import write_pfm
    
sys.path.insert(0, "third_party/detectron2")
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes as create_boxes

sys.path.insert(0, "third_party/detectron2/projects/DensePose")
from densepose import add_densepose_config
from densepose.modeling.cse.utils import squared_euclidean_distance_matrix
from densepose.data.build import get_class_to_mesh_name_mapping
from densepose.modeling import build_densepose_embedder
from densepose.vis.densepose_outputs_vertex import get_xyz_vertex_embedding
from densepose.vis.base import MatrixVisualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_cse(config_path, weight_path):
    """Create a Densepose CSE instance

    Args
        config_path [str]: Path to Densepose config
        weight_path [str]: Path to pretrained Densepose weights

    Returns (model, mesh_vertex_embeddings):
        model [GeneralizedRCNN]: Densepose network
        mesh_vertex_embeddings [Dict(str, Tensor(npts, 16))]: Canonical mesh vertex embeddings
    """
    config = get_cfg()
    add_densepose_config(config)
    config.merge_from_file(config_path)
    config.MODEL.WEIGHTS = weight_path
    model = build_model(config) # returns torch.nn.Module
    DetectionCheckpointer(model).load(config.MODEL.WEIGHTS)

    embedder = build_densepose_embedder(config)
    class_to_mesh_name = get_class_to_mesh_name_mapping(config)
    mesh_vertex_embeddings = {}
    for mesh_name in class_to_mesh_name.values():
        if embedder.has_embeddings(mesh_name):
            mesh_vertex_embeddings[mesh_name] = embedder(mesh_name).to(device)
    return model, mesh_vertex_embeddings


def run_cse(model, mesh_vertex_embeddings, image, mask, mesh_name="smpl_27554"):
    """Runs Densepose model to compute features

    Args:
        model
        embedder
        mesh_vertex_embeddings
        image
        mask
        mesh_name

    Returns (clst_verts, image_bgr, embedding, embedding_norm, bbox):
        clst_verts
        image_bgr
        embedding
        embedding_norm
        bbox
    """
    H, W, _ = image.shape

    # Pad
    H_ = (1 + H // 32) * 32
    W_ = (1 + W // 32) * 32
    image_tmp = np.zeros((H_, W_, 3), dtype=np.uint8)
    mask_tmp = np.zeros((H_, W_), dtype=np.uint8)
    image_tmp[:H, :W] = image
    mask_tmp[:H, :W] = mask
    image = image_tmp
    mask = mask_tmp

    # Preprocess image and bbox
    yid, xid = np.where(mask > 0)
    xmin, xmax, ymin, ymax = np.min(xid), np.max(xid), np.min(yid), np.max(yid)
    center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
    length = ((xmax - xmin) // 2, (ymax - ymin) // 2)
    bbox = [
        max(0, center[0] - length[0]),
        max(0, center[1] - length[1]),
        min(W_, center[0] + length[0]),
        min(H_, center[1] + length[1]),
    ]
    image_raw = image
    image = torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1)[None] # 1, 3, H_, W_
    image = (image - model.pixel_mean) / model.pixel_std
    pred_boxes = create_boxes(torch.tensor([bbox], dtype=torch.float32, device=device)) # 4, 2

    # Densepose inference
    model.eval()
    with torch.no_grad():
        features = model.backbone(image)
        features = [features[f] for f in model.roi_heads.in_features]
        features = [model.roi_heads.decoder(features)]
        features_dp = model.roi_heads.densepose_pooler(features, [pred_boxes])
        densepose_head_outputs = model.roi_heads.densepose_head(features_dp)
        densepose_predictor_outputs = model.roi_heads.densepose_predictor(densepose_head_outputs)
        coarse_segm_resized = densepose_predictor_outputs.coarse_segm[0]
        embedding_resized = densepose_predictor_outputs.embedding[0]

    # Use input mask
    x0, y0, x1, y1 = bbox
    mask_box = mask[y0:y1, x0:x1] # H_, W_
    mask_box = torch.tensor(mask_box, dtype=torch.float32, device=device)[None, None] # 1, 1, H_, W_
    mask_box = F.interpolate(mask_box, coarse_segm_resized.shape[1:3], mode="bilinear")[0, 0] > 0

    # Find the closest match in the cropped/resized coordinate
    clst_verts_pad = torch.zeros(H_, W_, dtype=torch.int64, device=device)
    clst_verts_box = torch.zeros(mask_box.shape, dtype=torch.int64, device=device)
    all_embeddings = embedding_resized[:, mask_box].t()
    assign_mat = squared_euclidean_distance_matrix(all_embeddings, mesh_vertex_embeddings[mesh_name])
    clst_verts_box[mask_box] = assign_mat.argmin(dim=1)

    clst_verts_box = F.interpolate(clst_verts_box[None, None].float(), (y1-y0, x1-x0), mode="nearest")[0, 0].long()
    clst_verts_pad[y0:y1, x0:x1] = clst_verts_box

    # Output embedding
    embedding = embedding_resized * mask_box.float()[None]

    # Embedding norm
    embedding_norm = torch.norm(embedding, p=2, dim=0)
    embedding_norm_pad = torch.zeros(H, W, dtype=torch.float32, device=device)
    embedding_norm_box = F.interpolate(embedding_norm[None, None], (y1-y0, x1-x0), mode="bilinear")[0, 0]
    embedding_norm_pad[y0:y1, x0:x1] = embedding_norm_box
    embedding_norm = embedding_norm_pad[:H, :W]
    embedding_norm = F.interpolate(embedding_norm[None, None], (H, W), mode="bilinear")[0, 0]

    embedding = embedding.cpu().numpy()
    embedding_norm = embedding_norm.cpu().numpy()

    # Visualization
    embed_map = get_xyz_vertex_embedding(mesh_name, device)
    vis = (torch.clip(embed_map[clst_verts_pad], 0, 1) * 255.0).cpu().numpy()
    mask_visualizer = MatrixVisualizer(inplace=False, cmap=cv2.COLORMAP_JET, val_scale=1.0, alpha=0.7)
    image_bgr = mask_visualizer.visualize(image_raw, mask, vis, [0, 0, W_, H_])

    image_bgr = image_bgr[:H, :W]
    image_bgr = cv2.resize(image_bgr, (W, H))
    clst_verts = clst_verts_pad[:H, :W]
    clst_verts = F.interpolate(clst_verts[None, None].float(), (H, W), mode="nearest")[0, 0].long()
    clst_verts = clst_verts.cpu().numpy()

    return clst_verts, image_bgr, embedding, embedding_norm, bbox


def compute_dpdir(data_path, seqname, is_human, max_size=1333):
    """Compute Densepose features for a RGB image sequence

    Args
        data_path [str]: Path to database/DAVIS
        seqname [str]: Seqname in {data_path}/JPEGImages
        is_human [bool]: Whether to use human or quadruped Densepose network
        max_size [int]: Resize image and mask to some empirical size before Densepose compute
    """
    imgdir = f"{data_path}/JPEGImages/Full-Resolution/{seqname}"
    maskdir = f"{data_path}/Annotations/Full-Resolution/{seqname}"
    dpdir = f"{data_path}/Densepose_out/Full-Resolution/{seqname}"
    if os.path.exists(dpdir):
        raise RuntimeError(f"dpdir {dpdir} already exists, cannot compute Densepose feats")
    os.mkdir(dpdir)

    # Compute Densepose
    if is_human:
        config_path = "third_party/detectron2/projects/DensePose/configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml"
        weight_path = "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x/250713061/model_final_1d3314.pkl"
        mesh_name = "smpl_27554"
    else:
        config_path = "third_party/detectron2/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k.yaml"
        weight_path = "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_CA_finetune_4k/253498611/model_final_6d69b7.pkl"
        mesh_name = "sheep_5004"

    predictor_dp, mesh_vertex_embeddings = create_cse(config_path, weight_path)

    print(f"Computing dpdir for {imgdir}...")
    for i, path in enumerate(tqdm.tqdm(sorted(glob.glob(f"{imgdir}/*.jpg")))):
        image = cv2.imread(path)
        mask = cv2.imread(path.replace("JPEGImages", "Annotations").replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE)
        H, W = image.shape[:2]

        # Recompute mask
        mask = mask / np.sort(np.unique(mask))[1]
        occluder = (mask == 255)
        mask[occluder] = 0
    
        # Resize to some empirical size
        if H > W:
            H_rszd, W_rszd = max_size, max_size * W // H
        else:
            H_rszd, W_rszd = max_size * H // W, max_size
        image_rszd = cv2.resize(image, (W_rszd, H_rszd))
        mask_rszd = cv2.resize(mask.astype(np.float32), (W_rszd, H_rszd)).astype(np.uint8)

        # Run Densepose
        clst_verts, image_bgr, embedding, embedding_norm, bbox = run_cse(
            predictor_dp, embedder, mesh_vertex_embeddings, image_rszd, mask_rszd, mesh_name=mesh_name
        )

        # Resize to original size
        bbox[0] *= W / clst_verts.shape[1]
        bbox[1] *= H / clst_verts.shape[0]
        bbox[2] *= W / clst_verts.shape[1]
        bbox[3] *= H / clst_verts.shape[0]
        np.savetxt(f"{dpdir}/bbox-{i:05d}.txt", bbox)

        clst_verts = cv2.resize(clst_verts, (W, H), interpolation=cv2.INTER_NEAREST)
        # Assume max 10k/200 max
        clst_verts = (clst_verts / 50).astype(np.float32)
        write_pfm(f"{dpdir}/{i:05d}.pfm", clst_verts)

        embedding = embedding.reshape((-1, embedding.shape[-1]))
        write_pfm(f"{dpdir}/feat-{i:05d}.pfm", embedding)

        vis = image_rszd
        alpha_mask = 0.8 * (mask_rszd > 0)[..., None]
        mask_result = vis * (1 - alpha_mask) + image_bgr * alpha_mask
        cv2.imwrite(f"{dpdir}/vis-{i:05d}.jpg", mask_result)

if __name__ == "__main__":
    compute_dpdir("database/DAVIS", "cat-pikachiu00", False)
    compute_dpdir("database/DAVIS", "cat-pexel20015", False)

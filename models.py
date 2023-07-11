import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pytorch3d.transforms import rotation_conversions as transforms
from torchvision import models
from torchvision.transforms.functional import resize

from banmo_utils import banmo
from geom_utils import zero_to_rest_bone, zero_to_rest_dpose, gl_projection
from render_utils import softras_render_mesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Banmo Feed Forward Model

class BanmoFeedForward(torch.nn.Module):
    """Feed forward banmo model"""
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        # CNN Regressors
        if opts.pose_and_bone_input == "rgb":
            n_channels_pose_and_bone = 3
        elif opts.pose_and_bone_input == "dpfeat":
            n_channels_pose_and_bone = 16
        elif opts.pose_and_bone_input == "rgb_dpfeat":
            n_channels_pose_and_bone = 19
        else:
            raise ValueError(f"Invalid pose_and_bone_input '{opts.pose_and_bone_input}'")

        pose_code_dim = (opts.n_bones - 1) * 3 if opts.pose_code_dim == 0 else opts.pose_code_dim

        self.pose_bone_regressor = Regressor(
            opts.regressor_type, opts.prefix_type,
            inp_dim=n_channels_pose_and_bone,
            out_dim=opts.pose_multiplex_count * opts.pose_feat_dim + pose_code_dim,
        ).to(device)
        
        if opts.env_code_input == "rgb":
            n_channels_env_code = 3
        elif opts.env_code_input == "dpfeat":
            n_channels_env_code = 16
        elif opts.env_code_input == "rgb_dpfeat":
            n_channels_env_code = 19
        else:
            raise ValueError(f"Invalid env_code_input '{opts.env_code_input}'")

        self.env_regressor = Regressor(
            opts.regressor_type, "conv",
            inp_dim=n_channels_env_code,
            out_dim=opts.env_code_dim,
        ).to(device)
        
        # Rotation decoder: Cp => M x (rot 6 | tra 3 | score 1)
        kwargs = json.loads(opts.rotation_decoder_kwargs)
        if opts.rotation_decoder_type == "banmo":
            self.rt_decoder = BanmoRotationDecoder(n_channels=opts.root_pose_dim, **kwargs).to(device)
        elif opts.rotation_decoder_type == "continuous_6d":
            self.rt_decoder = Cont6DRotationDecoder(n_channels=opts.root_pose_dim, **kwargs).to(device)
        else:
            raise ValueError(f"Invalid rotation_decoder_type '{opts.rotation_decoder_type}'")

        # Bone transform decoder: Cp => B bones * (rot 9 | tra 3)
        kwargs = json.loads(opts.bone_transform_decoder_kwargs)
        if opts.pose_code_dim == 0:
            bones_rst, bone_rts_rst = zero_to_rest_bone(banmo(), banmo().bones) # B, 10 | B, 12
            self.bone_transform_decoder = AngleBoneTransformDecoder(bone_rts_rst, **kwargs).to(device)
        elif opts.bone_transform_decoder_type == "banmo":
            bones_rst, bone_rts_rst = zero_to_rest_bone(banmo(), banmo().bones) # B, 10 | B, 12
            self.bone_transform_decoder = BanmoBoneTransformDecoder(bone_rts_rst, **kwargs).to(device)
        elif opts.bone_transform_decoder_type == "continuous_6d":
            n_bones = banmo().opts.num_bones
            self.bone_transform_decoder = Cont6DBoneTransformDecoder(n_bones, **kwargs).to(device)
        else:
            raise ValueError(f"Invalid bone_transform_decoder_type '{opts.bone_transform_decoder_type}'")

        # Temporal encoders: T * C => T * C
        def make_temporal_encoder(temporal_encoder_type, n_channels, kwargs, *, mplex=False, n_mplex=None):
            n_times = 2 * opts.temporal_radius + 1

            if temporal_encoder_type == "conv":
                if mplex:
                    out = MultiplexConvTemporalEncoder(n_mplex=n_mplex, n_channels=n_channels, **kwargs)
                else:
                    out = ConvTemporalEncoder(n_channels=n_channels, **kwargs)
            elif temporal_encoder_type == "mlp" and not mplex:
                out = MLPTemporalEncoder(n_channels=n_channels, n_times=n_times, **kwargs)
            elif temporal_encoder_type == "transformer":
                if mplex:
                    out = MultiplexTransformerTemporalEncoder(
                        n_mplex=n_mplex, n_channels=n_channels, n_times=n_times, **kwargs
                    )
                else:
                    out = TransformerTemporalEncoder(n_channels=n_channels, n_times=n_times, **kwargs)
            else:
                if mplex:
                    raise ValueError(f"Temporal encoder type '{opts.temporal_encoder_type}' can't multiplex")
                else:
                    raise ValueError(f"Invalid temporal encoder type '{opts.temporal_encoder_type}'")

            return out.to(device)

        kwargs = json.loads(opts.temporal_encoder_kwargs)
        self.pose_temporal_encoder = make_temporal_encoder(
            opts.temporal_encoder_type, opts.pose_feat_dim, kwargs,
            mplex=True, n_mplex=opts.pose_multiplex_count,
        )
        self.code_temporal_encoder = make_temporal_encoder(opts.temporal_encoder_type, pose_code_dim, kwargs)
        self.env_temporal_encoder = make_temporal_encoder(opts.temporal_encoder_type, opts.env_code_dim, kwargs)
        
        self.regressor_params = [
            {"params": self.pose_bone_regressor.parameters()},
            {"params": self.env_regressor.parameters()},
            {"params": self.rt_decoder.parameters()},
            {"params": self.bone_transform_decoder.parameters()},
        ]

        self.temporal_params = [
            {"params": self.pose_temporal_encoder.parameters()},
            {"params": self.code_temporal_encoder.parameters()},
            {"params": self.env_temporal_encoder.parameters()},
            {"params": self.rt_decoder.parameters()},
            {"params": self.bone_transform_decoder.parameters()},
        ]

        self.params = [
            {"params": self.pose_bone_regressor.parameters()},
            {"params": self.env_regressor.parameters()},
            {"params": self.pose_temporal_encoder.parameters()},
            {"params": self.code_temporal_encoder.parameters()},
            {"params": self.env_temporal_encoder.parameters()},
            {"params": self.rt_decoder.parameters()},
            {"params": self.bone_transform_decoder.parameters()},
        ]

    def regressor_forward(self, rgb_imgs, dp_feats):
        """Perform single-frame regressor forward pass

        Args
            rgb_imgs [bs, Crgb, H, W]: RGB images for root pose and pose code prediction
            dp_feats [bs, Cdpf, H, W]: Densepose features for texture prediction

        Returns: (out_pose_mplex, out_prob_vals, out_pose_feat, out_pose_code, out_env) where
            out_pose_mplex [bs, M, Cr]: Output multiplexed root body pose features
            out_prob_mplex [bs, M]: Log probabilities assigned to each multiplexed output pose
            out_pose_feat [bs, M*Cf]: Output root body pose features
            out_pose_code [bs, Cp]: Output pose codes, to be decoded by bone_transform_decoder
            out_env [bs, Ce]: Output env codes, to be interpreted by nerf-coarse
        """
        opts = self.opts
        bs = dp_feats.shape[0]
        M = opts.pose_multiplex_count
        Cr = opts.root_pose_dim
        Cp = pose_code_dim = (opts.n_bones - 1) * 3 if opts.pose_code_dim == 0 else opts.pose_code_dim

        if opts.pose_and_bone_input == "rgb":
            pose_bone_inp = rgb_imgs # bs, 3, H, W
        elif opts.pose_and_bone_input == "dpfeat":
            pose_bone_inp = dp_feats # bs, 16, H, W
        elif opts.pose_and_bone_input == "rgb_dpfeat":
            pose_bone_inp = torch.cat([rgb_imgs, dp_feats], dim=1) # bs, 19, H, W
        else:
            raise ValueError(f"Invalid pose_and_bone_input '{opts.pose_and_bone_input}'")

        if opts.env_code_input == "rgb":
            env_inp = rgb_imgs # bs, 3, H, W
        elif opts.env_code_input == "dpfeat":
            env_inp = dp_feats # bs, 16, H, W
        elif opts.env_code_input == "rgb_dpfeat":
            env_inp = torch.cat([rgb_imgs, dp_feats], dim=1) # bs, 19, H, W
        else:
            raise ValueError(f"Invalid env_code_input '{opts.env_code_input}'")

        del rgb_imgs, dp_feats

        out_pose_feat_code = self.pose_bone_regressor(pose_bone_inp) # bs, M*Cf+Cp
        out_pose_feat = out_pose_feat_code[:, :-Cp] # bs, M*Cf
        out_pose_code = out_pose_feat_code[:, -Cp:] # bs, Cp
        out_env = self.env_regressor(env_inp) # bs, Ce
        
        out_pose_mplex_wt = out_pose_feat.view(bs, M, -1) # bs, M, Cf
        out_pose_mplex_raw = out_pose_mplex_wt[:, :, :Cr] # bs, M, Cr
        out_prob_mplex = out_pose_mplex_wt[:, :, Cr] # bs, M

        out_pose_mplex = self.rt_decoder(out_pose_mplex_raw) # bs, M, 12

        return out_pose_mplex, out_prob_mplex, out_pose_feat, out_pose_code, out_env

    def temporal_forward(self, raw_pose_feat, raw_code_feat, raw_env_feat):
        """Performs multi-frame temporal encoder forawrd pass

        Args
            raw_pose_feat [..., T, M*Cf]: Raw root body pose features, from regressor
            raw_code_feat [..., T, Cp]: Raw pose code features, from regressor
            raw_env_feat [..., T, Ce]: Raw env code features, from regressor

        Returns: (out_pose, out_bone, out_angle, out_env) where
            out_pose [..., T, 12]: Output root body poses
            out_bone [..., T, B*12]: Output bone transforms
            out_angle [..., T, J, 3]: Output joint angles
            out_env [..., T, Ce]: Output env codes, to be interpreted by nerf-coarse
        """
        prefix_shape = raw_pose_feat.shape[:-2]
        T, _ = raw_pose_feat.shape[-2:]
        M = self.opts.pose_multiplex_count
        Cr = self.opts.root_pose_dim
        Cf = raw_pose_feat.shape[-1] // M
        Cp = raw_code_feat.shape[-1]
        Ce = raw_env_feat.shape[-1]
        raw_pose_feat = raw_pose_feat.reshape(-1, T, M, Cf) # bs, T, M, Cf
        raw_code_feat = raw_code_feat.reshape(-1, T, Cp) # bs, T, Cp
        raw_env_feat = raw_env_feat.reshape(-1, T, Ce) # bs, T, Ce

        out_pose_raw = self.pose_temporal_encoder(raw_pose_feat)[:, :, :Cr] # bs, T, Cr
        out_pose = self.rt_decoder(out_pose_raw); del out_pose_raw # bs, T, 12
        out_code = self.code_temporal_encoder(raw_code_feat) # bs, T, Cp
        out_bone, out_angle = self.bone_transform_decoder(out_code); del out_code # bs, T, B*12 | bs, T, J, 3
        out_env = self.env_temporal_encoder(raw_env_feat) # bs, T, Ce

        out_pose = out_pose.view(prefix_shape + (T, 12))
        out_bone = out_bone.view(prefix_shape + (T, -1))
        out_angle = out_angle.view(prefix_shape + (T, -1, 3))
        out_env = out_env.view(prefix_shape + (T, Ce))
        return out_pose, out_bone, out_angle, out_env

    def weight_mplex(self, x, mplex_wt, *, mode="softmax"):
        """Weight an input by the given multiplex weights

        Args
            x [..., M, ...]: Arbitary-shape input to weight
            mplex_wt [..., M]: M-dimensional vector of multiplex weights
            mode [str]: Multiplexing mode

        Returns
            x_out [..., ...]: Multiplex weighted output
        """
        assert x.shape[:mplex_wt.ndim] == mplex_wt.shape, \
            f"Expected x's prefix shape to match mplex_wt's shape, but found " \
            f"x shape '{x.shape}' and mplex_wt shape '{mplex_wt.shape}'"
        mplex_dim = mplex_wt.ndim - x.ndim - 1 # Dimension of M

        if mode == "softmax":
            mplex_wt_softmax = torch.nn.functional.softmax(mplex_wt, dim=-1) # ..., M
            for i in range(x.ndim - mplex_wt.ndim):
                mplex_wt_softmax = mplex_wt_softmax[..., None]

            x_out = mplex_wt_softmax * x; del mplex_wt_softmax # ..., M, ...
            x_out = torch.sum(x_out, dim=mplex_dim) # ..., ...

        elif mode == "max":
            mplex_wt_argmax = torch.argmax(mplex_wt, dim=-1)[..., None]; del mplex_wt # ..., 1
            target_shape = mplex_wt.shape[:-1] + (1,) + x.shape[mplex_wt.ndim:] # ..., 1, ...
            for i in range(x.ndim - mplex_wt.ndim):
                mplex_wt_argmax = mplex_wt_argmax[..., None]
            mplex_wt_argmax = mplex_wt_argmax.expand(target_shape) # ..., 1, ...

            x_out = torch.gather(x, mplex_dim, mplex_wt_argmax); del mplex_wt_argmax # ..., 1, ...
            x_out = x_out.squeeze(mplex_dim) # ..., ...

        else:
            raise ValueError(f"Invalid weight_multiplex mode '{mode}'")

        return x_out

    def forward(self, mode, *args, **kwargs):
        if mode == "regressor":
            return self.regressor_forward(*args, **kwargs)
        elif mode == "temporal":
            return self.temporal_forward(*args, **kwargs)
        else:
            raise ValueError(f"Invalid forward mode '{mode}'")


# ===== Helpers

def convert_relu_to_swish(model):
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU) or isinstance(child, nn.LeakyReLU):
            setattr(model, name, Swish())
        else:
            convert_relu_to_swish(child)

class Swish(nn.Module):
    """Implementation of Swish activation function: https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class BanmoRTHead(nn.Module):
    def __init__(
        self, use_quat, D=8, W=256, in_channels_xyz=63, in_channels_dir=27, out_channels=3, skips=[4],
        raw_feat=False, init_beta=0.01, activation=Swish(), in_channels_code=0, vid_code=None
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_code = in_channels_code
        self.skips = skips
        self.use_xyz = False

        # Video code
        self.vid_code = vid_code
        if vid_code is not None:
            self.num_vid, self.num_codedim = self.vid_code.weight.shape
            in_channels_xyz += self.num_codedim
            self.rand_ratio = 1. # 1: fully random

        # XYZ encoding layers
        self.weights_reg = []
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
                self.weights_reg.append(f"xyz_encoding_{i + 1}")
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
                self.weights_reg.append(f"xyz_encoding_{i + 1}")
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, activation)
            setattr(self, f"xyz_encoding_{i + 1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # Direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir, W // 2),
            activation
        )

        # Output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(nn.Linear(W // 2, out_channels))
        self.raw_feat = raw_feat

        self.beta = torch.tensor([init_beta], dtype=torch.float32) # logbeta
        self.beta = nn.Parameter(self.beta)
        self.symm_ratio = 0
        self.rand_ratio = 0

        # Use quaternion when estimating full rotation
        # Use exponential map when estiating delta rotation
        self.use_quat = use_quat
        if self.use_quat:
            self.num_output = 7
        else:
            self.num_output = 6
        self.scale_t = 0.1

        self.reinit(gain=1)

    def reinit(self, gain=1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.weight, "data"):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5 * gain))
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, 0], dim=-1)
        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], dim=-1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        sigma = self.sigma(xyz_)
        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding = self.dir_encoding(torch.cat([xyz_encoding_final, input_dir], dim=-1))
        rgb = self.rgb(dir_encoding)
        
        if self.raw_feat:
            x = rgb
        else:
            rgb = rgb.sigmoid()
            x = torch.cat([rgb, sigma], dim=-1)

        bs = x.shape[0]
        rts = x.view(-1, self.num_output) # bs*B, x
        B = rts.shape[0] // bs

        tmat = rts[:, 0:3] * self.scale_t # bs*B, 3 

        if self.use_quat:
            rquat = F.normalize(rts[:, 3:7], p=2, dim=-1) # bs*B, 4
            rmat = transforms.quaternion_to_matrix(rquat) # bs*B, 3, 3
        else:
            rot = rts[:, 3:6] # bs*B, 3
            rmat = transforms.so3_exponential_map(rot) # bs*B, 3, 3
        rmat = rmat.view(-1, 9) # bs*B, 9

        rts = torch.cat([rmat, tmat], dim=-1).view(bs, 1, -1) # bs*B, 1, 12
        return rts


# ===== Conv Temporal Encoder

class ConvTemporalEncoderBlock(nn.Module):
    """Building block for learned 1D convolutional temporal encoder backbone
    
    Args
        n_channels [int]: Number of feature channels
        kernel_size [int]: Size of Conv1d kernel
        use_groupnorm [bool]: Whether to use group norm or batch norm
        n_groups [int]: Number of groups to use with groupnorm, if applicable
        use_residual [bool]: Whether to use residual connection
    """
    def __init__(self, n_channels, kernel_size, use_groupnorm=False, n_groups=8, use_residual=False):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.use_groupnorm = use_groupnorm
        self.n_groups = n_groups
        self.use_residual = use_residual

        if use_groupnorm:
            self.norm1 = nn.GroupNorm(n_groups, n_channels)
            self.norm2 = nn.GroupNorm(n_groups, n_channels)
        else:
            self.norm1 = nn.BatchNorm1d(n_channels)
            self.norm2 = nn.BatchNorm1d(n_channels)

        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.swish = Swish()

    def forward(self, x_in):
        """x_in [batch_size, n_channels, n_times]"""
        x = x_in # bs, C, T

        x = self.norm1(x) # bs, C, T
        x = self.swish(x) # bs, C, T
        x = self.conv1(x) # bs, C, T
        x = self.norm2(x) # bs, C, T
        x = self.swish(x) # bs, C, T
        x = self.conv2(x) # bs, C, T

        if self.use_residual:
            x = x + x_in # bs, C, T
        return x


class ConvTemporalEncoder(nn.Module):
    """Single-multiplex temporal encoder with a learned 1D convolutional backbone
    
    Args
        n_blocks [int]: Number of convolutional blocks
        n_channels [int]: Dimensionality of output pose code / env code
        kernel_size [int]: Size of Conv1d kernel
        use_groupnorm [bool]: Whether to use group norm or batch norm
        n_groups [int]: Number of groups to use with groupnorm, if applicable
    """
    def __init__(self, n_blocks=3, n_channels=128, kernel_size=7, use_groupnorm=False, n_groups=8):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.use_groupnorm = use_groupnorm
        self.n_groups = n_groups

        self.blocks = nn.ModuleList([
            ConvTemporalEncoderBlock(n_channels, kernel_size, use_groupnorm, n_groups)
            for i in range(self.n_blocks)
        ])

    def forward(self, x):
        """x [batch_size, n_times, n_channels]"""
        x = x.permute(0, 2, 1) # bs, C, T
        for block in self.blocks:
            x = block(x) # bs, C, T
        x = x.permute(0, 2, 1) # bs, T, C
        return x


class MultiplexConvTemporalEncoder(nn.Module):
    """Multiplexed temporal encoder with a learned 1D convolutional backbone

    Args
        n_mplex [int]: Number of camera multiplexes
        n_blocks [int]: Number of convolutional blocks
        n_channels [int]: Dimensionality of output pose code / env code
        kernel_size [int]: Size of Conv1d kernel
        use_groupnorm [bool]: Whether to use group norm or batch norm
        n_groups [int]: Number of groups to use with groupnorm, if applicable
    """
    def __init__(self, n_mplex, n_blocks=3, n_channels=128, kernel_size=7, use_groupnorm=False, n_groups=8):
        super().__init__()
        self.n_mplex = n_mplex
        self.n_blocks = n_blocks
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.use_groupnorm = use_groupnorm
        self.n_groups = n_groups

        self.blocks = nn.ModuleList([
            ConvTemporalEncoderBlock(n_mplex * n_channels, kernel_size, use_groupnorm, n_groups)
            for i in range(self.n_blocks)
        ])
        self.linear_fc = nn.Linear(n_mplex * n_channels, n_channels)

    def forward(self, x):
        """x [batch_size, n_times, n_mplex, n_channels]"""
        bs, T, M, C = x.shape
        x = x.reshape(bs, T, M*C).permute(0, 2, 1) # bs, M*C, T
        for block in self.blocks:
            x = block(x) # bs, M*C, T
        x = x.permute(0, 2, 1) # bs, T, M*C
        x = self.linear_fc(x) # bs, T, C
        return x


# ===== MLP Temporal Encoder

class MLPTemporalEncoderBlock(nn.Module):
    """Building block for learned MLP temporal encoder backbone

    Args
        n_channels [int]: Number of feature channels
        n_times [int]: Total width of temporal window
        use_groupnorm [bool]: Whether to use group norm or batch norm
        n_groups [int]: Number of groups to use with groupnorm, if applicable
        use_residual [bool]: Whether to use residual connection
    """
    def __init__(self, n_channels=128, n_times=1, use_groupnorm=False, n_groups=None, use_residual=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.use_groupnorm = use_groupnorm
        self.n_groups = n_groups
        self.use_residual = use_residual

        if use_groupnorm:
            self.norm1 = nn.GroupNorm(n_groups, n_channels)
            self.norm2 = nn.GroupNorm(n_groups, n_channels)
        else:
            self.norm1 = nn.BatchNorm1d(n_channels)
            self.norm2 = nn.BatchNorm1d(n_channels)
           
        self.linear1 = nn.Linear(n_channels * n_times, n_channels * n_times)
        self.linear2 = nn.Linear(n_channels * n_times, n_channels * n_times)
        self.swish = Swish()

    def forward(self, x_in):
        """x_in [batch_size, n_channels, n_times]"""
        bs, C, T = x_in.shape
        x = x_in # bs, C, T
        
        x = self.norm1(x) # bs, C, T
        x = self.swish(x) # bs, C, T
        x = self.linear1(x.view(bs, C * T)).view(bs, C, T) # bs, C, T
        x = self.norm2(x) # bs, C, T
        x = self.swish(x) # bs, C, T
        x = self.linear2(x.view(bs, C * T)).view(bs, C, T) # bs, C, T
        
        if self.use_residual:
            x = x + x_in # bs, C, T
        return x


class MLPTemporalEncoder(nn.Module):
    """Single-multiplex temporal encoder with a learned MLP backbone

    Args
        n_blocks [int]: Number of linear blocks
        n_channels [int]: Dimensionality of output pose code / env code
        n_times [int]: Total width of temporal window
        use_groupnorm [bool]: Whether to use group norm or batch norm
        n_groups [int]: Number of groups to use with groupnorm, if applicable
    """
    def __init__(self, n_blocks=2, n_channels=128, n_times=1, use_groupnorm=False, n_groups=8):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_channels = n_channels
        self.n_times = n_times
        self.use_groupnorm = use_groupnorm
        self.n_groups = n_groups

        self.blocks = nn.ModuleList([
            MLPTemporalEncoderBlock(n_channels, n_times, use_groupnorm, n_groups)
            for i in range(self.n_blocks)
        ])

    def forward(self, x):
        """x [batch_size, n_times, n_channels]"""
        x = x.permute(0, 2, 1) # bs, C, T

        for block in self.blocks:
            x = block(x) # bs, C, T
        
        x = x.permute(0, 2, 1) # bs, T, C
        return x


# ===== Transformer Temporal Encoder

class SelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer with projection at the end,
    designed for camera-multiplexed inputs. Derived from github.com/karpathy/minGPT

    Args
        n_channels [int]: Embedding dimensionality
        n_heads [int]: Number of attention heads
        attn_pdrop [float]: Dropout probability for attention keys
        resid_pdrop [float]: Dropout probability for final output
    """
    def __init__(self, n_channels=128, n_heads=4, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

        # Key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(n_channels, 3 * n_heads * n_channels) 
        # Output projection
        self.c_proj = nn.Linear(n_heads * n_channels, n_channels)
        # Regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        """x [batch_size, n_times, n_channels]"""
        bs, T, C = x.shape
        assert C == self.n_channels, f"Expected C '{C}' to be equal to self.n_channels '{self.n_channels}'"

        # Calculate query, key, values for all heads in batch; move head forward next to batch dim
        att = self.c_attn(x); del x # bs, T, 3*nh*C
        nhC = self.n_heads * self.n_channels
        q = att[:, :, 0 * nhC : 1 * nhC].view(bs, T, self.n_heads, self.n_channels).transpose(1, 2) # bs, nh, T, C
        k = att[:, :, 1 * nhC : 2 * nhC].view(bs, T, self.n_heads, self.n_channels).transpose(1, 2) # bs, nh, T, C
        v = att[:, :, 2 * nhC : 3 * nhC].view(bs, T, self.n_heads, self.n_channels).transpose(1, 2) # bs, nh, T, C

        # Self-attention: (bs, nh, T, C) x (bs, nh, C, T) => (bs, nh, T, T)
        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(C)); del q, k # bs, nh, T, T
        att = F.softmax(att, dim=-1) # bs, nh, T, T
        att = self.attn_dropout(att) # bs, nh, T, T

        # Compute values and reassemble head outputs: (bs, nh, T, T) x (bs, nh, T, C) => (bs, nh, T, C)
        out = torch.matmul(att, v); del att, v # bs, nh, T, C
        out = out.transpose(1, 2).reshape(bs, T, nhC) # bs, T, nh*C

        # Output projection
        out = self.c_proj(out) # bs, T, C
        out = self.resid_dropout(out) # bs, T, C
        return out


class TransformerBlock(nn.Module):
    """Transformer block, derived from https://github.com/karpathy/minGPT

    Args
        n_channels [int]: Embedding dimensionality
        n_heads [int]: Number of attention heads
        attn_pdrop [float]: Dropout probability for attention keys
        resid_pdrop [float]: Dropout probability for final output
    """
    def __init__(self, n_channels=128, n_heads=4, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

        self.norm = nn.LayerNorm(n_channels)
        self.attn = SelfAttention(n_channels, n_heads, attn_pdrop, resid_pdrop)

        self.c_fc = nn.Linear(n_channels, 4 * n_channels)
        self.swish = Swish()
        self.c_proj = nn.Linear(4 * n_channels, n_channels)
        self.drop = nn.Dropout(resid_pdrop)
        self.mlpf = nn.Sequential(self.c_fc, self.swish, self.c_proj, self.drop)

    def forward(self, x):
        """x [batch_size, n_times, n_channels]"""
        # Use parallel attention layers from PaLM: https://arxiv.org/abs/2204.02311
        nx = self.norm(x) # bs, T, C
        x = x + self.mlpf(nx) + self.attn(nx) # bs, T, C
        return x


class TransformerTemporalEncoder(nn.Module):
    """Single-multiplex transformer temporal encoder that attends to embeddings
    within a fixed-size temporal window. Derived from github.com/karpathy/minGPT

    Args
        n_times [int]: Size of temporal window
        n_layers [int]: Number of transformer layers
        n_channels [int]: Embedding dimensionality
        n_heads [int]: Number of attention heads
        embed_pdrop [float]: Dropout probability for input embeddings
        attn_pdrop [float]: Dropout probability for attention keys
        resid_pdrop [float]: Dropout probability for final output
    """
    def __init__(
        self, n_times, *, n_layers=6, n_channels=128, n_heads=6,
        attn_pdrop=0.1, resid_pdrop=0.1, embed_pdrop=0.1,
    ):
        super().__init__()
        self.n_times = n_times
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embed_pdrop = embed_pdrop

        self.pos_enc = nn.Embedding(n_times, n_channels)
        self.drop = nn.Dropout(embed_pdrop)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_channels, n_heads, attn_pdrop, resid_pdrop)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(n_channels)

        self.transformer = nn.Sequential(self.pos_enc, self.drop, self.blocks, self.norm)
        
        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x):
        """x [batch_size, n_times, n_channels]: Per-frame embeddings"""
        bs, T, C = x.shape
        assert T == self.n_times, f"Expected T '{T}' and self.n_times '{self.n_times}' to be equal"
        
        # Compute positional encodings (trainable per frame)
        pos = torch.arange(0, T, dtype=torch.int64, device=device) # T,
        pos_emb = self.pos_enc(pos)[None] # 1, T, C

        # Transformer forward pass
        x = self.drop(x + pos_emb); del pos_emb # bs, T, C
        for block in self.blocks:
            x = block(x) # bs, T, C
        x = self.norm(x) # bs, T, C

        return x


class MultiplexTransformerTemporalEncoder(nn.Module):
    """Multiplexed transformer temporal encoder that attends to embeddings
    within a fixed-size temporal window. Derived from github.com/karpathy/minGPT
    Aims to model an order-invariant set aggregation function across multiplex axis

    Args
        n_times [int]: Temporal window size
        n_mplex [int]: Number of camera multiplexes
        n_layers [int]: Number of transformer layers
        n_channels [int]: Embedding dimensionality
        n_heads [int]: Number of attention heads
        embed_pdrop [float]: Dropout probability for input embeddings
        attn_pdrop [float]: Dropout probability for attention keys
        resid_pdrop [float]: Dropout probability for final output
    """
    def __init__(
        self, n_times, n_mplex, *, n_layers=6, n_channels=128, n_heads=6,
        attn_pdrop=0.1, resid_pdrop=0.1, embed_pdrop=0.1,
    ):
        super().__init__()
        self.n_times = n_times
        self.n_mplex = n_mplex
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embed_pdrop = embed_pdrop

        self.pos_enc = nn.Embedding(n_times, n_channels)
        self.drop = nn.Dropout(embed_pdrop)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_channels, n_heads, attn_pdrop, resid_pdrop)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(n_channels)

        self.transformer = nn.Sequential(self.pos_enc, self.drop, self.blocks, self.norm)

        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x):
        """x [batch_size, n_times, n_mples, n_channels]: Per-frame multiplexed embeddings"""
        bs, T, M, C = x.shape
        assert T == self.n_times, f"Expected T '{T}' and self.n_times '{self.n_times}' to be equal"

        # Compute positional embeddings (trainable per frame)
        pos = torch.arange(0, T, dtype=torch.int64, device=device) # T,
        pos_emb = self.pos_enc(pos)[None, :, None] # 1, T, 1, C

        # Transformer forward pass
        x = self.drop(x + pos_emb) # bs, T, M, C
        x = x.view(bs, T * M, C) # bs, T*M, C
        for block in self.blocks:
            x = block(x) # bs, T*M, C
        x = self.norm(x) # bs, T*M, C
        x = x.view(bs, T, M, C) # bs, T, M, C

        # Pooling along multiplex dimension. Models an order-invariant set
        # aggregation function: https://arxiv.org/abs/1810.00825
        x = torch.sum(x, dim=-2) # bs, T, C

        return x


# ===== CNN Regressors

class Conv(nn.Module):
    """Conv2d layer with optional batchnorm and Swish, for stacked hourglass

    Args
        inp_dim [int]: Input channels
        out_dim [int]: Output channels
        kernel_size [int]: Conv2d kernel size
        stride [int]: Conv2d Stride
        bn [bool]: Whether to apply batchnorm after conv
        swish [bool]: Whether to apply Swish after conv and batchnorm
    """
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, swish=True):
        super().__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.swish = Swish() if swish else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        """x [batch_size, Ci, H, W]"""
        x = self.conv(x) # bs, Co, H_, W_
        if self.bn is not None:
            x = self.bn(x) # bs, Co, H_, W_
        if self.swish is not None:
            x = self.swish(x) # bs, Co, H_, W_
        return x


class Residual(nn.Module):
    """Residual module with skip connection and three Conv2d layers, for stacked hourglass

    Args
        inp_dim [int]: Input channels
        out_dim [int]: Output channels
    """
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.swish = Swish()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, out_dim // 2, 1, swish=False)
        self.bn2 = nn.BatchNorm2d(out_dim // 2)
        self.conv2 = Conv(out_dim // 2, out_dim // 2, 3, swish=False)
        self.bn3 = nn.BatchNorm2d(out_dim // 2)
        self.conv3 = Conv(out_dim // 2, out_dim, 1, swish=False)
        self.skip = Conv(inp_dim, out_dim, 1, swish=False)
        self.need_skip = (inp_dim != out_dim)

    def forward(self, x):
        """x [batch_size, Ci, H, W]"""
        if self.need_skip:
            residual = self.skip(x) # bs, Co, H, W
        else:
            residual = x # bs, Co, H, W
        x = self.conv1(self.swish(self.bn1(x))) # bs, Co/2, H, W
        x = self.conv2(self.swish(self.bn2(x))) # bs, Co/2, H, W
        x = self.conv3(self.swish(self.bn3(x))) # bs, Co, H, W
        x = x + residual # bs, Co, H, W
        return x


class Hourglass(nn.Module):
    """Hourglass module with lower and upper branch, for stacked hourglass

    Args
        n_layers [int]: Number of recursive hourglass layers
        n_channels [int]: Input and output channels
    """
    def __init__(self, n_layers, n_channels):
        super().__init__()
        self.up1 = Residual(n_channels, n_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(n_channels, n_channels)
        self.low2 = Hourglass(n_layers - 1, n_channels) if n_layers > 1 else Residual(n_channels, n_channels)
        self.low3 = Residual(n_channels, n_channels)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        """x [batch_size, n_channels, H, W]"""
        up1 = self.up1(x) # bs, C, H, W
        pool1 = self.pool1(x); del x # bs, C, H/2, W/2
        low1 = self.low1(pool1); del pool1 # bs, C, H/2, W/2
        low2 = self.low2(low1); del low1 # bs, C, H/2, W/2
        low3 = self.low3(low2); del low2 # bs, C, H/2, W/2
        up2 = self.up2(low3); del low3 # bs, C, H, W
        return up1 + up2


class StackedHourglass(nn.Module):
    """Stacked hourglass network for pose estimation

    Args
        n_blocks [int]: Number of hourglass blocks
        img_channels [int]: Number of channels in input image
        inp_dim [int]: Number of input channels per stacked hourglass block
        out_dim [int]: Number of output channels per stacked hourglass block
    """
    def __init__(self, n_blocks, img_channels, inp_dim=256, out_dim=16):
        super().__init__()
        self.n_blocks = n_blocks

        self.preprocess = nn.Sequential(
            Conv(img_channels, 64, 7, 2, bn=True, swish=True),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim),
        )

        self.hourglass = nn.ModuleList([Hourglass(3, inp_dim) for i in range(n_blocks)])
        self.feats = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, swish=True)
            ) for i in range(n_blocks)
        ])
        self.preds = nn.ModuleList([
            Conv(inp_dim, out_dim, 1, bn=False, swish=False) for i in range(n_blocks)
        ])
        self.merge_feats = nn.ModuleList([
            Conv(inp_dim, inp_dim, 1, bn=False, swish=False) for i in range(n_blocks - 1)
        ])
        self.merge_preds = nn.ModuleList([
            Conv(out_dim, inp_dim, 1, bn=False, swish=False) for i in range(n_blocks - 1)
        ])

    def forward(self, x):
        """x [batch_size, img_channels, 224, 224]"""
        x = self.preprocess(x) # bs, Ci, 56, 56
        combined_preds = []
        for i in range(self.n_blocks):
            hg = self.hourglass[i](x) # bs, Ci, 56, 56
            feats = self.feats[i](hg); del hg # bs, Ci, 56, 56
            preds = self.preds[i](feats) # bs, Co, 56, 56
            combined_preds.append(preds)
            if i < self.n_blocks - 1:
                x = x + self.merge_feats[i](feats); del feats # bs, Ci, 56, 56
                x = x + self.merge_preds[i](preds); del preds # bs, Ci, 56, 56
        out = torch.cat(combined_preds, dim=1) # bs, N*Co, 56, 56
        return out


class Regressor(nn.Module):
    """Regressor with image encoder backbone.
    Given (16,224,224) densepose features, outputs pose code.
    Given (3,224,224) cropped rgb images, outputs environment code.

    Args
        regressor_type [string]: Type of CNN backbone to use
        inp_dim [int]: Number of channels in image input (rgb 3, dpfeat 16, rgb_dpfeat 19)
        out_dim [int]: Number of channels in output code (pose_code Cp, env_code Ce)
        pretrained [bool]: Whether to load pretrained weights from Posenet
        posenet_path [string]: Path to pretrained Posenet weights
    """
    def __init__(
        self, regressor_type, prefix_type, inp_dim=16, out_dim=128,
    ):
        super(Regressor, self).__init__()
        self.regressor_type = regressor_type
        self.prefix_type = prefix_type
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        if "resnet" in self.regressor_type:
            if self.regressor_type == "resnet18":
                self.resnet = models.resnet18()
            elif self.regressor_type == "resnet34":
                self.resnet = models.resnet34()
            elif self.regressor_type == "resnet50":
                self.resnet = models.resnet50()
            elif self.regressor_type == "resnext50_32x4d":
                self.resnet = models.resnext50_32x4d()
            else:
                raise NotImplementedError(f"CNN regressor type {self.regressor_type} is not available")

            # Instead of ReLU, use Swish activation
            convert_relu_to_swish(self.resnet)
            self.resnet.swish = Swish()
            
            # Instead of fully-connected output layer, use Conv1d to reduce dimension from 512 to out_dim
            self.cnn_output_size = self.resnet.fc.in_features
            del self.resnet.fc
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.cnn_output_size, self.out_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_dim),
                Swish(),
            )
            self.forward = self.resnet_forward

        else:
            raise ValueError(f"Invalid CNN regressor type '{self.regressor_type}'")
        
        # Preprocessing network
        # Instead of 3-dim RGB input, allow an arbitrary number of channels as input
        if self.prefix_type == "stacked_hourglass":
            self.posenet = StackedHourglass(8, inp_dim, 256, 16)
            self.resnet.conv1 = nn.Conv2d(8 * 16, 64, kernel_size=3, stride=1, padding=1, bias=False)
        elif self.prefix_type == "conv":
            self.resnet.conv1 = nn.Conv2d(inp_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise ValueError(f"Invalid prefix type '{self.prefix_type}'")


    def resnet_forward(self, x):
        """x [batch_size, img_channels, H, W]"""
        x = resize(x, (224, 224)) # bs, C, 224, 224

        # Preprocessing
        if self.prefix_type == "stacked_hourglass":
            x = self.posenet(x) # bs, 128, 56, 56
            x = self.resnet.conv1(x) # bs, 64, 56, 56
        elif self.prefix_type == "conv":
            x = self.resnet.conv1(x) # bs, 64, 112, 112
            x = self.resnet.bn1(x) # bs, 64, 112, 112
            x = self.resnet.swish(x) # bs, 64, 112, 112
            x = self.resnet.maxpool(x) # bs, 64, 56, 56

        # ResNet layers
        x = self.resnet.layer1(x) # bs, 64, 56, 56
        x = self.resnet.layer2(x) # bs, 128, 28, 28
        x = self.resnet.layer3(x) # bs, 256, 14, 14
        x = self.resnet.layer4(x) # bs, 512, 7, 7
        
        # After this point, we differ from standard ResNet forward pass. First
        # reduce dimensionality of output, then use maxpool instead of ResNet's avgpool
        x = self.conv1(x) # bs, out_dim, 7, 7
        x = F.max_pool2d(x, 4, 4) # bs, out_dim, 1, 1
        x = x[..., 0, 0] # bs, out_dim
        return x


# ===== Rotation Decoders

class BanmoRotationDecoder(nn.Module):
    """Uses BANMo RTHead (which is an MLP directly regressing quaternions from latent vectors)
    to decode rotation latent codes into 3x3 rotation and 1x3 translation

    Args
        n_channels [int]: Number of channels in input code
        pretrained [bool]: Whether to use pretrained BANMo RTHead weights
        posenet_path [str]: Path to pretrained PoseNet
        frozen [bool]: Whether to freeze BANMo's RTHead weights
    """
    def __init__(self, *, n_channels=128, pretrained=True, posenet_path=None, frozen=True):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = 2 * n_channels
        self.pretrained = pretrained
        self.posenet_path = posenet_path
        self.frozen = frozen

        self.rt_decoder = BanmoRTHead(
            use_quat=True, D=1, W=self.hidden_dim,
            in_channels_xyz=self.n_channels, in_channels_dir=0,
            out_channels=7, raw_feat=True
        )

        # Load pretrained RT decoder
        if pretrained:
            state_dict = torch.load(posenet_path)

            rthead_key_prefix = "module.nerf_root_rts.1."
            rthead_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if k.startswith(rthead_key_prefix):
                    rthead_state_dict[k[len(rthead_key_prefix):]] = v

            self.rt_decoder.load_state_dict(rthead_state_dict)

        # Freeze RT decoder weights
        if frozen:
            for param in self.rt_decoder.parameters():
                param.requires_grad = False
        else:
            for param in self.rt_decoder.parameters():
                param.requires_grad = True
        
    def forward(self, x):
        """x [..., n_channels]: Input codes"""
        prefix_shape = x.shape[:-1]
        out = self.rt_decoder(x) # ..., 1, 12
        out = out[..., 0, :] # ..., 12
        return out


class Cont6DRotationDecoder(nn.Module):
    """Uses MLP and 6D continuous rotation representation to decode rotation latent codes
    into 3x3 rotation and 1x3 translation. Derived from Zhou et al: https://arxiv.org/abs/1812.07035

    Args
        n_layers [int]: Number of fully connected layers before final downsampling layer
        n_channels [int]: Number of channels in input code
        use_groupnorm [bool]: Whether to use groupnorm or batchnorm
        n_groups [int]: Number of groups to use for groupnorm
        use_residual [bool]: Whether to include residual connection
    """
    def __init__(self, *, n_layers=0, n_channels=128, use_groupnorm=False, n_groups=8, use_residual=True):
        super().__init__()
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.use_groupnorm = use_groupnorm
        self.n_groups = n_groups
        self.use_residual = use_residual

        if use_groupnorm:
            self.norm_layers = nn.ModuleList([nn.GroupNorm(n_groups, n_channels) for i in range(self.n_layers)])
        else:
            self.norm_layers = nn.ModuleList([nn.BatchNorm1d(n_channels) for i in range(self.n_layers)])

        self.linear_layers = nn.ModuleList([nn.Linear(n_channels, n_channels) for i in range(self.n_layers)])

        self.swish = Swish()
        self.linear_fc = nn.Linear(n_channels, 9) # 6D rotation | 3D translation

    def forward(self, x_in):
        """x_in [..., C]: Input codes"""
        prefix_shape = x_in.shape[:-1]
        x_in = x_in.reshape(-1, x_in.shape[-1]) # -1, C
        x = x_in # -1, C

        for i in range(self.n_layers):
            x = self.norm_layers[i](x) # -1, C
            x = self.swish(x) # -1, C
            x = self.linear_layers[i](x) # -1, C
        
        if self.use_residual:
            x = x + x_in # -1, C
        x = self.linear_fc(x) # -1, 9
        
        # Decode 6D rotation and 3D translation
        rot6, tra3 = x[:, :6], x[:, 6:]; del x # -1, 6 | -1, 3
        a1, a2 = rot6[:, :3], rot6[:, 3:]; del rot6 # -1, 3 | -1, 3
        b1 = F.normalize(a1, dim=-1); del a1 # -1, 3
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1; del a2 # -1, 3
        b2 = F.normalize(b2, dim=-1) # -1, 3
        b3 = torch.cross(b1, b2, dim=-1) # -1, 3
        rot3x3 = torch.stack([b1, b2, b3], dim=-2).view(-1, 9); del b1, b2, b3 # -1, 9

        out = torch.cat([rot3x3, tra3], dim=-1); del rot3x3, tra3 # -1, 12
        out = out.view(prefix_shape + (12,)) # ..., 12
        return out


# ===== Multiplexed Rotation Decoders

class MultiplexBanmoRotationDecoder(nn.Module):
    """Uses BANMo's RTHead (which is an MLP directly regressing quaternions from latent vectors)
    to decode rotation latent codes into multiple 3x3 rotation and 1x3 translation predictions

    Args
        n_mplex [int]: NUmber of multiplex predictions to output
        n_channels [int]: Number of channels in input code
        hidden_dim [int]: Number of neurons in hidden dim of banmo RTHead
        pretrained [bool]: Whether to use pretrained BANMo RTHead weights
        posenet_path [str]: Path to pretrained PoseNet
        frozen [bool]: Whether to freeze BANMo's RTHead weights
    """
    def __init__(
        self, n_mplex, *, n_channels=128, hidden_dim=256,
        pretrained=True, posenet_path="mesh_material/posenet/quad.pth", frozen=True
    ):
        super().__init__()
        self.n_mplex = n_mplex
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.pretrained = pretrained
        self.posenet_path = posenet_path
        self.frozen = frozen

        self.rt_decoders = nn.ModuleList([])
        for i in range(self.n_mplex):
            rt_decoder = BanmoRTHead(
                use_quat=True, D=1, W=hidden_dim,
                in_channels_xyz=n_channels, in_channels_dir=0,
                out_channels=7, raw_feat=True
            )

            # Load pretrained RT decoder
            if pretrained:
                state_dict = torch.load(posenet_path)

                rthead_key_prefix = "module.nerf_root_rts.1."
                rthead_state_dict = OrderedDict()

                for k, v in state_dict.items():
                    if k.startswith(rthead_key_prefix):
                        rthead_state_dict[k[len(rthead_key_prefix):]] = v

                rt_decoder.load_state_dict(rthead_state_dict)

            # Freeze RT decoder weights
            if frozen:
                for param in rt_decoder.parameters():
                    param.requires_grad = False

            self.rt_decoders.append(rt_decoder)

        # Linear expansion layer for multiplexer
        self.mplex_expand = nn.Linear(n_channels, n_mplex * (n_channels + 1))
        
    def forward(self, x):
        """Run rotation decoder forward pass

        Args
            x [..., n_channels]: Input codes

        Returns: (out, mplex_wt) where
            out [..., M, 12]: Multiplexed 3x3 rotation and 1x3 translation matrices
            mplex_wt [..., M]: Multiplexing weights across matrices
        """
        prefix_shape = x.shape[:-1]
        M = self.n_mplex
        C = x.shape[-1]
        x_with_wt = self.mplex_expand(x).view(prefix_shape + (M, C + 1)) # ..., M, C + 1
        x, mplex_wt = x_with_wt[..., :-1], x_with_wt[..., -1]; del x_with_wt # ..., M, C | ..., M

        out = []
        for i, rt_decoder in enumerate(self.rt_decoders):
            out_part = rt_decoder(x[..., i, :]).view(prefix_shape + (12,)) # ..., 12
            out.append(out_part)
        out = torch.stack(out, dim=-2) # ..., M, 12
        return out, mplex_wt


class MultiplexCont6DRotationDecoder(nn.Module):
    """Uses MLP and 6D continuous rotation representation to decode rotation latent codes
    into multiple 3x3 rotation and 1x3 translation predictions.
    Derived from Zhou et al: https://arxiv.org/abs/1812.07035

    Args
        n_mplex [int]: Number of multiplex predictions to output
        n_layers [int]: Number of fully connected layers before final downsampling layer
        n_channels [int]: Number of channels in input code
        use_groupnorm [bool]: Whether to use groupnorm or batchnorm
        n_groups [int]: Number of groups to use for groupnorm
        use_residual [bool]: Whether to include residual connection
    """
    def __init__(self, n_mplex, *, n_layers=0, n_channels=128, use_groupnorm=False, n_groups=8, use_residual=True):
        super().__init__()
        self.n_mplex = n_mplex
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.use_groupnorm = use_groupnorm
        self.n_groups = n_groups
        self.use_residual = use_residual

        if use_groupnorm:
            self.norm_layers = nn.ModuleList([nn.GroupNorm(n_groups, n_channels) for i in range(self.n_layers)])
        else:
            self.norm_layers = nn.ModuleList([nn.BatchNorm1d(n_channels) for i in range(self.n_layers)])

        self.linear_layers = nn.ModuleList([nn.Linear(n_channels, n_channels) for i in range(self.n_layers)])

        self.swish = Swish()
        self.linear_fc = nn.Linear(n_channels, n_mplex * (6 + 3 + 1))

    def forward(self, x_in):
        """Perform MLP forward pass

        Args
            x_in [..., C]: Input codes

        Returns: (out, mplex_wt) where
            out [..., M, 12]: Multiplexed 3x3 rotation and 1x3 translation matrices
            mplex_wt [..., M]: Multiplexing weights across matrices
        """
        prefix_shape = x_in.shape[:-1]
        M = self.n_mplex
        C = x_in.shape[-1]
        x_in = x_in.view(-1, C) # -1, C
        x = x_in # -1, C

        for i in range(self.n_layers):
            x = self.norm_layers[i](x) # -1, C
            x = self.swish(x) # -1, C
            x = self.linear_layers[i](x) # -1, C
        
        if self.use_residual:
            x = x + x_in
        x = self.linear_fc(x).view(-1, M, 10) # -1, M, 10
        
        # Decode 6D rotation, 3D translation, and 1D multiplex weights
        rot6, tra3, mplex_wt = x[:, :, 0:6], x[:, :, 6:9], x[:, :, 9]; del x # -1, M, 6 | -1, M, 3 | -1, M
        a1, a2 = rot6[:, :, :3], rot6[:, :, 3:]; del rot6 # -1, M, 3 | -1, M, 3
        b1 = F.normalize(a1, dim=-1); del a1 # -1, M, 3
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1; del a2 # -1, M, 3
        b2 = F.normalize(b2, dim=-1) # -1, M, 3
        b3 = torch.cross(b1, b2, dim=-1) # -1, M, 3
        rot3x3 = torch.stack([b1, b2, b3], dim=-2).view(-1, self.n_mplex, 9); del b1, b2, b3 # -1, M, 9

        out = torch.cat([rot3x3, tra3], dim=-1); del rot3x3, tra3 # -1, M, 12
        out = out.view(prefix_shape + (M, 12)) # ..., M, 12
        mplex_wt = mplex_wt.view(prefix_shape + (M,)) # ..., M
        return out, mplex_wt

# ===== Bone Transform Decoder

class AngleBoneTransformDecoder(nn.Module):
    """Decodes joint angles into banmo bone transforms

    Args
        bone_rts_rst [B, 12]: Rest bone transforms
        frozen [bool]: Whether to freeze BANMo's RTHead weights
    """
    def __init__(self, bone_rts_rst, *, frozen=True):
        super().__init__()
        self.model = [banmo()] # store in list to hide params from nn.Module
        self.bone_rts_rst = bone_rts_rst
        self.frozen = frozen

        # Freeze weights
        if isinstance(self.model[0].nerf_body_rts, torch.nn.Sequential):
            for param in self.model[0].nerf_body_rts[1].parameters():
                param.requires_grad = (not frozen)
        else:
            for param in self.model[0].nerf_body_rts.parameters():
                param.requires_grad = (not frozen)

    def forward(self, x):
        """Run bone transform decoder forward pass given pose codes

        Args
            x [..., Cj]: Input joint angles

        Returns
            bone_rts_fw [..., B, 12]: Output 3x3 rotation and 1x3 translation matrix
            joint_angles [..., J, 3]: Output spherical joint angles per link
        """
        prefix_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]) # -1, Cp
        T = x.shape[0]
        B = self.bone_rts_rst.shape[0]
        J = B - 1
        assert x.shape[-1] == J * 3

        # Per-frame bone transforms corresponding to provided pose codes, without rest pose correction
        # In the old codebase, model.nerf_body_rts is a nn.Sequential that first
        # passes query_times through self.pose_code to compute ground-truth pose codes.
        # In the new codebase, model.nerf_body_rts is a module where the forward function
        # passes query_times through self.pose_code to compute ground-truth pose codes.
        # In both codebases, pose codes are passed to an RTHead to decode into per-bone rigid
        # transforms, which are used to deform input points by linear blend skinning.
        # We directly pass our pose codes into RTHead
        if isinstance(self.model[0].nerf_body_rts, torch.nn.Sequential):
            # Old cat70 RTHead impl: nn.Sequential(idx -> pose_code, pose_code -> bone_rts)
            bone_rts_fw = self.model[0].nerf_body_rts[1](x).view(T, B, 12) # -1, B, 12
            joint_angles = torch.zeros(T, J, 3, dtype=torch.float32, device=device) # -1, J, 3
        else:
            # New dog80 SkelHead impl: .pose_code() and .forward_decode()
            vid = None
            bone_rts_fw, joint_angles = self.model[0].nerf_body_rts.forward_decode(
                None, None, joint_angles=x
            ) # -1, B*12 | -1, J*3
            bone_rts_fw = bone_rts_fw.view(T, B, 12) # -1, B, 12
            joint_angles = joint_angles.view(T, J, 3) # -1, J, 3

        # Per-frame bone transforms, with rest pose correction
        bone_rts_rst = self.bone_rts_rst.detach()[None, :, :].expand(T, -1, -1) # -1, B, 12
        bone_rts_fw = zero_to_rest_dpose(bone_rts_fw, bone_rts_rst) # -1, B, 12
        bone_rts_fw = bone_rts_fw.view(prefix_shape + (B * 12,)) # ..., B*12
        joint_angles = joint_angles.view(prefix_shape + (J, 3)) # ..., J, 3
        return bone_rts_fw, joint_angles


class BanmoBoneTransformDecoder(nn.Module):
    """Uses BANMo's nerf_body_rts and nerf_root_rts to decode pose codes into
    bone rigid transforms, parameterized by B 3x3 rotations and 1x3 translations

    Args
        bone_rts_rst [B, 12]: Rest bone transforms
        frozen [bool]: Whether to freeze BANMo's RTHead weights
    """
    def __init__(self, bone_rts_rst, *, frozen=True):
        super().__init__()
        self.model = [banmo()] # store in list to hide params from nn.Module
        self.bone_rts_rst = bone_rts_rst
        self.frozen = frozen

        # Freeze weights
        if isinstance(self.model[0].nerf_body_rts, torch.nn.Sequential):
            for param in self.model[0].nerf_body_rts[1].parameters():
                param.requires_grad = (not frozen)
        else:
            for param in self.model[0].nerf_body_rts.parameters():
                param.requires_grad = (not frozen)
 
    def forward(self, x):
        """Run bone transform decoder forward pass given pose codes

        Args
            x [..., Cp]: Input pose codes

        Returns
            bone_rts_fw [..., B, 12]: Output 3x3 rotation and 1x3 translation matrix
            joint_angles [..., J, 3]: Output spherical joint angles per link
        """
        prefix_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]) # -1, Cp
        T = x.shape[0]
        B = self.bone_rts_rst.shape[0]
        J = B - 1

        # Per-frame bone transforms corresponding to provided pose codes, without rest pose correction
        # In the old codebase, model.nerf_body_rts is a nn.Sequential that first
        # passes query_times through self.pose_code to compute ground-truth pose codes.
        # In the new codebase, model.nerf_body_rts is a module where the forward function
        # passes query_times through self.pose_code to compute ground-truth pose codes.
        # In both codebases, pose codes are passed to an RTHead to decode into per-bone rigid
        # transforms, which are used to deform input points by linear blend skinning.
        # We directly pass our pose codes into RTHead
        if isinstance(self.model[0].nerf_body_rts, torch.nn.Sequential):
            # Old cat70 RTHead impl: nn.Sequential(idx -> pose_code, pose_code -> bone_rts)
            bone_rts_fw = self.model[0].nerf_body_rts[1](x).view(T, B, 12) # -1, B, 12
            joint_angles = torch.zeros(T, J, 3, dtype=torch.float32, device=device) # -1, J, 3
        else:
            # New dog80 SkelHead impl: .pose_code() and .forward_decode()
            vid = None
            bone_rts_fw, joint_angles = self.model[0].nerf_body_rts.forward_decode(x, None) # -1, B*12 | -1, J*3
            bone_rts_fw = bone_rts_fw.view(T, B, 12) # -1, B, 12
            joint_angles = joint_angles.view(T, J, 3) # -1, J, 3

        # Per-frame bone transforms, with rest pose correction
        bone_rts_rst = self.bone_rts_rst.detach()[None, :, :].expand(T, -1, -1) # -1, B, 12
        bone_rts_fw = zero_to_rest_dpose(bone_rts_fw, bone_rts_rst) # -1, B, 12
        bone_rts_fw = bone_rts_fw.view(prefix_shape + (B * 12,)) # ..., B*12
        joint_angles = joint_angles.view(prefix_shape + (J, 3)) # ..., J, 3
        return bone_rts_fw, joint_angles


class Cont6DBoneTransformDecoder(nn.Module):
    """Uses MLP and 6D continuous rotation representation to decode pose latent codes
    into B 3x3 rotation and 1x3 translations representing per-bone rigid transforms
    Derived from Zhou et al: https:/arxiv.org/abs/1812.07035

    Args
        n_bones [int]: Number of bones to output
        n_layers [int]: Number of fully connected layers before final downsampling layer
        n_channels [int]: Number of channels in input code
        use_groupnorm [bool]: Whether to use groupnorm or batchnorm
        n_groups [int]: Number of groups to use for groupnorm
        use_residual [bool]: Whether to include residual connection
    """
    def __init__(self, n_bones, *, n_layers=0, n_channels=128, use_groupnorm=False, n_groups=8, use_residual=True):
        super().__init__()
        self.n_bones = n_bones
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.use_groupnorm = use_groupnorm
        self.n_groups = n_groups
        self.use_residual = use_residual

        if use_groupnorm:
            self.norm_layers = nn.ModuleList([nn.GroupNorm(n_groups, n_channels) for i in range(self.n_layers)])
        else:
            self.norm_layers = nn.ModuleList([nn.BatchNorm1d(n_channels) for i in range(self.n_layers)])

        self.linear_layers = nn.ModuleList([nn.Linear(n_channels, n_channels) for i in range(self.n_layers)])

        self.swish = Swish()
        self.linear_fc = nn.Linear(n_channels, n_bones * 9)

    def forward(self, x_in):
        """Perform MLP forward pass

        Args
            x_in [..., C]: Input codes

        Returns
            out [..., B, 12]: 3x3 rotation and 1x3 matrices representing bone transforms
        """
        x_shape = x_in.shape # ..., C
        B = self.n_bones
        C = x_in.shape[-1]
        x_in = x_in.view(-1, C) # -1, C
        x = x_in # -1, C

        for i in range(self.n_layers):
            x = self.norm_layers[i](x) # -1, C
            x = self.swish(x) # -1, C
            x = self.linear_layers[i](x) # -1, C

        if self.use_residual:
            x = x + x_in
        x = self.linear_fc(x).view(-1, B, 9) # -1, B, 9
        
        # Decode 6D rotation and 3D translation
        rot6, tra3 = x[:, :, 0:6], x[:, :, 6:9]; del x # -1, B, 6 | -1, B, 3
        a1, a2 = rot6[:, :, :3], rot6[:, :, 3:]; del rot6 # -1, B, 3 | -1, B, 3
        b1 = F.normalize(a1, dim=-1); del a1 # -1, B, 3
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1; del a2 # -1, B, 3
        b2 = F.normalize(b2, dim=-1) # -1, B, 3
        b3 = torch.cross(b1, b2, dim=-1) # -1, B, 3
        rot3x3 = torch.stack([b1, b2, b3], dim=-2).view(-1, B, 9); del b1, b2, b3 # -1, B, 9

        out = torch.cat([rot3x3, tra3], dim=-1); del rot3x3, tra3 # -1, B*12
        out = out.view(x_shape[:-1] + (B * 12,)) # ..., B*12
        return out

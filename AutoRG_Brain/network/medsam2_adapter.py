import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from network.generic_UNet import Generic_UNet
from network.initialization import InitWeights_He
from network.neural_network import SegmentationNetwork


class _Fallback3DEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(embed_dim // 2, affine=True),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(embed_dim, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MedSAM2SegAdapter(SegmentationNetwork):
    """
    Drop-in segmentation backbone that preserves AutoRG-Brain interfaces.

    Output contract:
        forward(x, modal) -> (anatomy_logits, abnormal_logits)
        anatomy_logits shape: [B, num_classes_anatomy, D, H, W]
        abnormal_logits shape: [B, num_classes_abnormal, D, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        num_classes_anatomy: int = 96,
        num_classes_abnormal: int = 2,
        embed_dim: int = 128,
        deep_supervision: bool = True,
        use_medsam2_encoder: bool = False,
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        num_conv_per_stage: int = 2,
        max_num_features: int = 320,
        medsam2_repo_root: str = None,
        medsam2_config: str = None,
        medsam2_ckpt: str = None,
    ):
        super().__init__()

        self.conv_op = nn.Conv3d
        self.num_classes_anatomy = num_classes_anatomy
        self.num_classes_abnormal = num_classes_abnormal
        self.num_classes = num_classes_anatomy
        self.in_channels = in_channels

        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.medsam2_model = None
        self.medsam2_enabled = False
        self.medsam2_load_error = None

        # maps arbitrary MRI channels to 3-channel MedSAM2 image input
        self.input_proj_2d = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        self.encoder_name = "fallback_3d"
        self.encoder = _Fallback3DEncoder(in_channels, embed_dim)
        self._encoder_out_channels = embed_dim

        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [(2, 2, 2)] * 5
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3, 3)] * (len(pool_op_kernel_sizes) + 1)

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.num_pool = len(self.pool_op_kernel_sizes)
        self.input_shape_must_be_divisible_by = np.prod(np.vstack(self.pool_op_kernel_sizes), axis=0)

        if self.num_pool < 2:
            raise ValueError("pool_op_kernel_sizes must have at least 2 stages for decoder construction")

        self.conv_pad_sizes = [[1 if k == 3 else 0 for k in krnl] for krnl in self.conv_kernel_sizes]

        self.decoder_base_features = max(32, min(int(embed_dim), max_num_features))
        self.num_conv_per_stage = max(1, int(num_conv_per_stage))

        if use_medsam2_encoder:
            repo_root = medsam2_repo_root or os.environ.get("AUTORG_MEDSAM2_REPO")
            if repo_root is None:
                repo_root = str(Path(__file__).resolve().parents[2] / "external" / "MedSAM2")

            cfg_name = medsam2_config or os.environ.get("AUTORG_MEDSAM2_CONFIG", "configs/sam2.1_hiera_t512.yaml")
            ckpt_path = medsam2_ckpt or os.environ.get("AUTORG_MEDSAM2_CKPT")

            try:
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)

                from sam2.build_sam import build_sam2  # type: ignore

                self.medsam2_model = build_sam2(
                    config_file=cfg_name,
                    ckpt_path=ckpt_path,
                    mode="eval",
                    apply_postprocessing=False,
                )

                if os.environ.get("AUTORG_MEDSAM2_FREEZE", "1") == "1":
                    for parameter in self.medsam2_model.parameters():
                        parameter.requires_grad = False

                medsam2_hidden = int(self.medsam2_model.image_encoder.neck.d_model)
                self._encoder_out_channels = medsam2_hidden

                self.medsam2_enabled = True
                self.encoder_name = "medsam2"
            except ImportError as e:
                self.medsam2_model = None
                self.medsam2_enabled = False
                self.encoder_name = "fallback_3d"
                self.medsam2_load_error = f"import_error: {str(e)}"
                print(f"WARNING: MedSAM2 import failed: {e}. Using fallback 3D encoder.")
            except FileNotFoundError as e:
                self.medsam2_model = None
                self.medsam2_enabled = False
                self.encoder_name = "fallback_3d"
                self.medsam2_load_error = f"file_not_found: {str(e)}"
                print(f"WARNING: MedSAM2 checkpoint or config not found: {e}. Using fallback 3D encoder.")
            except Exception as e:
                self.medsam2_model = None
                self.medsam2_enabled = False
                self.encoder_name = "fallback_3d"
                self.medsam2_load_error = f"unknown_error: {str(e)}"
                print(f"WARNING: MedSAM2 initialization failed with unexpected error: {e}. Using fallback 3D encoder.")

        self._build_decoder_from_autorg(max_num_features=max_num_features)
        self._build_encoder_bridge()

        self.input_proj_2d.apply(InitWeights_He(1e-2))
        if not self.medsam2_enabled:
            self.encoder.apply(InitWeights_He(1e-2))
        self.encoder_entry_adapter.apply(InitWeights_He(1e-2))
        self.encoder_stage_adapters.apply(InitWeights_He(1e-2))
        self.encoder_downsamplers.apply(InitWeights_He(1e-2))
        self.encoder_to_bottleneck_down.apply(InitWeights_He(1e-2))
        self.encoder_bottleneck_adapter.apply(InitWeights_He(1e-2))

    @staticmethod
    def _extract_output_channels(block: nn.Module) -> int:
        if hasattr(block, "output_channels"):
            return int(block.output_channels)
        if isinstance(block, nn.Sequential):
            for module in reversed(list(block)):
                if hasattr(module, "output_channels"):
                    return int(module.output_channels)
        raise RuntimeError("Cannot infer output_channels from decoder scaffold block")

    def _build_decoder_from_autorg(self, max_num_features: int):
        scaffold = Generic_UNet(
            self.in_channels,
            self.decoder_base_features,
            self.num_classes_anatomy,
            self.num_classes_abnormal,
            self.num_pool,
            self.num_conv_per_stage,
            2,
            nn.Conv3d,
            nn.InstanceNorm3d,
            {"eps": 1e-5, "affine": True},
            nn.Dropout3d,
            {"p": 0.0, "inplace": True},
            nn.LeakyReLU,
            {"negative_slope": 1e-2, "inplace": True},
            self._deep_supervision,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.pool_op_kernel_sizes,
            self.conv_kernel_sizes,
            False,
            True,
            True,
            max_num_features,
        )

        self.tu = scaffold.tu
        self.conv_blocks_localization = scaffold.conv_blocks_localization
        self.seg_outputs_anatomy = scaffold.seg_outputs_anatomy
        self.seg_outputs_abnormal = scaffold.seg_outputs_abnormal
        self.upscale_logits_ops = scaffold.upscale_logits_ops

        context_blocks = getattr(scaffold, "conv_blocks_context", None)
        if context_blocks is None:
            context_blocks = scaffold.conv_blocks_context_a
        self.encoder_stage_channels = [self._extract_output_channels(block) for block in context_blocks[:-1]]
        self.decoder_bottleneck_channels = self._extract_output_channels(context_blocks[-1])

    def _build_encoder_bridge(self):
        self.encoder_entry_adapter = nn.Conv3d(
            self._encoder_out_channels,
            self.encoder_stage_channels[0],
            kernel_size=1,
            bias=False,
        )

        self.encoder_downsamplers = nn.ModuleList()
        self.encoder_stage_adapters = nn.ModuleList()

        for stage_idx in range(1, len(self.encoder_stage_channels)):
            pool_ks = tuple(int(k) for k in self.pool_op_kernel_sizes[stage_idx - 1])
            in_channels = self.encoder_stage_channels[stage_idx - 1]
            out_channels = self.encoder_stage_channels[stage_idx]

            self.encoder_downsamplers.append(
                nn.Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size=pool_ks,
                    stride=pool_ks,
                    bias=False,
                )
            )
            self.encoder_stage_adapters.append(nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))

        last_pool = tuple(int(k) for k in self.pool_op_kernel_sizes[-1])
        last_stage_channels = self.encoder_stage_channels[-1]
        self.encoder_to_bottleneck_down = nn.Conv3d(
            last_stage_channels,
            last_stage_channels,
            kernel_size=last_pool,
            stride=last_pool,
            bias=False,
        )
        self.encoder_bottleneck_adapter = nn.Conv3d(
            last_stage_channels,
            self.decoder_bottleneck_channels,
            kernel_size=1,
            bias=False,
        )

    def _forward_medsam2_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, height, width = x.shape
        
        # Ensure MedSAM2 model device matches input device
        if self.medsam2_model is not None:
            medsam2_device = next(self.medsam2_model.parameters()).device
            if x.device != medsam2_device:
                raise RuntimeError(
                    f"Input tensor device {x.device} does not match MedSAM2 model device {medsam2_device}. "
                    f"Please move input to the same device as the model."
                )
        
        x2d = x.permute(0, 2, 1, 3, 4).reshape(batch_size * depth, x.shape[1], height, width)
        x2d = self.input_proj_2d(x2d)

        image_size = int(os.environ.get("AUTORG_MEDSAM2_INPUT_SIZE", int(self.medsam2_model.image_size)))
        if x2d.shape[-2:] != (image_size, image_size):
            x2d = F.interpolate(x2d, size=(image_size, image_size), mode="bilinear", align_corners=False)

        slice_chunk = int(os.environ.get("AUTORG_MEDSAM2_SLICE_CHUNK", "8"))
        feat_chunks = []
        with torch.set_grad_enabled(self.training and os.environ.get("AUTORG_MEDSAM2_FREEZE", "1") != "1"):
            for start in range(0, x2d.shape[0], max(1, slice_chunk)):
                end = min(x2d.shape[0], start + max(1, slice_chunk))
                # enforce math-only SDPA for broader CUDA compatibility
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    backbone_out = self.medsam2_model.forward_image(x2d[start:end])
                feat_chunks.append(backbone_out["vision_features"])

        feat_2d = torch.cat(feat_chunks, dim=0)

        feat_2d = F.interpolate(feat_2d, size=(height, width), mode="bilinear", align_corners=False)
        feat_3d = feat_2d.reshape(batch_size, depth, feat_2d.shape[1], height, width).permute(0, 2, 1, 3, 4).contiguous()
        return feat_3d

    def _forward_encoder_bridge(self, features: torch.Tensor):
        x = self.encoder_entry_adapter(features)
        skips = [x]

        for stage_idx in range(len(self.encoder_downsamplers)):
            x = self.encoder_downsamplers[stage_idx](x)
            x = self.encoder_stage_adapters[stage_idx](x)
            skips.append(x)

        bottleneck = self.encoder_to_bottleneck_down(skips[-1])
        bottleneck = self.encoder_bottleneck_adapter(bottleneck)
        return skips, bottleneck

    def forward(self, x: torch.Tensor, modal=None):
        if self.medsam2_enabled and self.medsam2_model is not None:
            features = self._forward_medsam2_features(x)
        else:
            features = self.encoder(x)

        skips, x_dec = self._forward_encoder_bridge(features)

        seg_outputs_anatomy = []
        seg_outputs_abnormal = []

        for u in range(len(self.tu)):
            x_dec = self.tu[u](x_dec)
            skip = skips[-(u + 1)]
            if x_dec.shape[2:] != skip.shape[2:]:
                x_dec = F.interpolate(x_dec, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x_dec = torch.cat((x_dec, skip), dim=1)
            x_dec = self.conv_blocks_localization[u](x_dec)
            seg_outputs_anatomy.append(self.seg_outputs_anatomy[u](x_dec))
            seg_outputs_abnormal.append(self.seg_outputs_abnormal[u](x_dec))

        if self._deep_supervision and self.do_ds:
            anatomy_out = tuple(
                [seg_outputs_anatomy[-1]]
                + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs_anatomy[:-1][::-1])]
            )
            abnormal_out = tuple(
                [seg_outputs_abnormal[-1]]
                + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs_abnormal[:-1][::-1])]
            )
            return anatomy_out, abnormal_out
        return seg_outputs_anatomy[-1], seg_outputs_abnormal[-1]

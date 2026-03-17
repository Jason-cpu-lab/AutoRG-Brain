import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

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
        medsam2_repo_root: str = None,
        medsam2_config: str = None,
        medsam2_ckpt: str = None,
    ):
        super().__init__()

        self.conv_op = nn.Conv3d
        self.input_shape_must_be_divisible_by = None
        self.num_classes_anatomy = num_classes_anatomy
        self.num_classes_abnormal = num_classes_abnormal
        self.num_classes = num_classes_anatomy

        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.medsam2_model = None
        self.medsam2_enabled = False
        self.medsam2_load_error = None

        # maps arbitrary MRI channels to 3-channel MedSAM2 image input
        self.input_proj_2d = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        self.encoder_name = "fallback_3d"
        self.encoder = _Fallback3DEncoder(in_channels, embed_dim)

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
                self.anatomy_head = nn.Conv3d(medsam2_hidden, num_classes_anatomy, kernel_size=1)
                self.abnormal_head = nn.Conv3d(medsam2_hidden, num_classes_abnormal, kernel_size=1)

                self.medsam2_enabled = True
                self.encoder_name = "medsam2"
            except Exception:
                self.medsam2_model = None
                self.medsam2_enabled = False
                self.encoder_name = "fallback_3d"
                self.medsam2_load_error = "failed_to_build_sam2"

        if not self.medsam2_enabled:
            self.anatomy_head = nn.Conv3d(embed_dim, num_classes_anatomy, kernel_size=1)
            self.abnormal_head = nn.Conv3d(embed_dim, num_classes_abnormal, kernel_size=1)

    def _forward_medsam2_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, height, width = x.shape
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

    def forward(self, x: torch.Tensor, modal=None):
        if self.medsam2_enabled and self.medsam2_model is not None:
            features = self._forward_medsam2_features(x)
        else:
            features = self.encoder(x)

        anatomy_logits = self.anatomy_head(features)
        abnormal_logits = self.abnormal_head(features)

        if self._deep_supervision and self.do_ds:
            return (anatomy_logits,), (abnormal_logits,)
        return anatomy_logits, abnormal_logits

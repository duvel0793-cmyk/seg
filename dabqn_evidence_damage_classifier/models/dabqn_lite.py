from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.convnext_backbone import ConvNeXtV2Backbone
from models.damage.three_scale_damage_branch import ThreeScaleDamageBranch
from models.necks.fpn import FPN
from models.necks.pixel_decoder import PixelDecoder
from models.query.building_query_decoder import BuildingQueryDecoder, MLP
from models.query.mask_head import QueryMaskHead


class DABQNLite(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        pixel_dim = int(model_cfg.get("pixel_dim", hidden_dim))
        num_queries = int(model_cfg.get("num_queries", 150))
        decoder_layers = int(model_cfg.get("query_decoder_layers", 4))
        decoder_heads = int(model_cfg.get("query_decoder_heads", 8))
        dropout = float(model_cfg.get("dropout", 0.1))
        branch_type = str(model_cfg.get("damage_branch_type", "mask_pool"))
        damage_head_type = str(model_cfg.get("damage_head_type", "corn"))
        crop_feature_size = int(model_cfg.get("crop_feature_size", 7))
        scale_factors = dict(model_cfg.get("damage_scale_factors", {"tight": 1.05, "context": 2.0, "neighborhood": 4.0}))

        self.use_gt_mask_warmup = bool(model_cfg.get("use_gt_mask_warmup", True))
        self.gt_warmup_epochs = int(model_cfg.get("gt_warmup_epochs", 3))
        self.localization_pre_only = bool(model_cfg.get("localization_pre_only", False))
        self.return_full_masks_train = bool(model_cfg.get("return_full_masks_train", True))

        self.backbone = ConvNeXtV2Backbone(
            backbone_name=str(model_cfg.get("backbone", "convnextv2_tiny.fcmae_ft_in22k_in1k")),
            in_channels=3,
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
        self.fpn = FPN(self.backbone.feature_channels, out_channels=hidden_dim)
        self.pixel_decoder = PixelDecoder(in_channels=hidden_dim, pixel_channels=pixel_dim)
        self.query_decoder = BuildingQueryDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            num_feature_levels=3,
            dropout=dropout,
        )
        self.class_head = nn.Linear(hidden_dim, 2)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.mask_head = QueryMaskHead(hidden_dim, pixel_dim)
        self.damage_branch = ThreeScaleDamageBranch(
            feature_dim=hidden_dim,
            query_dim=hidden_dim,
            hidden_dim=hidden_dim,
            branch_type=branch_type,
            head_type=damage_head_type,
            dropout=dropout,
            crop_feature_size=crop_feature_size,
            scale_factors=scale_factors,
        )

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, width, height = boxes.unbind(dim=-1)
        x1 = cx - (width * 0.5)
        y1 = cy - (height * 0.5)
        x2 = cx + (width * 0.5)
        y2 = cy + (height * 0.5)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _resolve_compute_post(self, *, stage: str, compute_post: bool | None) -> bool:
        if compute_post is not None:
            return bool(compute_post)
        if str(stage) == "localization":
            return not self.localization_pre_only
        return True

    def _resolve_return_full_masks(self, *, stage: str, return_full_masks: bool | None) -> bool:
        if return_full_masks is not None:
            return bool(return_full_masks)
        if str(stage) == "localization":
            return bool(self.return_full_masks_train)
        return True

    def forward_localization(
        self,
        batch: dict[str, Any],
        *,
        stage: str = "joint",
        compute_post: bool | None = None,
        return_full_masks: bool | None = None,
    ) -> dict[str, Any]:
        pre_image = batch["pre_image"]
        compute_post = self._resolve_compute_post(stage=stage, compute_post=compute_post)
        return_full_masks = self._resolve_return_full_masks(stage=stage, return_full_masks=return_full_masks)
        pre_feats = self.backbone(pre_image)
        pre_pyramid = self.fpn(pre_feats)
        post_pyramid = None
        if compute_post:
            post_image = batch["post_image"]
            post_feats = self.backbone(post_image)
            post_pyramid = self.fpn(post_feats)
        pixel_outputs = self.pixel_decoder(pre_pyramid)
        query_outputs = self.query_decoder(pixel_outputs["memory_levels"])
        query_features = query_outputs["query_features"]

        pred_logits = self.class_head(query_features)
        pred_boxes = self.box_head(query_features).sigmoid()
        mask_outputs = self.mask_head(query_features, pixel_outputs["pixel_feature"])
        pred_masks_lowres = mask_outputs["mask_logits"]
        outputs = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_masks_lowres": pred_masks_lowres,
            "query_features": query_features,
            "aux_query_features": query_outputs["aux_query_features"],
            "mask_embeddings": mask_outputs["mask_embeddings"],
            "pre_pyramid": pre_pyramid,
            "post_pyramid": post_pyramid,
        }
        if return_full_masks:
            outputs["pred_masks"] = F.interpolate(
                pred_masks_lowres,
                size=pre_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            outputs["pred_masks"] = None
        return outputs

    def _build_guided_queries(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, torch.Tensor]] | None,
        matches: list[tuple[torch.Tensor, torch.Tensor]] | None,
        *,
        epoch: int,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_masks = outputs.get("pred_masks")
        if pred_masks is None:
            pred_masks = outputs["pred_masks_lowres"]
            if targets is not None and targets and "masks" in targets[0]:
                target_size = targets[0]["masks"].shape[-2:]
                pred_masks = F.interpolate(pred_masks, size=target_size, mode="bilinear", align_corners=False)
        masks = pred_masks.sigmoid()
        boxes = outputs["pred_boxes"]
        if (
            not self.use_gt_mask_warmup
            or targets is None
            or matches is None
            or str(stage) not in {"damage", "joint"}
            or int(epoch) >= self.gt_warmup_epochs
        ):
            return masks, boxes
        guided_masks = masks.clone()
        guided_boxes = boxes.clone()
        for batch_index, (pred_indices, target_indices) in enumerate(matches):
            if pred_indices.numel() == 0:
                continue
            target_masks = targets[batch_index]["masks"][target_indices]
            target_boxes = targets[batch_index]["boxes_norm"][target_indices]
            guided_masks[batch_index, pred_indices] = target_masks
            guided_boxes[batch_index, pred_indices] = target_boxes
        return guided_masks, guided_boxes

    def forward_damage(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Any],
        *,
        targets: list[dict[str, torch.Tensor]] | None = None,
        matches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        epoch: int = 0,
        stage: str = "joint",
    ) -> dict[str, Any]:
        del batch
        if str(stage) == "localization":
            outputs["damage_logits"] = None
            outputs["damage_probabilities"] = None
            outputs["damage_pred_labels"] = None
            outputs["damage_binary_logits"] = None
            outputs["damage_severity_logits"] = None
            return outputs

        guided_masks, guided_boxes = self._build_guided_queries(outputs, targets, matches, epoch=epoch, stage=stage)
        damage_outputs = self.damage_branch(
            pre_pyramid=outputs["pre_pyramid"],
            post_pyramid=outputs["post_pyramid"],
            query_features=outputs["query_features"],
            masks=guided_masks,
            boxes=guided_boxes,
        )
        outputs.update(damage_outputs)
        return outputs

    def forward(
        self,
        batch: dict[str, Any],
        *,
        targets: list[dict[str, torch.Tensor]] | None = None,
        matches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        epoch: int = 0,
        stage: str = "joint",
    ) -> dict[str, Any]:
        outputs = self.forward_localization(batch, stage=stage)
        return self.forward_damage(batch, outputs, targets=targets, matches=matches, epoch=epoch, stage=stage)


def build_dabqn_model(config: dict[str, Any]) -> DABQNLite:
    return DABQNLite(config)

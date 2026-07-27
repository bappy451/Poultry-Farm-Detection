"""
lpft_llrd_trainer.py
====================
Custom Ultralytics OBBTrainer implementing:
  1) Two-Stage LP-FT (Linear Probing followed by End-to-End Fine-Tuning)
  2) Layer-Wise Learning Rate Decay (LLRD) across backbone layers

Designed for fine-tuning SparK Trained-from-Scratch Vision Backbones on
downstream tasks without catastrophic forgetting of pre-trained representations.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model


class LPFT_LLRD_OBBTrainer(OBBTrainer):
    """
    Custom OBBTrainer for SparK Trained-from-Scratch Backbone fine-tuning.

    Features
    --------
    - **Stage 1 (Linear Probing - LP)**:
      For epochs 0 to `lp_epochs - 1`, backbone layers (0 to 10 in YOLO11l)
      are completely frozen (`requires_grad = False`). Only the detection neck
      and OBB head are trained.
    - **Stage 2 (End-to-End Fine-Tuning - FT)**:
      At epoch `lp_epochs`, backbone layers are unfrozen (`requires_grad = True`),
      and all parameters are fine-tuned end-to-end.
    - **Layer-Wise Learning Rate Decay (LLRD)**:
      Assigns a geometric learning rate decay factor to each layer depth l:
          eta_l = lr_base * (alpha ** (L + 1 - l))
      where L=10 is the number of backbone blocks and alpha is `layer_decay`.
      Detection neck and head receive 100% of base_lr.
    - **Decoupled Weight Decay**:
      1D bias vectors and Normalization weights are exempt from weight decay.
    """

    lp_epochs: int = 5
    layer_decay: float = 0.75

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lp_epochs = int(
            getattr(self.args, "lp_epochs", getattr(self.__class__, "lp_epochs", 5))
        )
        self.layer_decay = float(
            getattr(
                self.args,
                "layer_decay",
                getattr(self.__class__, "layer_decay", 0.75),
            )
        )

    def _setup_train(self, *args, **kwargs) -> None:
        """Initialize trainer and register LP-FT epoch callback."""
        super()._setup_train(*args, **kwargs)
        if self.lp_epochs > 0:
            self.add_callback("on_train_epoch_start", self._lpft_epoch_start_callback)

    def _lpft_epoch_start_callback(self, trainer: Any) -> None:
        """
        Callback invoked at the start of every training epoch (0-indexed).
        Controls freezing/unfreezing of the YOLO11l backbone (layers 0 to 10).
        """
        epoch = trainer.epoch
        # In YOLO11l-OBB, layers 0..10 are the backbone (Conv, C3k2, SPPF, C2PSA)
        backbone_layers = trainer.model.model[:11]

        if epoch < self.lp_epochs:
            # Stage 1: Freeze backbone layers 0..10
            for layer in backbone_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            if epoch == 0:
                LOGGER.info(
                    f"\n{'='*70}\n"
                    f"  [LP-FT Stage 1] LINEAR PROBING: Backbone (layers 0-10) FROZEN ❄️\n"
                    f"  Only detection head/neck are trained for epochs 1-{self.lp_epochs}.\n"
                    f"{'='*70}\n"
                )
        elif epoch == self.lp_epochs:
            # Stage 2: Unfreeze backbone layers 0..10
            for layer in backbone_layers:
                for param in layer.parameters():
                    param.requires_grad = True
            LOGGER.info(
                f"\n{'='*70}\n"
                f"  [LP-FT Stage 2] END-TO-END LLRD FINE-TUNING: Backbone UNFROZEN 🔥\n"
                f"  All layers fine-tuned with Layer-Wise Learning Rate Decay (alpha={self.layer_decay}).\n"
                f"{'='*70}\n"
            )

    def build_optimizer(
        self,
        model: torch.nn.Module,
        name: str = "AdamW",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: float = 1e5,
    ) -> torch.optim.Optimizer:
        """
        Construct an optimizer with Layer-Wise Learning Rate Decay (LLRD)
        and decoupled weight decay for bias/norm layers.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        param_groups_dict: dict[tuple[int, str, float, float], list[torch.nn.Parameter]] = {}
        L = 10  # max backbone layer index in YOLO11l (0 to 10)

        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name

                # Determine top-level sequential index (e.g. 'model.0.conv.weight' -> 0)
                layer_idx = 11  # default to neck/head
                parts = fullname.split(".")
                if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
                    idx_val = int(parts[1])
                    if idx_val <= L:
                        layer_idx = idx_val

                # Calculate LLRD effective learning rate for this layer
                if layer_idx <= L:
                    decay_factor = self.layer_decay ** (L + 1 - layer_idx)
                else:
                    decay_factor = 1.0

                lr_layer = lr * decay_factor

                # Determine parameter type and weight decay decoupling
                if "bias" in fullname:
                    ptype = "bias"
                    wd = 0.0
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    ptype = "bn"
                    wd = 0.0
                else:
                    ptype = "weight"
                    wd = decay

                group_key = (layer_idx, ptype, wd, lr_layer)
                if group_key not in param_groups_dict:
                    param_groups_dict[group_key] = []
                param_groups_dict[group_key].append(param)

        # Build sorted list of parameter groups for PyTorch Optimizer
        param_groups = []
        for (layer_idx, ptype, wd, lr_layer), params in sorted(
            param_groups_dict.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            group_name = f"layer_{layer_idx}_{ptype}"
            param_groups.append(
                {
                    "params": params,
                    "lr": lr_layer,
                    "weight_decay": wd,
                    "param_group": group_name,
                }
            )

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD"}
        opt_name = {x.lower(): x for x in optimizers}.get(name.lower(), "AdamW")

        if opt_name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, opt_name)(
                param_groups, lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif opt_name == "RMSProp":
            optimizer = optim.RMSProp(param_groups, lr=lr, momentum=momentum)
        else:
            optimizer = optim.SGD(
                param_groups, lr=lr, momentum=momentum, nesterov=True
            )

        # Log LLRD summary table
        LOGGER.info(
            f"\n[LLRD] Constructed {len(param_groups)} optimizer parameter groups "
            f"(layer_decay={self.layer_decay}, base_lr={lr}):"
        )
        LOGGER.info(f"  {'Layer / Depth':<25} {'Effective LR':<14} {'% of Base LR':<15}")
        LOGGER.info(f"  {'-'*54}")
        for idx in range(0, L + 2):
            label = f"Layer {idx}" if idx <= L else "Layer 11..23 (Neck/Head)"
            df = (self.layer_decay ** (L + 1 - idx)) if idx <= L else 1.0
            LOGGER.info(f"  {label:<25} {lr * df:<14.4e} {df * 100:>6.1f}%")
        LOGGER.info("")

        return optimizer

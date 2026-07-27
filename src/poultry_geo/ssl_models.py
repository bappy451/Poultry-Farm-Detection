from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch import nn


SUPPORTED_PRECISIONS = ("fp32", "fp16", "bf16", "nf4")
PRECISION_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "nf4": torch.bfloat16,
}

VIT_CONFIGS = {
    "S": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "B": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "L": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "H": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
}


def _bitsandbytes_supports_nf4() -> bool:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    return hasattr(getattr(bnb, "nn", None), "Linear4bit")


def build_nf4_quantization_config():
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return SimpleNamespace(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _validate_precision(precision: str) -> str:
    if precision not in SUPPORTED_PRECISIONS:
        supported = ", ".join(SUPPORTED_PRECISIONS[:-1]) + f", or {SUPPORTED_PRECISIONS[-1]}"
        raise ValueError(f"precision must be {supported}.")
    return precision


def _resolve_precision(precision: str, *, allow_nf4: bool) -> str:
    requested = _validate_precision(precision)
    if requested != "nf4":
        return requested
    return "nf4" if allow_nf4 and _bitsandbytes_supports_nf4() else "bf16"


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _runtime_compute_dtype(device: torch.device, precision: str) -> torch.dtype:
    if device.type == "cpu" and precision != "fp32":
        return torch.float32
    return PRECISION_DTYPES[precision]


def _output_dtype(precision: str) -> torch.dtype:
    return PRECISION_DTYPES[precision]


def _quantization_load_enabled(quantization_config: Any | None) -> bool:
    if quantization_config is None:
        return True
    return bool(getattr(quantization_config, "load_in_4bit", False))


def _quantization_compute_dtype(quantization_config: Any | None) -> torch.dtype:
    value = getattr(quantization_config, "bnb_4bit_compute_dtype", torch.bfloat16)
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.replace("torch.", "")
        if hasattr(torch, normalized):
            candidate = getattr(torch, normalized)
            if isinstance(candidate, torch.dtype):
                return candidate
    return torch.bfloat16


def _quantization_type(quantization_config: Any | None) -> str:
    value = getattr(quantization_config, "bnb_4bit_quant_type", "nf4")
    return str(value)


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return tensor.normal_(mean=0.0, std=std)


@dataclass(frozen=True)
class _ImageGeometry:
    image_size: tuple[int, int]
    patch_size: tuple[int, int]
    grid_size: tuple[int, int]

    @property
    def num_patches(self) -> int:
        return self.grid_size[0] * self.grid_size[1]


class _PrecisionLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, *, eps: float = 1e-6, precision: str = "fp32") -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.precision = precision
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        compute_dtype = _runtime_compute_dtype(inputs.device, self.precision)
        outputs = F.layer_norm(
            inputs.to(dtype=compute_dtype),
            (self.normalized_shape,),
            self.weight.to(device=inputs.device, dtype=compute_dtype),
            self.bias.to(device=inputs.device, dtype=compute_dtype),
            self.eps,
        )
        return outputs


class _PrecisionLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        precision: str = "fp32",
        quantization_config: Any | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision
        self.quantized: nn.Module | None = None

        if precision == "nf4" and _bitsandbytes_supports_nf4() and _quantization_load_enabled(quantization_config):
            import bitsandbytes as bnb

            self.quantized = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=bias,
                compute_dtype=_quantization_compute_dtype(quantization_config),
                quant_type=_quantization_type(quantization_config),
            )
            return

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.quantized is not None:
            return
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.quantized is not None:
            quantized = self.quantized.to(device=inputs.device)
            return quantized(inputs.to(dtype=torch.bfloat16))

        compute_dtype = _runtime_compute_dtype(inputs.device, self.precision)
        return F.linear(
            inputs.to(dtype=compute_dtype),
            self.weight.to(device=inputs.device, dtype=compute_dtype),
            None if self.bias is None else self.bias.to(device=inputs.device, dtype=compute_dtype),
        )


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        *,
        patch_size: tuple[int, int],
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.precision = precision
        self.weight = nn.Parameter(
            torch.empty(embed_dim, in_channels, patch_size[0], patch_size[1])
        )
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight.view(self.weight.shape[0], -1))
        nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        compute_dtype = _runtime_compute_dtype(inputs.device, self.precision)
        return F.conv2d(
            inputs.to(dtype=compute_dtype),
            self.weight.to(device=inputs.device, dtype=compute_dtype),
            self.bias.to(device=inputs.device, dtype=compute_dtype),
            stride=self.patch_size,
        )


class _PrecisionConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        kernel_hw = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.precision = precision
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_hw[0], kernel_hw[1]))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        compute_dtype = _runtime_compute_dtype(inputs.device, self.precision)
        return F.conv2d(
            inputs.to(dtype=compute_dtype),
            self.weight.to(device=inputs.device, dtype=compute_dtype),
            self.bias.to(device=inputs.device, dtype=compute_dtype),
            stride=self.stride,
            padding=self.padding,
        )


class _SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        precision: str = "fp32",
        quantization_config: Any | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = _PrecisionLinear(
            embed_dim,
            embed_dim * 3,
            precision=precision,
            quantization_config=quantization_config,
        )
        self.proj = _PrecisionLinear(
            embed_dim,
            embed_dim,
            precision=precision,
            quantization_config=quantization_config,
        )
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = inputs.shape
        qkv = self.qkv(inputs)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)

        attention_scores = (queries @ keys.transpose(-2, -1)) * self.scale
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attn_drop(attention_weights)

        attended = attention_weights @ values
        attended = attended.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)
        projected = self.proj(attended)
        return self.proj_drop(projected)


class _Mlp(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int,
        drop_rate: float = 0.0,
        precision: str = "fp32",
        quantization_config: Any | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = _PrecisionLinear(
            embed_dim,
            hidden_dim,
            precision=precision,
            quantization_config=quantization_config,
        )
        self.fc2 = _PrecisionLinear(
            hidden_dim,
            embed_dim,
            precision=precision,
            quantization_config=quantization_config,
        )
        self.drop = nn.Dropout(drop_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.fc1(inputs)
        outputs = F.gelu(outputs)
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        return self.drop(outputs)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class _DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        precision: str = "fp32",
        quantization_config: Any | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = max(embed_dim, int(embed_dim * mlp_ratio))
        self.norm1 = _PrecisionLayerNorm(embed_dim, precision=precision)
        self.attn = _SelfAttention(
            embed_dim,
            num_heads=num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            precision=precision,
            quantization_config=quantization_config,
        )
        self.drop_path = _DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = _PrecisionLayerNorm(embed_dim, precision=precision)
        self.mlp = _Mlp(
            embed_dim,
            hidden_dim=hidden_dim,
            drop_rate=drop_rate,
            precision=precision,
            quantization_config=quantization_config,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        outputs = outputs + self.drop_path(self.mlp(self.norm2(outputs)))
        return outputs


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        *,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be a positive integer.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if dropout < 0:
            raise ValueError("dropout cannot be negative.")
        in_features = getattr(base_layer, "in_features", None)
        out_features = getattr(base_layer, "out_features", None)
        if in_features is None or out_features is None:
            raise TypeError("LoRALinear expects a base layer with in_features and out_features.")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.precision = precision
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_up = nn.Parameter(torch.empty(self.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_down, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_outputs = self.base_layer(inputs)
        compute_dtype = _runtime_compute_dtype(inputs.device, self.precision)
        output_dtype = base_outputs.dtype
        down = F.linear(
            self.dropout(inputs).to(dtype=compute_dtype),
            self.lora_down.to(device=inputs.device, dtype=compute_dtype),
        )
        delta = F.linear(
            down,
            self.lora_up.to(device=inputs.device, dtype=compute_dtype),
        ) * self.scaling
        return (base_outputs.to(dtype=compute_dtype) + delta).to(dtype=output_dtype)


class BandAdapter(nn.Module):
    """
    Projects heterogeneous sensor bands (e.g., NAIP 4-band, Sentinel-2 12-band) 
    into a three-channel RGB-like space.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        *,
        precision: str = "fp32",
        bypass_rgb: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bypass_rgb = bypass_rgb
        self.requested_precision = _validate_precision(precision)
        self.resolved_precision = "bf16" if self.requested_precision == "nf4" else self.requested_precision

        self.proj: nn.Conv2d | None = None
        if not (self.bypass_rgb and in_channels == out_channels == 3):
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("BandAdapter expects inputs with shape [B, C, H, W].")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"BandAdapter expected {self.in_channels} channels, received {inputs.shape[1]}."
            )

        if self.proj is None:
            return inputs.to(dtype=_output_dtype(self.resolved_precision))

        compute_dtype = _runtime_compute_dtype(inputs.device, self.resolved_precision)
        outputs = F.conv2d(
            inputs.to(dtype=compute_dtype),
            self.proj.weight.to(device=inputs.device, dtype=compute_dtype),
            None if self.proj.bias is None else self.proj.bias.to(device=inputs.device, dtype=compute_dtype),
        )
        return outputs.to(dtype=_output_dtype(self.resolved_precision))


class RemoteSensingViT(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        embed_dim: int,
        depth: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        precision: str = "fp32",
        quantization_config: Any | None = None,
        gradient_checkpointing: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        requested_precision = _validate_precision(precision)
        image_hw = _pair(image_size)
        patch_hw = _pair(patch_size)

        if image_hw[0] % patch_hw[0] != 0 or image_hw[1] % patch_hw[1] != 0:
            raise ValueError(
                f"image_size {image_hw} must be divisible by patch_size {patch_hw}."
            )

        self.requested_precision = requested_precision
        self.resolved_precision = _resolve_precision(requested_precision, allow_nf4=True)
        self.gradient_checkpointing = gradient_checkpointing
        self.quantization_config = quantization_config
        self.embed_dim = embed_dim
        self.geometry = _ImageGeometry(
            image_size=image_hw,
            patch_size=patch_hw,
            grid_size=(image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1]),
        )
        self.grid_size = self.geometry.grid_size
        self.num_patches = self.geometry.num_patches

        block_precision = self.resolved_precision
        self.patch_embed = _PatchEmbed(3, embed_dim, patch_size=patch_hw, precision=block_precision)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        _trunc_normal_(self.pos_embed)
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList(
            _TransformerBlock(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                precision=block_precision,
                quantization_config=quantization_config,
            )
            for i in range(depth)
        )
        self.norm = _PrecisionLayerNorm(embed_dim, precision=block_precision)

    def embed_patches(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = int(inputs.shape[2]), int(inputs.shape[3])
        patch_height, patch_width = self.geometry.patch_size
        outputs = self.patch_embed(inputs)
        current_grid = (outputs.shape[2], outputs.shape[3])
        return outputs.flatten(2).transpose(1, 2), current_grid

    def add_position_embeddings(
        self,
        patch_tokens: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        outputs = patch_tokens + self._interpolate_positional_embeddings(
            grid_size,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        return self.pos_drop(outputs)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        outputs = tokens
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                outputs = checkpoint_utils.checkpoint(block, outputs)
            else:
                outputs = block(outputs)
        outputs = self.norm(outputs)
        return outputs.to(dtype=_output_dtype(self.resolved_precision))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[2] % self.geometry.patch_size[0] != 0 or inputs.shape[3] % self.geometry.patch_size[1] != 0:
            raise ValueError(
                f"Runtime input size {(inputs.shape[2], inputs.shape[3])} must be divisible by patch_size {self.geometry.patch_size}."
            )
        patch_tokens, current_grid = self.embed_patches(inputs)
        positioned_tokens = self.add_position_embeddings(patch_tokens, current_grid)
        return self.encode_tokens(positioned_tokens)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_features(inputs)

    def load_pretrained_weights(self) -> None:
        """
        Loads ImageNet pretrained weights from timm into the backbone.
        """
        try:
            import timm
        except ImportError:
            print("Warning: timm not installed. Cannot load pretrained weights.")
            return

        # Map internal config names to timm model names
        # Standard ViT models in timm
        timm_map = {
            384: "vit_small_patch16_224.dino",
            768: "vit_base_patch16_224.dino",
            1024: "vit_large_patch16_224",
            1280: "vit_huge_patch14_224_in21k",
        }
        
        timm_model_name = timm_map.get(self.embed_dim)
        if not timm_model_name:
            print(f"Warning: No pretrained mapping for embed_dim={self.embed_dim}")
            return
            
        print(f"Loading pretrained weights for {timm_model_name}...")
        try:
            # We only need the state_dict to load into our custom implementation
            timm_model = timm.create_model(timm_model_name, pretrained=True)
            timm_state = timm_model.state_dict()
            
            # Map timm names to our custom names
            # Our names: patch_embed.weight, pos_embed, blocks.i.norm1.weight, etc.
            # timm names: patch_embed.proj.weight, pos_embed, blocks.i.norm1.weight, etc.
            
            new_state = {}
            for k, v in timm_state.items():
                new_key = k
                # Handle patch_embed.proj -> patch_embed
                if k.startswith("patch_embed.proj."):
                    new_key = k.replace("patch_embed.proj.", "patch_embed.")
                
                # We skip things like cls_token if we don't have it (we use pure GAP or patch tokens)
                # We skip head weights
                if k.startswith("head.") or k == "cls_token" or k == "dist_token":
                    continue
                
                if new_key in self.state_dict():
                    # Check shape compatibility
                    target_shape = self.state_dict()[new_key].shape
                    if v.shape != target_shape:
                        if new_key == "pos_embed" and v.shape[1] == target_shape[1] + 1:
                            # Handle cls_token in timm (1, N+1, D) vs our (1, N, D)
                            v = v[:, 1:, :]
                        else:
                            print(f"  Skipping {new_key} due to shape mismatch: {v.shape} vs {target_shape}")
                            continue
                    
                    new_state[new_key] = v
            
            msg = self.load_state_dict(new_state, strict=False)
            print(f"  Pretrained weights loaded: {len(new_state)} tensors. Missing: {len(msg.missing_keys)}")
            
        except Exception as e:
            print(f"Error loading pretrained weights from timm: {e}")

    def _interpolate_positional_embeddings(
        self,
        target_grid: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if target_grid == self.grid_size:
            return self.pos_embed.to(device=device, dtype=dtype)
        embeddings = self.pos_embed.reshape(1, self.grid_size[0], self.grid_size[1], self.embed_dim)
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = F.interpolate(embeddings, size=target_grid, mode="bicubic", align_corners=False)
        embeddings = embeddings.permute(0, 2, 3, 1).reshape(1, target_grid[0] * target_grid[1], self.embed_dim)
        return embeddings.to(device=device, dtype=dtype)


def inject_lora_into_vit(
    model: RemoteSensingViT,
    *,
    rank: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: tuple[str, ...] = ("qkv", "proj"),
    freeze_backbone: bool = False,
) -> RemoteSensingViT:
    if freeze_backbone:
        model.requires_grad_(False)
    targets = set(target_modules)
    for block in model.blocks:
        if "qkv" in targets and not isinstance(block.attn.qkv, LoRALinear):
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha, dropout=dropout, precision=model.resolved_precision)
        if "proj" in targets and not isinstance(block.attn.proj, LoRALinear):
            block.attn.proj = LoRALinear(block.attn.proj, rank=rank, alpha=alpha, dropout=dropout, precision=model.resolved_precision)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
    model.use_lora = True
    return model


class DecoderHead(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        output_size: int | tuple[int, int],
        hidden_channels: int = 128,
        out_channels: int = 1,
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.output_size = _pair(output_size)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.requested_precision = _validate_precision(precision)
        self.resolved_precision = _resolve_precision(self.requested_precision, allow_nf4=False)
        mid_channels = max(1, hidden_channels // 2)

        self.proj = _PrecisionConv2d(embed_dim, hidden_channels, kernel_size=1, precision=self.resolved_precision)
        self.refine1 = _PrecisionConv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, precision=self.resolved_precision)
        self.refine2 = _PrecisionConv2d(hidden_channels, mid_channels, kernel_size=3, padding=1, precision=self.resolved_precision)
        self.head = _PrecisionConv2d(mid_channels, out_channels, kernel_size=1, precision=self.resolved_precision)

    def forward(self, patch_embeddings: torch.Tensor, *, grid_size: tuple[int, int] | None = None) -> torch.Tensor:
        if grid_size is None:
            side = round(patch_embeddings.shape[1] ** 0.5)
            grid_size = (side, side)
        if patch_embeddings.shape[1] != grid_size[0] * grid_size[1]:
            raise ValueError(
                f"patch_embeddings length {patch_embeddings.shape[1]} does not match grid_size {grid_size}."
            )
        outputs = patch_embeddings.transpose(1, 2).reshape(patch_embeddings.shape[0], self.embed_dim, grid_size[0], grid_size[1])
        outputs = self.proj(outputs)
        outputs = F.gelu(outputs)
        outputs = F.interpolate(outputs, size=self.output_size, mode="bilinear", align_corners=False)
        outputs = self.refine1(outputs)
        outputs = F.gelu(outputs)
        outputs = self.refine2(outputs)
        outputs = F.gelu(outputs)
        outputs = self.head(outputs)
        return outputs.to(dtype=_output_dtype(self.resolved_precision))


class SegmentationHead(DecoderHead):
    def __init__(self, *, embed_dim: int, output_size: int | tuple[int, int], hidden_channels: int = 128, precision: str = "fp32") -> None:
        super().__init__(embed_dim=embed_dim, output_size=output_size, hidden_channels=hidden_channels, out_channels=1, precision=precision)


class RemoteSensingSegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        embed_dim: int,
        depth: int,
        num_heads: int,
        precision: str = "fp32",
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
        freeze_backbone: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        decoder_hidden_channels: int = 128,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.requested_precision = _validate_precision(precision)
        self.use_lora = use_lora
        self.freeze_backbone = freeze_backbone
        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)
        self.adapter = BandAdapter(in_channels=in_channels, out_channels=3, precision=precision)
        self.backbone = RemoteSensingViT(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            precision=precision,
            gradient_checkpointing=gradient_checkpointing,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        if freeze_backbone and not use_lora:
            self.backbone.requires_grad_(False)
        if use_lora:
            inject_lora_into_vit(self.backbone, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, freeze_backbone=freeze_backbone)
        self.head = SegmentationHead(embed_dim=embed_dim, output_size=self.image_size, hidden_channels=decoder_hidden_channels, precision=precision)
        self.resolved_precision = self.backbone.resolved_precision

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter(inputs)
        tokens = self.backbone.forward_features(adapted)
        grid_size = (int(adapted.shape[2]) // self.patch_size[0], int(adapted.shape[3]) // self.patch_size[1])
        return self.head(tokens, grid_size=grid_size)


class RemoteSensingMIMModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        embed_dim: int,
        depth: int,
        num_heads: int,
        precision: str = "fp32",
        mask_ratio: float = 0.75,
        gradient_checkpointing: bool = False,
        use_lora: bool = False,
        freeze_backbone: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        quantization_config: Any | None = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.requested_precision = _validate_precision(precision)
        self.quantization_config = quantization_config
        if self.requested_precision == "nf4" and self.quantization_config is None:
            self.quantization_config = build_nf4_quantization_config()
        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)
        self.mask_ratio = float(mask_ratio)
        self.use_lora = bool(use_lora or self.requested_precision == "nf4")
        self.freeze_backbone = bool(freeze_backbone or self.requested_precision == "nf4")
        self.adapter = BandAdapter(in_channels=in_channels, out_channels=3, precision=precision)
        self.backbone = RemoteSensingViT(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            precision=precision,
            quantization_config=self.quantization_config,
            gradient_checkpointing=gradient_checkpointing,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        if pretrained:
            self.backbone.load_pretrained_weights()
            
        if self.freeze_backbone and not self.use_lora:
            self.backbone.requires_grad_(False)
        if self.use_lora:
            inject_lora_into_vit(self.backbone, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout, freeze_backbone=self.freeze_backbone)
        self.resolved_precision = self.backbone.resolved_precision
        self.patch_dim = 3 * self.patch_size[0] * self.patch_size[1]
        self.reconstruction_precision = "bf16" if self.requested_precision == "nf4" else self.resolved_precision
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        _trunc_normal_(self.mask_token)
        self.reconstruction_head = _PrecisionLinear(embed_dim, self.patch_dim, precision=self.reconstruction_precision)

    @staticmethod
    def compute_masked_loss(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, dim = target.shape
        C = 3
        P_sq = dim // C
        
        target_view = target.view(B, L, C, P_sq)
        recon_view = reconstruction.view(B, L, C, P_sq)

        # Patch-wise Normalization (Per Channel)
        mean = target_view.mean(dim=-1, keepdim=True)
        var = target_view.var(dim=-1, unbiased=False, keepdim=True)
        target_norm = (target_view - mean) / (var + 1e-6).sqrt()
        
        # L1 (Mean Absolute Error) is preferred over MSE for multispectral imagery 
        # (RGB+IR) as it is less sensitive to outliers caused by differing channel variances.
        loss = (recon_view.float() - target_norm.float()).abs().mean(dim=(2, 3))
        return loss[mask].mean()

    def forward_with_intermediates(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        adapted = self.adapter(inputs)
        patch_tokens, grid_size = self.backbone.embed_patches(adapted)
        num_masked = int(round(patch_tokens.shape[1] * self.mask_ratio))
        noise = torch.rand(patch_tokens.shape[0], patch_tokens.shape[1], device=patch_tokens.device)
        mask = torch.zeros(patch_tokens.shape[0], patch_tokens.shape[1], dtype=torch.bool, device=patch_tokens.device)
        mask.scatter_(1, noise.argsort(dim=1)[:, :num_masked], True)
        masked_tokens = torch.where(mask.unsqueeze(-1), self.mask_token.to(patch_tokens.dtype), patch_tokens)
        positioned = self.backbone.add_position_embeddings(masked_tokens, grid_size)
        encoded = self.backbone.encode_tokens(positioned)
        recon = self.reconstruction_head(encoded).to(dtype=_output_dtype(self.reconstruction_precision))
        target = F.unfold(adapted.detach().clamp(0.0, 1.0), kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2).to(recon.dtype)
        loss = self.compute_masked_loss(recon, target, mask)
        return {
            "adapted": adapted,
            "tokens": patch_tokens,
            "mask": mask,
            "reconstructed_patches": recon,
            "target_patches": target,
            "loss": loss,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(inputs)["loss"]


__all__ = [
    "BandAdapter",
    "DecoderHead",
    "LoRALinear",
    "RemoteSensingViT",
    "RemoteSensingMIMModel",
    "RemoteSensingSegmentationModel",
    "SegmentationHead",
    "build_nf4_quantization_config",
    "inject_lora_into_vit",
]

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None


def require_spconv() -> None:
    """Fail closed when a caller requests the sparse SparK implementation."""
    if spconv is None:
        raise RuntimeError(
            "SparK requires spconv for sparse convolution. Install a spconv build "
            "compatible with this PyTorch/CUDA environment; the old dense-zero "
            "fallback is intentionally unsupported because it is not SparK."
        )


class SparseTransformer(nn.Module):
    """Convert visible image patches into a native 2-D ``SparseConvTensor``."""

    def __init__(self, adapter: nn.Module, patch_size: int = 16, mask_ratio: float = 0.75):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1); at least one patch must remain visible.")
        self.adapter = adapter
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def generate_mask(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full-resolution and flattened patch masks (1=hidden, 0=visible)."""
        if batch_size <= 0 or height <= 0 or width <= 0:
            raise ValueError("Batch size and image dimensions must be positive.")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Image size {(height, width)} must be divisible by patch_size={self.patch_size}."
            )
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        patch_count = patch_height * patch_width
        hidden_count = int(patch_count * self.mask_ratio)
        noise = torch.rand(batch_size, patch_count, device=device)
        patch_mask = torch.zeros(batch_size, patch_count, device=device)
        if hidden_count:
            patch_mask.scatter_(1, noise.topk(hidden_count, dim=1).indices, 1.0)
        mask = F.interpolate(
            patch_mask.reshape(batch_size, 1, patch_height, patch_width),
            size=(height, width),
            mode="nearest",
        )
        return mask, patch_mask

    def forward(self, x: torch.Tensor):
        """Return a sparse tensor containing only visible pixels and its mask."""
        require_spconv()
        if x.ndim != 4:
            raise ValueError("SparseTransformer expects [batch, channels, height, width].")
        batch_size, _, height, width = x.shape
        device = x.device
        adapted = self.adapter(x)
        if adapted.ndim != 4 or adapted.shape[0] != batch_size or adapted.shape[2:] != (height, width):
            raise ValueError("Band adapter must preserve batch and spatial dimensions.")

        mask, patch_mask = self.generate_mask(batch_size, height, width, device)
        patch_width = width // self.patch_size
        visible_patches = (patch_mask == 0).nonzero(as_tuple=False)
        if not len(visible_patches):
            raise RuntimeError("Mask generation left no visible patches.")

        patch_pixels = self.patch_size * self.patch_size
        local_y, local_x = torch.meshgrid(
            torch.arange(self.patch_size, device=device),
            torch.arange(self.patch_size, device=device),
            indexing="ij",
        )
        local_y, local_x = local_y.flatten(), local_x.flatten()
        batch_index, patch_index = visible_patches[:, 0], visible_patches[:, 1]
        patch_y = (patch_index // patch_width) * self.patch_size
        patch_x = (patch_index % patch_width) * self.patch_size
        expanded_batch = batch_index.repeat_interleave(patch_pixels)
        y_index = patch_y.repeat_interleave(patch_pixels) + local_y.repeat(len(visible_patches))
        x_index = patch_x.repeat_interleave(patch_pixels) + local_x.repeat(len(visible_patches))
        indices = torch.stack((expanded_batch.int(), y_index.int(), x_index.int()), dim=1)
        features = adapted[expanded_batch, :, y_index, x_index]
        return spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=[height, width],
            batch_size=batch_size,
        ), mask

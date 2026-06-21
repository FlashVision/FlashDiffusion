"""Image utilities — save, load, grid, resize, tensor conversion."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


def save_image(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    path: str,
    quality: int = 95,
) -> str:
    """Save an image to disk.

    Args:
        image: PIL Image, tensor, or numpy array.
        path: Output file path.
        quality: JPEG quality (1-100).

    Returns:
        Path to saved image.
    """
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if path.lower().endswith((".jpg", ".jpeg")):
        image.save(path, quality=quality)
    else:
        image.save(path)

    return path


def load_image(
    path: str,
    resolution: Optional[int] = None,
) -> Image.Image:
    """Load an image from disk.

    Args:
        path: Image file path.
        resolution: If set, resize to this resolution (square).

    Returns:
        PIL Image in RGB mode.
    """
    image = Image.open(path).convert("RGB")
    if resolution is not None:
        image = image.resize((resolution, resolution), Image.BILINEAR)
    return image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to PIL Image.

    Handles tensors in [-1, 1] or [0, 1] range, and
    shapes (C, H, W) or (1, C, H, W).

    Args:
        tensor: Image tensor.

    Returns:
        PIL Image.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.detach().cpu().float()

    if tensor.min() < 0:
        tensor = (tensor + 1) / 2

    tensor = tensor.clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def make_image_grid(
    images: List[Image.Image],
    cols: int = 4,
    padding: int = 4,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Create a grid image from a list of PIL Images.

    Args:
        images: List of PIL Images (all should be same size).
        cols: Number of columns.
        padding: Pixel padding between images.
        bg_color: Background color.

    Returns:
        Grid PIL Image.
    """
    if not images:
        return Image.new("RGB", (100, 100), bg_color)

    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols

    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), bg_color)

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = padding + col * (w + padding)
        y = padding + row * (h + padding)
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        grid.paste(img, (x, y))

    return grid


def resize_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    method: str = "bilinear",
) -> Image.Image:
    """Resize an image.

    Args:
        image: Input PIL Image.
        size: Target size (int for square, or (w, h) tuple).
        method: Resize method ("bilinear", "nearest", "bicubic").

    Returns:
        Resized PIL Image.
    """
    if isinstance(size, int):
        size = (size, size)

    methods = {
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "bicubic": Image.BICUBIC,
    }
    resample = methods.get(method, Image.BILINEAR)
    return image.resize(size, resample)

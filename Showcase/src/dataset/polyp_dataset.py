from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

if hasattr(Image, "Resampling"):
    BILINEAR = Image.Resampling.BILINEAR
    NEAREST = Image.Resampling.NEAREST
else:
    BILINEAR = Image.BILINEAR
    NEAREST = Image.NEAREST


SamplePair = Tuple[Path, Path]


def collect_image_mask_pairs(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
) -> List[SamplePair]:
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)

    if not image_path.exists():
        raise FileNotFoundError("Image directory not found: {}".format(image_path))
    if not mask_path.exists():
        raise FileNotFoundError("Mask directory not found: {}".format(mask_path))

    pairs = []
    for file_path in sorted(image_path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        matched_mask = mask_path / file_path.name
        if matched_mask.exists():
            pairs.append((file_path, matched_mask))

    if not pairs:
        raise RuntimeError(
            "No paired image/mask files found in {} and {}".format(image_path, mask_path)
        )

    return pairs


class PolypDataset(Dataset):
    def __init__(
        self,
        image_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        samples: Optional[Sequence[SamplePair]] = None,
        image_size: int = 352,
        augment: bool = False,
    ) -> None:
        if samples is None:
            if image_dir is None or mask_dir is None:
                raise ValueError("image_dir and mask_dir are required when samples is None")
            samples = collect_image_mask_pairs(image_dir, mask_dir)

        if not samples:
            raise ValueError("At least one sample is required")

        self.samples = [(Path(image_path), Path(mask_path)) for image_path, mask_path in samples]
        self.image_size = int(image_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _resize(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        size = (self.image_size, self.image_size)
        return image.resize(size, BILINEAR), mask.resize(size, NEAREST)

    def _fit_to_canvas(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        width, height = image.size
        if width > self.image_size or height > self.image_size:
            max_left = max(0, width - self.image_size)
            max_top = max(0, height - self.image_size)
            left = random.randint(0, max_left)
            top = random.randint(0, max_top)
            image = image.crop((left, top, left + self.image_size, top + self.image_size))
            mask = mask.crop((left, top, left + self.image_size, top + self.image_size))
            return image, mask

        padded_image = Image.new("RGB", (self.image_size, self.image_size))
        padded_mask = Image.new("L", (self.image_size, self.image_size))
        max_left = self.image_size - width
        max_top = self.image_size - height
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)
        padded_image.paste(image, (left, top))
        padded_mask.paste(mask, (left, top))
        return padded_image, padded_mask

    def _random_erase(self, image: Image.Image) -> Image.Image:
        image_array = np.asarray(image, dtype=np.uint8).copy()
        erase_count = random.randint(1, 2)
        for _ in range(erase_count):
            erase_h = max(8, int(self.image_size * random.uniform(0.06, 0.18)))
            erase_w = max(8, int(self.image_size * random.uniform(0.06, 0.18)))
            top = random.randint(0, self.image_size - erase_h)
            left = random.randint(0, self.image_size - erase_w)
            fill_value = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
            image_array[top:top + erase_h, left:left + erase_w] = fill_value
        return Image.fromarray(image_array)

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        rotation_k = random.randint(0, 3)
        if rotation_k:
            angle = 90 * rotation_k
            image = image.rotate(angle, resample=BILINEAR)
            mask = mask.rotate(angle, resample=NEAREST)

        fine_angle = random.uniform(-15.0, 15.0)
        image = image.rotate(fine_angle, resample=BILINEAR, fillcolor=(0, 0, 0))
        mask = mask.rotate(fine_angle, resample=NEAREST, fillcolor=0)

        scale = random.uniform(0.7, 1.35)
        aspect_ratio = random.uniform(0.85, 1.15)
        scaled_width = max(32, int(round(self.image_size * scale * aspect_ratio)))
        scaled_height = max(32, int(round(self.image_size * scale / aspect_ratio)))
        image = image.resize((scaled_width, scaled_height), BILINEAR)
        mask = mask.resize((scaled_width, scaled_height), NEAREST)
        image, mask = self._fit_to_canvas(image, mask)

        if random.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))

        if random.random() < 0.25:
            image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.6, 1.6))

        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.6)))

        if random.random() < 0.25:
            image_array = np.asarray(image, dtype=np.float32)
            noise = np.random.normal(loc=0.0, scale=random.uniform(4.0, 12.0), size=image_array.shape)
            image_array = np.clip(image_array + noise, 0.0, 255.0).astype(np.uint8)
            image = Image.fromarray(image_array)

        if random.random() < 0.2:
            image = self._random_erase(image)

        return image, mask

    def __getitem__(self, index: int) -> dict:
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self._resize(image, mask)
        if self.augment:
            image, mask = self._augment_pair(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0
        mask_array = (mask_array >= 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).contiguous()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": image_path.name,
        }
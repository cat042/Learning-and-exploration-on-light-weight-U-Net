import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset.polyp_dataset import ALLOWED_EXTENSIONS
from models import MODEL_NAMES
from utils.training import load_model_from_checkpoint, resolve_device

if hasattr(Image, "Resampling"):
	BILINEAR = Image.Resampling.BILINEAR
	NEAREST = Image.Resampling.NEAREST
else:
	BILINEAR = Image.BILINEAR
	NEAREST = Image.NEAREST


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Predict segmentation masks with a trained UNet")
	parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument("--input", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
	parser.add_argument("--model", type=str, default=None, choices=MODEL_NAMES)
	parser.add_argument("--image-size", type=int, default=352)
	parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold; defaults to the best threshold saved during training")
	parser.add_argument("--device", type=str, default="auto")
	return parser.parse_args()


def collect_input_images(input_path: Path) -> list[Path]:
	if input_path.is_file() and input_path.suffix.lower() in ALLOWED_EXTENSIONS:
		return [input_path]

	if input_path.is_dir():
		return sorted(
			[file_path for file_path in input_path.iterdir() if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS]
		)

	raise FileNotFoundError("No valid input images found at {}".format(input_path))


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
	resized = image.resize((image_size, image_size), BILINEAR)
	image_array = np.asarray(resized, dtype=np.float32) / 255.0
	return torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)


def main() -> None:
	args = parse_args()
	project_root = args.project_root.resolve()
	checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
	input_path = args.input if args.input.is_absolute() else project_root / args.input
	output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	device = resolve_device(args.device)
	model, checkpoint = load_model_from_checkpoint(checkpoint_path, device, model_name=args.model)
	image_size = checkpoint.get("config", {}).get("image_size", args.image_size)
	threshold = checkpoint.get("config", {}).get("best_threshold", 0.5) if args.threshold is None else args.threshold

	for image_path in collect_input_images(input_path):
		image = Image.open(image_path).convert("RGB")
		original_size = image.size
		image_tensor = image_to_tensor(image, image_size).to(device)

		with torch.inference_mode():
			logits = model(image_tensor)
			probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

		mask = (probabilities >= threshold).astype(np.uint8) * 255
		mask_image = Image.fromarray(mask)
		mask_image = mask_image.resize(original_size, NEAREST)
		output_path = output_dir / "{}_mask.png".format(image_path.stem)
		mask_image.save(output_path)
		print("Saved {}".format(output_path))


if __name__ == "__main__":
	main()

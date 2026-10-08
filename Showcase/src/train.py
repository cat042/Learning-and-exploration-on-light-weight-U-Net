import argparse
from pathlib import Path

import torch

from dataset.polyp_dataset import collect_image_mask_pairs
from models import MODEL_NAMES
from utils.training import (
	DEFAULT_SEEDS,
	DEFAULT_NUM_WORKERS,
	DEFAULT_THRESHOLD,
	DEFAULT_THRESHOLD_CANDIDATES,
	create_data_loaders,
	default_data_paths,
	discover_test_datasets,
	evaluate_test_datasets,
	evaluate_loader,
	evaluate_loader_with_thresholds,
	instantiate_model,
	normalize_thresholds,
	print_dataset_metrics,
	print_summary_table,
	resolve_device,
	save_json,
	save_summary_csv,
	set_seed,
	split_samples,
	summarize_experiment_results,
	train_one_epoch,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a PyTorch segmentation model for polyp segmentation")
	parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
	parser.add_argument("--model", type=str, default="premier_model", choices=MODEL_NAMES)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--image-size", type=int, default=352)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--epochs", type=int, default=75)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--val-ratio", type=float, default=0.1)
	parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
	parser.add_argument("--device", type=str, default="auto")
	parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Default runs one seed only; pass multiple values to run repeated experiments")
	parser.add_argument("--base-channels", type=int, default=64)
	parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Fallback threshold used when no validation search is available")
	parser.add_argument("--threshold-candidates", type=float, nargs="+", default=DEFAULT_THRESHOLD_CANDIDATES, help="Validation thresholds to scan when selecting the best checkpoint")
	parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision on CUDA")
	parser.add_argument("--deterministic", action="store_true", help="Use deterministic cuDNN settings; slower but more reproducible")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	project_root = args.project_root.resolve()
	default_output_dir = Path("runs") / args.model
	output_dir_arg = args.output_dir if args.output_dir is not None else default_output_dir
	output_dir = output_dir_arg if output_dir_arg.is_absolute() else project_root / output_dir_arg
	output_dir.mkdir(parents=True, exist_ok=True)

	paths = default_data_paths(project_root)
	train_samples = collect_image_mask_pairs(paths["train_images"], paths["train_masks"])
	test_datasets = discover_test_datasets(paths["test_root"])
	device = resolve_device(args.device)
	pin_memory = device.type == "cuda"
	use_amp = device.type == "cuda" and not args.disable_amp
	threshold_candidates = normalize_thresholds(args.threshold_candidates)
	if hasattr(torch, "set_float32_matmul_precision"):
		torch.set_float32_matmul_precision("high")

	print("Device: {}".format(device))
	print("Model: {}".format(args.model))
	print("AMP: {}".format(use_amp))
	print("num_workers: {}".format(args.num_workers))
	print("Training samples: {}".format(len(train_samples)))
	print("Test datasets: {}".format(", ".join(test_datasets.keys())))
	print(
		"Hyperparameters -> optimizer: AdamW, lr: {}, batch_size: {}, epochs: {}".format(
			args.lr,
			args.batch_size,
			args.epochs,
		)
	)

	experiment_results = []
	model_kwargs = {"in_channels": 3, "out_channels": 1, "base_channels": args.base_channels}

	for seed in args.seeds:
		print("\n===== Seed {} =====".format(seed))
		set_seed(seed, deterministic=args.deterministic)

		run_dir = output_dir / "seed_{}".format(seed)
		run_dir.mkdir(parents=True, exist_ok=True)

		train_split, val_split = split_samples(train_samples, args.val_ratio, seed)
		train_loader, val_loader = create_data_loaders(
			train_samples=train_split,
			val_samples=val_split,
			image_size=args.image_size,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			pin_memory=pin_memory,
			seed=seed,
		)

		model = instantiate_model(device, model_name=args.model, model_kwargs=model_kwargs)
		scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
		optimizer = torch.optim.AdamW(
			model.parameters(),
			lr=args.lr,
			weight_decay=args.weight_decay,
		)

		best_score = -1.0
		best_threshold = float(args.threshold)
		best_path = run_dir / "best_model.pt"
		history = []

		for epoch in range(1, args.epochs + 1):
			train_metrics = train_one_epoch(model, train_loader, optimizer, device, use_amp=use_amp, scaler=scaler)
			val_metrics = (
				evaluate_loader_with_thresholds(
					model,
					val_loader,
					device,
					use_amp=use_amp,
					threshold=args.threshold,
					threshold_candidates=threshold_candidates,
				)
				if val_loader is not None
				else {**train_metrics, "threshold": float(args.threshold)}
			)
			monitor_metrics = val_metrics

			history_entry = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
			history.append(history_entry)

			if monitor_metrics["dice"] > best_score:
				best_score = monitor_metrics["dice"]
				best_threshold = float(monitor_metrics.get("threshold", args.threshold))
				torch.save(
					{
						"epoch": epoch,
						"seed": seed,
						"model_state_dict": model.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
						"config": {
							"model_name": args.model,
							"model_kwargs": model_kwargs,
							"image_size": args.image_size,
							"batch_size": args.batch_size,
							"epochs": args.epochs,
							"learning_rate": args.lr,
							"weight_decay": args.weight_decay,
							"val_ratio": args.val_ratio,
							"best_threshold": best_threshold,
							"threshold_candidates": threshold_candidates,
						},
						"best_metrics": monitor_metrics,
					},
					best_path,
				)

			if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
				print(
					"Epoch {}/{} | train loss {:.4f} dice {:.4f} iou {:.4f} | val loss {:.4f} dice {:.4f} iou {:.4f} thr {:.2f}".format(
						epoch,
						args.epochs,
						train_metrics["loss"],
						train_metrics["dice"],
						train_metrics["iou"],
						val_metrics["loss"],
						val_metrics["dice"],
						val_metrics["iou"],
						val_metrics.get("threshold", args.threshold),
					)
				)

		save_json(run_dir / "history.json", history)
		save_json(
			run_dir / "config.json",
			{
				"seed": seed,
				"train_samples": len(train_split),
				"val_samples": len(val_split),
				"train_images": str(paths["train_images"]),
				"train_masks": str(paths["train_masks"]),
				"test_root": str(paths["test_root"]),
				"image_size": args.image_size,
				"batch_size": args.batch_size,
				"epochs": args.epochs,
				"optimizer": "AdamW",
				"model_name": args.model,
				"base_channels": args.base_channels,
				"amp": use_amp,
				"deterministic": args.deterministic,
				"num_workers": args.num_workers,
				"best_threshold": best_threshold,
				"threshold_candidates": threshold_candidates,
				"learning_rate": args.lr,
				"weight_decay": args.weight_decay,
				"device": str(device),
			},
		)

		checkpoint = torch.load(best_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		test_metrics = evaluate_test_datasets(
			model=model,
			test_datasets=test_datasets,
			image_size=args.image_size,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			device=device,
			pin_memory=pin_memory,
			use_amp=use_amp,
			threshold=best_threshold,
		)
		save_json(run_dir / "test_metrics.json", test_metrics)
		print_dataset_metrics("Test results for seed {}".format(seed), test_metrics)

		experiment_results.append(
			{
				"seed": seed,
				"checkpoint": str(best_path),
				"best_epoch": checkpoint["epoch"],
				"test_metrics": test_metrics,
			}
		)

	summary = summarize_experiment_results(experiment_results)
	save_json(output_dir / "experiment_results.json", experiment_results)
	save_json(output_dir / "summary.json", summary)
	save_summary_csv(output_dir / "summary.csv", summary)
	print_summary_table(summary)


if __name__ == "__main__":
	main()

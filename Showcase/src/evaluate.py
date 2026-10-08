import argparse
from pathlib import Path

from models import MODEL_NAMES
from utils.training import (
	default_data_paths,
	evaluate_checkpoint,
	print_dataset_metrics,
	print_summary_table,
	resolve_device,
	save_json,
	save_summary_csv,
	summarize_experiment_results,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate UNet checkpoints on all test datasets")
	parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
	parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
	parser.add_argument("--checkpoint-dir", type=Path, default=None)
	parser.add_argument("--output-file", type=Path, default=None)
	parser.add_argument("--model", type=str, default=None, choices=MODEL_NAMES)
	parser.add_argument("--image-size", type=int, default=352)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold; defaults to the best threshold saved during training")
	parser.add_argument("--device", type=str, default="auto")
	return parser.parse_args()


def collect_checkpoints(project_root: Path, args: argparse.Namespace) -> list[Path]:
	checkpoints = []

	for checkpoint in args.checkpoint:
		checkpoint_path = checkpoint if checkpoint.is_absolute() else project_root / checkpoint
		checkpoints.append(checkpoint_path)

	if args.checkpoint_dir is not None:
		checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else project_root / args.checkpoint_dir
		direct_checkpoint = checkpoint_dir / "best_model.pt"
		if direct_checkpoint.exists():
			checkpoints.append(direct_checkpoint)
		else:
			checkpoints.extend(sorted(checkpoint_dir.glob("seed_*/best_model.pt")))

	deduplicated = []
	seen = set()
	for checkpoint in checkpoints:
		resolved = checkpoint.resolve()
		if resolved not in seen:
			seen.add(resolved)
			deduplicated.append(resolved)

	if not deduplicated:
		raise FileNotFoundError("No checkpoints provided. Use --checkpoint or --checkpoint-dir.")

	return deduplicated


def main() -> None:
	args = parse_args()
	project_root = args.project_root.resolve()
	device = resolve_device(args.device)
	test_root = default_data_paths(project_root)["test_root"]
	checkpoints = collect_checkpoints(project_root, args)

	results = []
	for checkpoint_path in checkpoints:
		result = evaluate_checkpoint(
			checkpoint_path=checkpoint_path,
			test_root=test_root,
			image_size=args.image_size,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			device=device,
			threshold=args.threshold,
			model_name=args.model,
		)
		print_dataset_metrics("Evaluation for {}".format(checkpoint_path.parent.name), result["test_metrics"])
		results.append(result)

	payload = {"runs": results}
	if len(results) > 1:
		summary = summarize_experiment_results(results)
		payload["summary"] = summary
		print_summary_table(summary)

	if args.output_file is not None:
		output_file = args.output_file if args.output_file.is_absolute() else project_root / args.output_file
		save_json(output_file, payload)
		if len(results) > 1:
			csv_path = output_file.with_suffix(".csv")
			save_summary_csv(csv_path, payload["summary"])


if __name__ == "__main__":
	main()

import csv
import json
import os
import random
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.polyp_dataset import PolypDataset, SamplePair, collect_image_mask_pairs
from models import create_model
from utils.metrics import batch_metrics, batch_metrics_from_logits, segmentation_loss

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_SEEDS = [1337]
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_THRESHOLD = 0.5
DEFAULT_THRESHOLD_CANDIDATES = [round(0.1 + 0.025 * index, 3) for index in range(33)]


def normalize_thresholds(thresholds: Optional[Sequence[float]]) -> List[float]:
    if not thresholds:
        return []

    normalized = []
    seen = set()
    for threshold in thresholds:
        threshold_value = round(float(threshold), 4)
        if threshold_value < 0.0 or threshold_value > 1.0:
            raise ValueError("Thresholds must be within [0, 1]")
        if threshold_value not in seen:
            seen.add(threshold_value)
            normalized.append(threshold_value)
    return sorted(normalized)


def default_data_paths(project_root: Path) -> Dict[str, Path]:
    data_root = project_root / "data"
    return {
        "train_images": data_root / "TrainDataset" / "TrainDataset" / "image",
        "train_masks": data_root / "TrainDataset" / "TrainDataset" / "masks",
        "test_root": data_root / "TestDataset" / "TestDataset",
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_samples(
    samples: Sequence[SamplePair],
    val_ratio: float,
    seed: int,
) -> Tuple[List[SamplePair], List[SamplePair]]:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("At least one sample is required for splitting")

    if val_ratio <= 0.0:
        return sample_list, []

    if len(sample_list) < 2:
        raise ValueError("Validation split requires at least two samples")

    shuffled = list(sample_list)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_size = min(val_size, len(shuffled) - 1)

    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    return train_samples, val_samples


def create_data_loaders(
    train_samples: Sequence[SamplePair],
    val_samples: Sequence[SamplePair],
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset = PolypDataset(samples=train_samples, image_size=image_size, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )

    if not val_samples:
        return train_loader, None

    val_dataset = PolypDataset(samples=val_samples, image_size=image_size, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    return train_loader, val_loader


def discover_test_datasets(test_root: Path) -> Dict[str, Tuple[Path, Path]]:
    if not test_root.exists():
        raise FileNotFoundError("Test root not found: {}".format(test_root))

    datasets = {}
    for dataset_dir in sorted(test_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        image_dir = dataset_dir / "images"
        mask_dir = dataset_dir / "masks"
        if image_dir.exists() and mask_dir.exists():
            datasets[dataset_dir.name] = (image_dir, mask_dir)

    if not datasets:
        raise RuntimeError("No test datasets found in {}".format(test_root))

    return datasets


def instantiate_model(
    device: torch.device,
    model_name: str = "premier_model",
    model_kwargs: Optional[Dict[str, int]] = None,
) -> torch.nn.Module:
    model_kwargs = model_kwargs or {"in_channels": 3, "out_channels": 1, "base_channels": 64}
    model = create_model(model_name, **model_kwargs)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
        with autocast_context:
            logits = model(images)
            loss = segmentation_loss(logits, masks)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            dice_values, iou_values = batch_metrics_from_logits(logits, masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_dice += dice_values.sum().item()
        running_iou += iou_values.sum().item()
        total_samples += batch_size

    return {
        "loss": running_loss / total_samples,
        "dice": running_dice / total_samples,
        "iou": running_iou / total_samples,
    }


def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = False) -> Dict[str, float]:
    return evaluate_loader_with_thresholds(model, loader, device, use_amp=use_amp)


def evaluate_loader_with_thresholds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    threshold_candidates: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_samples = 0
    amp_enabled = use_amp and device.type == "cuda"
    candidate_thresholds = normalize_thresholds(threshold_candidates)
    search_stats = {
        candidate: {"dice": 0.0, "iou": 0.0}
        for candidate in candidate_thresholds
    }

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            autocast_context = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
            with autocast_context:
                logits = model(images)
                loss = segmentation_loss(logits, masks)
            probabilities = torch.sigmoid(logits)
            if candidate_thresholds:
                for candidate in candidate_thresholds:
                    dice_values, iou_values = batch_metrics(probabilities, masks, threshold=candidate)
                    search_stats[candidate]["dice"] += dice_values.sum().item()
                    search_stats[candidate]["iou"] += iou_values.sum().item()

                dice_values, iou_values = batch_metrics(probabilities, masks, threshold=threshold)
            else:
                dice_values, iou_values = batch_metrics(probabilities, masks, threshold=threshold)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_dice += dice_values.sum().item()
            running_iou += iou_values.sum().item()
            total_samples += batch_size

    selected_threshold = float(threshold)
    if candidate_thresholds and total_samples > 0:
        best_threshold = max(
            candidate_thresholds,
            key=lambda candidate: (
                search_stats[candidate]["dice"] / total_samples,
                -abs(candidate - DEFAULT_THRESHOLD),
                -candidate,
            ),
        )
        selected_threshold = float(best_threshold)
        running_dice = search_stats[best_threshold]["dice"]
        running_iou = search_stats[best_threshold]["iou"]

    return {
        "loss": running_loss / total_samples,
        "dice": running_dice / total_samples,
        "iou": running_iou / total_samples,
        "threshold": selected_threshold,
    }


def evaluate_test_datasets(
    model: torch.nn.Module,
    test_datasets: Dict[str, Tuple[Path, Path]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    pin_memory: bool,
    use_amp: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Dict[str, float]]:
    dataset_metrics = {}
    for dataset_name, (image_dir, mask_dir) in test_datasets.items():
        samples = collect_image_mask_pairs(image_dir, mask_dir)
        dataset = PolypDataset(samples=samples, image_size=image_size, augment=False)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=seed_worker if num_workers > 0 else None,
        )

        metrics = evaluate_loader_with_thresholds(model, loader, device, use_amp=use_amp, threshold=threshold)
        dataset_metrics[dataset_name] = {
            "dice": float(metrics["dice"]),
            "iou": float(metrics["iou"]),
            "loss": float(metrics["loss"]),
            "threshold": float(metrics["threshold"]),
            "num_samples": len(dataset),
        }

    return dataset_metrics


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    model_name: Optional[str] = None,
) -> Tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_kwargs = checkpoint.get("config", {}).get(
        "model_kwargs",
        {"in_channels": 3, "out_channels": 1, "base_channels": 64},
    )
    resolved_model_name = model_name or checkpoint.get("config", {}).get("model_name", "premier_model")
    model = instantiate_model(device, model_name=resolved_model_name, model_kwargs=model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(
    checkpoint_path: Path,
    test_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool = False,
    threshold: Optional[float] = None,
    model_name: Optional[str] = None,
) -> Dict[str, object]:
    test_datasets = discover_test_datasets(test_root)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device, model_name=model_name)
    pin_memory = device.type == "cuda"
    checkpoint_threshold = checkpoint.get("config", {}).get("best_threshold", DEFAULT_THRESHOLD)
    threshold_to_use = float(checkpoint_threshold if threshold is None else threshold)
    test_metrics = evaluate_test_datasets(
        model=model,
        test_datasets=test_datasets,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        pin_memory=pin_memory,
        use_amp=use_amp,
        threshold=threshold_to_use,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "seed": checkpoint.get("seed"),
        "best_epoch": checkpoint.get("epoch"),
        "threshold": threshold_to_use,
        "test_metrics": test_metrics,
    }


def summarize_experiment_results(experiment_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not experiment_results:
        raise ValueError("At least one experiment result is required")

    dataset_names = sorted(
        {
            dataset_name
            for result in experiment_results
            for dataset_name in result["test_metrics"].keys()
        }
    )

    summary = {
        "num_runs": len(experiment_results),
        "datasets": {},
        "overall": {},
    }

    overall_dice_values = []
    overall_iou_values = []

    for dataset_name in dataset_names:
        dice_values = []
        iou_values = []
        for result in experiment_results:
            metrics = result["test_metrics"][dataset_name]
            dice_values.append(float(metrics["dice"]))
            iou_values.append(float(metrics["iou"]))

        summary["datasets"][dataset_name] = {
            "dice_mean": statistics.mean(dice_values),
            "dice_std": statistics.stdev(dice_values) if len(dice_values) > 1 else 0.0,
            "iou_mean": statistics.mean(iou_values),
            "iou_std": statistics.stdev(iou_values) if len(iou_values) > 1 else 0.0,
            "runs": len(dice_values),
        }

    for result in experiment_results:
        dice_values = [float(metrics["dice"]) for metrics in result["test_metrics"].values()]
        iou_values = [float(metrics["iou"]) for metrics in result["test_metrics"].values()]
        overall_dice_values.append(statistics.mean(dice_values))
        overall_iou_values.append(statistics.mean(iou_values))

    summary["overall"] = {
        "dice_mean": statistics.mean(overall_dice_values),
        "dice_std": statistics.stdev(overall_dice_values) if len(overall_dice_values) > 1 else 0.0,
        "iou_mean": statistics.mean(overall_iou_values),
        "iou_std": statistics.stdev(overall_iou_values) if len(overall_iou_values) > 1 else 0.0,
        "runs": len(overall_dice_values),
    }

    return summary


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_summary_csv(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "dice_mean", "dice_std", "iou_mean", "iou_std", "runs"])

        for dataset_name, metrics in summary["datasets"].items():
            writer.writerow(
                [
                    dataset_name,
                    metrics["dice_mean"],
                    metrics["dice_std"],
                    metrics["iou_mean"],
                    metrics["iou_std"],
                    metrics["runs"],
                ]
            )

        overall_metrics = summary["overall"]
        writer.writerow(
            [
                "overall",
                overall_metrics["dice_mean"],
                overall_metrics["dice_std"],
                overall_metrics["iou_mean"],
                overall_metrics["iou_std"],
                overall_metrics["runs"],
            ]
        )


def print_dataset_metrics(title: str, dataset_metrics: Dict[str, Dict[str, float]]) -> None:
    print(title)
    print("{:<18}{:>12}{:>12}".format("Dataset", "Dice", "IoU"))
    for dataset_name, metrics in dataset_metrics.items():
        print(
            "{:<18}{:>12.4f}{:>12.4f}".format(
                dataset_name,
                metrics["dice"],
                metrics["iou"],
            )
        )


def print_summary_table(summary: Dict[str, object]) -> None:
    print("\nMean +- Std across runs")
    print("{:<18}{:>24}{:>24}".format("Dataset", "Dice", "IoU"))

    for dataset_name, metrics in summary["datasets"].items():
        dice_text = "{:.4f} +- {:.4f}".format(metrics["dice_mean"], metrics["dice_std"])
        iou_text = "{:.4f} +- {:.4f}".format(metrics["iou_mean"], metrics["iou_std"])
        print("{:<18}{:>24}{:>24}".format(dataset_name, dice_text, iou_text))

    overall_metrics = summary["overall"]
    overall_dice = "{:.4f} +- {:.4f}".format(overall_metrics["dice_mean"], overall_metrics["dice_std"])
    overall_iou = "{:.4f} +- {:.4f}".format(overall_metrics["iou_mean"], overall_metrics["iou_std"])
    print("{:<18}{:>24}{:>24}".format("Overall", overall_dice, overall_iou))
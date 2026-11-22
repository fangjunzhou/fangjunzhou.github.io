"""Plot the first two principal components of ImageNet Inception features."""

import argparse
import hashlib
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from datasets import IterableDataset, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import hf_hub_download
from matplotlib.figure import Figure
from PIL import Image
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import interpolate

FEATURE_DIM = 2048
IMAGENET_REPO = "ILSVRC/imagenet-1k"
IMAGENETTE_REPO = "ShaomuTan/imagenette"
IMAGENET_V2_SIZE = 10_000
FID_WEIGHTS_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"


def load_imagenet_v2(limit: int | None, seed: int) -> IterableDataset:
    archive = hf_hub_download(
        repo_id="vaishaal/ImageNetV2",
        filename="imagenetv2-matched-frequency.tar.gz",
        repo_type="dataset",
        revision="d626240be2538720e83103a0e1178d24aca8b12c",
    )
    dataset = load_dataset(
        "webdataset", data_files={"test": archive}, split="test", streaming=True
    ).rename_column("jpeg", "image")
    if limit is not None and limit < IMAGENET_V2_SIZE:
        sample_indices = set(random.Random(seed).sample(range(IMAGENET_V2_SIZE), limit))
        dataset = dataset.filter(lambda _, index: index in sample_indices, with_indices=True)
    return dataset


def load_model() -> InceptionV3:
    weights = hf_hub_download(
        repo_id="FoundationVision/unitok_external",
        filename="weights-inception-2015-12-05-6726825d.pth",
        token=False,
    )
    with open(weights, "rb") as checkpoint:
        if hashlib.file_digest(checkpoint, "sha256").hexdigest() != FID_WEIGHTS_SHA256:
            raise ValueError("The Hugging Face checkpoint does not match the PyTorch FID release")

    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    block = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]
    # PyTorch FID 0.3.0 loads its checkpoint inside the constructor.
    with patch("pytorch_fid.inception.load_state_dict_from_url", return_value=state_dict):
        return InceptionV3([block], resize_input=False).eval()


def prepare_image(image: Image.Image) -> torch.Tensor:
    pixels = np.array(image.convert("RGB"))
    image_tensor = torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)
    # Match pytorch-fid's resize before batching images of different sizes.
    return interpolate(
        image_tensor.unsqueeze(0), size=(299, 299), mode="bilinear", align_corners=False
    )[0]


@torch.inference_mode()
def extract_features(
    dataset: IterableDataset, batch_size: int, device: torch.device
) -> np.ndarray:
    model = load_model().to(device)
    feature_batches = []
    total = 0

    for batch_number, batch in enumerate(dataset.batch(batch_size), start=1):
        images = torch.stack([prepare_image(image) for image in batch["image"]])
        pooled = model(images.to(device))[0]
        feature_batches.append(pooled.flatten(1).cpu().numpy())
        total += len(images)
        if batch_number % 10 == 0:
            print(f"Extracted features for {total:,} images", file=sys.stderr)

    if total < 2:
        raise ValueError("PCA requires at least two images")
    print(f"Extracted features for {total:,} images", file=sys.stderr)
    return np.concatenate(feature_batches)


def project_to_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = features.astype(np.float64)
    centered -= centered.mean(axis=0)
    # The common 1 / (n - 1) factor cancels in the explained-variance ratios.
    eigenvalues, axes = np.linalg.eigh(centered.T @ centered)
    scores = centered @ axes[:, -2:][:, ::-1]
    # Roundoff can give zero eigenvalues a small negative value.
    eigenvalues = eigenvalues.clip(min=0)
    total = eigenvalues.sum()
    explained = eigenvalues[-2:][::-1] / total if total > 0 else np.zeros(2)
    return scores, explained


def save_animation_data(
    dataset: IterableDataset, scores: np.ndarray, explained: np.ndarray, output: Path
) -> None:
    spreads = scores.std(axis=0)
    standardized = scores / np.where(spreads > 0, spreads, 1)
    selected = []
    # Pick real points near the main cloud, the left tail, and both PC2 extremes.
    targets = [(0.8, 0.1), (-2.8, 0), (0, 2.2), (-0.2, -2.2)]
    for target in targets[: len(scores)]:
        distances = np.sum((standardized - target) ** 2, axis=1)
        distances[selected] = np.inf
        selected.append(int(np.argmin(distances)))

    preview_positions = {index: position for position, index in enumerate(selected)}
    previews = {}
    for index, example in enumerate(dataset):
        if index in preview_positions:
            image = example["image"].convert("RGB")
            image.thumbnail((384, 384), Image.Resampling.LANCZOS)
            previews[f"preview_{preview_positions[index]}"] = np.array(image)
            if len(previews) == len(selected):
                break
    if len(previews) != len(selected):
        raise ValueError("Could not find the source images for the selected PCA points")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        np.savez_compressed(destination, scores=scores, explained=explained, selected=selected, **previews)


def save_scatter(
    scores: np.ndarray, explained: np.ndarray, dataset_name: str, output: Path
) -> None:
    background = "#1e1e2e"
    text = "#cdd6f4"
    figure = Figure(figsize=(8, 6), facecolor=background, layout="constrained")
    axis = figure.subplots()
    axis.set_facecolor(background)
    axis.scatter(scores[:, 0], scores[:, 1], s=9, color="#89b4fa", alpha=0.4, linewidths=0)
    axis.set_title(
        f"PCA of {dataset_name} Inception features ({len(scores):,} images)", color=text
    )
    axis.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)", color=text)
    axis.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)", color=text)
    axis.tick_params(colors=text)
    for spine in axis.spines.values():
        spine.set_color("#585b70")
    axis.set_axisbelow(True)
    axis.grid(color="#313244")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("imagenet", "imagenet-v2", "imagenette"),
        default="imagenet",
        help="Hugging Face evaluation set (default: imagenet)",
    )
    parser.add_argument("--output", type=Path, help="Scatter plot path, e.g. pca.png")
    parser.add_argument(
        "--animation-data", type=Path, help="Save PCA coordinates and selected source images to an NPZ"
    )
    parser.add_argument("--limit", type=int, help="Sample at most this many images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --limit (default: 0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size (default: 32)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()

    if args.output is None and args.animation_data is None:
        parser.error("specify --output, --animation-data, or both")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")

    if args.dataset == "imagenette":
        dataset_name = "Imagenette"
        split = "validation"
        dataset = load_dataset(IMAGENETTE_REPO, "320px", split="validation", streaming=True)
    elif args.dataset == "imagenet-v2":
        dataset_name = "ImageNetV2"
        split = "Matched Frequency test set"
        dataset = load_imagenet_v2(args.limit, args.seed)
    else:
        dataset_name = "ImageNet"
        split = "validation"
        try:
            dataset = load_dataset(IMAGENET_REPO, split="validation", streaming=True)
        except DatasetNotFoundError as error:
            parser.error(
                f"{error}\n"
                "If access is required, accept the terms at "
                f"https://huggingface.co/datasets/{IMAGENET_REPO} "
                "and authenticate with `hf auth login` or the HF_TOKEN environment variable."
            )
    if args.limit is not None and args.dataset != "imagenet-v2":
        # Mix row groups as well as rows so a short run can span multiple classes.
        dataset = dataset.reshard().shuffle(seed=args.seed, buffer_size=1000).take(args.limit)
    dataset = dataset.select_columns(["image"])

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Running Inception-v3 on {dataset_name} {split} using {device}", file=sys.stderr)
    features = extract_features(dataset, args.batch_size, torch.device(device))
    if args.dataset == "imagenet-v2":
        expected = min(args.limit, IMAGENET_V2_SIZE) if args.limit is not None else IMAGENET_V2_SIZE
        if len(features) != expected:
            raise ValueError(f"Expected {expected:,} ImageNetV2 images, got {len(features):,}")
    scores, explained = project_to_2d(features)
    if args.output is not None:
        save_scatter(scores, explained, dataset_name, args.output)
        print(f"Saved {args.output}")
    if args.animation_data is not None:
        save_animation_data(dataset, scores, explained, args.animation_data)
        print(f"Saved {args.animation_data}")


if __name__ == "__main__":
    main()

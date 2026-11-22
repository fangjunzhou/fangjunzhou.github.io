# FID visualizations

Each scene has its own file in `scenes/`. The Nix development shell provides uv, Python, Cairo, Pango, and the LaTeX tools Manim uses to render its Computer Modern labels.

From the repository root, pass a scene file to Manim to render it.

```sh
nix develop -c uv run --locked --package frechet-distance-visualizations \
  manim -qm assets/media/reading-notes/frechet-distance/scenes/average_log_likelihood.py
```

To render all the animations, run them in the same Nix shell.

```sh
nix develop -c bash -e <<'SH'
scene_dir=assets/media/reading-notes/frechet-distance/scenes
for scene in average_log_likelihood sparse_gaussian_mixture latent_variable_likelihood \
  frechet_monte_carlo gaussian_pairings identical_gaussian_pairings frechet_distance_breakdown \
  shared_gaussian_draw rotated_gaussian_draw svd_gaussian_pairing frechet_computation \
  imagenet_pca_inspection
do
  uv run --locked --package frechet-distance-visualizations manim -qm "$scene_dir/$scene.py"
done
SH
```

Manim writes each video to `media/videos/<scene-file-stem>/720p30/`. The scenes use the [Catppuccin Mocha palette](https://catppuccin.com/palette/#mocha).

To update the videos and preview images embedded in the post, run these FFmpeg commands from the repository root after rendering.

```sh
asset_dir=assets/media/reading-notes/frechet-distance

publish_animation() {
  video="media/videos/$1/720p30/$2.mp4"
  asset="$asset_dir/$3"
  ffmpeg -y -i "$video" -c copy -movflags +faststart "$asset.mp4"
  ffmpeg -y -ss "$4" -i "$video" -frames:v 1 -update 1 "$asset-poster.png"
}

publish_animation average_log_likelihood AverageLogLikelihood average-log-likelihood 10.4
publish_animation sparse_gaussian_mixture SparseGaussianMixture sparse-gaussian-mixture 8.6
publish_animation latent_variable_likelihood LatentVariableLikelihood latent-variable-likelihood 25.8
publish_animation frechet_monte_carlo FrechetMonteCarlo frechet-monte-carlo 16.8
publish_animation gaussian_pairings GaussianPairings gaussian-pairings 14.2
publish_animation identical_gaussian_pairings IdenticalGaussianPairings identical-gaussian-pairings 25.1
publish_animation frechet_distance_breakdown FrechetDistanceBreakdown frechet-distance-breakdown 4.9
publish_animation shared_gaussian_draw SharedGaussianDraw shared-gaussian-draw 20.3
publish_animation rotated_gaussian_draw RotatedGaussianDraw rotated-gaussian-draw 6.7
publish_animation svd_gaussian_pairing SvdGaussianPairing svd-gaussian-pairing 18.7
publish_animation frechet_computation FrechetComputation frechet-computation 22.2
publish_animation imagenet_pca_inspection ImageNetPCAInspection imagenet-pca-inspection 6.2
```

Use this command to render the figure comparing distributions with the same mean and covariance.

```sh
nix develop -c uv run --locked --package frechet-distance-visualizations \
  manim -qh -s -o matching-gaussian-moments \
  assets/media/reading-notes/frechet-distance/scenes/matching_gaussian_moments.py
cp media/images/matching_gaussian_moments/matching-gaussian-moments.png assets/media/reading-notes/frechet-distance/
```

If you already have uv, Manim's system dependencies, LaTeX, and dvisvgm, you can run the same command starting at `uv run`. [Manim installation guide](https://docs.manim.community/en/stable/installation/uv.html)

## ImageNet feature PCA

The script streams the ImageNet-1K validation split from Hugging Face. Accept the dataset's access terms and authenticate with `uvx --from huggingface-hub hf auth login` or set `HF_TOKEN`. [ImageNet-1K dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

```sh
nix develop -c uv run --locked --python 3.13 \
  --package frechet-distance-visualizations --extra imagenet-pca \
  python assets/media/reading-notes/frechet-distance/imagenet_pca.py \
  --limit 5000 --output imagenet-pca.png
```

For a run without signing in, use `--dataset imagenet-v2` for the public 10,000-image ImageNetV2 Matched Frequency test set. For a smaller run, use `--dataset imagenette` for the public 320 px validation split of the 10-class ImageNet subset. [ImageNetV2 test set](https://github.com/modestyachts/ImageNetV2), [Imagenette dataset](https://huggingface.co/datasets/ShaomuTan/imagenette)

Each point represents one image. The script extracts 2,048 features from the final average-pooling layer of the FID-compatible Inception-v3 model, fits PCA to the centered features, and labels each axis with its share of explained variance. It downloads the weights from Hugging Face and verifies that their SHA-256 matches the PyTorch FID release. [Hugging Face checkpoint](https://huggingface.co/FoundationVision/unitok_external/blob/main/weights-inception-2015-12-05-6726825d.pth), [PyTorch FID](https://github.com/mseitzer/pytorch-fid)

`--limit 5000` takes up to 5,000 images. ImageNetV2 uses a seeded sample without replacement after caching the matched-frequency archive. The other splits use a seeded streaming shuffle. Omit the limit to use the entire split. The script selects CUDA, MPS, or CPU automatically, and `--device cpu` overrides that selection.

The `ImageNetPCAInspection` scene reads the point coordinates and four source-image previews from the bundled NPZ. To regenerate that input, run the experiment with the same 8,192-image sample.

```sh
asset_dir=assets/media/reading-notes/frechet-distance
nix develop -c uv run --locked --python 3.13 \
  --package frechet-distance-visualizations --extra imagenet-pca \
  python "$asset_dir/imagenet_pca.py" --dataset imagenet-v2 --limit 8192 --seed 0 --batch-size 64 \
  --animation-data "$asset_dir/scenes/imagenet-v2-pca-data.npz"
```

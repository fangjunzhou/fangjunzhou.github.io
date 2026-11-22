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
  shared_gaussian_draw rotated_gaussian_draw svd_gaussian_pairing frechet_computation
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
```

Use this command to render the figure comparing distributions with the same mean and covariance.

```sh
nix develop -c uv run --locked --package frechet-distance-visualizations \
  manim -qh -s -o matching-gaussian-moments \
  assets/media/reading-notes/frechet-distance/scenes/matching_gaussian_moments.py
cp media/images/matching_gaussian_moments/matching-gaussian-moments.png assets/media/reading-notes/frechet-distance/
```

If you already have uv, Manim's system dependencies, LaTeX, and dvisvgm, you can run the same command starting at `uv run`. [Manim installation guide](https://docs.manim.community/en/stable/installation/uv.html)

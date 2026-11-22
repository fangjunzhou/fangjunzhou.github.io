# FID visualizations

Run this command from the repository root to render the animations. The Nix development shell provides uv, Python, Cairo, Pango, and the LaTeX tools Manim uses to render its Computer Modern labels.

```sh
nix develop -c uv run --locked --package frechet-distance-visualizations \
  manim -qm assets/media/reading-notes/frechet-distance/scenes.py \
  AverageLogLikelihood SparseGaussianMixture LatentVariableLikelihood FrechetMonteCarlo \
  GaussianPairings IdenticalGaussianPairings FrechetDistanceBreakdown
```

Manim writes the videos to `media/videos/scenes/720p30/`. The scenes use the [Catppuccin Mocha palette](https://catppuccin.com/palette/#mocha).

To update the videos and preview images embedded in the post, run these FFmpeg commands from the repository root after rendering.

```sh
video_dir=media/videos/scenes/720p30
asset_dir=assets/media/reading-notes/frechet-distance
ffmpeg -y -i "$video_dir/AverageLogLikelihood.mp4" -c copy -movflags +faststart "$asset_dir/average-log-likelihood.mp4"
ffmpeg -y -ss 10.4 -i "$video_dir/AverageLogLikelihood.mp4" -frames:v 1 "$asset_dir/average-log-likelihood-poster.png"
ffmpeg -y -i "$video_dir/SparseGaussianMixture.mp4" -c copy -movflags +faststart "$asset_dir/sparse-gaussian-mixture.mp4"
ffmpeg -y -ss 8.6 -i "$video_dir/SparseGaussianMixture.mp4" -frames:v 1 "$asset_dir/sparse-gaussian-mixture-poster.png"
ffmpeg -y -i "$video_dir/LatentVariableLikelihood.mp4" -c copy -movflags +faststart "$asset_dir/latent-variable-likelihood.mp4"
ffmpeg -y -ss 25.8 -i "$video_dir/LatentVariableLikelihood.mp4" -frames:v 1 "$asset_dir/latent-variable-likelihood-poster.png"
ffmpeg -y -i "$video_dir/FrechetMonteCarlo.mp4" -c copy -movflags +faststart "$asset_dir/frechet-monte-carlo.mp4"
ffmpeg -y -ss 16.8 -i "$video_dir/FrechetMonteCarlo.mp4" -frames:v 1 "$asset_dir/frechet-monte-carlo-poster.png"
ffmpeg -y -i "$video_dir/GaussianPairings.mp4" -c copy -movflags +faststart "$asset_dir/gaussian-pairings.mp4"
ffmpeg -y -ss 14.2 -i "$video_dir/GaussianPairings.mp4" -frames:v 1 -update 1 "$asset_dir/gaussian-pairings-poster.png"
ffmpeg -y -i "$video_dir/IdenticalGaussianPairings.mp4" -c copy -movflags +faststart "$asset_dir/identical-gaussian-pairings.mp4"
ffmpeg -y -ss 25.1 -i "$video_dir/IdenticalGaussianPairings.mp4" -frames:v 1 -update 1 "$asset_dir/identical-gaussian-pairings-poster.png"
ffmpeg -y -i "$video_dir/FrechetDistanceBreakdown.mp4" -c copy -movflags +faststart "$asset_dir/frechet-distance-breakdown.mp4"
ffmpeg -y -ss 4.9 -i "$video_dir/FrechetDistanceBreakdown.mp4" -frames:v 1 -update 1 "$asset_dir/frechet-distance-breakdown-poster.png"
```

Use this command to render the figure comparing distributions with the same mean and covariance.

```sh
nix develop -c uv run --locked --package frechet-distance-visualizations \
  manim -qh -s -o matching-gaussian-moments assets/media/reading-notes/frechet-distance/scenes.py \
  MatchingGaussianMoments
cp media/images/scenes/matching-gaussian-moments.png assets/media/reading-notes/frechet-distance/
```

If you already have uv, Manim's system dependencies, LaTeX, and dvisvgm, you can run the same command starting at `uv run`. [Manim installation guide](https://docs.manim.community/en/stable/installation/uv.html)

from itertools import product

import numpy as np
from manim import DOWN, RIGHT, Axes, Circle, DashedVMobject, Dot, Group, MathTex, Scene

from scene_helpers import (
    BASE,
    BLUE,
    CRUST,
    Covariance,
    GREEN,
    Mixture,
    SURFACE,
    TEXT,
    YELLOW,
    density_panel,
    mixture_density,
)


def matching_moment_distributions() -> tuple[Mixture, Mixture]:
    covariance: Covariance = ((1.0, 0.5), (0.5, 1.0))
    transform = np.linalg.cholesky(covariance)
    offset = 0.94
    # In whitened coordinates the spread of the means is offset by each component's variance.
    component_covariance = tuple(
        tuple(row) for row in (1 - offset**2) * np.asarray(covariance)
    )
    gaussian = ((1.0, (0.0, 0.0), covariance),)
    mixture = tuple(
        (0.25, tuple(transform @ mean), component_covariance)
        for mean in product((-offset, offset), repeat=2)
    )
    return gaussian, mixture


class MatchingGaussianMoments(Scene):
    """Compare two distinct densities with exactly the same Gaussian fit."""

    def construct(self):
        self.camera.background_color = BASE
        gaussian, mixture = matching_moment_distributions()
        covariance = gaussian[0][2]
        transform = np.linalg.cholesky(covariance)
        limit = 3.2
        axis = np.linspace(-limit, limit, 361)
        x, y = np.meshgrid(axis, axis[::-1])
        densities = [mixture_density(distribution, x, y) for distribution in (gaussian, mixture)]
        peak = max(float(density.max()) for density in densities)
        charts = Group()
        for density, label, color in zip(
            densities, ("Single Gaussian", "Mixture of four Gaussians"), (BLUE, GREEN), strict=True
        ):
            panel, image = density_panel(density, peak, label, color)
            axes = Axes(
                x_range=(-limit, limit, 1),
                y_range=(-limit, limit, 1),
                x_length=image.width,
                y_length=image.height,
                tips=False,
                axis_config={"color": SURFACE, "include_ticks": False, "stroke_width": 1.2},
            ).move_to(image)
            ellipse = Circle(radius=image.width / limit, color=YELLOW, stroke_width=1.8)
            ellipse.apply_matrix(transform).move_to(image)
            contour = DashedVMobject(ellipse, num_dashes=32, dashed_ratio=0.55, color=YELLOW)
            mean = Dot(image.get_center(), radius=0.04, color=YELLOW, stroke_color=CRUST, stroke_width=1)
            panel.add(axes, contour, mean)
            charts.add(panel.scale(1.42))
        charts.arrange(RIGHT, buff=0.70, aligned_edge=DOWN)

        matrix = r" \\ ".join(" & ".join(f"{value:g}" for value in row) for row in covariance)
        shared_moments = MathTex(
            r"\mu=(0,0),\qquad \Sigma=\begin{bmatrix}" + matrix + r"\end{bmatrix}",
            font_size=43,
            color=TEXT,
        )
        Group(charts, shared_moments).arrange(DOWN, buff=0.36)
        self.add(charts, shared_moments)

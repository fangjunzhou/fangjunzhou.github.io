"""Shared Gaussian data, plotting helpers, and Catppuccin colors."""

from collections.abc import Sequence
from math import pi
from random import Random

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Circle,
    Dot,
    Group,
    ImageMobject,
    ManimColor,
    MathTex,
    Square,
    Tex,
    VGroup,
)

# Catppuccin Mocha
BASE = "#1e1e2e"
CRUST = "#11111b"
SURFACE = "#45475a"
TEXT = "#cdd6f4"
SUBTEXT = "#bac2de"
BLUE = "#89b4fa"
GREEN = "#a6e3a1"
YELLOW = "#f9e2af"
RED = "#f38ba8"
MAUVE = "#cba6f7"
COMPONENT_COLORS = (BLUE, GREEN, YELLOW, RED, MAUVE)

LIMIT = 3.4
PANEL_WIDTH = 3.4
SAMPLE_COUNT = 64
TEST_COUNT = 4
TEST_DOT_RADIUS = 0.05

# Each component is (weight, mean, covariance).
type Point = tuple[float, ...]
type Covariance = tuple[tuple[float, ...], ...]
type Mixture = tuple[tuple[float, Point, Covariance], ...]

TRUE_DISTRIBUTION: Mixture = (
    (0.55, (-1.38, -0.82), ((0.38, 0.27), (0.27, 0.36))),
    (0.45, (1.24, 0.83), ((0.52, -0.24), (-0.24, 0.28))),
)
SPARSE_DISTRIBUTION: Mixture = (
    (
        0.2,
        (-2.85, -1.25, -0.85),
        ((0.10, 0.047, 0.036), (0.047, 0.056, 0.028), (0.036, 0.028, 0.052)),
    ),
    (
        0.2,
        (-1.2, 1.57, 1.2),
        ((0.035, -0.021, 0.024), (-0.021, 0.065, -0.034), (0.024, -0.034, 0.09)),
    ),
    (
        0.2,
        (0.05, -0.95, -0.4),
        ((0.038, 0.022, -0.023), (0.022, 0.065, -0.04), (-0.023, -0.04, 0.062)),
    ),
    (
        0.2,
        (2.7, -1.64, 1.56),
        ((0.13, -0.061, 0.038), (-0.061, 0.069, -0.025), (0.038, -0.025, 0.044)),
    ),
    (
        0.2,
        (2.7, 1.52, -1.13),
        ((0.065, 0.038, -0.041), (0.038, 0.073, -0.046), (-0.041, -0.046, 0.076)),
    ),
)


def mixture_density(mixture: Mixture, x, y):
    points = np.stack(np.broadcast_arrays(x, y), axis=-1)
    density = np.zeros(points.shape[:-1])
    for weight, mean, covariance in mixture:
        factor = np.linalg.cholesky(covariance)
        offsets = (points - mean).reshape(-1, 2)
        standardized = np.linalg.solve(factor, offsets.T)
        squared_distance = np.sum(standardized**2, axis=0).reshape(density.shape)
        density += weight * np.exp(-0.5 * squared_distance) / (2 * pi * np.prod(factor.diagonal()))
    return density


def draw_samples(mixture: Mixture, count: int, rng: Random) -> list[Point]:
    components = [(np.asarray(mean), np.linalg.cholesky(cov)) for _, mean, cov in mixture]
    picks = rng.choices(components, weights=[weight for weight, _, _ in mixture], k=count)
    return [
        tuple(mean + factor @ [rng.gauss(0, 1) for _ in mean])
        for mean, factor in picks
    ]


def text_label(text: str, size: int, color: str = TEXT, *, bold: bool = False) -> Tex:
    return Tex(r"\bfseries\boldmath " + text if bold else text, font_size=size, color=color)


def density_panel(density, peak: float, label: str, color: str):
    # A single square-root scale makes the three heatmaps directly comparable.
    level = np.sqrt(density / peak)[..., None]
    low = np.array([int(CRUST[i : i + 2], 16) for i in (1, 3, 5)])
    high = np.array([int(BLUE[i : i + 2], 16) for i in (1, 3, 5)])
    pixels = np.round(low + level * (high - low)).astype(np.uint8)

    image = ImageMobject(pixels).scale_to_fit_width(PANEL_WIDTH)
    frame = Square(side_length=PANEL_WIDTH, stroke_color=SURFACE, stroke_width=2).move_to(image)
    heading = text_label(label, 33, color).next_to(frame, UP, buff=0.17)
    return Group(image, frame, heading), image


def map_point(image: ImageMobject, point: Point, *, limit: float = LIMIT):
    x, y = point
    return image.get_center() + image.width / (2 * limit) * (x * RIGHT + y * UP)


def sample_dots(
    image: ImageMobject, points: list[Point], color: str, *, radius: float = 0.036,
    limit: float = LIMIT,
) -> VGroup:
    return VGroup(
        *[
            Dot(
                map_point(image, point, limit=limit), radius=radius, color=color, stroke_color=CRUST,
                stroke_width=1.5,
            )
            for point in points
        ]
    )


def gaussian_blob(color: str) -> ImageMobject:
    grid = np.linspace(-4, 4, 129)
    squared_radius = grid[:, None] ** 2 + grid[None, :] ** 2
    density = np.exp(-0.5 * squared_radius)
    pixels = np.empty((*density.shape, 4), dtype=np.uint8)
    pixels[..., :3] = ManimColor(color).to_int_rgb()
    pixels[..., 3] = np.rint(0.78 * 255 * density).astype(np.uint8)
    return ImageMobject(pixels)


def gaussian_contours(axes: Axes, mean, transform, color: str) -> VGroup:
    center = axes.c2p(*mean)
    unit = np.linalg.norm(axes.c2p(1, 0) - axes.c2p(0, 0))
    rings = VGroup(*[
        Circle(
            radius=unit * radius, color=color, stroke_width=1.5,
            stroke_opacity=stroke, fill_opacity=fill,
        ).apply_matrix(transform).move_to(center)
        for radius, stroke, fill in ((2.3, 0.45, 0.06), (1.5, 0.65, 0.08), (0.75, 0.85, 0.13))
    ])
    return VGroup(rings, Dot(center, radius=0.041, color=color)).set_z_index(1)


def convergence_chart(sample_count: int, y_values: Sequence[float]) -> tuple[Axes, VGroup]:
    graph = Axes(
        x_range=(0, np.log10(sample_count), 1),
        y_range=(y_values[0], y_values[-1], y_values[1] - y_values[0]),
        x_length=5.1, y_length=2.5, tips=False,
        axis_config={"color": SUBTEXT, "stroke_width": 1.55, "include_ticks": False},
    )
    x_ticks = VGroup(*[
        MathTex(f"{n:,}".replace(",", "{,}"), font_size=24, color=SUBTEXT)
        .next_to(graph.c2p(np.log10(n), 0), DOWN, buff=0.12)
        for n in (1, 10, 100, 1000, sample_count)
    ])
    y_ticks = VGroup(*[
        MathTex(f"{value:g}", font_size=24, color=SUBTEXT).next_to(graph.c2p(0, value), LEFT, buff=0.12)
        for value in y_values
    ])
    x_label = text_label("Pairs sampled (log scale)", 24, SUBTEXT).next_to(x_ticks, DOWN, buff=0.13)
    return graph, VGroup(graph, x_ticks, y_ticks, x_label)


def gaussian_axes(limit, width, height):
    return Axes(
        x_range=(-limit, limit, 1), y_range=(0, 0.45, 0.15),
        x_length=width, y_length=height, tips=False,
        axis_config={"color": SUBTEXT, "stroke_width": 1.45, "include_ticks": False},
        y_axis_config={"stroke_opacity": 0.25},
    )


def gaussian_density(offset, std):
    return np.exp(-0.5 * (offset / std)**2) / (std * np.sqrt(2 * pi))


def svd_with_rotations(matrix):
    """Factor a 2D matrix with positive determinant into rotations and scales."""
    p, singular_values, qt = np.linalg.svd(matrix)
    # Choose paired signs so Q's first axis points right and Q is a rotation.
    first_sign = np.copysign(1.0, qt[0, 0])
    signs = first_sign * np.array([1.0, np.sign(np.linalg.det(qt))])
    return p * signs, singular_values, qt * signs[:, None]


def axis_numbers(axes, values):
    return VGroup(*[
        MathTex(str(value), font_size=23, color=SUBTEXT).next_to(axes.c2p(value, 0), DOWN, buff=0.12)
        for value in values
    ])

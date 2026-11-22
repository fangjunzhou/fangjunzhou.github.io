from random import Random

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    LaggedStart,
    MathTex,
    ThreeDAxes,
    ThreeDCamera,
    ThreeDScene,
)

from scene_helpers import (
    BASE,
    BLUE,
    COMPONENT_COLORS,
    CRUST,
    Covariance,
    LIMIT,
    SAMPLE_COUNT,
    SPARSE_DISTRIBUTION,
    SUBTEXT,
    TEXT,
    TRUE_DISTRIBUTION,
    density_panel,
    draw_samples,
    gaussian_blob,
    mixture_density,
    sample_dots,
)


def projected_covariance(
    camera: ThreeDCamera, mean: np.ndarray, covariance: Covariance
) -> np.ndarray:
    # Linearize the perspective projection at the Gaussian's mean.
    rotation = camera.generate_rotation_matrix()
    view_mean = rotation @ (mean - camera.frame_center)
    focal_distance = camera.get_focal_distance()
    depth = focal_distance - view_mean[2]
    scale = camera.get_zoom() * focal_distance / depth
    projection = scale * (rotation[:2] + np.outer(view_mean[:2] / depth, rotation[2]))
    return projection @ covariance @ projection.T


class SparseGaussianMixture(ThreeDScene):
    """Move from the 2D heatmap to a 3D example with separated modes."""

    def construct(self):
        self.camera.background_color = BASE
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, focal_distance=23)

        grid = np.linspace(-LIMIT, LIMIT, 241)
        grid_x, grid_y = np.meshgrid(grid, grid[::-1])
        density = mixture_density(TRUE_DISTRIBUTION, grid_x, grid_y)
        flat_panel, flat_image = density_panel(
            density, float(density.max()), "2D Gaussian mixture", BLUE
        )
        flat_panel.scale(1.38, about_point=flat_image.get_center())
        flat_label = flat_panel[-1]
        flat_panel.remove(flat_label)
        flat_samples = draw_samples(TRUE_DISTRIBUTION, SAMPLE_COUNT, Random(19))
        flat_dots = sample_dots(flat_image, flat_samples, BLUE)
        flat_view = Group(flat_panel, flat_dots)
        flat_label.next_to(flat_view, UP, buff=0.23)
        self.add(flat_view, flat_label)
        self.wait(1.5)
        self.play(FadeOut(flat_label), run_time=0.25)

        axes = ThreeDAxes(
            x_range=(-4, 4, 1),
            y_range=(-3, 3, 1),
            z_range=(-2.6, 2.6, 1),
            x_length=8,
            y_length=6,
            z_length=5.2,
            axis_config={
                "color": SUBTEXT,
                "stroke_width": 1.7,
                "stroke_opacity": 0.62,
                "include_ticks": False,
                "tip_width": 0.16,
                "tip_height": 0.20,
            },
        )
        axis_labels = [
            MathTex(name, font_size=28, color=SUBTEXT).next_to(axis.get_end(), direction, buff=0.2)
            for name, axis, direction in zip("xyz", axes.get_axes(), (RIGHT, UP, OUT), strict=True)
        ]
        self.add_fixed_orientation_mobjects(*axis_labels)
        blobs = [
            gaussian_blob(color).move_to(mean)
            for (_, mean, _), color in zip(SPARSE_DISTRIBUTION, COMPONENT_COLORS, strict=True)
        ]
        self.add_fixed_orientation_mobjects(*blobs)
        # The texture spans four standard deviations from its center in each direction.
        corners = 4 * np.array([UP + LEFT, UP + RIGHT, DOWN + LEFT, DOWN + RIGHT])

        def update_blobs(_dt):
            for blob, (_, _, covariance) in zip(blobs, SPARSE_DISTRIBUTION, strict=True):
                mean = blob.get_center()
                factor = np.linalg.cholesky(projected_covariance(self.camera, mean, covariance))
                quad = corners.copy()
                quad[:, :2] = corners[:, :2] @ factor.T
                blob.set_points(quad + mean)

        self.add_updater(update_blobs)
        blob_reveals = [
            FadeIn(blob, shift=blob.get_center()[2] * OUT)
            for blob in blobs
        ]

        self.begin_ambient_camera_rotation(rate=18 * DEGREES)
        self.move_camera(
            phi=66 * DEGREES,
            zoom=0.78,
            frame_center=-0.28 * OUT,
            added_anims=[
                FadeOut(flat_view),
                LaggedStart(Create(axes), FadeIn(*axis_labels), lag_ratio=0.7),
                LaggedStart(*blob_reveals, lag_ratio=0.14),
            ],
            run_time=2.2,
        )
        self.wait(0.6)

        samples = [
            Dot(
                point=np.asarray(point),
                radius=0.044,
                color=TEXT,
                stroke_color=CRUST,
                stroke_width=1,
            )
            for point in draw_samples(SPARSE_DISTRIBUTION, 38, Random(41))
        ]
        self.add_fixed_orientation_mobjects(*samples)
        self.play(
            LaggedStart(*(FadeIn(dot, scale=0.7) for dot in samples), lag_ratio=0.035),
            run_time=1.5,
        )
        self.wait(8)
        self.stop_ambient_camera_rotation()

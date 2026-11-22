import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    MathTex,
    Scene,
    TransformFromCopy,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
)

from scene_helpers import (
    BASE,
    BLUE,
    GREEN,
    RED,
    SUBTEXT,
    TEXT,
    YELLOW,
    axis_numbers,
    gaussian_axes,
    gaussian_density,
    text_label,
)


class FrechetDistanceBreakdown(Scene):
    """Separate the squared mean shift and spread gap for two 1D Gaussians."""

    def construct(self):
        self.camera.background_color = BASE
        # Parameters are (real mean, generated mean, real std, generated std).
        states = np.array([
            [-1.5, 1.5, 2.0, 1.0],
            [-0.75, 0.75, 2.0, 1.0],
            [-0.75, 0.75, 1.5, 1.5],
            [0.0, 0.0, 1.5, 1.5],
        ])
        grid = np.linspace(-8, 8, 321)
        original_axes, centered_axes = (gaussian_axes(8, 6.6, 1.95) for _ in range(2))
        original_panel = VGroup(original_axes, axis_numbers(original_axes, (-6, -3, 0, 3, 6)))
        centered_panel = VGroup(centered_axes, axis_numbers(centered_axes, (-6, -3, 0, 3, 6)))
        original_title = text_label("Original fits", 32, SUBTEXT)
        centered_title = text_label("Centered at zero", 32, SUBTEXT)

        def distribution_readout(mean, std, color):
            mean_value = DecimalNumber(mean, unit=",", num_decimal_places=2, font_size=31, color=color)
            variance_value = DecimalNumber(std**2, num_decimal_places=2, font_size=31, color=color)
            label = VGroup(
                MathTex(r"\mathcal{N}(", font_size=31, color=color), mean_value, variance_value,
                MathTex(")", font_size=31, color=color),
            ).arrange(RIGHT, buff=0.06)
            return label, mean_value, variance_value

        real_label, real_mean, real_variance = distribution_readout(-1.5, 2, BLUE)
        generated_label, generated_mean, generated_variance = distribution_readout(1.5, 1, GREEN)
        legend = VGroup(real_label, generated_label).arrange(RIGHT, buff=0.85)
        top_plot = VGroup(original_title, legend, original_panel).arrange(DOWN, buff=0.17)
        bottom_plot = VGroup(centered_title, centered_panel).arrange(DOWN, buff=0.17)
        plots = VGroup(top_plot, bottom_plot).arrange(DOWN, buff=0.40)

        distance_value = DecimalNumber(10, num_decimal_places=2, font_size=53, color=TEXT)
        mean_value = DecimalNumber(9, num_decimal_places=2, font_size=36, color=YELLOW)
        spread_value = DecimalNumber(1, num_decimal_places=2, font_size=36, color=RED)
        distance_equation = MathTex(
            r"d_{\mathrm{F}}^2", "=", r"(\mu_r-\mu_g)^2", "+", r"(\sigma_r-\sigma_g)^2",
            font_size=38, color=TEXT,
        )
        distance_equation[2].set_color(YELLOW)
        distance_equation[4].set_color(RED)
        distance_equals = MathTex("=", font_size=38, color=TEXT)
        distance_row = VGroup(distance_equals, distance_value).arrange(RIGHT, buff=0.16)
        distance_block = VGroup(
            text_label(r"Squared Fr\'echet distance", 35), distance_equation, distance_row,
        ).arrange(DOWN, buff=0.23, aligned_edge=LEFT)
        distance_row.shift(RIGHT * (distance_equation[1].get_x() - distance_equals.get_x()))
        mean_row = VGroup(
            MathTex(r"(\mu_r-\mu_g)^2 =", font_size=35, color=YELLOW), mean_value,
        ).arrange(RIGHT, buff=0.13)
        spread_row = VGroup(
            MathTex(r"(\sigma_r-\sigma_g)^2 =", font_size=35, color=RED), spread_value,
        ).arrange(RIGHT, buff=0.13)
        score = VGroup(
            distance_block,
            VGroup(text_label("Mean term", 31, YELLOW), mean_row).arrange(DOWN, buff=0.18, aligned_edge=LEFT),
            VGroup(text_label("Spread term", 31, RED), spread_row).arrange(DOWN, buff=0.18, aligned_edge=LEFT),
        ).arrange(DOWN, buff=0.65, aligned_edge=LEFT)
        VGroup(plots, score).arrange(RIGHT, buff=1.1)
        real_label_center, generated_label_center = real_label.get_center(), generated_label.get_center()

        def curves():
            # The thicker blue stroke stays visible when the two densities coincide.
            return VGroup(
                VMobject(color=BLUE, stroke_width=5.0),
                VMobject(color=GREEN, stroke_width=2.6).set_z_index(1),
            )

        def gap_geometry(color):
            return VGroup(
                VMobject(color=color, stroke_width=4.0),
                VMobject(color=BLUE, stroke_width=1.5, stroke_opacity=0.58),
                VMobject(color=GREEN, stroke_width=1.5, stroke_opacity=0.58),
                Dot(radius=0.069, color=BLUE),
                Dot(radius=0.043, color=GREEN),
            ).set_z_index(2)

        original_curves, centered_curves = curves(), curves()
        mean_marks, spread_marks = gap_geometry(YELLOW), gap_geometry(RED)

        def plot_curves(graphs, axes, means, scales):
            for curve, mean, std in zip(graphs, means, scales, strict=True):
                coordinates = np.column_stack((grid, gaussian_density(grid - mean, std)))
                curve.set_points_as_corners(axes.c2p(coordinates))

        def plot_gap(marks, axes, positions, heights):
            gap, real_stem, generated_stem, real_dot, generated_dot = marks
            start, end = [axes.c2p(x, 0) for x in positions]
            gap.set_points_as_corners([start, end])
            real_stem.set_points_as_corners([start, axes.c2p(positions[0], heights[0])])
            generated_stem.set_points_as_corners([end, axes.c2p(positions[1], heights[1])])
            real_dot.move_to(start)
            generated_dot.move_to(end)

        def update_diagram(parameters):
            mu_r, mu_g, sigma_r, sigma_g = parameters
            scales = (sigma_r, sigma_g)
            plot_curves(original_curves, original_axes, (mu_r, mu_g), scales)
            plot_curves(centered_curves, centered_axes, (0, 0), scales)
            plot_gap(mean_marks, original_axes, (mu_r, mu_g), [gaussian_density(0, std) for std in scales])
            plot_gap(spread_marks, centered_axes, scales, [gaussian_density(std, std) for std in scales])
            real_mean.set_value(mu_r)
            generated_mean.set_value(mu_g)
            real_variance.set_value(sigma_r**2)
            generated_variance.set_value(sigma_g**2)
            real_label.arrange(RIGHT, buff=0.06).move_to(real_label_center)
            generated_label.arrange(RIGHT, buff=0.06).move_to(generated_label_center)
            mean_cost = (mu_r - mu_g)**2
            spread_cost = (sigma_r - sigma_g)**2
            mean_value.set_value(mean_cost)
            spread_value.set_value(spread_cost)
            distance_value.set_value(mean_cost + spread_cost)

        update_diagram(states[0])
        self.play(FadeIn(top_plot), Create(original_curves), run_time=0.9)
        self.wait(0.9)
        self.play(
            FadeIn(centered_panel), TransformFromCopy(original_curves, centered_curves), run_time=1.45,
        )
        self.play(FadeIn(centered_title, mean_marks, spread_marks, score), run_time=0.85)
        self.wait(1.6)

        live = VGroup(legend, original_curves, centered_curves, mean_marks, spread_marks, score)
        self.remove(legend, original_curves, centered_curves, mean_marks, spread_marks, score)
        self.add(live)
        for before, after in zip(states[:-1], states[1:], strict=True):
            self.play(
                UpdateFromAlphaFunc(live, lambda _live, alpha: update_diagram(before + alpha * (after - before))),
                run_time=2.3,
            )
            self.wait(1.4)

        self.wait(1)

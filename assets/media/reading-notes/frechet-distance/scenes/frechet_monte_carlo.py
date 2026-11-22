import numpy as np
from manim import (
    DEGREES,
    DOWN,
    RIGHT,
    UP,
    Axes,
    Create,
    DashedVMobject,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    Integer,
    Line,
    MathTex,
    Scene,
    Transform,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
    linear,
)

from scene_helpers import (
    BASE,
    BLUE,
    CRUST,
    GREEN,
    RED,
    SUBTEXT,
    SURFACE,
    TEXT,
    YELLOW,
    convergence_chart,
    gaussian_contours,
    text_label,
)


class FrechetMonteCarlo(Scene):
    """Estimate the squared Fréchet distance for Gaussians with shared principal axes."""

    def construct(self):
        self.camera.background_color = BASE
        angle = 28 * DEGREES
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        real_mean, real_std = np.array([-1.15, -0.60]), np.array([1.08, 0.34])
        generated_mean, generated_std = np.array([1.14, 0.72]), np.array([0.40, 1.17])
        closer_mean, closer_std = np.array([-0.52, 0.22]), np.array([0.87, 0.51])
        cases = (
            (generated_mean, generated_std, r"Use the same $Z \sim \mathcal{N}(0,I_2)$ for both points in each pair."),
            (closer_mean, generated_std, "Move the generated mean closer to the real mean."),
            (closer_mean, closer_std, "Bring the generated covariance closer to the real covariance."),
        )
        sample_count = 4096
        counts = np.arange(1, sample_count + 1)
        # Matching standard normal coordinates in the shared basis is optimal.
        noise = np.random.default_rng(240).standard_normal((sample_count, 2))
        real_samples = real_mean + (noise * real_std) @ rotation.T

        def distribution_axes():
            return Axes(
                x_range=(-5.5, 5.5, 1), y_range=(-3.6, 3.6, 1),
                x_length=6.05, y_length=3.96, tips=False,
                axis_config={"color": SURFACE, "stroke_width": 2, "include_ticks": False},
            )

        real_axes, generated_axes = VGroup(distribution_axes(), distribution_axes()).arrange(RIGHT, buff=0.66)

        def contours(axes, mean, std, color):
            return gaussian_contours(axes, mean, rotation @ np.diag(std), color)

        real = contours(real_axes, real_mean, real_std, BLUE)
        generated = contours(generated_axes, generated_mean, generated_std, GREEN)
        real_label = text_label(r"Real features $X$", 35, BLUE).next_to(real_axes, UP, buff=0.38)
        generated_label = text_label(r"Generated features $Y$", 35, GREEN).next_to(generated_axes, UP, buff=0.38)
        legend = VGroup(real_label.copy(), generated_label.copy()).arrange(RIGHT, buff=0.48)
        legend.next_to(real_axes, UP, buff=0.38)
        note = text_label(cases[0][2], 32, SUBTEXT).to_edge(DOWN, buff=0.42)

        self.play(FadeIn(real_axes, real, real_label), FadeIn(generated_axes, generated, generated_label))
        self.wait(1.5)
        self.play(
            Transform(generated, contours(real_axes, generated_mean, generated_std, GREEN)),
            Transform(real_label, legend[0]),
            Transform(generated_label, legend[1]),
            FadeOut(generated_axes),
            run_time=1.4,
        )
        self.play(FadeIn(note), run_time=0.6)

        max_log_count = np.log10(sample_count)
        graph, chart = convergence_chart(sample_count, range(0, 11, 2))
        average = DecimalNumber(8, num_decimal_places=2, font_size=40, color=YELLOW)
        count = Integer(sample_count, font_size=28, color=SUBTEXT)
        cost = DecimalNumber(8, num_decimal_places=2, font_size=28, color=YELLOW)
        expected = DecimalNumber(8, num_decimal_places=2, font_size=30, color=RED)
        average_row = VGroup(
            MathTex(r"\frac{1}{n}\sum_{i=1}^{n}\lVert X_i-Y_i\rVert_2^2 =", font_size=36, color=TEXT),
            average,
        ).arrange(RIGHT, buff=0.12)
        pair_count = VGroup(text_label("Pairs", 28, SUBTEXT), count).arrange(RIGHT, buff=0.12)
        pair_cost = VGroup(
            MathTex(r"\lVert X_n-Y_n\rVert_2^2 =", font_size=28, color=TEXT), cost,
        ).arrange(RIGHT, buff=0.12)
        pair_row = VGroup(pair_count, pair_cost).arrange(RIGHT, buff=0.54)
        exact_row = VGroup(text_label(r"Squared Fr\'echet distance $=$", 30, RED), expected).arrange(RIGHT, buff=0.18)
        VGroup(average_row, chart, exact_row, pair_row).arrange(DOWN, buff=0.25).move_to(generated_axes)

        log_counts = np.log10(counts)
        checkpoints = np.unique(np.geomspace(1, sample_count, 44).astype(int)) - 1
        curve_indices = np.unique(np.geomspace(1, sample_count, 480).astype(int)) - 1

        for case, (mean, std, description) in enumerate(cases):
            generated_samples = mean + (noise * std) @ rotation.T
            offsets = real_samples - generated_samples
            costs = np.sum(offsets**2, axis=1)
            running_means = np.cumsum(costs) / counts
            exact = np.sum((real_mean - mean)**2) + np.sum((real_std - std)**2)

            start, end = real_axes.c2p(*real_samples[0]), real_axes.c2p(*generated_samples[0])
            gap = Line(start, end, color=YELLOW, stroke_width=2.4).set_z_index(2)
            x_dot = Dot(start, radius=0.075, color=BLUE, stroke_color=CRUST, stroke_width=1.5)
            y_dot = Dot(end, radius=0.075, color=GREEN, stroke_color=CRUST, stroke_width=1.5)
            pair = VGroup(x_dot, y_dot).set_z_index(3)
            benchmark = DashedVMobject(
                Line(graph.c2p(0, exact), graph.c2p(max_log_count, exact), color=RED, stroke_width=2.1),
                num_dashes=26,
            )
            average.set_value(running_means[0])
            count.set_value(1)
            cost.set_value(costs[0])
            expected.set_value(exact)

            if case == 0:
                dx, dy = offsets[0]
                first_cost = VGroup(
                    MathTex(r"\lVert X_1 - Y_1\rVert_2^2", font_size=44, color=TEXT),
                    MathTex(rf"= ({dx:.3f})^2 + ({dy:.3f})^2 \approx {costs[0]:.2f}", font_size=38, color=YELLOW),
                ).arrange(DOWN, buff=0.24).move_to(chart)
                self.play(FadeIn(pair), Create(gap), FadeIn(first_cost))
                self.wait(2.3)
                self.play(FadeOut(first_cost), FadeIn(chart, average_row, exact_row, pair_row, benchmark))
            else:
                self.play(
                    Transform(generated, contours(real_axes, mean, std, GREEN)),
                    Transform(note, text_label(description, 32, SUBTEXT).move_to(note)),
                )
                self.play(FadeIn(pair, gap, average_row, exact_row, pair_row, benchmark), run_time=0.55)

            curve_points = np.array([graph.c2p(log_counts[i], running_means[i]) for i in curve_indices])
            trace = VMobject(color=YELLOW, stroke_width=3).set_z_index(2)
            live = VGroup(average_row, pair_row, gap, pair, trace)
            self.remove(average_row, pair_row, gap, pair)
            self.add(live)
            last_index = -1

            def advance(_live, alpha):
                nonlocal last_index
                index = checkpoints[int(alpha * (len(checkpoints) - 1))]
                if index == last_index:
                    return
                last_index = index
                start = real_axes.c2p(*real_samples[index])
                end = real_axes.c2p(*generated_samples[index])
                x_dot.move_to(start)
                y_dot.move_to(end)
                gap.put_start_and_end_on(start, end)
                average.set_value(running_means[index])
                count.set_value(counts[index])
                cost.set_value(costs[index])
                stop = np.searchsorted(curve_indices, index, side="right")
                endpoint = graph.c2p(log_counts[index], running_means[index])
                trace.set_points_as_corners(np.vstack((curve_points[:stop], endpoint)))

            self.play(UpdateFromAlphaFunc(live, advance), rate_func=linear, run_time=6.5 if case == 0 else 4.1)
            self.wait(1.7)
            if case < len(cases) - 1:
                self.play(FadeOut(live, benchmark, exact_row), run_time=0.45)

        self.wait(1.4)

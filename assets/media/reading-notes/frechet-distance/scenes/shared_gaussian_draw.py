import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Create,
    CurvedArrow,
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
    TransformFromCopy,
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


def covariance_root(std, angle_degrees):
    angle = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (rotation * std) @ rotation.T


class SharedGaussianDraw(Scene):
    """Estimate the squared cost of pairing two Gaussian draws through the same Z."""

    def construct(self):
        self.camera.background_color = BASE
        origin = np.zeros(2)
        means = np.array([[-1, -0.45], [1, 0.45]])
        closer_means = np.array([[-0.45, -0.23], [0.45, 0.23]])
        roots = (covariance_root((1.20, 0.40), 32), covariance_root((0.95, 0.48), -29))
        closer_roots = (covariance_root((1.03, 0.67), 32), covariance_root((0.98, 0.67), -29))
        matching_root = (closer_roots[0] + closer_roots[1]) / 2
        fit_targets = (
            (closer_means, roots),
            (closer_means, closer_roots),
            ((origin, origin), (matching_root, matching_root)),
        )
        colors = (BLUE, GREEN)
        sample_count = 8192
        counts = np.arange(1, sample_count + 1)
        log_counts = np.log10(counts)
        noise = np.random.default_rng(44).standard_normal((sample_count, 2))
        checkpoints = np.unique(np.rint(np.geomspace(1, sample_count, 44)).astype(int)) - 1
        curve_indices = np.unique(np.rint(np.geomspace(1, sample_count, 480)).astype(int)) - 1

        def axes(x_limit, y_limit, unit):
            return Axes(
                x_range=(-x_limit, x_limit, 1), y_range=(-y_limit, y_limit, 1),
                x_length=2 * x_limit * unit, y_length=2 * y_limit * unit, tips=False,
                axis_config={"color": SURFACE, "stroke_width": 1.6, "include_ticks": False},
            )

        original_axes, centered_axes = axes(4, 3.2, 0.405), axes(4, 3.2, 0.405)
        original_panel = VGroup(text_label("Original fits", 32, SUBTEXT), original_axes).arrange(DOWN, buff=0.14)
        centered_panel = VGroup(text_label("Centered at zero", 32, SUBTEXT), centered_axes).arrange(DOWN, buff=0.14)
        fits = VGroup(original_panel, centered_panel).arrange(RIGHT, buff=0.46)
        legend = VGroup(text_label("Real fit", 31, BLUE), text_label("Generated fit", 31, GREEN)).arrange(
            RIGHT, buff=0.48,
        )
        prior_axes = axes(3.4, 3.4, 0.33)
        prior_panel = VGroup(
            prior_axes, MathTex(r"Z \sim \mathcal{N}(0,I_2)", font_size=32, color=SUBTEXT),
        ).arrange(DOWN, buff=0.13)
        maps = VGroup(
            MathTex(r"X = \Sigma_r^{1/2}Z", font_size=36, color=BLUE),
            MathTex(r"Y = \Sigma_g^{1/2}Z", font_size=36, color=GREEN),
        ).arrange(DOWN, buff=0.24)
        sampling = VGroup(maps, prior_panel).arrange(RIGHT, buff=0.60)
        left = VGroup(legend, fits, sampling).arrange(DOWN, buff=0.43)

        graph, chart = convergence_chart(sample_count, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        average = DecimalNumber(0, num_decimal_places=3, font_size=37, color=YELLOW)
        expected_value = DecimalNumber(0, num_decimal_places=3, font_size=28, color=RED)
        count = Integer(sample_count, font_size=27, color=SUBTEXT)
        pair_cost = DecimalNumber(0, num_decimal_places=3, font_size=27, color=YELLOW)
        mean_cost = float(np.sum((means[0] - means[1])**2))
        mean_value = DecimalNumber(mean_cost, num_decimal_places=3, font_size=35, color=TEXT)
        spread_value = DecimalNumber(0, num_decimal_places=3, font_size=35, color=YELLOW)
        total = DecimalNumber(0, num_decimal_places=3, font_size=36, color=TEXT)
        average_row = VGroup(
            MathTex(r"\frac{1}{n}\sum_{i=1}^n \lVert X_i - Y_i\rVert_2^2 =", font_size=35, color=TEXT), average,
        ).arrange(RIGHT, buff=0.12)
        expectation = VGroup(
            MathTex(r"\mathbb{E}[\lVert X-Y\rVert_2^2] =", font_size=28, color=RED), expected_value,
        ).arrange(RIGHT, buff=0.12)
        pair_row = VGroup(
            VGroup(text_label("Pairs", 26, SUBTEXT), count).arrange(RIGHT, buff=0.12),
            VGroup(MathTex(r"\lVert X_n-Y_n\rVert_2^2 =", font_size=26, color=TEXT), pair_cost).arrange(
                RIGHT, buff=0.12,
            ),
        ).arrange(RIGHT, buff=0.48)
        total_row = VGroup(
            mean_value, MathTex("+", font_size=34, color=TEXT),
            spread_value, MathTex(r"\approx", font_size=34, color=TEXT), total,
        ).arrange(RIGHT, buff=0.16)
        total_block = VGroup(text_label("Mean term + spread estimate", 29, SUBTEXT), total_row).arrange(
            DOWN, buff=0.18,
        )
        right = VGroup(average_row, chart, expectation, pair_row, total_block).arrange(DOWN, buff=0.25)
        VGroup(left, right).arrange(RIGHT, buff=0.68)

        def contours(plot_axes, centers, scales):
            fits = VGroup(*[
                gaussian_contours(plot_axes, center, root, color)
                for center, root, color in zip(centers, scales, colors, strict=True)
            ])
            # The blue outline remains visible when the contours coincide.
            fits[0][0].set_stroke(width=2.6)
            return fits

        def mean_segment(centers):
            return VMobject(color=YELLOW, stroke_width=2.6).set_points_as_corners([
                original_axes.c2p(*mean) for mean in centers
            ])

        original = contours(original_axes, means, roots)
        centered = contours(centered_axes, (origin, origin), roots)
        prior = gaussian_contours(prior_axes, origin, np.eye(2), SUBTEXT)
        mean_gap = mean_segment(means).set_z_index(2)
        self.play(FadeIn(legend, original_panel, original, mean_gap), run_time=0.85)
        self.wait(1.25)
        self.play(FadeIn(centered_panel), TransformFromCopy(original, centered), run_time=1.35)
        self.play(FadeIn(prior_panel, prior, maps), run_time=0.7)

        real_root, generated_root = roots
        xs, ys = noise @ real_root.T, noise @ generated_root.T
        costs = np.sum((xs - ys)**2, axis=1)
        running_means = np.cumsum(costs) / counts
        exact = np.sum((real_root - generated_root)**2)
        curve_points = np.array([graph.c2p(log_counts[i], running_means[i]) for i in curve_indices])
        trace = VMobject(color=YELLOW, stroke_width=3).set_z_index(2)
        benchmark = DashedVMobject(
            Line(graph.c2p(0, exact), graph.c2p(log_counts[-1], exact), color=RED, stroke_width=2.1),
            num_dashes=26,
        )
        z_dot = Dot(radius=0.065, color=TEXT, stroke_color=CRUST, stroke_width=1.5)
        x_dot = Dot(radius=0.077, color=BLUE, stroke_color=CRUST, stroke_width=1.5)
        y_dot = Dot(radius=0.077, color=GREEN, stroke_color=CRUST, stroke_width=1.5)
        dots = VGroup(z_dot, x_dot, y_dot).set_z_index(4)
        gap = VMobject(color=YELLOW, stroke_width=2.6).set_z_index(3)
        expected_value.set_value(exact)
        last_index = -1

        def show_sample(index):
            z_dot.move_to(prior_axes.c2p(*noise[index]))
            start, end = centered_axes.c2p(*xs[index]), centered_axes.c2p(*ys[index])
            x_dot.move_to(start)
            y_dot.move_to(end)
            gap.set_points_as_corners([start, end])
            count.set_value(counts[index])
            pair_cost.set_value(costs[index])
            average.set_value(running_means[index])
            spread_value.set_value(running_means[index])
            total.set_value(mean_cost + running_means[index])

        def advance(_live, alpha):
            nonlocal last_index
            index = checkpoints[int(alpha * (len(checkpoints) - 1))]
            if index == last_index:
                return
            last_index = index
            show_sample(index)
            stop = np.searchsorted(curve_indices, index, side="right")
            endpoint = graph.c2p(log_counts[index], running_means[index])
            trace.set_points_as_corners(np.vstack((curve_points[:stop], endpoint)))

        show_sample(0)
        arrows = VGroup(*[
            CurvedArrow(
                z_dot.get_center() + 0.075 * UP, dot.get_center() + 0.085 * DOWN,
                angle=angle, color=color, stroke_width=2.3, tip_length=0.15,
            )
            for dot, color, angle in zip((x_dot, y_dot), colors, (0.55, -0.55), strict=True)
        ]).set_z_index(2)
        self.play(FadeIn(z_dot), run_time=0.45)
        self.play(Create(arrows), FadeIn(x_dot, y_dot), run_time=0.9)
        self.play(Create(gap), FadeIn(pair_row), run_time=0.55)
        self.wait(1.3)
        self.play(FadeIn(chart), run_time=0.45)
        self.play(FadeOut(arrows), FadeIn(average_row, benchmark, expectation, total_block), run_time=0.65)
        live = VGroup(dots, gap, pair_row, average_row, total_block, trace)
        self.remove(z_dot, x_dot, y_dot, gap, pair_row, average_row, total_block)
        self.add(live)
        self.play(UpdateFromAlphaFunc(live, advance), rate_func=linear, run_time=5.6)
        self.wait(1.5)

        exact_mean = DecimalNumber(mean_cost, num_decimal_places=3, font_size=37, color=YELLOW)
        exact_spread = DecimalNumber(exact, num_decimal_places=3, font_size=37, color=RED)
        mean_addend, spread_addend = exact_mean.copy(), exact_spread.copy()
        exact_total = DecimalNumber(mean_cost + exact, num_decimal_places=3, font_size=44, color=TEXT)
        mean_formula = VGroup(
            MathTex(r"\lVert \mu_r - \mu_g \rVert_2^2 =", font_size=35, color=YELLOW), exact_mean,
        ).arrange(RIGHT, buff=0.14)
        spread_formula = VGroup(
            MathTex(
                r"\left\lVert \Sigma_r^{1/2} - \Sigma_g^{1/2} \right\rVert_F^2 =",
                font_size=35, color=RED,
            ),
            exact_spread,
        ).arrange(RIGHT, buff=0.14)
        sum_formula = VGroup(
            mean_addend, MathTex("+", font_size=36, color=TEXT),
            spread_addend, MathTex(r"\approx", font_size=36, color=TEXT), exact_total,
        ).arrange(RIGHT, buff=0.15)
        analytic = VGroup(
            VGroup(text_label("Mean term", 32, YELLOW), mean_formula).arrange(DOWN, buff=0.17, aligned_edge=LEFT),
            VGroup(text_label("Spread cost for this pairing", 32, RED), spread_formula).arrange(
                DOWN, buff=0.17, aligned_edge=LEFT,
            ),
            VGroup(text_label("Mean term + spread cost", 32), sum_formula).arrange(
                DOWN, buff=0.17, aligned_edge=LEFT,
            ),
        ).arrange(DOWN, buff=0.56, aligned_edge=LEFT).move_to(right)

        plots = VGroup(legend, fits, original, centered, mean_gap)
        self.remove(legend, original_panel, centered_panel, original, centered, mean_gap)
        self.add(plots)
        self.play(
            FadeOut(live, chart, benchmark, expectation, sampling, prior),
            plots.animate.set_y(0),
            FadeIn(analytic),
            run_time=0.9,
        )
        self.wait(1.0)

        for next_means, next_roots in fit_targets:
            start_mean_gap = means[0] - means[1]
            next_mean_gap = next_means[0] - next_means[1]
            start_root_gap = roots[0] - roots[1]
            next_root_gap = next_roots[0] - next_roots[1]

            def update_costs(_analytic, alpha):
                mean_delta = (1 - alpha) * start_mean_gap + alpha * next_mean_gap
                root_delta = (1 - alpha) * start_root_gap + alpha * next_root_gap
                mean = np.sum(mean_delta**2)
                spread = np.sum(root_delta**2)
                exact_mean.set_value(mean)
                mean_addend.set_value(mean)
                exact_spread.set_value(spread)
                spread_addend.set_value(spread)
                exact_total.set_value(mean + spread)

            self.play(
                Transform(original, contours(original_axes, next_means, next_roots)),
                Transform(centered, contours(centered_axes, (origin, origin), next_roots)),
                Transform(mean_gap, mean_segment(next_means).set_z_index(2)),
                UpdateFromAlphaFunc(analytic, update_costs),
                rate_func=linear, run_time=2.2,
            )
            means, roots = next_means, next_roots
            self.wait(1.35)

        self.wait(1.1)

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
    Rotate,
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
    svd_with_rotations,
    text_label,
)
from shared_gaussian_draw import covariance_root
from svd_gaussian_pairing import latex_matrix


def example_fits():
    means = np.array([[-0.62, -0.36], [0.73, 0.45]])
    roots = np.array([covariance_root((1.50, 0.18), 27), covariance_root((1.40, 0.22), -25)])
    return means, roots


def distance_terms(means, roots, singular_values):
    mean = np.sum((means[0] - means[1])**2)
    spread = np.sum(roots**2) - 2 * np.sum(singular_values)
    return mean, max(0.0, spread)


class FrechetComputation(Scene):
    """Compute the squared Gaussian Fréchet distance using the SVD pairing."""

    def construct(self):
        self.camera.background_color = BASE
        means, roots = example_fits()
        a, b = roots
        p, singular_values, qt = svd_with_rotations(b @ a)
        rotation = p @ qt
        mean_cost, spread_cost = distance_terms(means, roots, singular_values)
        exact = mean_cost + spread_cost
        shared_spread = np.sum((a - b)**2)
        sample_count = 32_768
        counts = np.arange(1, sample_count + 1)
        log_counts = np.log10(counts)
        draws = np.random.default_rng(32).standard_normal((sample_count, 2))
        paired_draws = draws @ rotation.T
        real_points = means[0] + draws @ a.T
        generated_points = means[1] + paired_draws @ b.T
        costs = np.sum((real_points - generated_points)**2, axis=1)
        running_means = np.cumsum(costs) / counts
        checkpoints = np.unique(np.rint(np.geomspace(1, sample_count, 44)).astype(int)) - 1
        curve_indices = np.unique(np.rint(np.geomspace(1, sample_count, 480)).astype(int)) - 1
        colors = (BLUE, GREEN)

        def plane(x_limit, y_limit, unit):
            return Axes(
                x_range=(-x_limit, x_limit, 1), y_range=(-y_limit, y_limit, 1),
                x_length=2 * x_limit * unit, y_length=2 * y_limit * unit, tips=False,
                axis_config={"color": SURFACE, "stroke_width": 1.6, "include_ticks": False},
            )

        output_axes = plane(4.3, 2.4, 0.70)
        prior_axes = plane(3.2, 3.2, 0.32)
        legend = VGroup(text_label("Real fit", 32, BLUE), text_label("Generated fit", 32, GREEN)).arrange(
            RIGHT, buff=0.48,
        )
        outputs = VGroup(legend, output_axes).arrange(DOWN, buff=0.18)
        mappings = VGroup(
            MathTex(r"x=\mu_r+AU", font_size=35, color=BLUE),
            MathTex(r"y=\mu_g+BV", font_size=35, color=GREEN),
        ).arrange(DOWN, buff=0.20)
        draw_labels = VGroup(
            MathTex(r"U\sim\mathcal{N}(0,I_2)", font_size=29, color=BLUE),
            MathTex(r"V=PQ^\top U", font_size=29, color=GREEN),
        ).arrange(RIGHT, buff=0.35)
        source = VGroup(prior_axes, draw_labels).arrange(DOWN, buff=0.13)
        sampling = VGroup(mappings, source).arrange(RIGHT, buff=0.63)
        left = VGroup(outputs, sampling).arrange(DOWN, buff=0.35)

        graph, chart = convergence_chart(sample_count, range(0, 7))
        average = DecimalNumber(0, num_decimal_places=3, font_size=36, color=YELLOW)
        count = Integer(sample_count, font_size=27, color=SUBTEXT)
        pair_cost = DecimalNumber(0, num_decimal_places=3, font_size=26, color=YELLOW)
        average_row = VGroup(
            MathTex(r"\frac1n\sum_{i=1}^n\lVert x_i-y_i\rVert_2^2 =", font_size=35, color=TEXT),
            average,
        ).arrange(RIGHT, buff=0.12)
        exact_row = VGroup(
            text_label("Analytical squared distance", 29, RED),
            MathTex(rf"d_{{\mathrm{{F}}}}^2\approx {exact:.3f}", font_size=32, color=RED),
        ).arrange(DOWN, buff=0.13)
        pair_row = VGroup(
            VGroup(text_label("Pairs", 27, SUBTEXT), count).arrange(RIGHT, buff=0.12),
            VGroup(MathTex(r"\lVert x_n-y_n\rVert_2^2 =", font_size=26, color=TEXT), pair_cost).arrange(
                RIGHT, buff=0.12,
            ),
        ).arrange(RIGHT, buff=0.45)
        right = VGroup(average_row, chart, exact_row, pair_row).arrange(DOWN, buff=0.26)
        VGroup(left, right).arrange(RIGHT, buff=0.74)

        def fits(centers, square_roots):
            plots = VGroup(*[
                gaussian_contours(output_axes, center, root, color)
                for center, root, color in zip(centers, square_roots, colors, strict=True)
            ])
            # Both colors remain visible when the curves overlap.
            plots[0][0].set_stroke(width=2.7)
            return plots

        output_fits = fits(means, roots)
        prior_fit = gaussian_contours(prior_axes, np.zeros(2), np.eye(2), SUBTEXT)
        square_roots = VGroup(
            MathTex(r"A\approx" + latex_matrix(a, 2), font_size=32, color=BLUE),
            MathTex(r"B\approx" + latex_matrix(b, 2), font_size=32, color=GREEN),
        ).arrange(RIGHT, buff=0.40)
        svd_formula = MathTex(r"BA=PDQ^\top", font_size=39, color=TEXT)
        spread_label = text_label("Expected squared gap after centering", 28, SUBTEXT)
        shared_label = MathTex(rf"V=U\qquad {shared_spread:.3f}", font_size=34, color=SUBTEXT)
        optimal_label = MathTex(rf"V=PQ^\top U\qquad {spread_cost:.3f}", font_size=34, color=YELLOW)
        intro = VGroup(square_roots, svd_formula, spread_label, shared_label, optimal_label).arrange(
            DOWN, buff=0.28,
        ).move_to(right)
        self.play(FadeIn(outputs, output_fits, square_roots), run_time=0.85)
        self.play(FadeIn(spread_label, shared_label), run_time=0.55)
        self.wait(1.4)
        self.play(FadeIn(svd_formula, optimal_label), run_time=0.85)
        self.wait(2.1)

        def coordinate_row(label, point, color):
            return MathTex(label + latex_matrix(point[:, None], 3), font_size=35, color=color)

        u_step = coordinate_row(r"U\approx", draws[0], BLUE)
        q_step = coordinate_row(r"Q^\top U\approx", qt @ draws[0], SUBTEXT)
        p_step = coordinate_row(r"V=P(Q^\top U)\approx", paired_draws[0], GREEN)
        steps = VGroup(u_step, q_step, p_step).arrange(DOWN, buff=0.38, aligned_edge=LEFT).move_to(right)
        u_dot = Dot(prior_axes.c2p(*draws[0]), radius=0.062, color=BLUE, stroke_color=CRUST, stroke_width=1)
        v_dot = Dot(prior_axes.c2p(*draws[0]), radius=0.052, color=SUBTEXT, stroke_color=CRUST, stroke_width=1)
        prior_center = prior_axes.c2p(0, 0)
        self.play(FadeOut(intro), FadeIn(prior_axes, prior_fit, draw_labels[0]), run_time=0.7)
        self.play(FadeIn(u_dot, u_step), run_time=0.6)
        self.wait(0.7)
        self.add(v_dot)
        self.play(
            Rotate(v_dot, angle=np.arctan2(qt[1, 0], qt[0, 0]), about_point=prior_center),
            FadeIn(q_step), run_time=1.1,
        )
        self.wait(0.7)
        self.play(
            Rotate(v_dot, angle=np.arctan2(p[1, 0], p[0, 0]), about_point=prior_center),
            FadeIn(p_step, draw_labels[1]), run_time=1.1,
        )
        v_dot.set_color(GREEN)
        self.wait(0.8)

        x_dot = Dot(output_axes.c2p(*real_points[0]), radius=0.080, color=BLUE, stroke_color=CRUST, stroke_width=1.5)
        y_dot = Dot(output_axes.c2p(*generated_points[0]), radius=0.073, color=GREEN, stroke_color=CRUST, stroke_width=1.5)
        dots = VGroup(u_dot, v_dot, x_dot, y_dot).set_z_index(4)
        arrows = VGroup(*[
            CurvedArrow(
                start.get_center() + 0.075 * UP, end.get_center() + 0.09 * DOWN, color=color,
                angle=angle, stroke_width=2.2, tip_length=0.15,
            )
            for start, end, color, angle in zip((u_dot, v_dot), (x_dot, y_dot), colors, (0.58, -0.58), strict=True)
        ]).set_z_index(2)
        gap = VMobject(color=YELLOW, stroke_width=2.8).set_z_index(3)
        gap.set_points_as_corners([x_dot.get_center(), y_dot.get_center()])
        self.play(Create(arrows), FadeIn(mappings, x_dot, y_dot), run_time=1.0)
        self.play(Create(gap), run_time=0.5)
        self.wait(1.2)

        expected_line = DashedVMobject(
            Line(graph.c2p(0, exact), graph.c2p(log_counts[-1], exact), color=RED, stroke_width=2.1),
            num_dashes=26,
        )
        trace = VMobject(color=YELLOW, stroke_width=3).set_z_index(2)
        curve_points = np.array([graph.c2p(log_counts[i], running_means[i]) for i in curve_indices])
        last_index = -1

        def show_pair(index):
            u_dot.move_to(prior_axes.c2p(*draws[index]))
            v_dot.move_to(prior_axes.c2p(*paired_draws[index]))
            x_dot.move_to(output_axes.c2p(*real_points[index]))
            y_dot.move_to(output_axes.c2p(*generated_points[index]))
            gap.set_points_as_corners([x_dot.get_center(), y_dot.get_center()])
            count.set_value(counts[index])
            pair_cost.set_value(costs[index])
            average.set_value(running_means[index])

        def advance(_live, alpha):
            nonlocal last_index
            index = checkpoints[int(alpha * (len(checkpoints) - 1))]
            if index == last_index:
                return
            last_index = index
            show_pair(index)
            stop = np.searchsorted(curve_indices, index, side="right")
            endpoint = graph.c2p(log_counts[index], running_means[index])
            trace.set_points_as_corners(np.vstack((curve_points[:stop], endpoint)))

        show_pair(0)
        self.play(
            FadeOut(steps, arrows), FadeIn(chart, expected_line, average_row, exact_row, pair_row), run_time=0.75,
        )
        live = VGroup(dots, gap, average_row, pair_row, trace)
        self.remove(u_dot, v_dot, x_dot, y_dot, gap, average_row, pair_row)
        self.add(live)
        self.play(UpdateFromAlphaFunc(live, advance), rate_func=linear, run_time=6.5)
        self.wait(1.65)

        def readout(label, value, color, size=34):
            number = DecimalNumber(value, num_decimal_places=3, font_size=size, color=color)
            row = VGroup(MathTex(label, font_size=size, color=color), number).arrange(RIGHT, buff=0.14)
            return row, number

        mean_row, mean_value = readout(r"\lVert\mu_r-\mu_g\rVert_2^2 =", mean_cost, TEXT)
        spread_row, spread_value = readout(
            r"\operatorname{tr}(\Sigma_r+\Sigma_g)-2\sum_i s_i =", spread_cost, TEXT, size=31,
        )
        squared_row, squared_value = readout(r"d_{\mathrm{F}}^2 =", exact, RED, size=42)
        mc_value = DecimalNumber(running_means[-1], num_decimal_places=3, font_size=33, color=YELLOW)
        mc_row = VGroup(text_label("Monte Carlo", 30, YELLOW), mc_value).arrange(RIGHT, buff=0.18)
        summary = VGroup(mean_row, spread_row, squared_row, mc_row).arrange(
            DOWN, buff=0.48, aligned_edge=LEFT,
        ).move_to(right)
        fit_view = VGroup(outputs, output_fits)
        self.remove(outputs, output_fits)
        self.add(fit_view)
        self.play(
            FadeOut(live, chart, exact_row, expected_line, sampling, prior_fit),
            fit_view.animate.set_y(0),
            FadeIn(summary),
            run_time=0.9,
        )
        self.wait(1.0)

        merged_means = np.zeros_like(means)
        merged_roots = np.stack([roots.mean(axis=0)] * 2)

        def merge_readout(_summary, alpha):
            centers = (1 - alpha) * means + alpha * merged_means
            square_roots = (1 - alpha) * roots + alpha * merged_roots
            real_root, gen_root = square_roots
            current_p, current_s, current_qt = svd_with_rotations(gen_root @ real_root)
            current_rotation = current_p @ current_qt
            mean, spread = distance_terms(centers, square_roots, current_s)
            delta = (centers[0] - centers[1]) + draws @ (real_root - gen_root @ current_rotation).T
            mean_value.set_value(mean)
            spread_value.set_value(spread)
            squared_value.set_value(mean + spread)
            mc_value.set_value(np.mean(np.sum(delta**2, axis=1)))

        self.play(
            Transform(output_fits, fits(merged_means, merged_roots)),
            UpdateFromAlphaFunc(summary, merge_readout),
            rate_func=linear, run_time=3.6,
        )
        self.wait(2.0)

from math import log, pi
from random import Random
from statistics import NormalDist

import numpy as np
from manim import (
    DEGREES,
    DL,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    UR,
    Arrow,
    Circle,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    Integer,
    LaggedStart,
    Line,
    MathTex,
    MobjectTable,
    Scene,
    Square,
    SurroundingRectangle,
    Transform,
    TransformFromCopy,
    TransformMatchingTex,
    UpdateFromAlphaFunc,
    VGroup,
    linear,
)

from scene_helpers import (
    BASE,
    BLUE,
    COMPONENT_COLORS,
    CRUST,
    GREEN,
    LIMIT,
    Mixture,
    Point,
    SPARSE_DISTRIBUTION,
    SUBTEXT,
    SURFACE,
    TEST_COUNT,
    TEST_DOT_RADIUS,
    TEXT,
    YELLOW,
    density_panel,
    draw_samples,
    gaussian_blob,
    mixture_density,
    sample_dots,
    text_label,
)


def latent_prediction(z: Point, components: Mixture, normal: NormalDist) -> tuple[int, Point]:
    # Map three disjoint prior bands to each component. The within-band
    # quantile and the second latent coordinate set the prediction.
    band_count = 3 * len(components)
    band, quantile = divmod(band_count * normal.cdf(z[0]), 1)
    component = int(band) % len(components)
    _, mean, covariance = components[component]
    local_z = normal.inv_cdf(min(max(quantile, 1e-12), 1 - 1e-12)), z[1]
    prediction = np.asarray(mean) + np.linalg.cholesky(covariance) @ local_z
    return component, tuple(prediction)


class LatentVariableLikelihood(Scene):
    """Estimate the likelihood of four test points by sampling the latent prior."""

    def construct(self):
        self.camera.background_color = BASE
        prior: Mixture = ((1.0, (0, 0), ((1, 0), (0, 1))),)
        target: Mixture = tuple(
            (weight, mean[:2], tuple(row[:2] for row in covariance[:2]))
            for weight, mean, covariance in SPARSE_DISTRIBUTION
        )
        shifts = ((0.18, 0.06), (0.11, -0.18), (-0.08, 0.18), (-0.18, 0.14), (-0.18, -0.16))
        model: Mixture = tuple(
            (weight, tuple(np.asarray(mean) + shift), covariance)
            for (weight, mean, covariance), shift in zip(target, shifts, strict=True)
        )
        normal = NormalDist()
        rng = Random(19)
        latent_points = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(48)]
        mapped = [latent_prediction(point, model, normal) for point in latent_points]
        predictions = [point for _, point in mapped]
        test_points = draw_samples(target, TEST_COUNT, Random(67))

        output_limit = 4.1
        prior_grid = np.linspace(-LIMIT, LIMIT, 241)
        output_grid = np.linspace(-output_limit, output_limit, 241)
        prior_density = mixture_density(prior, *np.meshgrid(prior_grid, prior_grid[::-1]))
        output_x, output_y = np.meshgrid(output_grid, output_grid[::-1])
        target_density, prediction_density = [
            mixture_density(mixture, output_x, output_y) for mixture in (target, model)
        ]
        output_peak = max(float(target_density.max()), float(prediction_density.max()))
        prior_panel, prior_image = density_panel(
            prior_density, float(prior_density.max()), r"Prior $p(z)=\mathcal{N}(0,I)$", BLUE
        )
        target_panel, target_image = density_panel(
            target_density, output_peak, r"True density $p^\star(x)$", BLUE
        )
        prediction_panel, prediction_image = density_panel(
            prediction_density, output_peak, r"Predictions $G_\theta(z)$", GREEN
        )
        for panel in (prior_panel, target_panel, prediction_panel):
            panel.scale(1.26)
        arrow = Arrow(LEFT * 1.02, RIGHT * 1.02, buff=0, color=SUBTEXT, stroke_width=2.3)
        generator = MathTex(r"G_\theta", font_size=33, color=GREEN)
        mapping = VGroup(arrow, generator).arrange(UP, buff=0.16)
        diagram = Group(prior_panel, mapping, prediction_panel).arrange(RIGHT, buff=0.52)
        footer = text_label("Draw latent points from a standard Gaussian.", 31, SUBTEXT)
        Group(diagram, footer).arrange(DOWN, buff=1.0)
        target_panel.shift(prediction_image.get_center() - target_image.get_center())

        prior_dots = sample_dots(prior_image, latent_points, BLUE)
        prediction_dots = sample_dots(prediction_image, predictions, GREEN, limit=output_limit)
        for source, prediction, (component, _) in zip(
            prior_dots, prediction_dots, mapped, strict=True
        ):
            source.set_fill(COMPONENT_COLORS[component])
            prediction.set_fill(COMPONENT_COLORS[component])

        self.play(FadeIn(prior_panel), FadeIn(target_panel), FadeIn(footer), run_time=0.75)
        self.play(FadeIn(prior_dots), run_time=0.75)
        self.wait(0.9)
        model_note = text_label("The model maps the draws into sparse clusters.", 31, SUBTEXT)
        self.play(
            FadeOut(target_panel), FadeIn(prediction_panel), FadeIn(mapping),
            Transform(footer, model_note.move_to(footer)), run_time=0.9,
        )
        self.play(
            LaggedStart(
                *(
                    TransformFromCopy(source, prediction, path_arc=-12 * DEGREES)
                    for source, prediction in zip(prior_dots, prediction_dots, strict=True)
                ),
                lag_ratio=0.032,
            ),
            run_time=2.15,
        )
        self.wait(0.75)

        sigma = 0.16
        offsets = np.asarray(test_points)[:, None, :] - np.asarray(predictions)
        squared_distances = np.sum((offsets / sigma) ** 2, axis=-1)
        conditional_values = np.exp(-0.5 * squared_distances) / (2 * pi * sigma**2)
        nearest = int(np.argmin(squared_distances[0]))
        match_count = int(np.count_nonzero(np.all(offsets[0] == 0, axis=1)))
        mapped_view = (prior_panel, mapping, prediction_panel, prior_dots, prediction_dots)
        self.play(
            *(FadeOut(part) for part in mapped_view), FadeOut(footer), run_time=0.45,
        )
        self.compare_decoders(
            offsets[0, nearest], nearest + 1, conditional_values[0, nearest],
            sigma, match_count, len(latent_points),
        )

        integral = MathTex(
            r"p_\theta(x_i)", "=", r"\int", r"p_\theta(x_i\mid z)", r"p(z)\,\mathrm dz",
            font_size=39,
            color=TEXT,
        ).move_to(footer)
        self.play(*(FadeIn(part) for part in mapped_view), FadeIn(integral), run_time=0.65)
        self.wait(1.5)
        estimate = MathTex(
            r"p_\theta(x_i)", r"\approx", r"\frac{1}{M}\sum_{j=1}^{M}",
            r"p_\theta(x_i\mid z_j)", r"\qquad z_j\sim p(z),\quad M=48",
            font_size=33,
            color=TEXT,
        )
        prior_view = Group(prior_panel, prior_dots)
        prediction_view = Group(prediction_panel, prediction_dots)
        prior_slot = prior_view.copy().scale(0.75)
        mapping_slot = mapping.copy()
        prediction_slot = prediction_view.copy().scale(0.75)
        compact_row = Group(prior_slot, mapping_slot, prediction_slot)
        compact_row.arrange(RIGHT, buff=0.92).to_edge(UP, buff=0.3)
        estimate.next_to(compact_row, DOWN, buff=0.15)
        self.play(
            prior_view.animate.scale(0.75).move_to(prior_slot),
            mapping.animate.move_to(mapping_slot),
            prediction_view.animate.scale(0.75).move_to(prediction_slot),
            TransformMatchingTex(integral, estimate),
            run_time=1.3,
        )
        self.wait(0.8)

        test_dots = sample_dots(
            prediction_image, test_points, YELLOW, radius=TEST_DOT_RADIUS, limit=output_limit
        )
        self.play(
            prior_dots.animate.set_fill(TEXT, opacity=0.62),
            prediction_dots.animate.set_fill(TEXT, opacity=0.62),
            FadeIn(test_dots),
            run_time=0.45,
        )
        kernel_width = 8 * sigma * prediction_image.width / (2 * output_limit)
        self.evaluate_test_points(
            prior_view, prediction_view, estimate, test_dots, conditional_values, sigma, kernel_width
        )
        self.play(
            FadeOut(prior_view), FadeOut(prediction_view), FadeOut(mapping),
            FadeOut(estimate), FadeOut(test_dots), run_time=0.7,
        )
        self.show_likelihood_table(conditional_values)

    def compare_decoders(self, offset, draw_number, density, sigma, matches, sample_count):
        title = text_label("Scoring the same test point", 43, TEXT, bold=True).to_edge(UP, buff=0.45)
        frame = Square(
            side_length=4.2, stroke_color=SURFACE, stroke_width=1.8,
            fill_color=CRUST, fill_opacity=1,
        ).move_to(LEFT * 3.25 + DOWN * 0.25)
        plot_label = text_label("Closest sampled prediction", 28, SUBTEXT).next_to(frame, UP, buff=0.18)

        # The close-up spans four standard deviations on each side of the prediction.
        scale = frame.width / (8 * sigma)
        prediction = Dot(frame.get_center(), radius=0.07, color=GREEN)
        test = Dot(
            prediction.get_center() + scale * (offset[0] * RIGHT + offset[1] * UP),
            radius=0.07, color=YELLOW,
        )
        gap = Line(prediction.get_center(), test.get_center(), color=SUBTEXT, stroke_width=1.5)
        prediction_label = MathTex(
            rf"G_\theta(z_{{{draw_number}}})", font_size=33, color=GREEN,
        ).next_to(prediction, DL, buff=0.15)
        test_label = MathTex(r"x_1", font_size=36, color=YELLOW).next_to(test, UR, buff=0.15)
        markers = VGroup(prediction, test, prediction_label, test_label).set_z_index(2)

        deterministic = VGroup(
            text_label("Deterministic decoder", 37, TEXT, bold=True),
            text_label("Each latent gives one prediction.", 30, SUBTEXT),
            MathTex(rf"{matches}\text{{ of }}{sample_count}\text{{ exact matches}}", font_size=36, color=YELLOW),
            text_label("Even the closest prediction misses.", 28, SUBTEXT),
        ).arrange(DOWN, buff=0.30, aligned_edge=LEFT).move_to(RIGHT * 0.05, aligned_edge=LEFT)
        self.play(
            FadeIn(title), FadeIn(frame), FadeIn(plot_label), FadeIn(gap),
            FadeIn(markers), FadeIn(deterministic), run_time=0.65,
        )
        self.wait(2.2)

        gaussian = VGroup(
            text_label("Narrow Gaussian decoder", 37, TEXT, bold=True),
            text_label(rf"Same prediction, with $\sigma={sigma:g}$.", 30, SUBTEXT),
            MathTex(rf"p_\theta(x_1\mid z_{{{draw_number}}})\approx {density:.2f}", font_size=36, color=GREEN),
            text_label("The near miss now contributes.", 28, SUBTEXT),
        ).arrange(DOWN, buff=0.30, aligned_edge=LEFT)
        gaussian.match_x(deterministic, LEFT).match_y(deterministic, UP)
        kernel = gaussian_blob(GREEN).scale_to_fit_width(frame.width).move_to(prediction).set_z_index(1)
        self.play(FadeOut(deterministic), FadeOut(gap), FadeIn(kernel), FadeIn(gaussian), run_time=0.65)
        self.wait(2.4)
        self.play(
            FadeOut(title), FadeOut(frame), FadeOut(plot_label), FadeOut(markers),
            FadeOut(kernel), FadeOut(gaussian), run_time=0.45,
        )

    def evaluate_test_points(
        self, prior_view, prediction_view, estimate, test_dots, conditional_values, sigma, kernel_width
    ):
        _, prior_dots = prior_view
        _, prediction_dots = prediction_view
        sample_count = len(prior_dots)
        max_density = 1 / (2 * pi * sigma**2)
        tile_heading = text_label("Conditional densities", 29, SUBTEXT)
        tile_caption = text_label("Brighter squares contribute more.", 25, SUBTEXT)
        tile_template = VGroup(
            *(
                Square(side_length=0.35, stroke_color=SURFACE, stroke_width=1.15)
                for _ in range(sample_count)
            )
        ).arrange_in_grid(cols=12, buff=0.068)
        density_column = VGroup(tile_heading, tile_template, tile_caption)
        density_column.arrange(DOWN, buff=0.17).next_to(estimate, DOWN, buff=0.17).match_x(prior_view)

        for i, (test_dot, values) in enumerate(zip(test_dots, conditional_values, strict=True), start=1):
            test_label = MathTex(rf"\text{{Test point }}x_{i}", font_size=33, color=YELLOW)
            test_ring = Circle(radius=0.13, color=YELLOW, stroke_width=2.2).move_to(test_dot)
            tile_grid = tile_template.copy()
            count = Integer(0, font_size=27, color=TEXT, edge_to_fix=RIGHT)
            count_line = VGroup(
                count, text_label(f"of {sample_count} terms", 26, SUBTEXT)
            ).arrange(RIGHT, buff=0.14)
            running_sum = DecimalNumber(
                0, num_decimal_places=3, font_size=28, color=GREEN, edge_to_fix=LEFT
            )
            sum_line = VGroup(text_label("Running sum", 27, SUBTEXT), running_sum).arrange(RIGHT, buff=0.19)
            score = values.mean()
            result = VGroup(
                MathTex(
                    rf"\widehat p_\theta(x_{i})=\frac{{{values.sum():.3f}}}{{{sample_count}}}\approx {score:.4f}",
                    font_size=27, color=TEXT,
                ),
                MathTex(rf"\log\widehat p_\theta(x_{i})\approx {log(score):.3f}", font_size=27, color=TEXT),
            ).arrange(DOWN, buff=0.16)
            score_column = VGroup(test_label, count_line, sum_line, result)
            score_column.arrange(DOWN, buff=0.20).next_to(estimate, DOWN, buff=0.17).match_x(prediction_view)

            narrow_gaussian = gaussian_blob(GREEN).scale_to_fit_width(kernel_width)
            latent_ring = Circle(radius=0.11, color=GREEN, stroke_width=2)
            prediction_ring = Circle(radius=0.10, color=GREEN, stroke_width=2)
            probe = Group(narrow_gaussian, latent_ring, prediction_ring)

            def position_probe(index):
                latent_ring.move_to(prior_dots[index])
                prediction_ring.move_to(prediction_dots[index])
                narrow_gaussian.move_to(prediction_dots[index])

            first_index = int(np.argmin(values)) if i == 1 else 0
            position_probe(first_index)
            self.play(FadeIn(test_label), FadeIn(test_ring), FadeIn(probe), run_time=0.45)
            if i == 1:
                far = MathTex(
                    rf"p_\theta(x_1\mid z_{{{first_index + 1}}})\approx 0",
                    font_size=31, color=SUBTEXT,
                )
                far_caption = text_label("A distant prediction contributes very little.", 25, SUBTEXT)
                VGroup(far, far_caption).arrange(DOWN, buff=0.3).next_to(test_label, DOWN, buff=0.52)
                self.play(FadeIn(far), FadeIn(far_caption), run_time=0.35)
                self.wait(1.05)
                nearest = int(np.argmax(values))
                near_label = MathTex(
                    rf"p_\theta(x_1\mid z_{{{nearest + 1}}})\approx {values[nearest]:.3f}",
                    font_size=31, color=GREEN,
                ).move_to(far)
                near_caption = text_label("A nearby prediction contributes much more.", 25, SUBTEXT)
                self.play(
                    latent_ring.animate.move_to(prior_dots[nearest]),
                    prediction_ring.animate.move_to(prediction_dots[nearest]),
                    narrow_gaussian.animate.move_to(prediction_dots[nearest]),
                    Transform(far, near_label), Transform(far_caption, near_caption.move_to(far_caption)),
                    run_time=0.65,
                )
                self.wait(1.15)
                self.play(FadeOut(far), FadeOut(far_caption), run_time=0.35)
                self.play(FadeIn(tile_heading), FadeIn(tile_caption), run_time=0.45)

            position_probe(0)
            cursor = SurroundingRectangle(tile_grid[0], color=GREEN, stroke_width=1.8, buff=0.026)
            cumulative = np.r_[0, np.cumsum(values)]
            visited = 0

            def reveal_terms(_group, alpha):
                nonlocal visited
                scanned = min(sample_count, int(alpha * sample_count + 1e-8))
                for j in range(visited, scanned):
                    opacity = 0.11 + 0.88 * np.sqrt(values[j] / max_density)
                    tile_grid[j].set_fill(GREEN, opacity=float(opacity))
                if scanned > visited:
                    cursor.move_to(tile_grid[scanned - 1])
                    position_probe(scanned - 1)
                    count.set_value(scanned)
                    running_sum.set_value(cumulative[scanned])
                    visited = scanned

            self.play(FadeIn(tile_grid), FadeIn(cursor), FadeIn(count_line), FadeIn(sum_line), run_time=0.35)
            sweep = Group(tile_grid, cursor, probe, count_line, sum_line)
            self.play(UpdateFromAlphaFunc(sweep, reveal_terms, rate_func=linear), run_time=3.3 if i == 1 else 2.2)
            self.play(FadeIn(result), run_time=0.55)
            if i == 1:
                largest = np.argsort(values)[-3:]
                share = 100 * float(values[largest].sum() / values.sum())
                contribution = text_label(
                    rf"3 draws supply {share:.1f}\% of the sum.", 25, YELLOW,
                ).move_to(tile_caption)
                original_caption = tile_caption.copy()
                self.play(
                    *(tile_grid[j].animate.set_stroke(YELLOW, width=1.8) for j in largest),
                    Transform(tile_caption, contribution), run_time=0.45,
                )
                self.wait(1.9)
                self.play(Transform(tile_caption, original_caption), run_time=0.35)
            else:
                self.wait(0.65)
            self.play(FadeOut(sweep), FadeOut(result), FadeOut(test_ring), FadeOut(test_label), run_time=0.35)
        self.play(FadeOut(tile_heading), FadeOut(tile_caption), run_time=0.35)

    def show_likelihood_table(self, conditional_values):
        likelihoods = conditional_values.mean(axis=1)
        log_likelihoods = np.log(likelihoods)
        table = MobjectTable(
            [
                [
                    MathTex(rf"x_{i}", font_size=36, color=YELLOW),
                    MathTex(f"{likelihood:.4f}", font_size=35, color=GREEN),
                    MathTex(f"{score:.3f}", font_size=35, color=TEXT),
                ]
                for i, (likelihood, score) in enumerate(
                    zip(likelihoods, log_likelihoods, strict=True), start=1
                )
            ],
            col_labels=[
                text_label("Test point", 33, SUBTEXT, bold=True),
                MathTex(r"\widehat p_\theta(x_i)", font_size=35, color=SUBTEXT),
                MathTex(r"\log \widehat p_\theta(x_i)", font_size=35, color=SUBTEXT),
            ],
            include_outer_lines=True,
            h_buff=1.1,
            v_buff=0.28,
            line_config={"stroke_color": SURFACE, "stroke_width": 1.5},
        )
        score = MathTex(
            rf"\frac{{1}}{{{len(likelihoods)}}}\sum_{{i=1}}^{{{len(likelihoods)}}}"
            rf"\log\widehat p_\theta(x_i)\approx {log_likelihoods.mean():.3f}",
            font_size=41, color=YELLOW,
        )
        average = VGroup(text_label("Average log-likelihood", 31, SUBTEXT), score).arrange(DOWN, buff=0.2)
        VGroup(table, average).arrange(DOWN, buff=0.44)
        self.play(FadeIn(table), run_time=0.85)
        self.play(FadeIn(average), run_time=0.8)
        self.wait(2.8)

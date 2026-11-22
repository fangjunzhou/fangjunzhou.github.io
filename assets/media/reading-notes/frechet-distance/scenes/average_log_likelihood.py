from math import fsum, log
from random import Random

import numpy as np
from manim import (
    DOWN,
    RIGHT,
    UP,
    Circle,
    FadeIn,
    FadeOut,
    Group,
    LaggedStart,
    MobjectTable,
    Scene,
    Transform,
    TransformFromCopy,
    VGroup,
)

from scene_helpers import (
    BASE,
    BLUE,
    GREEN,
    LIMIT,
    Mixture,
    RED,
    SAMPLE_COUNT,
    SUBTEXT,
    SURFACE,
    TEST_COUNT,
    TEST_DOT_RADIUS,
    TRUE_DISTRIBUTION,
    YELLOW,
    density_panel,
    draw_samples,
    map_point,
    mixture_density,
    sample_dots,
    text_label,
)


MODEL_ONE: Mixture = (
    (0.53, (-1.25, -0.72), ((0.48, 0.28), (0.28, 0.42))),
    (0.47, (1.15, 0.74), ((0.58, -0.22), (-0.22, 0.35))),
)
MODEL_TWO: Mixture = (
    (1.0, (0.05, 0.05), ((1.48, 0.38), (0.38, 1.04))),
)


class AverageLogLikelihood(Scene):
    """Score two image models on the same four held-out samples."""

    def construct(self):
        self.camera.background_color = BASE
        rng = Random(19)
        real_samples = draw_samples(TRUE_DISTRIBUTION, SAMPLE_COUNT, rng)
        test_samples = real_samples[:TEST_COUNT]
        models = (MODEL_ONE, MODEL_TWO)
        model_samples = [draw_samples(model, SAMPLE_COUNT, rng) for model in models]

        grid = np.linspace(-LIMIT, LIMIT, 241)
        grid_x, grid_y = np.meshgrid(grid, grid[::-1])
        densities = [mixture_density(mix, grid_x, grid_y) for mix in (TRUE_DISTRIBUTION, *models)]
        peak = max(float(density.max()) for density in densities)

        footer = text_label("Brighter regions have higher density.", 33, SUBTEXT).to_edge(DOWN, buff=0.96)
        true_panel, true_image = density_panel(
            densities[0], peak, r"True density $p^\star(x)$", BLUE
        )
        model_one_panel, model_one_image = density_panel(
            densities[1], peak, r"Model 1 $p_{\theta_1}(x)$", GREEN
        )
        model_two_panel, model_two_image = density_panel(
            densities[2], peak, r"Model 2 $p_{\theta_2}(x)$", RED
        )
        true_slot = true_panel.copy()
        comparison_row = Group(true_slot, model_one_panel, model_two_panel)
        comparison_row.arrange(RIGHT, buff=0.78, aligned_edge=DOWN).next_to(footer, UP, buff=1)
        true_panel.scale(1.26).next_to(footer, UP, buff=0.56)
        real_dots = sample_dots(true_image, real_samples, BLUE)

        self.play(FadeIn(true_panel), FadeIn(footer), run_time=0.75)
        self.play(
            LaggedStart(*(FadeIn(dot, scale=0.7) for dot in real_dots), lag_ratio=0.025),
            run_time=1.35,
        )
        self.wait(0.9)

        one_dots = sample_dots(model_one_image, model_samples[0], GREEN)
        two_dots = sample_dots(model_two_image, model_samples[1], RED)
        real_group = Group(true_panel, real_dots)
        model_note = text_label("Each panel shows samples drawn from its own density.", 33, SUBTEXT)
        model_note.move_to(footer)

        self.play(
            real_group.animate.scale(1 / 1.26).move_to(true_slot),
            FadeIn(model_one_panel),
            Transform(footer, model_note),
            run_time=0.95,
        )
        self.play(FadeIn(one_dots), FadeIn(model_two_panel), run_time=0.65)
        self.play(FadeIn(two_dots), run_time=0.65)
        self.wait(1.2)

        test_note = text_label("Hold out four real images as the test set.", 33, YELLOW).move_to(footer)
        true_test = VGroup(*real_dots[:TEST_COUNT])
        rest = VGroup(*real_dots[TEST_COUNT:])
        self.play(
            *[
                dot.animate.scale_to_fit_width(2 * TEST_DOT_RADIUS).set_fill(YELLOW, opacity=1)
                for dot in true_test
            ],
            rest.animate.set_opacity(0.32),
            Transform(footer, test_note),
            run_time=0.8,
        )
        self.wait(1.05)

        tests_one = sample_dots(model_one_image, test_samples, YELLOW, radius=TEST_DOT_RADIUS)
        tests_two = sample_dots(model_two_image, test_samples, YELLOW, radius=TEST_DOT_RADIUS)
        common_test_note = text_label("Score both models at the same four test points.", 33, YELLOW)
        common_test_note.move_to(footer)
        self.play(FadeOut(one_dots), FadeOut(two_dots), run_time=0.45)
        self.play(
            TransformFromCopy(true_test, tests_one),
            TransformFromCopy(true_test, tests_two),
            Transform(footer, common_test_note),
            run_time=1.2,
        )
        self.wait(1.05)

        charts = (real_group, Group(model_one_panel, tests_one), Group(model_two_panel, tests_two))
        images = (true_image, model_one_image, model_two_image)
        model_colors = (GREEN, RED)
        log_terms = [[], []]
        rows = []
        for index, (x, y) in enumerate(test_samples, start=1):
            row = [text_label(rf"$x_{index}\quad ({x:.2f}, {y:.2f})$", 29, YELLOW)]
            for terms, model, color in zip(log_terms, models, model_colors, strict=True):
                density = float(mixture_density(model, x, y))
                log_density = log(density)
                terms.append(log_density)
                value = text_label(f"{density:.3f}", 29)
                logged = text_label(rf"$\to {log_density:.3f}$", 29, color)
                row.append(VGroup(value, logged).arrange(RIGHT, buff=0.18))
            rows.append(row)

        sums = [fsum(terms) for terms in log_terms]
        averages = [total / TEST_COUNT for total in sums]
        average_entries = [
            text_label(rf"\({total:.3f} \div {TEST_COUNT} \approx {average:.3f}\)", 29, color, bold=True)
            for total, average, color in zip(sums, averages, model_colors, strict=True)
        ]
        table = MobjectTable(
            [*rows, [text_label("Average log-likelihood", 28, SUBTEXT, bold=True), *average_entries]],
            col_labels=[
                text_label("Test image", 28, SUBTEXT, bold=True),
                text_label(r"Model 1 \quad $p \to \log p$", 28, GREEN, bold=True),
                text_label(r"Model 2 \quad $p \to \log p$", 28, RED, bold=True),
            ],
            include_outer_lines=True,
            h_buff=0.95,
            v_buff=0.19,
            line_config={"stroke_color": SURFACE, "stroke_width": 1.5},
        )
        math = text_label(
            r"Average log-likelihood $= \frac{1}{N}\sum_{i=1}^{N}\log p_{\theta}(x_i)$", 35
        )
        better = int(averages[1] > averages[0])
        difference = abs(averages[0] - averages[1])
        result = text_label(f"Model {better + 1} scores {difference:.3f} nats higher per image.", 33, YELLOW)
        chart_slots = Group(*(chart.copy().scale(0.76) for chart in charts))
        chart_slots.arrange(RIGHT, buff=1.58, aligned_edge=DOWN)
        Group(chart_slots, math, table, result).arrange(DOWN, buff=0.18)
        self.play(
            *[
                chart.animate.scale(0.76).move_to(slot)
                for chart, slot in zip(charts, chart_slots, strict=True)
            ],
            FadeOut(footer),
            FadeIn(math),
            run_time=1.0,
        )

        header, *score_rows, average_row = table.get_rows()
        self.play(
            FadeIn(table.get_horizontal_lines()),
            FadeIn(table.get_vertical_lines()),
            FadeIn(header),
            run_time=0.65,
        )

        rings = VGroup(*(Circle(radius=0.13, color=YELLOW, stroke_width=2.4) for _ in images))
        for index, (point, row) in enumerate(zip(test_samples, score_rows, strict=True)):
            points = [map_point(image, point) for image in images]
            if index == 0:
                for ring, position in zip(rings, points, strict=True):
                    ring.move_to(position)
                self.play(FadeIn(rings), run_time=0.3)
            else:
                self.play(
                    *(ring.animate.move_to(pos) for ring, pos in zip(rings, points, strict=True)),
                    run_time=0.3,
                )

            test_label, *score_cells = row
            self.play(FadeIn(test_label, *(cell[0] for cell in score_cells)), run_time=0.43)
            self.play(FadeIn(*(cell[1] for cell in score_cells)), run_time=0.45)
            self.wait(0.38)

        self.play(FadeOut(rings), FadeIn(average_row), run_time=0.8)
        self.wait(1.05)

        highlight = table.get_cell((TEST_COUNT + 2, better + 2), color=YELLOW, stroke_width=2)
        self.play(FadeIn(highlight), FadeIn(result), run_time=0.65)
        self.wait(2.4)

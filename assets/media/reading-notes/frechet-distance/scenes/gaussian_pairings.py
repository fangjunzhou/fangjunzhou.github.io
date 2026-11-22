import numpy as np
from manim import (
    DOWN,
    RIGHT,
    UP,
    Arrow,
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
    TEXT,
    YELLOW,
    axis_numbers,
    convergence_chart,
    gaussian_axes,
    gaussian_density,
    text_label,
)


class GaussianPairings(Scene):
    """Compare Monte Carlo squared gaps for three pairings of two 1D Gaussians."""

    generated_std = 1
    convergence_ticks = range(0, 13, 3)

    def construct(self):
        self.camera.background_color = BASE
        real_std, generated_std = 2, self.generated_std
        generated_scale = "" if generated_std == 1 else str(generated_std)
        sample_count = 16_384
        counts = np.arange(1, sample_count + 1)
        log_counts = np.log10(counts)
        rng = np.random.default_rng(32)
        independent_x, independent_y, shared = rng.standard_normal((3, sample_count))
        # Reuse the shared draws to show what changes when Y changes sign.
        pairings = (
            (
                "Independent draws", (independent_x, independent_y), 1, real_std**2 + generated_std**2,
                r"Z_x,Z_y \overset{\mathrm{iid}}{\sim} \mathcal{N}(0,1)",
                rf"X={real_std}Z_x", rf"Y={generated_scale}Z_y",
            ),
            (
                "Shared draw", (shared,), 1, (real_std - generated_std)**2,
                r"Z\sim\mathcal{N}(0,1)", rf"X={real_std}Z", rf"Y={generated_scale}Z",
            ),
            (
                "Shared draw with a sign flip", (shared,), -1, (real_std + generated_std)**2,
                r"Z\sim\mathcal{N}(0,1)", rf"X={real_std}Z", rf"Y=-{generated_scale}Z",
            ),
        )

        def curve(axes, std, color):
            return axes.plot(lambda x: gaussian_density(x, std), color=color, stroke_width=3)

        def generated_density(axes):
            line = curve(axes, generated_std, GREEN)
            if real_std == generated_std:
                return DashedVMobject(line, num_dashes=44, dashed_ratio=0.55).set_z_index(1)
            return line

        real_axes, generated_axes = VGroup(gaussian_axes(8, 5.75, 2.3), gaussian_axes(8, 5.75, 2.3)).arrange(
            RIGHT, buff=0.8,
        )
        real_curve = curve(real_axes, real_std, BLUE)
        generated_curve = generated_density(generated_axes)
        real_ticks = axis_numbers(real_axes, (-6, -3, 0, 3, 6))
        generated_ticks = axis_numbers(generated_axes, (-6, -3, 0, 3, 6))
        real_plot = VGroup(real_axes, real_ticks, real_curve)
        generated_plot = VGroup(generated_axes, generated_ticks, generated_curve)
        real_label = MathTex(rf"X\sim\mathcal{{N}}(0,{real_std**2})", font_size=36, color=BLUE)
        generated_label = MathTex(rf"Y\sim\mathcal{{N}}(0,{generated_std**2})", font_size=36, color=GREEN)
        real_label.next_to(real_plot, UP, buff=0.28)
        generated_label.next_to(generated_plot, UP, buff=0.28)
        legend = VGroup(real_label.copy(), generated_label.copy()).arrange(RIGHT, buff=0.65)
        source_axes = gaussian_axes(3.5, 3.45, 0.92)
        source_plot = VGroup(source_axes, curve(source_axes, 1, SUBTEXT), axis_numbers(source_axes, (-3, 0, 3)))

        formulas = [
            VGroup(
                MathTex(prior, font_size=31, color=SUBTEXT),
                VGroup(MathTex(x_map, font_size=33, color=BLUE), MathTex(y_map, font_size=33, color=GREEN))
                .arrange(RIGHT, buff=0.65),
            ).arrange(DOWN, buff=0.18)
            for _, _, _, _, prior, x_map, y_map in pairings
        ]
        density_slot = real_plot.copy()
        VGroup(legend, density_slot, source_plot, formulas[0]).arrange(DOWN, buff=0.35).set_x(real_axes.get_x())
        for formula in formulas[1:]:
            formula.move_to(formulas[0])

        self.play(FadeIn(real_plot, real_label), FadeIn(generated_plot, generated_label), run_time=0.85)
        self.wait(1.25)
        self.play(
            real_plot.animate.move_to(density_slot),
            Transform(generated_curve, generated_density(density_slot[0])),
            Transform(real_label, legend[0]),
            Transform(generated_label, legend[1]),
            FadeOut(generated_axes, generated_ticks),
            run_time=1.4,
        )

        graph, chart = convergence_chart(sample_count, self.convergence_ticks)
        phase_labels = [text_label(name, 36) for name, *_ in pairings]
        average = DecimalNumber(9, num_decimal_places=2, font_size=39, color=YELLOW)
        pair_count = Integer(sample_count, font_size=28, color=SUBTEXT)
        pair_cost = DecimalNumber(9, num_decimal_places=2, font_size=28, color=YELLOW)
        average_row = VGroup(
            MathTex(r"\frac{1}{n}\sum_{i=1}^{n}(X_i-Y_i)^2 =", font_size=35, color=TEXT), average,
        ).arrange(RIGHT, buff=0.12)
        count_readout = VGroup(text_label("Pairs", 28, SUBTEXT), pair_count).arrange(RIGHT, buff=0.12)
        cost_readout = VGroup(
            MathTex(r"(X_n-Y_n)^2 =", font_size=28, color=TEXT), pair_cost,
        ).arrange(RIGHT, buff=0.12)
        pair_row = VGroup(count_readout, cost_readout).arrange(RIGHT, buff=0.50)
        expected_labels = [
            MathTex(rf"\mathbb{{E}}[(X-Y)^2] = {expected}", font_size=32, color=RED)
            for _, _, _, expected, *_ in pairings
        ]
        VGroup(phase_labels[0], average_row, chart, expected_labels[0], pair_row).arrange(
            DOWN, buff=0.25,
        ).set_x(generated_axes.get_x())
        for phase, expected_label in zip(phase_labels[1:], expected_labels[1:], strict=True):
            phase.move_to(phase_labels[0])
            expected_label.move_to(expected_labels[0])

        checkpoints = np.unique(np.rint(np.geomspace(1, sample_count, 44)).astype(int)) - 1
        curve_indices = np.unique(np.rint(np.geomspace(1, sample_count, 480)).astype(int)) - 1
        self.play(FadeIn(source_plot), run_time=0.55)

        for stage, (_, draws, sign, expected, *_labels) in enumerate(pairings):
            xs = real_std * draws[0]
            ys = sign * generated_std * draws[-1]
            costs = (xs - ys)**2
            running_means = np.cumsum(costs) / counts
            curve_points = np.array([graph.c2p(log_counts[i], running_means[i]) for i in curve_indices])
            trace = VMobject(color=YELLOW, stroke_width=3).set_z_index(2)
            benchmark = DashedVMobject(
                Line(graph.c2p(0, expected), graph.c2p(log_counts[-1], expected), color=RED, stroke_width=2.1),
                num_dashes=26,
            )
            colors = (BLUE, GREEN) if len(draws) == 2 else (TEXT,)
            sources = VGroup(*[Dot(radius=0.065, color=color) for color in colors]).set_z_index(3)
            x_dot = Dot(radius=0.072, color=BLUE, stroke_color=CRUST, stroke_width=1.5).set_z_index(3)
            y_dot = Dot(radius=0.072, color=GREEN, stroke_color=CRUST, stroke_width=1.5).set_z_index(3)
            x_stem = Line(color=BLUE, stroke_width=1.5, stroke_opacity=0.50)
            y_stem = Line(color=GREEN, stroke_width=1.5, stroke_opacity=0.50)
            if expected == 0:
                y_dot.scale(0.60).set_z_index(4)
                gap = None
                gap_marks = VGroup(x_stem)
            else:
                gap = Line(color=YELLOW, stroke_width=4).set_z_index(2)
                gap_marks = VGroup(x_stem, y_stem, gap)
            sample_marks = VGroup(sources, x_dot, y_dot, gap_marks)
            last_index = -1

            def show_sample(index):
                for dot, noise in zip(sources, draws, strict=True):
                    z = noise[index]
                    dot.move_to(source_axes.c2p(z, gaussian_density(z, 1)))
                x, y = xs[index], ys[index]
                real_point = real_axes.c2p(x, gaussian_density(x, real_std))
                generated_point = real_axes.c2p(y, gaussian_density(y, generated_std))
                real_x, generated_x = real_axes.c2p(x, 0), real_axes.c2p(y, 0)
                x_dot.move_to(real_point)
                y_dot.move_to(generated_point)
                x_stem.put_start_and_end_on(real_point, real_x)
                if gap is not None:
                    y_stem.put_start_and_end_on(generated_point, generated_x)
                    gap.put_start_and_end_on(real_x, generated_x)
                average.set_value(running_means[index])
                pair_count.set_value(counts[index])
                pair_cost.set_value(costs[index])

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
            if expected == 0:
                mappings = ((sources[0], x_dot, SUBTEXT),)
            else:
                mappings = ((sources[0], x_dot, BLUE), (sources[-1], y_dot, GREEN))
            arrows = VGroup(*[
                Arrow(source.get_center(), target.get_center(), color=color, stroke_width=2.2, buff=0.08)
                for source, target, color in mappings
            ])
            phase, formula, exact_label = phase_labels[stage], formulas[stage], expected_labels[stage]
            self.play(FadeIn(phase, formula, sources), run_time=0.55)
            self.play(Create(arrows), FadeIn(x_dot, y_dot), run_time=0.8)
            self.play(FadeIn(gap_marks, pair_row), run_time=0.5)
            self.wait(1.25)
            graph_reveal = FadeIn(chart) if stage == 0 else FadeIn(benchmark)
            self.play(FadeOut(arrows), graph_reveal, FadeIn(average_row, exact_label), run_time=0.5)
            if stage == 0:
                self.play(FadeIn(benchmark), run_time=0.3)

            live = VGroup(sample_marks, average_row, pair_row, trace)
            self.remove(sources, x_dot, y_dot, gap_marks, average_row, pair_row)
            self.add(live)
            self.play(UpdateFromAlphaFunc(live, advance), rate_func=linear, run_time=5.5)
            self.wait(1.55)
            if stage < len(pairings) - 1:
                self.play(FadeOut(live, formula, phase, benchmark, exact_label), run_time=0.45)

        self.wait(1)

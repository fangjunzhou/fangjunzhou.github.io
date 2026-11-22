import numpy as np
from manim import (
    DEGREES,
    DOWN,
    RIGHT,
    Arc,
    Arrow,
    Axes,
    DashedVMobject,
    DecimalMatrix,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    MathTex,
    Scene,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
)

from scene_helpers import BASE, BLUE, CRUST, GREEN, RED, SUBTEXT, SURFACE, TEXT, YELLOW, gaussian_contours, text_label


class RotatedGaussianDraw(Scene):
    """Rotate one draw in a pair from two identical circular Gaussians."""

    def construct(self):
        self.camera.background_color = BASE
        draws = np.random.default_rng(44).standard_normal((4, 2))
        sample = draws[0]
        axes = Axes(
            x_range=(-3.2, 3.2, 1), y_range=(-3.2, 3.2, 1), x_length=5.3, y_length=5.3, tips=False,
            axis_config={"color": SURFACE, "stroke_width": 1.6, "include_ticks": False},
        )
        legend = VGroup(
            MathTex(r"X=Z", font_size=38, color=BLUE),
            MathTex(r"Y=R_\theta Z", font_size=38, color=GREEN),
        ).arrange(RIGHT, buff=0.75)
        distribution = MathTex(r"X,Y \sim \mathcal{N}(0,I_2)", font_size=32, color=SUBTEXT)
        pair_cost = DecimalNumber(0, num_decimal_places=2, font_size=34, color=YELLOW)
        pair_row = VGroup(
            MathTex(r"\lVert X-Y\rVert_2^2 =", font_size=34, color=YELLOW), pair_cost,
        ).arrange(RIGHT, buff=0.15)
        left = VGroup(legend, axes, distribution, pair_row).arrange(DOWN, buff=0.21)

        rotation_formula = MathTex(
            r"R_\theta=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}",
            font_size=39, color=TEXT,
        )
        angle_value = DecimalNumber(180, num_decimal_places=0, unit=r"^\circ", font_size=34, color=TEXT)
        angle_row = VGroup(MathTex(r"\theta =", font_size=34, color=TEXT), angle_value).arrange(RIGHT, buff=0.12)
        matrix = DecimalMatrix(
            np.eye(2), h_buff=1.1, v_buff=0.53, bracket_h_buff=0.10,
            element_to_mobject_config={"num_decimal_places": 2, "font_size": 29},
        ).set_color(TEXT)
        rotation = VGroup(angle_row, matrix).arrange(RIGHT, buff=0.36)
        expected_formula = MathTex(
            r"\mathbb{E}[\lVert X-Y\rVert_2^2] = 4(1-\cos\theta)", font_size=34, color=RED,
        )
        expected_value = DecimalNumber(0, num_decimal_places=2, font_size=43, color=RED)
        expected_equals = MathTex("=", font_size=34, color=RED)
        expected_row = VGroup(expected_equals, expected_value).arrange(RIGHT, buff=0.15)
        expected = VGroup(expected_formula, expected_row).arrange(DOWN, buff=0.17)
        distance = VGroup(
            text_label(r"Squared Fr\'echet distance", 31),
            MathTex(r"d_{\mathrm{F}}^2=0", font_size=36, color=TEXT),
        ).arrange(DOWN, buff=0.17)
        right = VGroup(rotation_formula, rotation, expected, distance).arrange(DOWN, buff=0.46)
        VGroup(left, right).arrange(RIGHT, buff=0.95)
        matrix_entries = matrix.get_entries()
        entry_centers = [entry.get_center() for entry in matrix_entries]

        zero = np.zeros(2)
        center = axes.c2p(*zero)
        real_density = gaussian_contours(axes, zero, np.eye(2), BLUE)
        generated_density = gaussian_contours(axes, zero, np.eye(2), GREEN)
        generated_rings = VGroup(*[
            DashedVMobject(ring.set_fill(opacity=0), num_dashes=32, dashed_ratio=0.55)
            for ring in generated_density[0]
        ]).set_z_index(2)
        x_arrow = Arrow(center, axes.c2p(*sample), buff=0, color=BLUE, stroke_width=4.4, tip_length=0.17)
        y_arrow = Arrow(center, axes.c2p(*sample), buff=0, color=GREEN, stroke_width=2.6, tip_length=0.13)
        x_dot = Dot(radius=0.081, color=BLUE, stroke_color=CRUST, stroke_width=1.5).set_z_index(4)
        y_dot = Dot(radius=0.052, color=GREEN, stroke_color=CRUST, stroke_width=1).set_z_index(5)
        gap = VMobject(color=YELLOW, stroke_width=3.2).set_z_index(3)
        angle_arc = Arc(radius=0.56, color=SUBTEXT, stroke_width=2).set_z_index(3)
        angle_mark = MathTex(r"\theta", font_size=27, color=SUBTEXT).set_z_index(3)
        pair = VGroup(x_arrow, y_arrow, gap, angle_arc, angle_mark, x_dot, y_dot)
        x_arrow.set_z_index(3)
        y_arrow.set_z_index(3)

        def show_rotation(angle):
            cosine, sine = np.cos(angle), np.sin(angle)
            transform = np.array([[cosine, -sine], [sine, cosine]])
            rotated = transform @ sample
            x_tip, y_tip = axes.c2p(*sample), axes.c2p(*rotated)
            x_arrow.put_start_and_end_on(center, x_tip)
            y_arrow.put_start_and_end_on(center, y_tip)
            x_dot.move_to(x_tip)
            y_dot.move_to(y_tip)
            gap.set_points_as_corners([x_tip, y_tip])
            heading = np.arctan2(sample[1], sample[0])
            middle = heading + angle / 2
            angle_arc.become(Arc(
                radius=0.56, start_angle=heading, angle=angle, arc_center=center, color=SUBTEXT, stroke_width=2,
            ))
            angle_mark.move_to(center + 0.75 * np.array([np.cos(middle), np.sin(middle), 0]))
            angle_mark.set_opacity(min(angle / (18 * DEGREES), 1))
            for entry, position, value in zip(matrix_entries, entry_centers, transform.ravel(), strict=True):
                entry.set_value(value).move_to(position)
            angle_value.set_value(angle / DEGREES)
            pair_cost.set_value(np.sum((sample - rotated)**2))
            expected_value.set_value(4 * (1 - cosine))

        show_rotation(0)
        self.play(FadeIn(axes, legend, real_density, generated_rings, distribution), run_time=0.9)
        self.play(FadeIn(pair, pair_row, rotation_formula, rotation, expected, distance), run_time=0.85)
        self.wait(1.25)

        live = VGroup(pair, pair_row, rotation, expected_row)
        self.remove(pair, pair_row, rotation, expected_row)
        self.add(live)
        quarter_turn = 90 * DEGREES
        self.play(UpdateFromAlphaFunc(live, lambda _live, alpha: show_rotation(alpha * quarter_turn)), run_time=2.8)
        self.wait(1.6)

        self.remove(live)
        self.add(pair, pair_row, rotation, expected_row)
        for sample in draws[1:]:
            self.play(FadeOut(pair, pair_row), run_time=0.22)
            show_rotation(quarter_turn)
            self.play(FadeIn(pair, pair_row), run_time=0.22)
            self.wait(0.55)

        self.remove(pair, pair_row, rotation, expected_row)
        self.add(live)
        self.play(
            UpdateFromAlphaFunc(live, lambda _live, alpha: show_rotation((1 + alpha) * quarter_turn)),
            run_time=2.5,
        )
        self.wait(1.2)
        self.play(
            UpdateFromAlphaFunc(live, lambda _live, alpha: show_rotation(2 * (1 - alpha) * quarter_turn)),
            run_time=3.4,
        )
        self.wait(1.7)

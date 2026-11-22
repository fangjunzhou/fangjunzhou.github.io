import numpy as np
from manim import (
    DOWN,
    RIGHT,
    Arrow,
    Axes,
    Dot,
    FadeIn,
    FadeOut,
    MathTex,
    Rotate,
    Scene,
    TransformFromCopy,
    VGroup,
)

from scene_helpers import (
    BASE, BLUE, CRUST, GREEN, SUBTEXT, SURFACE, TEXT, YELLOW, gaussian_contours, svd_with_rotations, text_label,
)


def pairing_example():
    a = np.diag([3.0, 1.0])
    b = np.array([[1.0, 1.0], [1.0, 2.0]])
    m = b @ a
    p, singular_values, qt = svd_with_rotations(m)
    u = np.random.default_rng(44).standard_normal(2)
    return a, b, m, p, np.diag(singular_values), qt, u


def latex_matrix(matrix, digits):
    rows = [" & ".join(f"{value:.{digits}f}" for value in row) for row in matrix]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


class SvdGaussianPairing(Scene):
    """Pair one standard Gaussian draw by matching its SVD coordinates."""

    def construct(self):
        self.camera.background_color = BASE
        a, b, m, p, d, qt, u = pairing_example()
        rotation = p @ qt
        v = rotation @ u
        u_prime, v_prime = qt @ u, p.T @ v

        def matrix_readout(label, values, digits, *, size=37):
            return MathTex(label + latex_matrix(values, digits), font_size=size, color=TEXT)

        inputs = VGroup(
            matrix_readout("A=", a, 0),
            matrix_readout("B=", b, 0),
            matrix_readout("M=BA=", m, 0),
        ).arrange(RIGHT, buff=0.52)
        factors = VGroup(
            matrix_readout(r"P\approx", p, 3, size=34),
            matrix_readout(r"D\approx", d, 3, size=34),
            matrix_readout(r"Q^\top\approx", qt, 3, size=34),
        ).arrange(RIGHT, buff=0.47)
        factorization = VGroup(MathTex(r"M=PDQ^\top", font_size=40, color=TEXT), factors).arrange(
            DOWN, buff=0.26,
        )
        matrix_panel = VGroup(inputs, factorization).arrange(DOWN, buff=0.68)

        def coordinates():
            return Axes(
                x_range=(-2.65, 2.65, 1), y_range=(-2.65, 2.65, 1),
                x_length=4.3, y_length=4.3, tips=False,
                axis_config={"color": SURFACE, "stroke_width": 1.6, "include_ticks": False},
            )

        def vector_readout(label, point, color):
            return MathTex(label + latex_matrix(point[:, None], 3), font_size=32, color=color)

        original_axes, svd_axes = coordinates(), coordinates()
        original_title = text_label("Original coordinates", 34, SUBTEXT)
        svd_title = text_label("SVD coordinates", 34, SUBTEXT)
        draw_label = MathTex(r"U \sim \mathcal{N}(0,I_2)", font_size=30, color=SUBTEXT)
        u_value = vector_readout(r"U\approx", u, BLUE)
        v_value = vector_readout(r"V\approx", v, GREEN)
        original_values = VGroup(u_value, v_value).arrange(RIGHT, buff=0.52)
        u_prime_value = vector_readout(r"U'=Q^\top U\approx", u_prime, BLUE)
        v_prime_value = vector_readout(r"V'=P^\top V\approx", v_prime, GREEN)
        svd_values = VGroup(u_prime_value, v_prime_value).arrange(RIGHT, buff=0.40)
        equality = MathTex(r"U'=V'", font_size=33, color=YELLOW)
        original_panel = VGroup(original_title, original_axes, draw_label, original_values).arrange(DOWN, buff=0.17)
        svd_panel = VGroup(svd_title, svd_axes, equality, svd_values).arrange(DOWN, buff=0.17)
        VGroup(original_panel, svd_panel).arrange(RIGHT, buff=0.9, aligned_edge=DOWN)

        rotation_panel = VGroup(
            MathTex(r"V=PQ^\top U", font_size=43, color=GREEN),
            matrix_readout(r"PQ^\top\approx", rotation, 3),
        ).arrange(DOWN, buff=0.30).move_to(svd_axes)

        def vector(axes, point, color, radius, stroke):
            center, tip = axes.c2p(0, 0), axes.c2p(*point)
            return VGroup(
                Arrow(center, tip, buff=0, color=color, stroke_width=stroke, tip_length=0.15),
                Dot(tip, radius=radius, color=color, stroke_color=CRUST, stroke_width=1.2),
            ).set_z_index(3)

        original_density = gaussian_contours(original_axes, np.zeros(2), np.eye(2), SUBTEXT)
        svd_density = gaussian_contours(svd_axes, np.zeros(2), np.eye(2), SUBTEXT)
        u_vector = vector(original_axes, u, BLUE, 0.083, 4.2)
        v_vector = vector(original_axes, u, GREEN, 0.055, 2.5).set_z_index(4)
        u_prime_vector = vector(svd_axes, u_prime, BLUE, 0.083, 4.2)
        v_prime_vector = vector(svd_axes, v_prime, GREEN, 0.055, 2.5).set_z_index(4)
        angle = np.arctan2(rotation[1, 0], rotation[0, 0])

        initial_draw = VGroup(original_title, original_axes, original_density, draw_label, u_vector, u_value)
        self.play(FadeIn(original_title, original_axes, original_density, draw_label), run_time=0.8)
        self.play(FadeIn(u_vector, u_value), run_time=0.65)
        self.wait(1.1)
        self.play(FadeOut(initial_draw), FadeIn(inputs), run_time=0.8)
        self.wait(1.3)
        self.play(FadeIn(factorization), run_time=0.9)
        self.wait(2.4)
        self.play(
            FadeOut(inputs, factorization),
            FadeIn(initial_draw),
            run_time=0.85,
        )
        self.play(FadeIn(rotation_panel), run_time=0.85)
        self.wait(1.05)
        self.play(FadeIn(v_vector), run_time=0.35)
        self.play(
            Rotate(v_vector, angle=angle, about_point=original_axes.c2p(0, 0)),
            FadeIn(v_value),
            run_time=1.5,
        )
        self.wait(1.35)
        self.play(
            FadeOut(rotation_panel), FadeIn(svd_title, svd_axes, svd_density), run_time=0.8,
        )
        self.play(TransformFromCopy(u_vector, u_prime_vector), FadeIn(u_prime_value), run_time=0.95)
        self.wait(0.75)
        self.play(TransformFromCopy(v_vector, v_prime_vector), FadeIn(v_prime_value), run_time=0.95)
        self.play(FadeIn(equality), run_time=0.5)
        self.wait(2.1)

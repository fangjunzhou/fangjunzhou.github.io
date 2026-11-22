"""Inspect the source images behind points in the ImageNetV2 PCA plane."""

from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Axes,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    Line,
    ManimColor,
    MathTex,
    Rectangle,
    Scene,
    SurroundingRectangle,
    UpdateFromAlphaFunc,
    VGroup,
    linear,
)
from PIL import Image, ImageDraw

from scene_helpers import BASE, BLUE, GREEN, MAUVE, RED, SUBTEXT, SURFACE, YELLOW, text_label


def pca_plane(scores, explained):
    x_min, y_min = np.floor(scores.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(scores.max(axis=0)).astype(int)
    unit = min(7.6 / (x_max - x_min), 5.2 / (y_max - y_min))
    axes = Axes(
        x_range=(x_min, x_max, 5),
        y_range=(y_min, y_max, 2),
        x_length=unit * (x_max - x_min),
        y_length=unit * (y_max - y_min),
        tips=False,
        axis_config={"include_ticks": False, "stroke_opacity": 0},
    )
    x_ticks = range(((x_min + 4) // 5) * 5, x_max + 1, 5)
    y_ticks = range(((y_min + 1) // 2) * 2, y_max + 1, 2)
    grid = VGroup(*[
        Line(axes.c2p(x, y_min), axes.c2p(x, y_max), color=SURFACE, stroke_width=0.8)
        for x in x_ticks
    ], *[
        Line(axes.c2p(x_min, y), axes.c2p(x_max, y), color=SURFACE, stroke_width=0.8)
        for y in y_ticks
    ]).set_stroke(opacity=0.68)
    horizontal = Line(axes.c2p(x_min, y_min), axes.c2p(x_max, y_min), color=SUBTEXT, stroke_width=1.5)
    vertical = Line(axes.c2p(x_min, y_min), axes.c2p(x_min, y_max), color=SUBTEXT, stroke_width=1.5)
    x_numbers = VGroup(*[
        MathTex(str(x), font_size=22, color=SUBTEXT).next_to(axes.c2p(x, y_min), DOWN, buff=0.12)
        for x in x_ticks
    ])
    y_numbers = VGroup(*[
        MathTex(str(y), font_size=22, color=SUBTEXT).next_to(axes.c2p(x_min, y), LEFT, buff=0.12)
        for y in y_ticks
    ])
    x_label = text_label(rf"PC1 ({explained[0]:.1%} of variance)".replace("%", r"\%"), 28)
    x_label.next_to(horizontal, DOWN, buff=0.47)
    y_label = text_label(rf"PC2 ({explained[1]:.1%} of variance)".replace("%", r"\%"), 28)
    y_label.rotate(np.pi / 2).next_to(y_numbers, LEFT, buff=0.17)
    guide = VGroup(axes, grid, horizontal, vertical, x_numbers, y_numbers, x_label, y_label)

    width = 1200
    height = round(width * axes.y_length / axes.x_length)
    points = (scores - (x_min, y_min)) / (x_max - x_min, y_max - y_min)
    pixels = points * (width - 1, 1 - height) + (0, height - 1)
    canvas = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(canvas)
    point_color = (*ManimColor(BLUE).to_int_rgb(), 158)
    frames = [np.array(canvas)]
    order = np.random.default_rng(0).permutation(len(scores))
    for batch in np.array_split(order, 24):
        for x, y in pixels[batch]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=point_color)
        frames.append(np.array(canvas))

    # Rasterize the dense point layer once so Manim can move it with the axes.
    cloud = ImageMobject(frames[0])
    cloud.stretch_to_fit_width(axes.x_length).stretch_to_fit_height(axes.y_length)
    cloud.move_to(axes.c2p((x_min + x_max) / 2, (y_min + y_max) / 2))
    return axes, guide, cloud, frames


class ImageNetPCAInspection(Scene):
    def construct(self):
        self.camera.background_color = BASE
        data_file = Path(__file__).with_name("imagenet-v2-pca-data.npz")
        with np.load(data_file) as data:
            scores, explained = data["scores"], data["explained"]
            selected = data["selected"]
            previews = [data[f"preview_{i}"] for i in range(len(selected))]

        axes, guide, cloud, frames = pca_plane(scores, explained)
        chart = Group(guide, cloud).move_to(np.zeros(3))
        self.play(FadeIn(guide), run_time=0.6)

        def reveal_points(image, alpha):
            image.get_pixel_array()[:] = frames[round(alpha * (len(frames) - 1))]

        self.play(UpdateFromAlphaFunc(cloud, reveal_points, rate_func=linear), run_time=2.4)
        self.wait(0.6)

        image_slot = Rectangle(width=3.8, height=3.8)
        layout = Group(chart.copy(), image_slot).arrange(RIGHT, buff=1.05).move_to(np.zeros(3))
        self.play(chart.animate.move_to(layout[0]), run_time=0.85)

        for index, pixels, color in zip(selected, previews, (GREEN, YELLOW, RED, MAUVE), strict=True):
            point = axes.c2p(*scores[index])
            ring = Circle(radius=0.16, color=color, stroke_width=2.5).move_to(point)
            dot = Dot(point, radius=0.058, color=color, stroke_color=BASE, stroke_width=1.2)
            marker = VGroup(ring, dot).set_z_index(3)

            image = ImageMobject(pixels)
            image.scale(min(image_slot.width / image.width, image_slot.height / image.height))
            image.move_to(image_slot)
            border = SurroundingRectangle(image, buff=0, color=color, stroke_width=2)
            card = Group(image, border).set_z_index(2)
            connector = Line(point, image.get_left(), color=color, stroke_width=1.7)
            connector.set_stroke(opacity=0.8).set_z_index(1)

            self.play(Create(ring), FadeIn(dot), run_time=0.4)
            self.play(Create(connector), FadeIn(card, shift=RIGHT * 0.14), run_time=0.52)
            self.wait(2.05)
            self.play(FadeOut(marker), FadeOut(connector), FadeOut(card), run_time=0.38)

        self.play(chart.animate.move_to(np.zeros(3)), run_time=0.8)
        self.wait(0.7)

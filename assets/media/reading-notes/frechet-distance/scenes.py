from itertools import product
from math import fsum, log, pi
from random import Random
from statistics import NormalDist

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Circle,
    Create,
    DashedVMobject,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    Integer,
    LaggedStart,
    Line,
    ManimColor,
    MathTex,
    MobjectTable,
    Scene,
    Square,
    SurroundingRectangle,
    Tex,
    ThreeDAxes,
    ThreeDCamera,
    ThreeDScene,
    Transform,
    TransformFromCopy,
    TransformMatchingTex,
    UpdateFromAlphaFunc,
    VGroup,
    VMobject,
    linear,
)

# Catppuccin Mocha
BASE = "#1e1e2e"
CRUST = "#11111b"
SURFACE = "#45475a"
TEXT = "#cdd6f4"
SUBTEXT = "#bac2de"
BLUE = "#89b4fa"
GREEN = "#a6e3a1"
YELLOW = "#f9e2af"
RED = "#f38ba8"
MAUVE = "#cba6f7"
COMPONENT_COLORS = (BLUE, GREEN, YELLOW, RED, MAUVE)

LIMIT = 3.4
PANEL_WIDTH = 3.4
SAMPLE_COUNT = 64
TEST_COUNT = 4
TEST_DOT_RADIUS = 0.05

# Each component is (weight, mean, covariance).
type Point = tuple[float, ...]
type Covariance = tuple[tuple[float, ...], ...]
type Mixture = tuple[tuple[float, Point, Covariance], ...]

TRUE_DISTRIBUTION: Mixture = (
    (0.55, (-1.38, -0.82), ((0.38, 0.27), (0.27, 0.36))),
    (0.45, (1.24, 0.83), ((0.52, -0.24), (-0.24, 0.28))),
)
MODEL_ONE: Mixture = (
    (0.53, (-1.25, -0.72), ((0.48, 0.28), (0.28, 0.42))),
    (0.47, (1.15, 0.74), ((0.58, -0.22), (-0.22, 0.35))),
)
MODEL_TWO: Mixture = (
    (1.0, (0.05, 0.05), ((1.48, 0.38), (0.38, 1.04))),
)
SPARSE_DISTRIBUTION: Mixture = (
    (
        0.2,
        (-2.85, -1.25, -0.85),
        ((0.10, 0.047, 0.036), (0.047, 0.056, 0.028), (0.036, 0.028, 0.052)),
    ),
    (
        0.2,
        (-1.2, 1.57, 1.2),
        ((0.035, -0.021, 0.024), (-0.021, 0.065, -0.034), (0.024, -0.034, 0.09)),
    ),
    (
        0.2,
        (0.05, -0.95, -0.4),
        ((0.038, 0.022, -0.023), (0.022, 0.065, -0.04), (-0.023, -0.04, 0.062)),
    ),
    (
        0.2,
        (2.7, -1.64, 1.56),
        ((0.13, -0.061, 0.038), (-0.061, 0.069, -0.025), (0.038, -0.025, 0.044)),
    ),
    (
        0.2,
        (2.7, 1.52, -1.13),
        ((0.065, 0.038, -0.041), (0.038, 0.073, -0.046), (-0.041, -0.046, 0.076)),
    ),
)


def mixture_density(mixture: Mixture, x, y):
    points = np.stack(np.broadcast_arrays(x, y), axis=-1)
    density = np.zeros(points.shape[:-1])
    for weight, mean, covariance in mixture:
        factor = np.linalg.cholesky(covariance)
        offsets = (points - mean).reshape(-1, 2)
        standardized = np.linalg.solve(factor, offsets.T)
        squared_distance = np.sum(standardized**2, axis=0).reshape(density.shape)
        density += weight * np.exp(-0.5 * squared_distance) / (2 * pi * np.prod(factor.diagonal()))
    return density


def draw_samples(mixture: Mixture, count: int, rng: Random) -> list[Point]:
    components = [(np.asarray(mean), np.linalg.cholesky(cov)) for _, mean, cov in mixture]
    picks = rng.choices(components, weights=[weight for weight, _, _ in mixture], k=count)
    return [
        tuple(mean + factor @ [rng.gauss(0, 1) for _ in mean])
        for mean, factor in picks
    ]


def text_label(text: str, size: int, color: str = TEXT, *, bold: bool = False) -> Tex:
    return Tex(r"\bfseries\boldmath " + text if bold else text, font_size=size, color=color)


def density_panel(density, peak: float, label: str, color: str):
    # A single square-root scale makes the three heatmaps directly comparable.
    level = np.sqrt(density / peak)[..., None]
    low = np.array([int(CRUST[i : i + 2], 16) for i in (1, 3, 5)])
    high = np.array([int(BLUE[i : i + 2], 16) for i in (1, 3, 5)])
    pixels = np.round(low + level * (high - low)).astype(np.uint8)

    image = ImageMobject(pixels).scale_to_fit_width(PANEL_WIDTH)
    frame = Square(side_length=PANEL_WIDTH, stroke_color=SURFACE, stroke_width=2).move_to(image)
    heading = text_label(label, 33, color).next_to(frame, UP, buff=0.17)
    return Group(image, frame, heading), image


def map_point(image: ImageMobject, point: Point, *, limit: float = LIMIT):
    x, y = point
    return image.get_center() + image.width / (2 * limit) * (x * RIGHT + y * UP)


def sample_dots(
    image: ImageMobject, points: list[Point], color: str, *, radius: float = 0.036,
    limit: float = LIMIT,
) -> VGroup:
    return VGroup(
        *[
            Dot(
                map_point(image, point, limit=limit), radius=radius, color=color, stroke_color=CRUST,
                stroke_width=1.5,
            )
            for point in points
        ]
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


def gaussian_blob(color: str) -> ImageMobject:
    grid = np.linspace(-4, 4, 129)
    squared_radius = grid[:, None] ** 2 + grid[None, :] ** 2
    density = np.exp(-0.5 * squared_radius)
    pixels = np.empty((*density.shape, 4), dtype=np.uint8)
    pixels[..., :3] = ManimColor(color).to_int_rgb()
    pixels[..., 3] = np.rint(0.78 * 255 * density).astype(np.uint8)
    return ImageMobject(pixels)


def projected_covariance(
    camera: ThreeDCamera, mean: np.ndarray, covariance: Covariance
) -> np.ndarray:
    # Linearize the perspective projection at the Gaussian's mean.
    rotation = camera.generate_rotation_matrix()
    view_mean = rotation @ (mean - camera.frame_center)
    focal_distance = camera.get_focal_distance()
    depth = focal_distance - view_mean[2]
    scale = camera.get_zoom() * focal_distance / depth
    projection = scale * (rotation[:2] + np.outer(view_mean[:2] / depth, rotation[2]))
    return projection @ covariance @ projection.T


class SparseGaussianMixture(ThreeDScene):
    """Move from the 2D heatmap to a 3D example with separated modes."""

    def construct(self):
        self.camera.background_color = BASE
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, focal_distance=23)

        grid = np.linspace(-LIMIT, LIMIT, 241)
        grid_x, grid_y = np.meshgrid(grid, grid[::-1])
        density = mixture_density(TRUE_DISTRIBUTION, grid_x, grid_y)
        flat_panel, flat_image = density_panel(
            density, float(density.max()), "2D Gaussian mixture", BLUE
        )
        flat_panel.scale(1.38, about_point=flat_image.get_center())
        flat_label = flat_panel[-1]
        flat_panel.remove(flat_label)
        flat_samples = draw_samples(TRUE_DISTRIBUTION, SAMPLE_COUNT, Random(19))
        flat_dots = sample_dots(flat_image, flat_samples, BLUE)
        flat_view = Group(flat_panel, flat_dots)
        flat_label.next_to(flat_view, UP, buff=0.23)
        self.add(flat_view, flat_label)
        self.wait(1.5)
        self.play(FadeOut(flat_label), run_time=0.25)

        axes = ThreeDAxes(
            x_range=(-4, 4, 1),
            y_range=(-3, 3, 1),
            z_range=(-2.6, 2.6, 1),
            x_length=8,
            y_length=6,
            z_length=5.2,
            axis_config={
                "color": SUBTEXT,
                "stroke_width": 1.7,
                "stroke_opacity": 0.62,
                "include_ticks": False,
                "tip_width": 0.16,
                "tip_height": 0.20,
            },
        )
        axis_labels = [
            MathTex(name, font_size=28, color=SUBTEXT).next_to(axis.get_end(), direction, buff=0.2)
            for name, axis, direction in zip("xyz", axes.get_axes(), (RIGHT, UP, OUT), strict=True)
        ]
        self.add_fixed_orientation_mobjects(*axis_labels)
        blobs = [
            gaussian_blob(color).move_to(mean)
            for (_, mean, _), color in zip(SPARSE_DISTRIBUTION, COMPONENT_COLORS, strict=True)
        ]
        self.add_fixed_orientation_mobjects(*blobs)
        # The texture spans four standard deviations from its center in each direction.
        corners = 4 * np.array([UP + LEFT, UP + RIGHT, DOWN + LEFT, DOWN + RIGHT])

        def update_blobs(_dt):
            for blob, (_, _, covariance) in zip(blobs, SPARSE_DISTRIBUTION, strict=True):
                mean = blob.get_center()
                factor = np.linalg.cholesky(projected_covariance(self.camera, mean, covariance))
                quad = corners.copy()
                quad[:, :2] = corners[:, :2] @ factor.T
                blob.set_points(quad + mean)

        self.add_updater(update_blobs)
        blob_reveals = [
            FadeIn(blob, shift=blob.get_center()[2] * OUT)
            for blob in blobs
        ]

        self.begin_ambient_camera_rotation(rate=18 * DEGREES)
        self.move_camera(
            phi=66 * DEGREES,
            zoom=0.78,
            frame_center=-0.28 * OUT,
            added_anims=[
                FadeOut(flat_view),
                LaggedStart(Create(axes), FadeIn(*axis_labels), lag_ratio=0.7),
                LaggedStart(*blob_reveals, lag_ratio=0.14),
            ],
            run_time=2.2,
        )
        self.wait(0.6)

        samples = [
            Dot(
                point=np.asarray(point),
                radius=0.044,
                color=TEXT,
                stroke_color=CRUST,
                stroke_width=1,
            )
            for point in draw_samples(SPARSE_DISTRIBUTION, 38, Random(41))
        ]
        self.add_fixed_orientation_mobjects(*samples)
        self.play(
            LaggedStart(*(FadeIn(dot, scale=0.7) for dot in samples), lag_ratio=0.035),
            run_time=1.5,
        )
        self.wait(8)
        self.stop_ambient_camera_rotation()


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

        # The same within-band quantile in three bands gives the exact same prediction.
        component = 2
        anchors = [
            (normal.inv_cdf((repeat * len(model) + component + 0.5) / (3 * len(model))), 0.25)
            for repeat in range(3)
        ]
        _, common_prediction = latent_prediction(anchors[0], model, normal)
        sources = sample_dots(prior_image, anchors, TEXT, radius=0.073)
        destination = sample_dots(
            prediction_image, [common_prediction], TEXT, radius=0.073, limit=output_limit
        )[0]
        destination_ring = Circle(radius=0.15, color=TEXT, stroke_width=2.1)
        destination_ring.move_to(destination)
        many_to_one_note = text_label("Different latent points can give the same prediction.", 31, TEXT)
        self.play(
            FadeIn(sources),
            Transform(footer, many_to_one_note.move_to(footer)),
            run_time=0.55,
        )
        copies = [destination.copy() for _ in sources]
        self.play(
            LaggedStart(
                *(TransformFromCopy(source, copy, path_arc=-15 * DEGREES)
                  for source, copy in zip(sources, copies, strict=True)),
                lag_ratio=0.28,
            ),
            run_time=1.65,
        )
        self.remove(*copies)
        self.add(destination)
        self.play(FadeIn(destination_ring), run_time=0.4)
        self.wait(1.0)

        sigma = 0.16
        kernel = gaussian_blob(GREEN).scale_to_fit_width(
            8 * sigma * prediction_image.width / (2 * output_limit)
        ).move_to(destination)
        conditional = MathTex(
            r"p_\theta(x_i\mid z_j)=\mathcal{N}(x_i;\,G_\theta(z_j),\,\sigma^2 I)",
            r"\qquad \sigma=0.16",
            font_size=33,
            color=TEXT,
        ).move_to(footer)
        self.play(
            FadeOut(sources), FadeOut(destination_ring), FadeOut(footer),
            FadeIn(kernel), FadeIn(conditional), run_time=0.7,
        )
        self.wait(1.5)
        integral = MathTex(
            r"p_\theta(x_i)", "=", r"\int", r"p_\theta(x_i\mid z)", r"p(z)\,\mathrm dz",
            font_size=39,
            color=TEXT,
        ).move_to(conditional)
        self.play(FadeOut(kernel), FadeOut(destination), FadeOut(conditional), FadeIn(integral), run_time=0.65)
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

        offsets = np.asarray(test_points)[:, None, :] - np.asarray(predictions)
        conditional_values = np.exp(-0.5 * np.sum((offsets / sigma) ** 2, axis=-1)) / (2 * pi * sigma**2)
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
            self.wait(1.1 if i == 1 else 0.65)
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


def matching_moment_distributions() -> tuple[Mixture, Mixture]:
    covariance: Covariance = ((1.0, 0.5), (0.5, 1.0))
    transform = np.linalg.cholesky(covariance)
    offset = 0.94
    # In whitened coordinates the spread of the means is offset by each component's variance.
    component_covariance = tuple(
        tuple(row) for row in (1 - offset**2) * np.asarray(covariance)
    )
    gaussian = ((1.0, (0.0, 0.0), covariance),)
    mixture = tuple(
        (0.25, tuple(transform @ mean), component_covariance)
        for mean in product((-offset, offset), repeat=2)
    )
    return gaussian, mixture


class MatchingGaussianMoments(Scene):
    """Compare two distinct densities with exactly the same Gaussian fit."""

    def construct(self):
        self.camera.background_color = BASE
        gaussian, mixture = matching_moment_distributions()
        covariance = gaussian[0][2]
        transform = np.linalg.cholesky(covariance)
        limit = 3.2
        axis = np.linspace(-limit, limit, 361)
        x, y = np.meshgrid(axis, axis[::-1])
        densities = [mixture_density(distribution, x, y) for distribution in (gaussian, mixture)]
        peak = max(float(density.max()) for density in densities)
        charts = Group()
        for density, label, color in zip(
            densities, ("Single Gaussian", "Mixture of four Gaussians"), (BLUE, GREEN), strict=True
        ):
            panel, image = density_panel(density, peak, label, color)
            axes = Axes(
                x_range=(-limit, limit, 1),
                y_range=(-limit, limit, 1),
                x_length=image.width,
                y_length=image.height,
                tips=False,
                axis_config={"color": SURFACE, "include_ticks": False, "stroke_width": 1.2},
            ).move_to(image)
            ellipse = Circle(radius=image.width / limit, color=YELLOW, stroke_width=1.8)
            ellipse.apply_matrix(transform).move_to(image)
            contour = DashedVMobject(ellipse, num_dashes=32, dashed_ratio=0.55, color=YELLOW)
            mean = Dot(image.get_center(), radius=0.04, color=YELLOW, stroke_color=CRUST, stroke_width=1)
            panel.add(axes, contour, mean)
            charts.add(panel.scale(1.42))
        charts.arrange(RIGHT, buff=0.70, aligned_edge=DOWN)

        matrix = r" \\ ".join(" & ".join(f"{value:g}" for value in row) for row in covariance)
        shared_moments = MathTex(
            r"\mu=(0,0),\qquad \Sigma=\begin{bmatrix}" + matrix + r"\end{bmatrix}",
            font_size=43,
            color=TEXT,
        )
        Group(charts, shared_moments).arrange(DOWN, buff=0.36)
        self.add(charts, shared_moments)


def convergence_chart(sample_count: int, y_values: range) -> tuple[Axes, VGroup]:
    graph = Axes(
        x_range=(0, np.log10(sample_count), 1), y_range=(y_values[0], y_values[-1], y_values.step),
        x_length=5.1, y_length=2.5, tips=False,
        axis_config={"color": SUBTEXT, "stroke_width": 1.55, "include_ticks": False},
    )
    x_ticks = VGroup(*[
        MathTex(f"{n:,}".replace(",", "{,}"), font_size=24, color=SUBTEXT)
        .next_to(graph.c2p(np.log10(n), 0), DOWN, buff=0.12)
        for n in (1, 10, 100, 1000, sample_count)
    ])
    y_ticks = VGroup(*[
        MathTex(str(value), font_size=24, color=SUBTEXT).next_to(graph.c2p(0, value), LEFT, buff=0.12)
        for value in y_values
    ])
    x_label = text_label("Pairs sampled (log scale)", 24, SUBTEXT).next_to(x_ticks, DOWN, buff=0.13)
    return graph, VGroup(graph, x_ticks, y_ticks, x_label)


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
            center = axes.c2p(*mean)
            unit = np.linalg.norm(axes.c2p(1, 0) - axes.c2p(0, 0))
            transform = rotation @ np.diag(std)
            rings = VGroup(*[
                Circle(
                    radius=unit * radius, color=color, stroke_width=1.5,
                    stroke_opacity=stroke, fill_opacity=fill,
                ).apply_matrix(transform).move_to(center)
                for radius, stroke, fill in ((2.3, 0.45, 0.06), (1.5, 0.65, 0.08), (0.75, 0.85, 0.13))
            ])
            return VGroup(rings, Dot(center, radius=0.041, color=color)).set_z_index(1)

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


def gaussian_axes(limit, width, height):
    return Axes(
        x_range=(-limit, limit, 1), y_range=(0, 0.45, 0.15),
        x_length=width, y_length=height, tips=False,
        axis_config={"color": SUBTEXT, "stroke_width": 1.45, "include_ticks": False},
        y_axis_config={"stroke_opacity": 0.25},
    )


def gaussian_density(offset, std):
    return np.exp(-0.5 * (offset / std)**2) / (std * np.sqrt(2 * pi))


def axis_numbers(axes, values):
    return VGroup(*[
        MathTex(str(value), font_size=23, color=SUBTEXT).next_to(axes.c2p(value, 0), DOWN, buff=0.12)
        for value in values
    ])


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


class IdenticalGaussianPairings(GaussianPairings):
    generated_std = 2
    convergence_ticks = range(0, 21, 4)


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

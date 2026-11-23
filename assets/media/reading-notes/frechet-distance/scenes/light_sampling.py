from itertools import product
from math import cos, inf, sin, sqrt, tau
from random import Random

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    ImageMobject,
    LaggedStart,
    Line,
    ManimColor,
    MathTex,
    Square,
    ThreeDCamera,
    ThreeDScene,
    VGroup,
    VMobject,
)

from scene_helpers import BASE, BLUE, CRUST, GREEN, MAUVE, RED, SUBTEXT, TEXT, YELLOW, text_label


SCENE_OFFSET = np.array([0.0, 0.0, -0.98])
LIGHT = SCENE_OFFSET + np.array([-2.20, -0.25, 3.15])
SHADING_POINT = SCENE_OFFSET + np.array([-0.82, -1.68, 0.0])
CAMERA_CENTER = SCENE_OFFSET + np.array([-3.72, -3.05, 1.42])
OBJECTS = (
    (SCENE_OFFSET + np.array([-1.08, 0.44, 0.96]), 0.96, "#c89170"),
    (SCENE_OFFSET + np.array([0.82, -0.54, 0.73]), 0.73, "#6babb0"),
)
ROOM_MIN = SCENE_OFFSET + np.array([-3.40, -2.70, 0.0])
ROOM_MAX = SCENE_OFFSET + np.array([2.64, 2.00, 3.67])
LIGHT_RADIUS = 0.28
SAMPLE_COUNT = 24


def sphere_distance(origin, direction, center, radius):
    """Return the first positive hit distance along a unit direction."""
    offset = origin - center
    along = float(direction @ offset)
    discriminant = along**2 - float(offset @ offset) + radius**2
    if discriminant < 0:
        return inf
    near, far = -along - sqrt(discriminant), -along + sqrt(discriminant)
    return near if near > 0 else far if far > 0 else inf


def obstruction_distance(origin, direction):
    return min(sphere_distance(origin, direction, center, radius) for center, radius, _ in OBJECTS)


def smooth_sphere(camera: ThreeDCamera, center, radius, color, *, emissive=False):
    """Shade the sphere's analytic surface from the same perspective as the Manim scene."""
    rotation = camera.generate_rotation_matrix()
    view_center = rotation @ center
    focal_distance = camera.get_focal_distance()
    zoom = camera.get_zoom()
    corners = view_center + radius * np.array(list(product((-1, 1), repeat=3)))
    projected = zoom * focal_distance * corners[:, :2] / (focal_distance - corners[:, 2, None])
    left, bottom = projected.min(axis=0)
    right, top = projected.max(axis=0)

    # Each texel traces a ray through the image plane, so lighting varies smoothly across the sphere.
    pixels = (np.arange(512) + 0.5) / 512
    x, y = np.meshgrid(left + (right - left) * pixels, top - (top - bottom) * pixels)
    rays = np.stack((x / zoom, y / zoom, np.full_like(x, -focal_distance)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    offset = np.array([0, 0, focal_distance]) - view_center
    along = rays @ offset
    discriminant = along**2 - offset @ offset + radius**2
    distance = -along - np.sqrt(np.maximum(discriminant, 0))
    normals = (offset + distance[..., None] * rays) / radius

    if emissive:
        brightness = 0.78 + 0.22 * np.clip(normals[..., 2], 0, 1)
    else:
        to_light = rotation @ LIGHT - view_center - radius * normals
        to_light /= np.linalg.norm(to_light, axis=-1, keepdims=True)
        brightness = 0.28 + 0.72 * np.clip(np.sum(normals * to_light, axis=-1), 0, 1)

    rgba = np.empty((*x.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(255 * brightness[..., None] * ManimColor(color).to_rgb()).astype(np.uint8)
    rgba[..., 3] = 255 * (discriminant >= 0)
    return (
        ImageMobject(rgba)
        .stretch_to_fit_width(right - left)
        .stretch_to_fit_height(top - bottom)
        .move_to(np.array([(left + right) / 2, (bottom + top) / 2, 0]))
    )


class LightSampling(ThreeDScene):
    """Compare random bounces and explicit light sampling in the same room."""

    def construct(self):
        self.camera.background_color = BASE
        self.camera.light_source.move_to(LIGHT)
        self.set_camera_orientation(
            phi=61 * DEGREES, theta=-102 * DEGREES, zoom=1.10, focal_distance=24,
        )

        tile_size = 0.67
        floor = VGroup(*(
            Square(
                side_length=tile_size, stroke_width=0,
                fill_color=("#525665", "#3c4050")[(column + row) % 2], fill_opacity=1,
            ).move_to(ROOM_MIN + tile_size * np.array([column + 0.5, row + 0.5, 0]))
            for column in range(9)
            for row in range(7)
        )).set_shade_in_3d(True)
        self.add(floor)
        spheres = [smooth_sphere(self.camera, *sphere) for sphere in OBJECTS]
        self.add_fixed_in_frame_mobjects(*spheres)
        previous_angle = self.camera.get_theta()

        def update_spheres(_dt):
            nonlocal previous_angle
            angle = self.camera.get_theta()
            if angle != previous_angle:
                for image, sphere in zip(spheres, OBJECTS, strict=True):
                    image.become(smooth_sphere(self.camera, *sphere))
                previous_angle = angle

        self.add_updater(update_spheres)
        intro = self.overlay("Sampling light on a diffuse surface", 42, UP, bold=True)

        point_light = VGroup(
            Circle(radius=0.14, color=YELLOW, stroke_width=1.4, stroke_opacity=0.57),
            Dot(radius=0.066, color=YELLOW),
        ).move_to(LIGHT).set_z_index(4)
        light_label = text_label("Light", 26, YELLOW).move_to(LIGHT + np.array([-0.64, -0.26, 0.1]))
        shading_dot = Dot(
            SHADING_POINT, radius=0.068, color=BLUE, stroke_color=CRUST, stroke_width=1.5,
        ).set_z_index(4)
        shading_label = MathTex("x", font_size=34, color=BLUE).move_to(
            SHADING_POINT + np.array([-0.26, -0.12, 0.1])
        ).set_stroke(CRUST, width=1.2, background=True)
        light_label.set_z_index(4)
        shading_label.set_z_index(4)
        outline = VMobject(stroke_color=TEXT, stroke_width=2, fill_color=CRUST, fill_opacity=1)
        outline.start_new_path(0.48 * LEFT)
        outline.add_cubic_bezier_curve_to(0.23 * LEFT + 0.35 * UP, 0.23 * RIGHT + 0.35 * UP, 0.48 * RIGHT)
        outline.add_cubic_bezier_curve_to(0.23 * RIGHT + 0.35 * DOWN, 0.23 * LEFT + 0.35 * DOWN, 0.48 * LEFT)
        outline.close_path()
        eye = VGroup(
            outline,
            Circle(radius=0.15, stroke_color=BLUE, stroke_width=1.8, fill_color=BLUE, fill_opacity=0.6),
            Dot(radius=0.073, color=CRUST),
            Dot(0.045 * LEFT + 0.055 * UP, radius=0.026, color=TEXT),
        ).move_to(CAMERA_CENTER).set_z_index(4)
        camera_label = text_label("Camera", 29, SUBTEXT).move_to(CAMERA_CENTER + 0.64 * OUT)
        camera_label.set_stroke(CRUST, width=2, background=True)
        camera_ray = Line(CAMERA_CENTER, SHADING_POINT, color=MAUVE, stroke_width=3.6).set_z_index(2)
        ray_label = text_label("Camera ray", 26, MAUVE).move_to(
            0.65 * CAMERA_CENTER + 0.35 * SHADING_POINT + 0.46 * OUT,
        ).set_stroke(CRUST, width=2, background=True)
        self.add_fixed_orientation_mobjects(point_light, light_label, eye, camera_label, ray_label)
        self.play(
            FadeIn(intro), FadeIn(eye), FadeIn(camera_label), FadeIn(point_light),
            FadeIn(light_label), run_time=0.65,
        )
        self.play(Create(camera_ray), FadeIn(ray_label), run_time=1.25)
        self.add_fixed_orientation_mobjects(shading_dot, shading_label)
        camera_result = self.overlay("The camera ray hits the floor at $x$.", 30, DOWN, MAUVE)
        self.play(FadeIn(shading_dot), FadeIn(shading_label), FadeIn(camera_result), run_time=0.45)
        self.move_camera(theta=-69 * DEGREES, run_time=2.4)
        self.wait(0.75)

        # A fixed seed lets both random methods use exactly the same cosine-weighted draws.
        rng = Random(48)
        directions = []
        for _ in range(SAMPLE_COUNT):
            u, azimuth = rng.random(), tau * rng.random()
            directions.append(np.array([sqrt(u) * cos(azimuth), sqrt(u) * sin(azimuth), sqrt(1 - u)]))

        point_rays, point_hits = self.random_bounces(directions, 0)
        point_heading = self.overlay("Random bounces toward a point light", 40, UP, bold=True)
        self.play(
            FadeOut(intro), FadeOut(camera_result), FadeOut(ray_label), FadeIn(point_heading),
            camera_ray.animate.set_stroke(opacity=0.32), run_time=0.6,
        )
        self.play(LaggedStart(*(Create(ray) for ray in point_rays), lag_ratio=0.12), run_time=3.0)
        point_result = self.overlay(
            f"{point_hits} of {SAMPLE_COUNT} bounces reach the light.", 31, DOWN, YELLOW,
        )
        self.play(FadeIn(point_result), run_time=0.45)
        self.wait(2.4)

        sphere_light = smooth_sphere(self.camera, LIGHT, LIGHT_RADIUS, YELLOW, emissive=True)
        self.add_fixed_in_frame_mobjects(sphere_light)
        sphere_heading = self.overlay("Random bounces toward a small sphere light", 40, UP, bold=True)
        self.play(
            FadeOut(point_heading), FadeOut(point_result), FadeOut(point_rays),
            FadeOut(point_light), FadeIn(sphere_light), FadeIn(sphere_heading), run_time=0.7,
        )
        self.wait(0.65)
        sphere_rays, sphere_hits = self.random_bounces(directions, LIGHT_RADIUS)
        self.play(LaggedStart(*(Create(ray) for ray in sphere_rays), lag_ratio=0.12), run_time=3.0)
        sphere_result = self.overlay(
            f"{sphere_hits} of the same {SAMPLE_COUNT} bounces reaches the light.", 31, DOWN, GREEN,
        )
        self.play(FadeIn(sphere_result), run_time=0.5)
        self.wait(2.45)

        direct_heading = self.overlay("Direct light sampling", 40, UP, bold=True)
        self.play(
            FadeOut(sphere_heading), FadeOut(sphere_result), FadeOut(sphere_rays),
            FadeOut(sphere_light), FadeIn(point_light), FadeIn(direct_heading), run_time=0.7,
        )

        visible_ray = Line(SHADING_POINT, LIGHT, color=GREEN, stroke_width=4).set_z_index(2)
        visible_result = self.overlay("Aim a shadow ray from $x$ at the light.", 31, DOWN, GREEN)
        self.play(Create(visible_ray), run_time=1.2)
        self.play(FadeIn(visible_result), run_time=0.45)
        self.wait(1.45)

        # Project the front sphere's center through the light onto the floor.
        occluder = OBJECTS[1][0]
        shadow_point = LIGHT + (occluder - LIGHT) * ((LIGHT[2] - SHADING_POINT[2]) / (LIGHT[2] - occluder[2]))
        toward_light = LIGHT - shadow_point
        direction = toward_light / np.linalg.norm(toward_light)
        blocked_at = shadow_point + direction * obstruction_distance(shadow_point, direction)
        shadow_dot = Dot(
            shadow_point, radius=0.068, color=BLUE, stroke_color=CRUST, stroke_width=1.5,
        ).set_z_index(4)
        shadow_label = MathTex("x'", font_size=34, color=BLUE).move_to(
            shadow_point + np.array([0.27, -0.12, 0.05])
        ).set_stroke(CRUST, width=1.2, background=True).set_z_index(4)
        self.add_fixed_orientation_mobjects(shadow_dot, shadow_label)
        blocked_ray = Line(shadow_point, blocked_at, color=RED, stroke_width=4).set_z_index(2)
        self.play(
            FadeIn(shadow_dot), FadeIn(shadow_label), Create(blocked_ray), FadeOut(visible_result),
            run_time=1.05,
        )
        blocked_result = self.overlay("The teal sphere blocks the shadow ray from $x'$.", 31, DOWN, RED)
        self.play(FadeIn(blocked_result), run_time=0.45)
        self.move_camera(theta=-48 * DEGREES, run_time=2.15)
        self.wait(1.75)

    def overlay(self, text, size, edge, color=TEXT, *, bold=False):
        label = text_label(text, size, color, bold=bold).to_edge(edge, buff=0.32).set_z_index(7)
        self.add_fixed_in_frame_mobjects(label)
        return label

    def random_bounces(self, directions, light_radius):
        rays = VGroup()
        hits = 0
        for direction in directions:
            edges = np.where(direction > 0, ROOM_MAX, ROOM_MIN)
            exit_distance = min((edges - SHADING_POINT) / direction)
            stop = min(4.0, exit_distance, obstruction_distance(SHADING_POINT, direction))
            light_distance = sphere_distance(SHADING_POINT, direction, LIGHT, light_radius)
            hit = light_distance < stop
            hits += hit
            rays.add(Line(
                SHADING_POINT, SHADING_POINT + direction * min(stop, light_distance),
                stroke_color=GREEN if hit else BLUE, stroke_width=4 if hit else 1.7,
                stroke_opacity=1 if hit else 0.41,
            ).set_z_index(3 if hit else 1))
        return rays, hits

from __future__ import annotations

import numpy as np

from vision_ranging import (
    CameraParams,
    ExtrinsicsEstimator,
    ExtrinsicsEstimatorConfig,
    MarkerObservation,
    Plane,
    Pose,
    measure_distance_between_pixels,
    pixel_to_camera_ray,
    pixel_to_world_on_plane,
    ray_plane_intersection,
    rotation_angle_deg,
    undistort_points,
)


def _camera() -> CameraParams:
    return CameraParams(
        name="test_cam",
        model="pinhole",
        width=1280,
        height=720,
        K=np.array([[980.0, 0.0, 640.0], [0.0, 980.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        D=np.zeros(5, dtype=np.float64),
    )


def _synthetic_scene(seed: int = 0, noise_px: float = 0.2):
    rng = np.random.default_rng(seed)
    camera = _camera()
    plane = Plane.from_height(0.0)

    rvec = np.array(
        [
            np.deg2rad(rng.uniform(-8.0, 8.0)),
            np.deg2rad(rng.uniform(-15.0, 15.0)),
            np.deg2rad(rng.uniform(-8.0, 8.0)),
        ],
        dtype=np.float64,
    )
    import cv2

    R, _ = cv2.Rodrigues(rvec)
    t = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(2.0, 3.5)], dtype=np.float64)
    pose = Pose(R=R, t=t)

    rows, cols, spacing = 3, 4, 0.25
    points_w = []
    for r in range(rows):
        for c in range(cols):
            points_w.append([c * spacing, r * spacing, 0.0])
    points_w = np.asarray(points_w, dtype=np.float64)

    proj, _ = cv2.projectPoints(points_w, rvec, t.reshape(3, 1), camera.K, camera.D)
    points_uv = proj.reshape(-1, 2)
    if noise_px > 0:
        points_uv = points_uv + rng.normal(0.0, noise_px, size=points_uv.shape)

    return camera, plane, pose, points_w, points_uv


def test_undistort_and_center_ray() -> None:
    cam = _camera()
    pts = np.array([[320.0, 200.0], [640.0, 360.0], [900.0, 620.0]], dtype=np.float64)
    out = undistort_points(pts, cam)
    assert np.allclose(out, pts, atol=1e-7)

    ray = pixel_to_camera_ray((640.0, 360.0), cam)
    assert np.allclose(ray, np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_ray_plane_intersection_basic_and_parallel() -> None:
    plane = Plane.from_height(0.0)

    ok, p, status = ray_plane_intersection(
        ray_origin_w=np.array([0.0, 0.0, 1.0]),
        ray_dir_w=np.array([0.0, 0.0, -1.0]),
        plane=plane,
    )
    assert ok and status == "ok"
    assert p is not None and np.allclose(p, np.array([0.0, 0.0, 0.0]))

    ok2, p2, status2 = ray_plane_intersection(
        ray_origin_w=np.array([0.0, 0.0, 1.0]),
        ray_dir_w=np.array([1.0, 0.0, 0.0]),
        plane=plane,
    )
    assert not ok2 and p2 is None and status2 == "ray_parallel_to_plane"


def test_distance_simple_case() -> None:
    cam = CameraParams(
        name="simple",
        model="pinhole",
        width=640,
        height=480,
        K=np.eye(3, dtype=np.float64),
        D=np.zeros(5, dtype=np.float64),
    )
    pose = Pose.identity()
    plane = Plane.from_height(1.0)

    m = measure_distance_between_pixels(
        pixel_a_uv=(0.0, 0.0),
        pixel_b_uv=(1.0, 0.0),
        camera=cam,
        pose_w2c=pose,
        plane=plane,
    )
    assert m.success
    assert m.distance_m is not None
    assert abs(m.distance_m - 1.0) < 1e-9



def test_pixel_to_world_expected_point() -> None:
    cam = CameraParams(
        name="simple",
        model="pinhole",
        width=640,
        height=480,
        K=np.eye(3, dtype=np.float64),
        D=np.zeros(5, dtype=np.float64),
    )
    pose = Pose.identity()
    plane = Plane.from_height(2.0)

    ok, point_w, status, _ = pixel_to_world_on_plane((0.0, 0.0), cam, pose, plane)
    assert ok and status == "ok"
    assert point_w is not None and np.allclose(point_w, np.array([0.0, 0.0, 2.0]))


def test_pnp_update_and_measurement_regression() -> None:
    cam, plane, pose_gt, points_w, points_uv = _synthetic_scene(seed=7, noise_px=0.3)
    obs = MarkerObservation(
        object_points=points_w,
        image_points=points_uv,
        ids=np.arange(points_w.shape[0]),
        marker_type="synthetic",
    )

    estimator = ExtrinsicsEstimator(
        ExtrinsicsEstimatorConfig(
            min_inliers=6,
            min_inlier_ratio=0.5,
            max_reprojection_error_px=6.0,
            smoothing_alpha=1.0,
        )
    )

    update = estimator.estimate(obs, cam, previous_pose=None)
    assert update.pose is not None
    assert update.quality.success
    assert update.quality.reprojection_error_px <= 6.0

    rot_err = rotation_angle_deg(update.pose.R, pose_gt.R)
    trans_err = float(np.linalg.norm(update.pose.t - pose_gt.t))
    assert rot_err < 2.0
    assert trans_err < 0.2

    idx_a, idx_b = 1, points_w.shape[0] - 2
    true_dist = float(np.linalg.norm(points_w[idx_a] - points_w[idx_b]))
    m = measure_distance_between_pixels(
        points_uv[idx_a],
        points_uv[idx_b],
        cam,
        update.pose,
        plane,
        extrinsics_score=update.quality.score,
    )

    assert m.success and m.distance_m is not None
    relative_error = abs(m.distance_m - true_dist) / true_dist
    assert relative_error <= 0.05

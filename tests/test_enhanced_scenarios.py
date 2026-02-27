"""Enhanced synthetic scenario tests for contract validation.

Covers: pixel jitter, water level dynamics, extended Monte Carlo,
marker fallback, degenerate geometry, distortion, and recovery metric.
"""
from __future__ import annotations

import numpy as np
import cv2
from datetime import datetime, timezone, timedelta

from vision_ranging import (
    CameraParams,
    Plane,
    Pose,
    measure_distance_between_pixels,
    ExtrinsicsEstimator,
    ExtrinsicsEstimatorConfig,
    auto_update_pose,
    build_marker_detectors,
    pixel_to_world_on_plane,
    ray_plane_intersection,
    undistort_points,
    rotation_angle_deg,
)
from water_level import (
    WaterLevelFusion,
    InMemoryWaterLevelSource,
    WaterLevelReading,
    StaticWaterLevelSource,
)
from test_contract_metrics import (
    look_at_pose,
    project_world_points,
    euler_deg_to_matrix,
    disturbed_pose,
    make_camera_models,
    make_marker_configuration,
    render_mock_image,
    setup_pose_estimator,
    MAX_REL_ERROR,
    POSE_SUCCESS_THRESHOLD,
    RECOVERY_THRESHOLD,
    EPS,
)

SEED = 20240601


def _default_camera() -> CameraParams:
    return make_camera_models()["Hikvision_Bullet_4mm"]


def _default_pose() -> Pose:
    return look_at_pose(
        np.array([0.0, -8.0, 8.0], dtype=np.float64),
        np.array([0.0, 28.0, 0.0], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# 1. Distance coverage with pixel jitter
# ---------------------------------------------------------------------------
def test_distance_with_pixel_jitter() -> None:
    """Measure distance at 2–50 m with ±2 px Gaussian jitter; error ≤ 5%."""
    rng = np.random.default_rng(SEED)
    camera = _default_camera()
    plane = Plane.from_height(0.0)

    distances = [2.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    n_jitter_trials = 10

    for d_true in distances:
        # Scale geometry so pixel resolution handles jitter at all distances
        cam_range = max(8.0, d_true * 1.5)
        cam_height = cam_range * 0.3
        y_mid = cam_range * 0.8
        center = np.array([0.0, y_mid - cam_range * 0.9, cam_height], dtype=np.float64)
        pose = look_at_pose(center, np.array([0.0, y_mid, 0.0], dtype=np.float64))

        p1_w = np.array([-0.5 * d_true, y_mid, 0.0], dtype=np.float64)
        p2_w = np.array([0.5 * d_true, y_mid, 0.0], dtype=np.float64)
        uv_clean = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

        for trial in range(n_jitter_trials):
            noise = rng.normal(0.0, 2.0, size=(2, 2))
            uv_noisy = uv_clean + noise
            uv_noisy[:, 0] = np.clip(uv_noisy[:, 0], 0, camera.width - 1)
            uv_noisy[:, 1] = np.clip(uv_noisy[:, 1], 0, camera.height - 1)

            result = measure_distance_between_pixels(
                uv_noisy[0], uv_noisy[1], camera, pose, plane
            )
            assert result.success, (
                f"Jitter test failed: d={d_true}, trial={trial}, msg={result.message}"
            )
            assert result.distance_m is not None
            rel_error = abs(result.distance_m - d_true) / d_true
            assert rel_error <= MAX_REL_ERROR, (
                f"Jitter error too large: d_true={d_true}, measured={result.distance_m:.4f}, "
                f"rel_error={rel_error:.4f}, trial={trial}"
            )


# ---------------------------------------------------------------------------
# 2. Water level dynamic changes
# ---------------------------------------------------------------------------
def test_water_level_dynamic_compensation() -> None:
    """Water level jumps and drifts; measurement error ≤ 5% at each step."""
    camera = _default_camera()

    # Water level schedule: sudden jump then gradual drift
    base_time = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    water_levels = [0.0, 0.0, 0.5, 0.5, 0.6, 0.7, 0.8]
    readings = [
        WaterLevelReading(
            height_m=h,
            timestamp=base_time + timedelta(minutes=i * 10),
            source="test",
        )
        for i, h in enumerate(water_levels)
    ]

    source = InMemoryWaterLevelSource(readings)
    fusion = WaterLevelFusion(source=source, base_plane=Plane.from_height(0.0))

    d_true = 10.0
    y_mid = 30.0
    center = np.array([0.0, -8.0, 10.0], dtype=np.float64)
    pose = look_at_pose(center, np.array([0.0, y_mid, 0.0], dtype=np.float64))

    for i, expected_h in enumerate(water_levels):
        ref_time = base_time + timedelta(minutes=i * 10)
        plane, reading = fusion.current_plane(reference_time=ref_time)
        assert abs(reading.height_m - expected_h) < 1e-6, (
            f"Step {i}: water level mismatch: got {reading.height_m}, expected {expected_h}"
        )

        # Place world points on the actual water plane
        p1_w = np.array([-0.5 * d_true, y_mid, expected_h], dtype=np.float64)
        p2_w = np.array([0.5 * d_true, y_mid, expected_h], dtype=np.float64)
        uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

        result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane)
        assert result.success, (
            f"Step {i}: measurement failed at h={expected_h}: {result.message}"
        )
        assert result.distance_m is not None
        rel_error = abs(result.distance_m - d_true) / d_true
        assert rel_error <= MAX_REL_ERROR, (
            f"Step {i}: h={expected_h}, measured={result.distance_m:.4f}, "
            f"rel_error={rel_error:.4f} > 5%"
        )


# ---------------------------------------------------------------------------
# 3. Pose drift Monte Carlo extended (N=200)
# ---------------------------------------------------------------------------
def test_pose_drift_monte_carlo_extended() -> None:
    """200 random perturbations; ≥ 95% success (pose + distance error ≤ 5%)."""
    rng = np.random.default_rng(SEED + 1)

    camera = _default_camera()
    plane = Plane.from_height(0.0)
    marker_cfg, aruco_world_map, circle_world = make_marker_configuration()
    detectors = build_marker_detectors(marker_cfg)
    estimator = setup_pose_estimator()
    base_pose = _default_pose()

    # Known measurement target
    d_true = 20.0
    p1_w = np.array([-10.0, 30.0, 0.0], dtype=np.float64)
    p2_w = np.array([10.0, 30.0, 0.0], dtype=np.float64)

    n_trials = 200
    success_count = 0

    for _ in range(n_trials):
        dist_pose, shift_m, tilt_deg = disturbed_pose(base_pose, rng)
        assert shift_m > 0.05 or tilt_deg > 2.0

        frame = render_mock_image(
            camera,
            dist_pose,
            aruco_world_map,
            circle_world,
            use_aruco=True,
            use_circle=True,
        )

        log: dict = {}
        success, pose_est, _ = auto_update_pose(
            frame=frame,
            camera=camera,
            previous_pose=base_pose,
            marker_cfg=marker_cfg,
            estimator=estimator,
            log=log,
            detectors=detectors,
        )

        if not success or pose_est is None:
            continue

        # Measure distance with corrected pose
        uv = project_world_points(np.vstack([p1_w, p2_w]), camera, dist_pose)
        result = measure_distance_between_pixels(uv[0], uv[1], camera, pose_est, plane)
        if result.success and result.distance_m is not None:
            rel_err = abs(result.distance_m - d_true) / d_true
            if rel_err <= MAX_REL_ERROR:
                success_count += 1

    rate = success_count / n_trials
    assert rate >= POSE_SUCCESS_THRESHOLD, (
        f"Monte Carlo extended: success rate={rate:.3f} < 0.95 "
        f"({success_count}/{n_trials})"
    )


# ---------------------------------------------------------------------------
# 4. Marker fallback under ArUco occlusion
# ---------------------------------------------------------------------------
def test_aruco_occlusion_circle_fallback() -> None:
    """ArUco occluded → circle fallback succeeds; distance error ≤ 5%."""
    camera = _default_camera()
    plane = Plane.from_height(0.0)
    marker_cfg, aruco_world_map, circle_world = make_marker_configuration()
    detectors = build_marker_detectors(marker_cfg)
    estimator = setup_pose_estimator()
    base_pose = _default_pose()

    rng = np.random.default_rng(SEED + 2)
    dist_pose, _, _ = disturbed_pose(base_pose, rng)

    # ArUco occluded, circle available
    frame = render_mock_image(
        camera,
        dist_pose,
        aruco_world_map,
        circle_world,
        use_aruco=True,
        use_circle=True,
        aruco_occluded=True,
    )

    log: dict = {}
    success, pose_est, _ = auto_update_pose(
        frame=frame,
        camera=camera,
        previous_pose=base_pose,
        marker_cfg=marker_cfg,
        estimator=estimator,
        log=log,
        detectors=detectors,
    )

    assert success and pose_est is not None, (
        f"Circle fallback failed when ArUco occluded: log={log}"
    )
    assert log.get("marker_type") == "circle", (
        f"Expected marker_type='circle', got '{log.get('marker_type')}'"
    )

    # Verify distance measurement with corrected pose
    d_true = 16.0
    p1_w = np.array([-8.0, 30.0, 0.0], dtype=np.float64)
    p2_w = np.array([8.0, 30.0, 0.0], dtype=np.float64)
    uv = project_world_points(np.vstack([p1_w, p2_w]), camera, dist_pose)

    result = measure_distance_between_pixels(uv[0], uv[1], camera, pose_est, plane)
    assert result.success and result.distance_m is not None, (
        f"Distance measurement failed after circle fallback: {result.message}"
    )
    rel_error = abs(result.distance_m - d_true) / d_true
    assert rel_error <= MAX_REL_ERROR, (
        f"Fallback distance error too large: d_true={d_true}, "
        f"measured={result.distance_m:.4f}, rel_error={rel_error:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Degenerate geometry – grazing rays
# ---------------------------------------------------------------------------
def test_degenerate_grazing_rays() -> None:
    """Camera nearly parallel to water plane; must not crash or return NaN/Inf."""
    camera = _default_camera()
    plane = Plane.from_height(0.0)

    # Camera at low height, looking nearly horizontally (grazing angle ~2°)
    cam_center = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    target = np.array([0.0, 200.0, 0.3], dtype=np.float64)
    pose = look_at_pose(cam_center, target)

    # Pick pixels near the top of the image (close to horizon)
    horizon_pixels = [
        (camera.width // 2 - 100, 10),
        (camera.width // 2 + 100, 10),
    ]

    result = measure_distance_between_pixels(
        horizon_pixels[0], horizon_pixels[1], camera, pose, plane
    )

    if result.success:
        # If the system returns a result it must be finite
        assert result.distance_m is not None
        assert np.isfinite(result.distance_m), (
            f"Grazing ray returned non-finite distance: {result.distance_m}"
        )
        assert result.distance_m > 0, (
            f"Grazing ray returned non-positive distance: {result.distance_m}"
        )
    else:
        # Graceful failure is acceptable – verify no crash and has a message
        assert isinstance(result.message, str) and len(result.message) > 0, (
            "Grazing ray failure should include a descriptive message"
        )
        assert result.distance_m is None or np.isfinite(result.distance_m), (
            f"Failed result must not contain NaN/Inf: {result.distance_m}"
        )

    # Also verify individual pixel-to-world calls don't crash
    for px in horizon_pixels:
        ok, pt_w, status, grazing = pixel_to_world_on_plane(px, camera, pose, plane)
        if ok:
            assert pt_w is not None
            assert np.all(np.isfinite(pt_w)), (
                f"World point has non-finite values: {pt_w}"
            )
        else:
            assert isinstance(status, str) and len(status) > 0


# ---------------------------------------------------------------------------
# 6. Noise and distortion robustness
# ---------------------------------------------------------------------------
def test_distortion_robustness() -> None:
    """Non-zero distortion coefficients; undistortion pipeline keeps error ≤ 5%."""
    camera = CameraParams.from_dict(
        {
            "name": "Distorted_Test",
            "model": "pinhole",
            "width": 1920,
            "height": 1080,
            "K": [[1380.0, 0.0, 960.0], [0.0, 1385.0, 540.0], [0.0, 0.0, 1.0]],
            "D": [0.1, -0.25, 0.001, 0.001, 0.05],
        }
    )
    plane = Plane.from_height(0.0)

    center = np.array([0.0, -8.0, 10.0], dtype=np.float64)
    pose = look_at_pose(center, np.array([0.0, 28.0, 0.0], dtype=np.float64))

    test_cases = [
        (5.0, 28.0),
        (10.0, 28.0),
        (20.0, 35.0),
    ]

    for d_true, y_mid in test_cases:
        p1_w = np.array([-0.5 * d_true, y_mid, 0.0], dtype=np.float64)
        p2_w = np.array([0.5 * d_true, y_mid, 0.0], dtype=np.float64)

        # Project using distorted camera model
        uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

        # Ensure projected points are within frame
        if not (
            np.all(uv[:, 0] >= 0)
            and np.all(uv[:, 0] < camera.width)
            and np.all(uv[:, 1] >= 0)
            and np.all(uv[:, 1] < camera.height)
        ):
            continue  # skip if distortion pushes points out of frame

        result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane)
        assert result.success, (
            f"Distortion test failed: d_true={d_true}, msg={result.message}"
        )
        assert result.distance_m is not None
        rel_error = abs(result.distance_m - d_true) / d_true
        assert rel_error <= MAX_REL_ERROR, (
            f"Distortion error too large: d_true={d_true}, "
            f"measured={result.distance_m:.4f}, rel_error={rel_error:.4f}"
        )


# ---------------------------------------------------------------------------
# 7. Recovery metric verification
# ---------------------------------------------------------------------------
def test_recovery_metric_multiple_scenarios() -> None:
    """3+ disturbance magnitudes; recovery ≥ 0.9 and post_rel_error ≤ 5%."""
    camera = _default_camera()
    plane = Plane.from_height(0.0)
    marker_cfg, aruco_world_map, circle_world = make_marker_configuration()
    detectors = build_marker_detectors(marker_cfg)
    estimator = setup_pose_estimator()
    base_pose = _default_pose()

    d_true = 24.0
    p1_w = np.array([-12.0, 32.0, 0.0], dtype=np.float64)
    p2_w = np.array([12.0, 32.0, 0.0], dtype=np.float64)

    # Different disturbance magnitudes
    disturbances = [
        # (rx, ry, rz, translation_vector)
        (4.0, -3.0, 1.0, np.array([0.3, -0.2, 0.1])),
        (7.0, -6.0, 1.5, np.array([0.7, -0.45, 0.25])),
        (5.0, 4.0, -1.0, np.array([-0.3, 0.35, -0.15])),
    ]

    for idx, (rx, ry, rz, t_shift) in enumerate(disturbances):
        R_delta = euler_deg_to_matrix(rx, ry, rz)
        disturbed_R = R_delta @ base_pose.R
        disturbed_center = base_pose.camera_center_world() + t_shift
        dist_pose = Pose(R=disturbed_R, t=-disturbed_R @ disturbed_center)

        # Project points using the disturbed pose
        uv_pair = project_world_points(np.vstack([p1_w, p2_w]), camera, dist_pose)

        # Pre-correction: measure with the wrong (base) pose
        pre = measure_distance_between_pixels(
            uv_pair[0], uv_pair[1], camera, base_pose, plane
        )
        assert pre.success and pre.distance_m is not None, (
            f"Scenario {idx}: pre-measurement failed: {pre.message}"
        )

        # Correct using markers
        frame = render_mock_image(
            camera,
            dist_pose,
            aruco_world_map,
            circle_world,
            use_aruco=True,
            use_circle=True,
        )

        log: dict = {}
        success, corrected_pose, _ = auto_update_pose(
            frame=frame,
            camera=camera,
            previous_pose=base_pose,
            marker_cfg=marker_cfg,
            estimator=estimator,
            log=log,
            detectors=detectors,
        )
        assert success and corrected_pose is not None, (
            f"Scenario {idx}: pose correction failed, log={log}"
        )

        # Post-correction measurement
        post = measure_distance_between_pixels(
            uv_pair[0], uv_pair[1], camera, corrected_pose, plane
        )
        assert post.success and post.distance_m is not None, (
            f"Scenario {idx}: post-measurement failed: {post.message}"
        )

        pre_err = abs(pre.distance_m - d_true)
        post_err = abs(post.distance_m - d_true)
        recovery = 1.0 - post_err / (pre_err + EPS)
        post_rel = post_err / d_true

        assert recovery >= RECOVERY_THRESHOLD, (
            f"Scenario {idx}: recovery={recovery:.4f} < 0.9, "
            f"pre_err={pre_err:.4f}, post_err={post_err:.4f}"
        )
        assert post_rel <= MAX_REL_ERROR, (
            f"Scenario {idx}: post_rel_error={post_rel:.4f} > 5%"
        )

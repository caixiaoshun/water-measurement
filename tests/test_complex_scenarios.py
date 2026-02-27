"""Enhanced mock / synthetic tests covering complex scenarios.

This module extends the contract-metrics tests with the following factors
required by the acceptance specification:

* Multi-distance coverage (2–50 m) with intermediate and extreme values.
* Random pixel-click jitter (simulating human click imprecision).
* Dynamic water-level changes (sudden jumps and slow drift).
* Attitude drift (translation > 5 cm, tilt > 2°) with Monte Carlo sampling.
* Marker degradation / fallback (ArUco occluded → circle takes over).
* Degenerate geometry (ray nearly parallel to plane, near-horizon points).
* Lens distortion and image noise (non-zero D coefficients).

Every check is assertion-controlled; no "print ✅ passed" shortcuts.
"""

from __future__ import annotations

import numpy as np
import cv2

from vision_ranging import (
    CameraParams,
    ExtrinsicsEstimator,
    ExtrinsicsEstimatorConfig,
    MarkerObservation,
    Plane,
    Pose,
    auto_update_pose,
    build_marker_detectors,
    measure_distance_between_pixels,
    pixel_to_world_on_plane,
    ray_plane_intersection,
    rotation_angle_deg,
    undistort_points,
)
from water_level import (
    InMemoryWaterLevelSource,
    WaterLevelFusion,
    WaterLevelReading,
)

from datetime import datetime, timezone, timedelta

MAX_REL_ERROR = 0.05
EPS = 1e-9


# ---------------------------------------------------------------------------
# Helpers (shared with test_contract_metrics but self-contained here)
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def look_at_pose(
    camera_center_w: np.ndarray,
    target_w: np.ndarray,
    up_w: np.ndarray | None = None,
) -> Pose:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up_w is None else np.asarray(up_w, dtype=np.float64)
    C = np.asarray(camera_center_w, dtype=np.float64).reshape(3)
    T = np.asarray(target_w, dtype=np.float64).reshape(3)
    z_cam_w = _normalize(T - C)
    x_cam_w = _normalize(np.cross(z_cam_w, up))
    y_cam_w = _normalize(np.cross(z_cam_w, x_cam_w))
    R_cw = np.column_stack([x_cam_w, y_cam_w, z_cam_w])
    R = R_cw.T
    t = -R @ C
    return Pose(R=R, t=t)


def project_world_points(
    points_w: np.ndarray,
    camera: CameraParams,
    pose_w2c: Pose,
) -> np.ndarray:
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(pose_w2c.R)
    uv, _ = cv2.projectPoints(pts, rvec, pose_w2c.t.reshape(3, 1), camera.K, camera.D)
    return uv.reshape(-1, 2)


def euler_deg_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    rxr, ryr, rzr = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rxr), -np.sin(rxr)], [0, np.sin(rxr), np.cos(rxr)]], dtype=np.float64)
    Ry = np.array([[np.cos(ryr), 0, np.sin(ryr)], [0, 1, 0], [-np.sin(ryr), 0, np.cos(ryr)]], dtype=np.float64)
    Rz = np.array([[np.cos(rzr), -np.sin(rzr), 0], [np.sin(rzr), np.cos(rzr), 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _make_camera(distortion: np.ndarray | None = None) -> CameraParams:
    D = np.zeros(5, dtype=np.float64) if distortion is None else np.asarray(distortion, dtype=np.float64)
    return CameraParams(
        name="test_cam",
        model="pinhole",
        width=1920,
        height=1080,
        K=np.array([[1380.0, 0.0, 960.0], [0.0, 1385.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        D=D,
    )


def _make_marker_config():
    marker_length = 1.2
    marker_sep = 0.35
    rows, cols = 2, 3
    origin = np.array([-2.1, 26.0, 0.0], dtype=np.float64)
    aruco_world_map: dict[int, np.ndarray] = {}
    marker_id = 0
    step = marker_length + marker_sep
    for r in range(rows):
        for c in range(cols):
            x0 = origin[0] + c * step
            y0 = origin[1] + r * step
            corners = np.array([
                [x0, y0, 0.0],
                [x0 + marker_length, y0, 0.0],
                [x0 + marker_length, y0 + marker_length, 0.0],
                [x0, y0 + marker_length, 0.0],
            ], dtype=np.float64)
            aruco_world_map[marker_id] = corners
            marker_id += 1
    circle_world = np.array([
        [-1.5, 24.2, 0.0], [0.0, 24.2, 0.0], [1.5, 24.2, 0.0],
        [-1.5, 25.8, 0.0], [0.0, 25.8, 0.0], [1.5, 25.8, 0.0],
    ], dtype=np.float64)
    cfg = {
        "type": "aruco",
        "fallback_types": ["circle"],
        "aruco": {
            "dictionary": "DICT_4X4_50",
            "world_corners_by_id": {str(k): v.tolist() for k, v in aruco_world_map.items()},
        },
        "circle": {
            "expected_count": int(circle_world.shape[0]),
            "world_points": circle_world.tolist(),
            "min_radius_px": 8, "max_radius_px": 120,
            "min_dist_px": 28, "param1": 80, "param2": 12,
            "blur_kernel": 3, "dp": 1.2,
        },
    }
    return cfg, aruco_world_map, circle_world


def _render_mock_image(camera, pose, aruco_world_map, circle_world, *, use_aruco=True, use_circle=True):
    frame: dict = {}
    if use_aruco:
        obj, img, ids = [], [], []
        for mid, corners_w in aruco_world_map.items():
            uv = project_world_points(corners_w, camera, pose)
            obj.append(corners_w)
            img.append(uv)
            ids.extend([mid] * 4)
        frame["synthetic_aruco_observation"] = {
            "object_points": np.vstack(obj),
            "image_points": np.vstack(img),
            "ids": np.asarray(ids, dtype=np.int32),
            "metadata": {},
        }
    else:
        frame["synthetic_aruco_observation"] = None
    if use_circle:
        circle_uv = project_world_points(circle_world, camera, pose)
        frame["synthetic_circle_observation"] = {
            "object_points": circle_world.copy(),
            "image_points": circle_uv,
            "ids": np.arange(circle_world.shape[0], dtype=np.int32),
            "metadata": {},
        }
    else:
        frame["synthetic_circle_observation"] = None
    return frame


def _setup_estimator() -> ExtrinsicsEstimator:
    return ExtrinsicsEstimator(ExtrinsicsEstimatorConfig(
        ransac_reprojection_error_px=4.0,
        ransac_confidence=0.999,
        ransac_iterations=300,
        min_inliers=6,
        min_inlier_ratio=0.5,
        max_reprojection_error_px=4.5,
        smoothing_alpha=1.0,
        max_rotation_jump_deg=35.0,
        max_translation_jump_m=3.0,
    ))


# ===================================================================
# Test: Multi-distance coverage with random pixel jitter
# ===================================================================

class TestPixelJitterMultiDistance:
    """Verify measurement accuracy across 2-50 m with simulated user click jitter."""

    def test_distance_accuracy_with_pixel_jitter(self) -> None:
        rng = np.random.default_rng(42)
        camera = _make_camera()
        plane = Plane.from_height(0.0)

        heights = [3.0, 8.0, 15.0, 30.0]
        distances = [2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
        jitter_px = 2.0  # ±2 pixel random jitter

        errors: list[float] = []
        for h in heights:
            for d_true in distances:
                center = np.array([0.0, -10.0, h], dtype=np.float64)
                y_mid = max(25.0, 1.6 * d_true)
                pose = look_at_pose(center, np.array([0.0, y_mid, 0.0]))

                p1_w = np.array([-0.5 * d_true, y_mid, 0.0], dtype=np.float64)
                p2_w = np.array([0.5 * d_true, y_mid, 0.0], dtype=np.float64)
                uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

                # Add random pixel jitter to simulate imprecise human clicks
                uv_jittered = uv + rng.normal(0.0, jitter_px, size=uv.shape)

                result = measure_distance_between_pixels(
                    uv_jittered[0], uv_jittered[1], camera, pose, plane
                )
                assert result.success, (
                    f"Measurement failed h={h}, D={d_true}: {result.message}"
                )
                assert result.distance_m is not None
                rel_err = abs(result.distance_m - d_true) / d_true
                errors.append(rel_err)
                # For short distances with jitter, allow relaxed threshold
                # (pixel jitter has larger geometric impact at short range)
                if d_true >= 10.0:
                    threshold = MAX_REL_ERROR
                elif d_true >= 5.0:
                    threshold = 0.08
                else:
                    threshold = 0.20
                assert rel_err <= threshold, (
                    f"Pixel jitter test: rel_err={rel_err:.4f} > {threshold}, "
                    f"h={h}, D_true={d_true}, D_est={result.distance_m:.4f}"
                )

        # Overall: median error should be well below 5%
        median_err = float(np.median(errors))
        assert median_err <= MAX_REL_ERROR, f"Median error across all jitter tests: {median_err:.4f}"


# ===================================================================
# Test: Dynamic water level changes
# ===================================================================

class TestWaterLevelDynamic:
    """Verify measurement accuracy when water level changes over time."""

    def test_sudden_water_level_jump(self) -> None:
        """Water level jumps from 0.0 to 1.5 m; system should compensate."""
        camera = _make_camera()
        base_height = 0.0
        new_height = 1.5

        center = np.array([0.0, -5.0, 12.0], dtype=np.float64)
        target = np.array([0.0, 30.0, 0.0], dtype=np.float64)
        pose = look_at_pose(center, target)

        d_true = 10.0
        p1_w = np.array([-5.0, 30.0, new_height], dtype=np.float64)
        p2_w = np.array([5.0, 30.0, new_height], dtype=np.float64)
        uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

        # Correct plane with updated water level
        plane_correct = Plane.from_height(new_height)
        result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane_correct)
        assert result.success and result.distance_m is not None
        rel_err = abs(result.distance_m - d_true) / d_true
        assert rel_err <= MAX_REL_ERROR, (
            f"Water level jump test: rel_err={rel_err:.4f} with correct plane"
        )

    def test_slow_drift_water_level(self) -> None:
        """Water level slowly drifts; each measurement uses current level."""
        camera = _make_camera()
        center = np.array([0.0, -5.0, 15.0], dtype=np.float64)
        pose = look_at_pose(center, np.array([0.0, 30.0, 0.0]))

        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        readings = []
        drift_heights = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2]
        for i, h in enumerate(drift_heights):
            readings.append(WaterLevelReading(
                height_m=h,
                timestamp=base_time + timedelta(minutes=i * 10),
                source="drift_test",
            ))

        source = InMemoryWaterLevelSource(readings)
        base_plane = Plane.from_height(0.0)
        fusion = WaterLevelFusion(source=source, base_plane=base_plane)

        d_true = 8.0
        for i, h in enumerate(drift_heights):
            ref_time = base_time + timedelta(minutes=i * 10)
            plane, _ = fusion.current_plane(reference_time=ref_time)
            actual_height = plane.as_height()
            assert actual_height is not None
            assert abs(actual_height - h) < 1e-9, (
                f"Water level fusion mismatch: expected {h}, got {actual_height}"
            )

            p1_w = np.array([-4.0, 30.0, h], dtype=np.float64)
            p2_w = np.array([4.0, 30.0, h], dtype=np.float64)
            uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

            result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane)
            assert result.success and result.distance_m is not None
            rel_err = abs(result.distance_m - d_true) / d_true
            assert rel_err <= MAX_REL_ERROR, (
                f"Slow drift test at h={h}: rel_err={rel_err:.4f}"
            )


# ===================================================================
# Test: Attitude drift with Monte Carlo sampling
# ===================================================================

class TestAttitudeDrift:
    """Monte Carlo: random camera pose disturbances + correction via markers."""

    def test_attitude_drift_correction_rate(self) -> None:
        """translation > 5cm and tilt > 2° random disturbances, ≥95% correction."""
        rng = np.random.default_rng(12345)
        camera = _make_camera()
        marker_cfg, aruco_world_map, circle_world = _make_marker_config()
        detectors = build_marker_detectors(marker_cfg)
        estimator = _setup_estimator()

        base_pose = look_at_pose(
            np.array([0.0, -8.0, 8.0]),
            np.array([0.0, 28.0, 0.0]),
        )

        n_trials = 100
        success_count = 0

        for _ in range(n_trials):
            # Generate random disturbance
            translation = rng.uniform([-0.3, -0.3, -0.2], [0.3, 0.3, 0.2])
            for _retry in range(100):
                if float(np.linalg.norm(translation)) > 0.05:
                    break
                translation = rng.uniform([-0.3, -0.3, -0.2], [0.3, 0.3, 0.2])
            tilt_deg = float(rng.uniform(2.3, 8.0))
            tilt_axis = rng.normal(0.0, 1.0, size=3)
            tilt_axis[2] = 0.0
            tilt_axis = _normalize(tilt_axis)
            R_tilt, _ = cv2.Rodrigues(tilt_axis * np.deg2rad(tilt_deg))
            R_new = R_tilt @ base_pose.R
            center_new = base_pose.camera_center_world() + translation
            t_new = -R_new @ center_new
            disturbed = Pose(R=R_new, t=t_new)

            frame = _render_mock_image(
                camera, disturbed, aruco_world_map, circle_world,
                use_aruco=True, use_circle=True,
            )
            log: dict = {}
            success, pose_est, _ = auto_update_pose(
                frame=frame, camera=camera, previous_pose=base_pose,
                marker_cfg=marker_cfg, estimator=estimator,
                log=log, detectors=detectors,
            )
            if success and pose_est is not None:
                rot_err = rotation_angle_deg(pose_est.R, disturbed.R)
                center_err = float(np.linalg.norm(
                    pose_est.camera_center_world() - disturbed.camera_center_world()
                ))
                if rot_err <= 2.5 and center_err <= 0.40:
                    success_count += 1

        rate = success_count / n_trials
        assert rate >= 0.95, (
            f"Attitude drift correction rate: {rate:.3f} < 0.95 "
            f"({success_count}/{n_trials})"
        )


# ===================================================================
# Test: Marker degradation / fallback
# ===================================================================

class TestMarkerDegradation:
    """ArUco occluded or missing → circle fallback must provide usable pose."""

    def test_aruco_occluded_circle_fallback(self) -> None:
        camera = _make_camera()
        marker_cfg, aruco_world_map, circle_world = _make_marker_config()
        detectors = build_marker_detectors(marker_cfg)
        estimator = _setup_estimator()

        base_pose = look_at_pose(
            np.array([0.0, -8.0, 9.0]),
            np.array([0.0, 28.0, 0.0]),
        )

        R_delta = euler_deg_to_matrix(5.0, -4.0, 1.0)
        disturbed_R = R_delta @ base_pose.R
        disturbed_center = base_pose.camera_center_world() + np.array([0.5, -0.3, 0.15])
        disturbed = Pose(R=disturbed_R, t=-disturbed_R @ disturbed_center)

        # Only circle markers available (ArUco occluded)
        frame = _render_mock_image(
            camera, disturbed, aruco_world_map, circle_world,
            use_aruco=False, use_circle=True,
        )

        log: dict = {}
        success, pose_est, _ = auto_update_pose(
            frame=frame, camera=camera, previous_pose=base_pose,
            marker_cfg=marker_cfg, estimator=estimator,
            log=log, detectors=detectors,
        )

        assert success, f"Circle fallback failed: {log}"
        assert pose_est is not None, "Pose should be updated by circle detector"
        assert log.get("marker_type") == "circle", (
            f"Expected circle detector, got {log.get('marker_type')}"
        )
        # Verify pose is not just the old pose (it was actually updated)
        rot_diff = rotation_angle_deg(pose_est.R, base_pose.R)
        assert rot_diff > 0.5, (
            "Pose appears unchanged from base; circle fallback may not have "
            "actually updated the pose"
        )

    def test_both_markers_missing_returns_failure(self) -> None:
        """When all markers are missing, system should not claim success."""
        camera = _make_camera()
        marker_cfg, aruco_world_map, circle_world = _make_marker_config()
        detectors = build_marker_detectors(marker_cfg)
        estimator = _setup_estimator()

        base_pose = look_at_pose(
            np.array([0.0, -8.0, 9.0]),
            np.array([0.0, 28.0, 0.0]),
        )

        frame = _render_mock_image(
            camera, base_pose, aruco_world_map, circle_world,
            use_aruco=False, use_circle=False,
        )
        log: dict = {}
        success, pose_est, update = auto_update_pose(
            frame=frame, camera=camera, previous_pose=base_pose,
            marker_cfg=marker_cfg, estimator=estimator,
            log=log, detectors=detectors,
        )

        assert not success, "Should fail when no markers are available"
        assert update.quality.used_previous_pose, (
            "Should fall back to previous pose when markers unavailable"
        )


# ===================================================================
# Test: Degenerate geometry
# ===================================================================

class TestDegenerateGeometry:
    """Near-parallel rays, near-horizon points, numerical edge cases."""

    def test_ray_nearly_parallel_to_plane(self) -> None:
        """When ray is nearly parallel to the water plane, system should
        either fail gracefully or report a warning / very low confidence."""
        camera = _make_camera()

        # Camera looking nearly horizontally along the water surface
        center = np.array([0.0, 0.0, 0.01], dtype=np.float64)
        target = np.array([0.0, 100.0, 0.01], dtype=np.float64)
        pose = look_at_pose(center, target)
        plane = Plane.from_height(0.0)

        # Center pixel → ray nearly parallel to plane
        ok, point_w, status, grazing = pixel_to_world_on_plane(
            (960.0, 540.0), camera, pose, plane
        )
        if ok:
            # If intersection found, grazing angle should be very small
            assert grazing < 0.05, (
                f"Expected near-zero grazing angle, got {grazing:.6f}"
            )
        else:
            # Graceful failure is also acceptable
            assert status in ("ray_parallel_to_plane", "intersection_behind_camera")

    def test_intersection_behind_camera(self) -> None:
        """Ray pointing away from the plane should fail cleanly."""
        plane = Plane.from_height(0.0)
        origin = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        ok, point, status = ray_plane_intersection(origin, direction, plane)
        assert not ok, "Should fail when intersection is behind the camera"
        assert status == "intersection_behind_camera"

    def test_far_horizon_low_confidence(self) -> None:
        """Points near the horizon produce low-confidence measurements."""
        camera = _make_camera()
        center = np.array([0.0, -5.0, 5.0], dtype=np.float64)
        pose = look_at_pose(center, np.array([0.0, 50.0, 0.0]))
        plane = Plane.from_height(0.0)

        # Project two points that are very far away
        p1_w = np.array([-5.0, 500.0, 0.0], dtype=np.float64)
        p2_w = np.array([5.0, 500.0, 0.0], dtype=np.float64)
        uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

        result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane)
        if result.success:
            # Confidence should be low for far-horizon points
            assert result.confidence < 0.8, (
                f"Expected low confidence for horizon points, got {result.confidence:.3f}"
            )


# ===================================================================
# Test: Lens distortion and noise
# ===================================================================

class TestDistortionAndNoise:
    """Non-zero distortion coefficients and synthetic noise."""

    def test_measurement_with_barrel_distortion(self) -> None:
        """Camera with barrel distortion still produces ≤5% error."""
        D = np.array([-0.08, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)
        camera = _make_camera(distortion=D)
        plane = Plane.from_height(0.0)

        center = np.array([0.0, -5.0, 10.0], dtype=np.float64)
        pose = look_at_pose(center, np.array([0.0, 30.0, 0.0]))

        distances = [5.0, 10.0, 20.0]
        for d_true in distances:
            p1_w = np.array([-0.5 * d_true, 30.0, 0.0], dtype=np.float64)
            p2_w = np.array([0.5 * d_true, 30.0, 0.0], dtype=np.float64)
            uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

            result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane)
            assert result.success and result.distance_m is not None, (
                f"Distortion test failed for D={d_true}: {result.message}"
            )
            rel_err = abs(result.distance_m - d_true) / d_true
            assert rel_err <= MAX_REL_ERROR, (
                f"Distortion test: rel_err={rel_err:.4f} for D_true={d_true}"
            )

    def test_measurement_with_noisy_pixels(self) -> None:
        """Multiple noise realizations; median error ≤ 5%."""
        rng = np.random.default_rng(99)
        camera = _make_camera()
        plane = Plane.from_height(0.0)
        center = np.array([0.0, -5.0, 10.0], dtype=np.float64)
        pose = look_at_pose(center, np.array([0.0, 25.0, 0.0]))

        d_true = 12.0
        p1_w = np.array([-6.0, 25.0, 0.0], dtype=np.float64)
        p2_w = np.array([6.0, 25.0, 0.0], dtype=np.float64)
        uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

        errors = []
        for _ in range(50):
            noise = rng.normal(0.0, 1.5, size=uv.shape)
            uv_noisy = uv + noise
            result = measure_distance_between_pixels(
                uv_noisy[0], uv_noisy[1], camera, pose, plane
            )
            if result.success and result.distance_m is not None:
                errors.append(abs(result.distance_m - d_true) / d_true)

        assert len(errors) > 0, "All noisy measurements failed"
        median_err = float(np.median(errors))
        assert median_err <= MAX_REL_ERROR, (
            f"Noisy pixel test: median relative error = {median_err:.4f} > 5%"
        )

    def test_pnp_with_distorted_camera(self) -> None:
        """PnP estimation works with distorted camera model."""
        D = np.array([-0.05, 0.005, 0.0, 0.0, 0.0], dtype=np.float64)
        camera = _make_camera(distortion=D)
        marker_cfg, aruco_world_map, circle_world = _make_marker_config()
        detectors = build_marker_detectors(marker_cfg)
        estimator = _setup_estimator()

        base_pose = look_at_pose(
            np.array([0.0, -8.0, 8.0]),
            np.array([0.0, 28.0, 0.0]),
        )

        frame = _render_mock_image(
            camera, base_pose, aruco_world_map, circle_world,
            use_aruco=True, use_circle=True,
        )
        log: dict = {}
        success, pose_est, _ = auto_update_pose(
            frame=frame, camera=camera, previous_pose=None,
            marker_cfg=marker_cfg, estimator=estimator,
            log=log, detectors=detectors,
        )
        assert success and pose_est is not None, (
            f"PnP with distortion failed: {log}"
        )

        # Verify recovered pose is close to ground truth
        rot_err = rotation_angle_deg(pose_est.R, base_pose.R)
        assert rot_err < 3.0, f"Rotation error {rot_err:.2f}° too large"


# ===================================================================
# Test: Extreme value edge cases
# ===================================================================

class TestExtremeValues:
    """Edge cases at the boundaries of the operational range."""

    def test_minimum_distance_2m(self) -> None:
        camera = _make_camera()
        plane = Plane.from_height(0.0)
        center = np.array([0.0, -2.0, 5.0], dtype=np.float64)
        pose = look_at_pose(center, np.array([0.0, 5.0, 0.0]))

        d_true = 2.0
        p1_w = np.array([-1.0, 5.0, 0.0], dtype=np.float64)
        p2_w = np.array([1.0, 5.0, 0.0], dtype=np.float64)
        uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

        result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane)
        assert result.success and result.distance_m is not None
        rel_err = abs(result.distance_m - d_true) / d_true
        assert rel_err <= MAX_REL_ERROR, f"2m edge case: rel_err={rel_err:.4f}"

    def test_maximum_distance_50m(self) -> None:
        camera = _make_camera()
        plane = Plane.from_height(0.0)
        center = np.array([0.0, -20.0, 40.0], dtype=np.float64)
        pose = look_at_pose(center, np.array([0.0, 80.0, 0.0]))

        d_true = 50.0
        p1_w = np.array([-25.0, 80.0, 0.0], dtype=np.float64)
        p2_w = np.array([25.0, 80.0, 0.0], dtype=np.float64)
        uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

        result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane)
        assert result.success and result.distance_m is not None
        rel_err = abs(result.distance_m - d_true) / d_true
        assert rel_err <= MAX_REL_ERROR, f"50m edge case: rel_err={rel_err:.4f}"

    def test_very_close_points(self) -> None:
        """Two points very close together (sub-meter)."""
        camera = _make_camera()
        plane = Plane.from_height(0.0)
        center = np.array([0.0, -3.0, 5.0], dtype=np.float64)
        pose = look_at_pose(center, np.array([0.0, 8.0, 0.0]))

        d_true = 0.5
        p1_w = np.array([-0.25, 8.0, 0.0], dtype=np.float64)
        p2_w = np.array([0.25, 8.0, 0.0], dtype=np.float64)
        uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

        result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane)
        assert result.success and result.distance_m is not None
        # For very close points, allow slightly relaxed threshold
        rel_err = abs(result.distance_m - d_true) / d_true
        assert rel_err <= 0.10, f"Close-point test: rel_err={rel_err:.4f}"

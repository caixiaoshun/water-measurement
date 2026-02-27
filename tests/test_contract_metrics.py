from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np

from vision_ranging import (
    ArucoMarkerDetector,
    BaseMarkerDetector,
    CameraParams,
    CircleMarkerDetector,
    ExtrinsicsEstimator,
    ExtrinsicsEstimatorConfig,
    Plane,
    Pose,
    auto_update_pose,
    build_marker_detectors,
    measure_distance_between_pixels,
    rotation_angle_deg,
    undistort_points,
)

EPS = 1e-9
MAX_REL_ERROR = 0.05
POSE_SUCCESS_THRESHOLD = 0.95
RECOVERY_THRESHOLD = 0.9
LATENCY_THRESHOLD_MS = 1000.0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def look_at_pose(camera_center_w: np.ndarray, target_w: np.ndarray, up_w: np.ndarray | None = None) -> Pose:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up_w is None else np.asarray(up_w, dtype=np.float64)
    C = np.asarray(camera_center_w, dtype=np.float64).reshape(3)
    T = np.asarray(target_w, dtype=np.float64).reshape(3)

    z_cam_w = _normalize(T - C)
    x_cam_w = _normalize(np.cross(z_cam_w, up))
    y_up_w = _normalize(np.cross(z_cam_w, x_cam_w))
    y_cam_w = y_up_w

    R_cw = np.column_stack([x_cam_w, y_cam_w, z_cam_w])
    R = R_cw.T
    t = -R @ C
    return Pose(R=R, t=t)


def project_world_points(points_w: np.ndarray, camera: CameraParams, pose_w2c: Pose) -> np.ndarray:
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(pose_w2c.R)
    uv, _ = cv2.projectPoints(pts, rvec, pose_w2c.t.reshape(3, 1), camera.K, camera.D)
    return uv.reshape(-1, 2)


def euler_deg_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    rxr, ryr, rzr = np.deg2rad([rx, ry, rz])
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(rxr), -np.sin(rxr)], [0.0, np.sin(rxr), np.cos(rxr)]],
        dtype=np.float64,
    )
    Ry = np.array(
        [[np.cos(ryr), 0.0, np.sin(ryr)], [0.0, 1.0, 0.0], [-np.sin(ryr), 0.0, np.cos(ryr)]],
        dtype=np.float64,
    )
    Rz = np.array(
        [[np.cos(rzr), -np.sin(rzr), 0.0], [np.sin(rzr), np.cos(rzr), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return Rz @ Ry @ Rx


def disturbed_pose(base_pose: Pose, rng: np.random.Generator) -> tuple[Pose, float, float]:
    base_center = base_pose.camera_center_world()

    translation = rng.uniform([-0.25, -0.25, -0.20], [0.25, 0.25, 0.20])
    while float(np.linalg.norm(translation)) <= 0.05:
        translation = rng.uniform([-0.25, -0.25, -0.20], [0.25, 0.25, 0.20])

    tilt_deg = float(rng.uniform(2.3, 7.5))
    tilt_axis = rng.normal(0.0, 1.0, size=3)
    tilt_axis[2] = 0.0
    tilt_axis = _normalize(tilt_axis)
    tilt_rotvec = tilt_axis * np.deg2rad(tilt_deg)
    R_tilt, _ = cv2.Rodrigues(tilt_rotvec)

    yaw_deg = float(rng.uniform(-4.0, 4.0))
    R_yaw = euler_deg_to_matrix(0.0, 0.0, yaw_deg)

    R_new = R_tilt @ R_yaw @ base_pose.R
    center_new = base_center + translation
    t_new = -R_new @ center_new
    return Pose(R=R_new, t=t_new), float(np.linalg.norm(translation)), tilt_deg


def make_camera_models() -> dict[str, CameraParams]:
    return {
        "Hikvision_Bullet_4mm": CameraParams.from_dict(
            {
                "name": "Hikvision_Bullet_4mm",
                "model": "pinhole",
                "width": 1920,
                "height": 1080,
                "K": [[1380.0, 0.0, 960.0], [0.0, 1385.0, 540.0], [0.0, 0.0, 1.0]],
                "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        "Dahua_Dome_2p8mm": CameraParams.from_dict(
            {
                "name": "Dahua_Dome_2p8mm",
                "model": "pinhole",
                "width": 2560,
                "height": 1440,
                "K": [[1820.0, 0.0, 1280.0], [0.0, 1815.0, 720.0], [0.0, 0.0, 1.0]],
                "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        "Axis_PTZ_Optical": CameraParams.from_dict(
            {
                "name": "Axis_PTZ_Optical",
                "model": "pinhole",
                "width": 1280,
                "height": 720,
                "K": [[930.0, 0.0, 640.0], [0.0, 932.0, 360.0], [0.0, 0.0, 1.0]],
                "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
    }


def make_marker_configuration() -> tuple[dict, dict[int, np.ndarray], np.ndarray]:
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
            corners = np.array(
                [
                    [x0, y0, 0.0],
                    [x0 + marker_length, y0, 0.0],
                    [x0 + marker_length, y0 + marker_length, 0.0],
                    [x0, y0 + marker_length, 0.0],
                ],
                dtype=np.float64,
            )
            aruco_world_map[marker_id] = corners
            marker_id += 1

    circle_world = np.array(
        [
            [-1.5, 24.2, 0.0],
            [0.0, 24.2, 0.0],
            [1.5, 24.2, 0.0],
            [-1.5, 25.8, 0.0],
            [0.0, 25.8, 0.0],
            [1.5, 25.8, 0.0],
        ],
        dtype=np.float64,
    )

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
            "min_radius_px": 8,
            "max_radius_px": 120,
            "min_dist_px": 28,
            "param1": 80,
            "param2": 12,
            "blur_kernel": 3,
            "dp": 1.2,
        },
    }
    return cfg, aruco_world_map, circle_world


def render_mock_image(
    camera: CameraParams,
    pose: Pose,
    aruco_world_map: dict[int, np.ndarray],
    circle_world: np.ndarray,
    use_aruco: bool,
    use_circle: bool,
    aruco_occluded: bool = False,
) -> dict:
    # 可解释 mock：由世界点几何投影生成“伪图像对象”，供检测器读取。
    frame: dict = {}

    if use_aruco and not aruco_occluded:
        object_points = []
        image_points = []
        ids = []
        for marker_id, corners_w in aruco_world_map.items():
            uv = project_world_points(corners_w, camera, pose)
            object_points.append(corners_w)
            image_points.append(uv)
            ids.extend([marker_id] * 4)
        frame["synthetic_aruco_observation"] = {
            "object_points": np.vstack(object_points),
            "image_points": np.vstack(image_points),
            "ids": np.asarray(ids, dtype=np.int32),
            "metadata": {"generated_from_geometry": True},
        }
    else:
        frame["synthetic_aruco_observation"] = None

    if use_circle:
        circle_uv = project_world_points(circle_world, camera, pose)
        frame["synthetic_circle_observation"] = {
            "object_points": circle_world.copy(),
            "image_points": circle_uv,
            "ids": np.arange(circle_world.shape[0], dtype=np.int32),
            "metadata": {"generated_from_geometry": True},
        }
    else:
        frame["synthetic_circle_observation"] = None

    return frame


def setup_pose_estimator() -> ExtrinsicsEstimator:
    return ExtrinsicsEstimator(
        ExtrinsicsEstimatorConfig(
            ransac_reprojection_error_px=4.0,
            ransac_confidence=0.999,
            ransac_iterations=300,
            min_inliers=6,
            min_inlier_ratio=0.5,
            max_reprojection_error_px=4.5,
            smoothing_alpha=1.0,
            max_rotation_jump_deg=35.0,
            max_translation_jump_m=3.0,
        )
    )


def test_distance_error_under_5pct_over_range() -> None:
    """合同指标(1): 在2~50m工况下，相对误差<=5%。"""
    camera = make_camera_models()["Hikvision_Bullet_4mm"]
    plane = Plane.from_height(0.0)

    height_list = [2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    distance_list = [2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    for h in height_list:
        for d_true in distance_list:
            center = np.array([0.0, -10.0, h], dtype=np.float64)
            y_mid = max(25.0, 1.6 * d_true)
            pose = look_at_pose(center, np.array([0.0, y_mid, 0.0], dtype=np.float64))

            p1_w = np.array([-0.5 * d_true, y_mid, 0.0], dtype=np.float64)
            p2_w = np.array([0.5 * d_true, y_mid, 0.0], dtype=np.float64)
            uv = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

            result = measure_distance_between_pixels(uv[0], uv[1], camera, pose, plane)
            assert result.success, f"指标(1)失败: 测距失败, h={h}, D={d_true}, msg={result.message}"
            assert result.distance_m is not None
            rel_error = abs(result.distance_m - d_true) / d_true
            assert rel_error <= MAX_REL_ERROR, (
                f"指标(1)失败: 相对误差={rel_error:.4f} 超过5%, h={h}, D_true={d_true}, D_est={result.distance_m}"
            )


def test_pose_correction_success_rate_ge_95pct() -> None:
    """合同指标(2): 姿态扰动后，外参自动修正成功率>=95%。"""
    rng = np.random.default_rng(20260227)

    camera = make_camera_models()["Hikvision_Bullet_4mm"]
    marker_cfg, aruco_world_map, circle_world = make_marker_configuration()
    detectors = build_marker_detectors(marker_cfg)
    estimator = setup_pose_estimator()

    base_pose = look_at_pose(np.array([0.0, -8.0, 8.0], dtype=np.float64), np.array([0.0, 28.0, 0.0], dtype=np.float64))

    n_trials = 120
    success_count = 0

    for _ in range(n_trials):
        disturbed, shift_m, tilt_deg = disturbed_pose(base_pose, rng)
        assert (shift_m > 0.05) or (tilt_deg > 2.0)

        frame = render_mock_image(
            camera,
            disturbed,
            aruco_world_map,
            circle_world,
            use_aruco=True,
            use_circle=True,
            aruco_occluded=False,
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

        rot_err = rotation_angle_deg(pose_est.R, disturbed.R)
        center_err = float(np.linalg.norm(pose_est.camera_center_world() - disturbed.camera_center_world()))
        if rot_err <= 2.5 and center_err <= 0.40:
            success_count += 1

    success_rate = success_count / n_trials
    assert success_rate >= POSE_SUCCESS_THRESHOLD, (
        f"指标(2)失败: success_rate={success_rate:.3f} < 0.95 (success={success_count}/{n_trials})"
    )


def test_recovery_ge_90pct_when_disturbed_then_corrected() -> None:
    """合同指标(3)+(6): 扰动后修正恢复>=90%，且Aruco失效时Circle可接管。"""
    camera = make_camera_models()["Hikvision_Bullet_4mm"]
    plane = Plane.from_height(0.0)
    marker_cfg, aruco_world_map, circle_world = make_marker_configuration()
    detectors = build_marker_detectors(marker_cfg)
    estimator = setup_pose_estimator()

    base_pose = look_at_pose(np.array([0.0, -8.0, 8.0], dtype=np.float64), np.array([0.0, 28.0, 0.0], dtype=np.float64))

    # 强扰动：保证修正前误差明显
    R_delta = euler_deg_to_matrix(7.0, -6.0, 1.5)
    disturbed_R = R_delta @ base_pose.R
    disturbed_center = base_pose.camera_center_world() + np.array([0.7, -0.45, 0.25], dtype=np.float64)
    disturbed_pose_w2c = Pose(R=disturbed_R, t=-disturbed_R @ disturbed_center)

    d_true = 24.0
    p1_w = np.array([-12.0, 32.0, 0.0], dtype=np.float64)
    p2_w = np.array([12.0, 32.0, 0.0], dtype=np.float64)
    uv_pair = project_world_points(np.vstack([p1_w, p2_w]), camera, disturbed_pose_w2c)

    pre = measure_distance_between_pixels(uv_pair[0], uv_pair[1], camera, base_pose, plane)
    assert pre.success and pre.distance_m is not None

    # ArUco不可识别，仅保留circle，必须靠fallback修正
    frame = render_mock_image(
        camera,
        disturbed_pose_w2c,
        aruco_world_map,
        circle_world,
        use_aruco=False,
        use_circle=True,
        aruco_occluded=False,
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

    assert success and corrected_pose is not None, f"指标(3)/(6)失败: 无法完成外参修正, log={log}"
    assert log.get("marker_type") == "circle", f"指标(6)失败: fallback未切到circle, log={log}"

    post = measure_distance_between_pixels(uv_pair[0], uv_pair[1], camera, corrected_pose, plane)
    assert post.success and post.distance_m is not None

    pre_err = abs(pre.distance_m - d_true)
    post_err = abs(post.distance_m - d_true)
    recovery = 1.0 - post_err / (pre_err + EPS)

    assert recovery >= RECOVERY_THRESHOLD, (
        f"指标(3)失败: recovery={recovery:.4f} < 0.9, pre_err={pre_err:.4f}, post_err={post_err:.4f}"
    )

    post_rel = post_err / d_true
    assert post_rel <= MAX_REL_ERROR, f"指标(3)失败: 修正后误差={post_rel:.4f} > 5%"


def test_latency_le_1s() -> None:
    """合同指标(4): 采用P95口径，测距耗时<=1000ms。"""
    camera = make_camera_models()["Axis_PTZ_Optical"]
    plane = Plane.from_height(0.0)
    pose = look_at_pose(np.array([0.0, -6.0, 12.0], dtype=np.float64), np.array([0.0, 34.0, 0.0], dtype=np.float64))

    p1_w = np.array([-8.0, 34.0, 0.0], dtype=np.float64)
    p2_w = np.array([8.0, 34.0, 0.0], dtype=np.float64)
    uv_pair = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)

    times = []
    for _ in range(60):
        out = measure_distance_between_pixels(uv_pair[0], uv_pair[1], camera, pose, plane)
        assert out.success and out.elapsed_ms >= 0.0
        times.append(out.elapsed_ms)

    p95 = float(np.percentile(np.asarray(times, dtype=np.float64), 95))
    assert p95 <= LATENCY_THRESHOLD_MS, f"指标(4)失败: P95={p95:.3f}ms > 1000ms"


def test_multi_camera_models_ge_3() -> None:
    """合同指标(5): 至少3套相机模型参数接入并跑通链路。"""
    camera_models = make_camera_models()
    assert len(camera_models) >= 3, "指标(5)失败: 相机模型数不足3"

    plane = Plane.from_height(0.0)
    pose = look_at_pose(np.array([0.0, -7.0, 10.0], dtype=np.float64), np.array([0.0, 30.0, 0.0], dtype=np.float64))

    p1_w = np.array([-5.0, 30.0, 0.0], dtype=np.float64)
    p2_w = np.array([5.0, 30.0, 0.0], dtype=np.float64)
    d_true = 10.0

    for model_name, camera in camera_models.items():
        uv_pair = project_world_points(np.vstack([p1_w, p2_w]), camera, pose)
        undist = undistort_points(uv_pair, camera)
        assert undist.shape == (2, 2), f"指标(5)失败: {model_name} 去畸变未跑通"

        out = measure_distance_between_pixels(uv_pair[0], uv_pair[1], camera, pose, plane)
        assert out.success and out.distance_m is not None, f"指标(5)失败: {model_name} 测距失败"
        rel = abs(out.distance_m - d_true) / d_true
        assert rel <= MAX_REL_ERROR, f"指标(5)失败: {model_name} 相对误差={rel:.4f} > 5%"


def test_marker_types_ge_2() -> None:
    """合同指标(6): 至少两类基准点，且Aruco失败时circle仍可更新pose。"""
    camera = make_camera_models()["Dahua_Dome_2p8mm"]
    marker_cfg, aruco_world_map, circle_world = make_marker_configuration()
    detectors: list[BaseMarkerDetector] = build_marker_detectors(marker_cfg)

    detector_classes = {d.__class__.__name__ for d in detectors}
    assert {"ArucoMarkerDetector", "CircleMarkerDetector"}.issubset(detector_classes), (
        f"指标(6)失败: 检测器类型不足, got={detector_classes}"
    )

    base_pose = look_at_pose(np.array([0.0, -8.0, 9.0], dtype=np.float64), np.array([0.0, 28.0, 0.0], dtype=np.float64))
    disturbed, shift_m, tilt_deg = disturbed_pose(base_pose, np.random.default_rng(66))
    assert shift_m > 0.05 or tilt_deg > 2.0

    frame = render_mock_image(
        camera,
        disturbed,
        aruco_world_map,
        circle_world,
        use_aruco=False,
        use_circle=True,
        aruco_occluded=False,
    )

    estimator = setup_pose_estimator()
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

    assert success and pose_est is not None, f"指标(6)失败: circle fallback无法更新pose, log={log}"
    assert log.get("marker_type") == "circle", f"指标(6)失败: 实际未使用circle, log={log}"


def test_delivery_artifacts_present() -> None:
    """合同指标(7): 技术报告与核心源码存在性检查。"""
    root = Path(__file__).resolve().parents[1]

    report_candidates = [
        root / "TECH_REPORT.md",
        root / "docs" / "report.md",
        root / "docs" / "TECH_REPORT.md",
    ]
    existing_reports = [p for p in report_candidates if p.exists() and p.is_file()]
    assert existing_reports, "指标(7)失败: TECH_REPORT.md 或 docs/report.md 或 docs/TECH_REPORT.md 不存在"

    non_empty = [p for p in existing_reports if p.stat().st_size > 0]
    assert non_empty, "指标(7)失败: 技术报告存在但为空"

    # Check source files (support both root and src/ layout)
    core_modules = ["main.py", "vision_ranging.py", "water_level.py"]
    for name in core_modules:
        found = (root / name).exists() or (root / "src" / name).exists()
        assert found, f"指标(7)失败: 缺少核心文件 {name}"
    assert (root / "config.yaml").exists(), "指标(7)失败: 缺少 config.yaml"

    # 额外做可导入性检查
    importlib.import_module("vision_ranging")
    importlib.import_module("water_level")
    importlib.import_module("main")

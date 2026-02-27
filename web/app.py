"""FastAPI backend for the water-measurement Web demo.

Endpoints:
    GET  /health            – liveness check
    GET  /config            – current camera & system configuration summary
    POST /measure           – accepts two pixel coordinates, returns distance + 3D data
    GET  /test_cases        – list synthetic test cases with full 3D scene data
    POST /test_cases/{id}/run – run a test case measurement, return result + 3D data
    GET  /                  – serves the static frontend (index.html)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root and src/ are importable
_project_root = str(Path(__file__).resolve().parents[1])
_src_dir = str(Path(__file__).resolve().parents[1] / "src")
for _p in [_project_root, _src_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vision_ranging import (  # noqa: E402
    CameraParams,
    ExtrinsicsEstimator,
    ExtrinsicsEstimatorConfig,
    Plane,
    Pose,
    measure_distance_between_pixels,
)

app = FastAPI(
    title="水面视觉测距系统演示",
    description="自适应单目视觉测距 – Web 演示",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Geometry helpers (defined first so _DEFAULT_POSE can use them)
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def _look_at_pose(
    camera_center_w: np.ndarray,
    target_w: np.ndarray,
    up_w: Optional[np.ndarray] = None,
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


def _project_world_points(
    points_w: np.ndarray,
    camera: CameraParams,
    pose_w2c: Pose,
) -> np.ndarray:
    pts = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(pose_w2c.R)
    uv, _ = cv2.projectPoints(pts, rvec, pose_w2c.t.reshape(3, 1), camera.K, camera.D)
    return uv.reshape(-1, 2)


def _euler_deg_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    rxr, ryr, rzr = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rxr), -np.sin(rxr)], [0, np.sin(rxr), np.cos(rxr)]], dtype=np.float64)
    Ry = np.array([[np.cos(ryr), 0, np.sin(ryr)], [0, 1, 0], [-np.sin(ryr), 0, np.cos(ryr)]], dtype=np.float64)
    Rz = np.array([[np.cos(rzr), -np.sin(rzr), 0], [np.sin(rzr), np.cos(rzr), 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Default camera / pose / plane (demo configuration)
# ---------------------------------------------------------------------------

_DEFAULT_CAMERA = CameraParams(
    name="demo_cam",
    model="pinhole",
    width=1280,
    height=720,
    K=np.array([[920.0, 0.0, 640.0], [0.0, 920.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    D=np.zeros(5, dtype=np.float64),
)

_DEFAULT_WATER_LEVEL = 0.0
_DEFAULT_PLANE = Plane.from_height(_DEFAULT_WATER_LEVEL)

# Camera looking down at a water surface at z=0 from [0, -4.8, 6.4] toward [0, 10, 0].
# Use _look_at_pose so camera_center_world() returns a visualizable above-water position.
_DEFAULT_POSE = _look_at_pose(
    camera_center_w=np.array([0.0, -4.8, 6.4], dtype=np.float64),
    target_w=np.array([0.0, 10.0, 0.0]),
)

_estimator_cfg = ExtrinsicsEstimatorConfig()
_estimator = ExtrinsicsEstimator(_estimator_cfg)


# ---------------------------------------------------------------------------
# Test case definitions (mirrors patterns from tests/ directory)
# ---------------------------------------------------------------------------

def _build_test_cases() -> List[Dict[str, Any]]:
    """Build synthetic test cases matching patterns from tests/ directory."""
    cases: List[Dict[str, Any]] = []

    # Shared base camera (1280x720, 920px focal length) – same as _DEFAULT_CAMERA
    base_camera = _DEFAULT_CAMERA

    # High-resolution camera for complex tests (mirrors test_complex_scenarios.py)
    hd_camera = CameraParams(
        name="hd_cam",
        model="pinhole",
        width=1920,
        height=1080,
        K=np.array([[1380.0, 0.0, 960.0], [0.0, 1385.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        D=np.zeros(5, dtype=np.float64),
    )

    # --- Case 1: 基线测量 5m (test_basic.py: test_distance_simple_case) ---
    c1_center = np.array([0.0, -4.8, 6.4], dtype=np.float64)
    c1_pose = _look_at_pose(c1_center, np.array([0.0, 10.0, 0.0]))
    c1_p1 = np.array([-2.5, 10.0, 0.0], dtype=np.float64)
    c1_p2 = np.array([2.5, 10.0, 0.0], dtype=np.float64)
    c1_uv = _project_world_points(np.vstack([c1_p1, c1_p2]), base_camera, c1_pose)
    cases.append({
        "id": "baseline_5m",
        "name": "基线测量 5m",
        "description": "标准相机配置，水面 z=0，y=10m 处两点水平距离 5m。对应 test_basic.py 基础测距场景。",
        "category": "基础测试",
        "camera": base_camera,
        "pose": c1_pose,
        "plane_height_m": 0.0,
        "point1_world": c1_p1.tolist(),
        "point2_world": c1_p2.tolist(),
        "point1_pixel": {"x": float(c1_uv[0, 0]), "y": float(c1_uv[0, 1])},
        "point2_pixel": {"x": float(c1_uv[1, 0]), "y": float(c1_uv[1, 1])},
        "expected_distance_m": 5.0,
        "tags": ["基础", "无噪声", "标准姿态"],
    })

    # --- Case 2: 基线测量 10m ---
    c2_p1 = np.array([-5.0, 15.0, 0.0], dtype=np.float64)
    c2_p2 = np.array([5.0, 15.0, 0.0], dtype=np.float64)
    c2_uv = _project_world_points(np.vstack([c2_p1, c2_p2]), base_camera, c1_pose)
    cases.append({
        "id": "baseline_10m",
        "name": "基线测量 10m",
        "description": "标准相机配置，水面 z=0，y=15m 处两点水平距离 10m。验证中等距离精度。",
        "category": "基础测试",
        "camera": base_camera,
        "pose": c1_pose,
        "plane_height_m": 0.0,
        "point1_world": c2_p1.tolist(),
        "point2_world": c2_p2.tolist(),
        "point1_pixel": {"x": float(c2_uv[0, 0]), "y": float(c2_uv[0, 1])},
        "point2_pixel": {"x": float(c2_uv[1, 0]), "y": float(c2_uv[1, 1])},
        "expected_distance_m": 10.0,
        "tags": ["基础", "无噪声", "标准姿态"],
    })

    # --- Case 3: 远距离 50m (test_complex_scenarios.py: TestExtremeValues.test_maximum_distance_50m) ---
    c3_center = np.array([0.0, -20.0, 40.0], dtype=np.float64)
    c3_pose = _look_at_pose(c3_center, np.array([0.0, 80.0, 0.0]))
    c3_p1 = np.array([-25.0, 80.0, 0.0], dtype=np.float64)
    c3_p2 = np.array([25.0, 80.0, 0.0], dtype=np.float64)
    c3_uv = _project_world_points(np.vstack([c3_p1, c3_p2]), hd_camera, c3_pose)
    cases.append({
        "id": "long_range_50m",
        "name": "远距离测量 50m",
        "description": "相机高度 40m，测量 50m 跨度目标。对应 test_complex_scenarios.py TestExtremeValues.test_maximum_distance_50m。",
        "category": "极值测试",
        "camera": hd_camera,
        "pose": c3_pose,
        "plane_height_m": 0.0,
        "point1_world": c3_p1.tolist(),
        "point2_world": c3_p2.tolist(),
        "point1_pixel": {"x": float(c3_uv[0, 0]), "y": float(c3_uv[0, 1])},
        "point2_pixel": {"x": float(c3_uv[1, 0]), "y": float(c3_uv[1, 1])},
        "expected_distance_m": 50.0,
        "tags": ["极值", "远距", "高空"],
    })

    # --- Case 4: 近距离 2m (test_complex_scenarios.py: TestExtremeValues.test_minimum_distance_2m) ---
    c4_center = np.array([0.0, -2.0, 5.0], dtype=np.float64)
    c4_pose = _look_at_pose(c4_center, np.array([0.0, 5.0, 0.0]))
    c4_p1 = np.array([-1.0, 5.0, 0.0], dtype=np.float64)
    c4_p2 = np.array([1.0, 5.0, 0.0], dtype=np.float64)
    c4_uv = _project_world_points(np.vstack([c4_p1, c4_p2]), hd_camera, c4_pose)
    cases.append({
        "id": "short_range_2m",
        "name": "近距离测量 2m",
        "description": "相机高度 5m，测量近距离 2m 目标。对应 test_complex_scenarios.py TestExtremeValues.test_minimum_distance_2m。",
        "category": "极值测试",
        "camera": hd_camera,
        "pose": c4_pose,
        "plane_height_m": 0.0,
        "point1_world": c4_p1.tolist(),
        "point2_world": c4_p2.tolist(),
        "point1_pixel": {"x": float(c4_uv[0, 0]), "y": float(c4_uv[0, 1])},
        "point2_pixel": {"x": float(c4_uv[1, 0]), "y": float(c4_uv[1, 1])},
        "expected_distance_m": 2.0,
        "tags": ["极值", "近距"],
    })

    # --- Case 5: 水位变化 +1.5m (test_complex_scenarios.py: TestWaterLevelDynamic) ---
    c5_center = np.array([0.0, -5.0, 12.0], dtype=np.float64)
    c5_pose = _look_at_pose(c5_center, np.array([0.0, 30.0, 0.0]))
    c5_wl = 1.5
    c5_p1 = np.array([-5.0, 30.0, c5_wl], dtype=np.float64)
    c5_p2 = np.array([5.0, 30.0, c5_wl], dtype=np.float64)
    c5_uv = _project_world_points(np.vstack([c5_p1, c5_p2]), hd_camera, c5_pose)
    cases.append({
        "id": "water_level_1_5m",
        "name": "水位上升至 1.5m",
        "description": "水面从 z=0 上升至 z=1.5m，系统需用正确水位平面解算。对应 test_complex_scenarios.py TestWaterLevelDynamic.test_sudden_water_level_jump。",
        "category": "水位变化",
        "camera": hd_camera,
        "pose": c5_pose,
        "plane_height_m": c5_wl,
        "point1_world": c5_p1.tolist(),
        "point2_world": c5_p2.tolist(),
        "point1_pixel": {"x": float(c5_uv[0, 0]), "y": float(c5_uv[0, 1])},
        "point2_pixel": {"x": float(c5_uv[1, 0]), "y": float(c5_uv[1, 1])},
        "expected_distance_m": 10.0,
        "tags": ["水位变化", "动态"],
    })

    # --- Case 6: 像素噪声 ±2px (test_complex_scenarios.py: TestPixelJitterMultiDistance) ---
    c6_center = np.array([0.0, -10.0, 15.0], dtype=np.float64)
    c6_pose = _look_at_pose(c6_center, np.array([0.0, 40.0, 0.0]))
    c6_p1 = np.array([-5.0, 40.0, 0.0], dtype=np.float64)
    c6_p2 = np.array([5.0, 40.0, 0.0], dtype=np.float64)
    c6_uv_clean = _project_world_points(np.vstack([c6_p1, c6_p2]), hd_camera, c6_pose)
    rng6 = np.random.default_rng(42)
    c6_uv = c6_uv_clean + rng6.normal(0.0, 2.0, size=c6_uv_clean.shape)
    cases.append({
        "id": "pixel_noise_10m",
        "name": "像素噪声 ±2px（10m 基线）",
        "description": "模拟用户点击误差 ±2px 高斯噪声，距离 10m。对应 test_complex_scenarios.py TestPixelJitterMultiDistance。",
        "category": "噪声测试",
        "camera": hd_camera,
        "pose": c6_pose,
        "plane_height_m": 0.0,
        "point1_world": c6_p1.tolist(),
        "point2_world": c6_p2.tolist(),
        "point1_pixel": {"x": float(c6_uv[0, 0]), "y": float(c6_uv[0, 1])},
        "point2_pixel": {"x": float(c6_uv[1, 0]), "y": float(c6_uv[1, 1])},
        "expected_distance_m": 10.0,
        "tags": ["噪声", "像素抖动"],
    })

    # --- Case 7: 姿态扰动（3° 倾斜 + 8cm 偏移）(mirrors test_complex_scenarios.py: TestAttitudeDrift) ---
    c7_base_center = np.array([0.0, -4.8, 6.4], dtype=np.float64)
    c7_base_pose = _look_at_pose(c7_base_center, np.array([0.0, 10.0, 0.0]))
    R_tilt = _euler_deg_to_matrix(3.0, 0.0, 0.0)
    c7_R = R_tilt @ c7_base_pose.R
    c7_center = c7_base_center + np.array([0.06, 0.04, 0.032], dtype=np.float64)
    c7_pose = Pose(R=c7_R, t=-c7_R @ c7_center)
    c7_p1 = np.array([-2.5, 10.0, 0.0], dtype=np.float64)
    c7_p2 = np.array([2.5, 10.0, 0.0], dtype=np.float64)
    c7_uv = _project_world_points(np.vstack([c7_p1, c7_p2]), base_camera, c7_pose)
    cases.append({
        "id": "attitude_perturb",
        "name": "姿态扰动（3° 倾斜 + 8cm 偏移）",
        "description": "相机倾斜 3°、平移 8cm，模拟安装漂移。对应 test_complex_scenarios.py TestAttitudeDrift 场景。",
        "category": "姿态扰动",
        "camera": base_camera,
        "pose": c7_pose,
        "plane_height_m": 0.0,
        "point1_world": c7_p1.tolist(),
        "point2_world": c7_p2.tolist(),
        "point1_pixel": {"x": float(c7_uv[0, 0]), "y": float(c7_uv[0, 1])},
        "point2_pixel": {"x": float(c7_uv[1, 0]), "y": float(c7_uv[1, 1])},
        "expected_distance_m": 5.0,
        "tags": ["姿态扰动", "漂移"],
    })

    # --- Case 8: 畸变相机（桶形畸变）(test_complex_scenarios.py: TestDistortionAndNoise) ---
    D_barrel = np.array([-0.08, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    c8_camera = CameraParams(
        name="distorted_cam",
        model="pinhole",
        width=1920,
        height=1080,
        K=np.array([[1380.0, 0.0, 960.0], [0.0, 1385.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        D=D_barrel,
    )
    c8_center = np.array([0.0, -5.0, 10.0], dtype=np.float64)
    c8_pose = _look_at_pose(c8_center, np.array([0.0, 30.0, 0.0]))
    c8_p1 = np.array([-5.0, 30.0, 0.0], dtype=np.float64)
    c8_p2 = np.array([5.0, 30.0, 0.0], dtype=np.float64)
    c8_uv = _project_world_points(np.vstack([c8_p1, c8_p2]), c8_camera, c8_pose)
    cases.append({
        "id": "barrel_distortion_10m",
        "name": "桶形畸变相机（10m 基线）",
        "description": "k1=-0.08 桶形畸变，系统应通过去畸变校正后精度 ≤5%。对应 test_complex_scenarios.py TestDistortionAndNoise.test_measurement_with_barrel_distortion。",
        "category": "镜头畸变",
        "camera": c8_camera,
        "pose": c8_pose,
        "plane_height_m": 0.0,
        "point1_world": c8_p1.tolist(),
        "point2_world": c8_p2.tolist(),
        "point1_pixel": {"x": float(c8_uv[0, 0]), "y": float(c8_uv[0, 1])},
        "point2_pixel": {"x": float(c8_uv[1, 0]), "y": float(c8_uv[1, 1])},
        "expected_distance_m": 10.0,
        "tags": ["畸变", "镜头校正"],
    })

    # --- Case 9: 退化几何（近水平射线）(test_complex_scenarios.py: TestDegenerateGeometry) ---
    c9_camera = base_camera
    c9_center = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    c9_target = np.array([0.0, 200.0, 0.3], dtype=np.float64)
    c9_pose = _look_at_pose(c9_center, c9_target)
    # Near-horizon pixels
    c9_px1 = {"x": float(c9_camera.width // 2 - 100), "y": 10.0}
    c9_px2 = {"x": float(c9_camera.width // 2 + 100), "y": 10.0}
    # Approximate world points for display (not measurable accurately)
    c9_p1 = np.array([-1.0, 50.0, 0.0], dtype=np.float64)
    c9_p2 = np.array([1.0, 50.0, 0.0], dtype=np.float64)
    cases.append({
        "id": "degenerate_geometry",
        "name": "退化几何（近水平射线）",
        "description": "相机高度 0.5m 近乎水平观测，射线几乎平行于水面，预期测量失败或置信度极低。对应 TestDegenerateGeometry。",
        "category": "退化场景",
        "camera": c9_camera,
        "pose": c9_pose,
        "plane_height_m": 0.0,
        "point1_world": c9_p1.tolist(),
        "point2_world": c9_p2.tolist(),
        "point1_pixel": c9_px1,
        "point2_pixel": c9_px2,
        "expected_distance_m": None,
        "tags": ["退化", "低置信度", "预期失败"],
    })

    return cases


_TEST_CASES = _build_test_cases()
_TEST_CASE_MAP: Dict[str, Dict[str, Any]] = {tc["id"]: tc for tc in _TEST_CASES}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PixelPoint(BaseModel):
    x: float = Field(..., description="像素 x 坐标")
    y: float = Field(..., description="像素 y 坐标")


class MeasureRequest(BaseModel):
    point1: PixelPoint
    point2: PixelPoint
    water_level_m: Optional[float] = Field(None, description="水位覆盖（米）")


class MeasureResponse(BaseModel):
    success: bool
    distance_m: Optional[float]
    confidence: float
    elapsed_ms: float
    message: str
    diagnostics: Dict[str, Any]
    # 3D 可视化坐标
    camera_center_world: Optional[List[float]] = None
    camera_pose_R: Optional[List[List[float]]] = None
    point1_world: Optional[List[float]] = None
    point2_world: Optional[List[float]] = None
    water_level_m: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigResponse(BaseModel):
    camera: Dict[str, Any]
    water_level_m: float
    plane: Dict[str, Any]
    image_width: int
    image_height: int
    pose: PoseData


class PoseData(BaseModel):
    R: List[List[float]]
    t: List[float]
    camera_center_world: List[float]


class TestCaseScene(BaseModel):
    id: str
    name: str
    description: str
    category: str
    camera: Dict[str, Any]
    pose: PoseData
    plane_height_m: float
    point1_world: List[float]
    point2_world: List[float]
    point1_pixel: Dict[str, float]
    point2_pixel: Dict[str, float]
    expected_distance_m: Optional[float]
    tags: List[str]


class TestCaseRunResponse(BaseModel):
    id: str
    name: str
    measure_result: MeasureResponse
    expected_distance_m: Optional[float]
    relative_error: Optional[float]
    passed: Optional[bool]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="2.0.0")


@app.get("/config", response_model=ConfigResponse)
def config():
    cam_center = _DEFAULT_POSE.camera_center_world()
    return ConfigResponse(
        camera=_DEFAULT_CAMERA.to_dict(),
        water_level_m=_DEFAULT_WATER_LEVEL,
        plane=_DEFAULT_PLANE.to_dict(),
        image_width=_DEFAULT_CAMERA.width,
        image_height=_DEFAULT_CAMERA.height,
        pose=PoseData(
            R=_DEFAULT_POSE.R.tolist(),
            t=_DEFAULT_POSE.t.tolist(),
            camera_center_world=cam_center.tolist(),
        ),
    )


@app.post("/measure", response_model=MeasureResponse)
def measure(req: MeasureRequest):
    camera = _DEFAULT_CAMERA
    pose = _DEFAULT_POSE
    water_level = req.water_level_m if req.water_level_m is not None else _DEFAULT_WATER_LEVEL
    plane = Plane.from_height(water_level)

    pixel_a = (req.point1.x, req.point1.y)
    pixel_b = (req.point2.x, req.point2.y)

    result = measure_distance_between_pixels(
        pixel_a_uv=pixel_a,
        pixel_b_uv=pixel_b,
        camera=camera,
        pose_w2c=pose,
        plane=plane,
        extrinsics_score=1.0,
    )

    cam_center = pose.camera_center_world()
    p1_world = result.point1_world.tolist() if result.point1_world is not None else None
    p2_world = result.point2_world.tolist() if result.point2_world is not None else None

    return MeasureResponse(
        success=result.success,
        distance_m=result.distance_m,
        confidence=result.confidence,
        elapsed_ms=result.elapsed_ms,
        message=result.message,
        diagnostics=dict(result.diagnostics),
        camera_center_world=cam_center.tolist(),
        camera_pose_R=pose.R.tolist(),
        point1_world=p1_world,
        point2_world=p2_world,
        water_level_m=water_level,
    )


@app.get("/test_cases", response_model=List[TestCaseScene])
def list_test_cases():
    result = []
    for tc in _TEST_CASES:
        cam: CameraParams = tc["camera"]
        pose: Pose = tc["pose"]
        center_w = pose.camera_center_world()
        result.append(TestCaseScene(
            id=tc["id"],
            name=tc["name"],
            description=tc["description"],
            category=tc["category"],
            camera=cam.to_dict(),
            pose=PoseData(
                R=pose.R.tolist(),
                t=pose.t.tolist(),
                camera_center_world=center_w.tolist(),
            ),
            plane_height_m=tc["plane_height_m"],
            point1_world=tc["point1_world"],
            point2_world=tc["point2_world"],
            point1_pixel=tc["point1_pixel"],
            point2_pixel=tc["point2_pixel"],
            expected_distance_m=tc["expected_distance_m"],
            tags=tc["tags"],
        ))
    return result


@app.post("/test_cases/{case_id}/run", response_model=TestCaseRunResponse)
def run_test_case(case_id: str):
    tc = _TEST_CASE_MAP.get(case_id)
    if tc is None:
        raise HTTPException(status_code=404, detail=f"测试用例 '{case_id}' 不存在")

    camera: CameraParams = tc["camera"]
    pose: Pose = tc["pose"]
    water_level: float = tc["plane_height_m"]
    plane = Plane.from_height(water_level)

    pixel_a = (tc["point1_pixel"]["x"], tc["point1_pixel"]["y"])
    pixel_b = (tc["point2_pixel"]["x"], tc["point2_pixel"]["y"])

    result = measure_distance_between_pixels(
        pixel_a_uv=pixel_a,
        pixel_b_uv=pixel_b,
        camera=camera,
        pose_w2c=pose,
        plane=plane,
        extrinsics_score=1.0,
    )

    cam_center = pose.camera_center_world()
    p1_world = result.point1_world.tolist() if result.point1_world is not None else None
    p2_world = result.point2_world.tolist() if result.point2_world is not None else None

    measure_resp = MeasureResponse(
        success=result.success,
        distance_m=result.distance_m,
        confidence=result.confidence,
        elapsed_ms=result.elapsed_ms,
        message=result.message,
        diagnostics=dict(result.diagnostics),
        camera_center_world=cam_center.tolist(),
        camera_pose_R=pose.R.tolist(),
        point1_world=p1_world,
        point2_world=p2_world,
        water_level_m=water_level,
    )

    expected = tc["expected_distance_m"]
    rel_error: Optional[float] = None
    passed: Optional[bool] = None
    if expected is not None and result.success and result.distance_m is not None and expected > 0:
        rel_error = abs(result.distance_m - expected) / expected
        passed = rel_error <= 0.05
    elif expected is None:
        # Degenerate case: pass if system fails or reports low confidence
        passed = (not result.success) or (result.confidence < 0.3)

    return TestCaseRunResponse(
        id=tc["id"],
        name=tc["name"],
        measure_result=measure_resp,
        expected_distance_m=expected,
        relative_error=rel_error,
        passed=passed,
    )


# Serve static frontend
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = _static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>水面测距系统演示</h1><p>前端未找到。</p>")

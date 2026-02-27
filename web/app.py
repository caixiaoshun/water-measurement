"""FastAPI backend for the water-measurement Web demo.

Endpoints:
    GET  /health   – liveness check
    GET  /config   – current camera & system configuration summary
    POST /measure  – accepts two pixel coordinates, returns distance + diagnostics
    GET  /scenarios – list all demo scenarios
    POST /scenarios/{scenario_id}/run – run all preset cases for a scenario
    GET  /         – serves the static frontend (index.html)
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
    title="Water Measurement Demo",
    description="Adaptive monocular visual ranging – Web demo",
    version="1.0.0",
)

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

# Camera looking down at a water surface at z=0
_DEFAULT_POSE = Pose(
    R=np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.6, -0.8],
        [0.0, 0.8, 0.6],
    ], dtype=np.float64),
    t=np.array([0.0, -4.8, 6.4], dtype=np.float64),
)

_DEFAULT_PLANE = Plane.from_height(0.0)
_DEFAULT_WATER_LEVEL = 0.0

_estimator_cfg = ExtrinsicsEstimatorConfig()
_estimator = ExtrinsicsEstimator(_estimator_cfg)


# ---------------------------------------------------------------------------
# Geometry helpers (mirrors test helpers)
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
# Scenario definitions
# ---------------------------------------------------------------------------

def _build_scenarios() -> List[Dict[str, Any]]:
    """Build the 6 demo scenarios with preset measurement cases."""
    scenarios: List[Dict[str, Any]] = []

    # Shared base camera (1280x720, 920px focal length)
    base_camera = _DEFAULT_CAMERA
    base_pose = _look_at_pose(
        np.array([0.0, -4.8, 6.4], dtype=np.float64),
        np.array([0.0, 10.0, 0.0], dtype=np.float64),
    )

    # --- Scenario 1: init (baseline) ---
    s1_camera = base_camera
    s1_pose = base_pose
    s1_plane = Plane.from_height(0.0)
    s1_wl = 0.0

    s1_p1a = np.array([-2.5, 10.0, 0.0], dtype=np.float64)
    s1_p1b = np.array([2.5, 10.0, 0.0], dtype=np.float64)
    s1_uv1 = _project_world_points(np.vstack([s1_p1a, s1_p1b]), s1_camera, s1_pose)

    s1_p2a = np.array([-5.0, 15.0, 0.0], dtype=np.float64)
    s1_p2b = np.array([5.0, 15.0, 0.0], dtype=np.float64)
    s1_uv2 = _project_world_points(np.vstack([s1_p2a, s1_p2b]), s1_camera, s1_pose)

    scenarios.append({
        "id": "init",
        "name": "初始化基线 / Initialization (Baseline)",
        "description": "Standard camera setup with ArUco markers, water at z=0. Validates basic measurement accuracy.",
        "camera": s1_camera,
        "pose": s1_pose,
        "plane": s1_plane,
        "water_level_m": s1_wl,
        "meta": {
            "scenario_type": "initialization",
            "marker_type": "aruco",
            "water_level_m": 0.0,
        },
        "preset_cases": [
            {
                "name": "5m baseline at y=10",
                "point1": {"x": float(s1_uv1[0, 0]), "y": float(s1_uv1[0, 1])},
                "point2": {"x": float(s1_uv1[1, 0]), "y": float(s1_uv1[1, 1])},
                "expected_distance_m": 5.0,
                "threshold_rel": 0.05,
            },
            {
                "name": "10m baseline at y=15",
                "point1": {"x": float(s1_uv2[0, 0]), "y": float(s1_uv2[0, 1])},
                "point2": {"x": float(s1_uv2[1, 0]), "y": float(s1_uv2[1, 1])},
                "expected_distance_m": 10.0,
                "threshold_rel": 0.05,
            },
        ],
    })

    # --- Scenario 2: attitude_perturb ---
    R_tilt = _euler_deg_to_matrix(3.0, 0.0, 0.0)  # 3° tilt around x-axis
    perturbed_R = R_tilt @ base_pose.R
    base_center = base_pose.camera_center_world()
    shift = np.array([0.06, 0.04, 0.032], dtype=np.float64)  # norm ≈ 0.08 m
    perturbed_center = base_center + shift
    s2_pose = Pose(R=perturbed_R, t=-perturbed_R @ perturbed_center)
    s2_camera = base_camera
    s2_plane = Plane.from_height(0.0)

    s2_p1a = np.array([-2.5, 10.0, 0.0], dtype=np.float64)
    s2_p1b = np.array([2.5, 10.0, 0.0], dtype=np.float64)
    s2_uv1 = _project_world_points(np.vstack([s2_p1a, s2_p1b]), s2_camera, s2_pose)

    s2_p2a = np.array([-5.0, 15.0, 0.0], dtype=np.float64)
    s2_p2b = np.array([5.0, 15.0, 0.0], dtype=np.float64)
    s2_uv2 = _project_world_points(np.vstack([s2_p2a, s2_p2b]), s2_camera, s2_pose)

    scenarios.append({
        "id": "attitude_perturb",
        "name": "姿态扰动 / Attitude Perturbation",
        "description": "Camera shifted 8cm and tilted 3° to simulate drift. Shows measurement error from pose mismatch.",
        "camera": s2_camera,
        "pose": s2_pose,
        "plane": s2_plane,
        "water_level_m": 0.0,
        "meta": {
            "scenario_type": "attitude_perturbation",
            "translation_shift_m": 0.08,
            "tilt_deg": 3.0,
            "marker_type": "aruco",
        },
        "preset_cases": [
            {
                "name": "5m baseline (perturbed pose)",
                "point1": {"x": float(s2_uv1[0, 0]), "y": float(s2_uv1[0, 1])},
                "point2": {"x": float(s2_uv1[1, 0]), "y": float(s2_uv1[1, 1])},
                "expected_distance_m": 5.0,
                "threshold_rel": 0.05,
            },
            {
                "name": "10m baseline (perturbed pose)",
                "point1": {"x": float(s2_uv2[0, 0]), "y": float(s2_uv2[0, 1])},
                "point2": {"x": float(s2_uv2[1, 0]), "y": float(s2_uv2[1, 1])},
                "expected_distance_m": 10.0,
                "threshold_rel": 0.05,
            },
        ],
    })

    # --- Scenario 3: aruco_fallback ---
    # Slightly different pose simulating post-correction with circle markers
    s3_center = base_center + np.array([0.01, -0.01, 0.005], dtype=np.float64)
    s3_pose = _look_at_pose(s3_center, np.array([0.0, 10.0, 0.0], dtype=np.float64))
    s3_camera = base_camera
    s3_plane = Plane.from_height(0.0)

    s3_p1a = np.array([-2.5, 10.0, 0.0], dtype=np.float64)
    s3_p1b = np.array([2.5, 10.0, 0.0], dtype=np.float64)
    s3_uv1 = _project_world_points(np.vstack([s3_p1a, s3_p1b]), s3_camera, s3_pose)

    s3_p2a = np.array([-5.0, 15.0, 0.0], dtype=np.float64)
    s3_p2b = np.array([5.0, 15.0, 0.0], dtype=np.float64)
    s3_uv2 = _project_world_points(np.vstack([s3_p2a, s3_p2b]), s3_camera, s3_pose)

    scenarios.append({
        "id": "aruco_fallback",
        "name": "ArUco失败回退圆标 / ArUco Failure + Circle Fallback",
        "description": "ArUco markers occluded; system falls back to circle markers with slightly lower accuracy.",
        "camera": s3_camera,
        "pose": s3_pose,
        "plane": s3_plane,
        "water_level_m": 0.0,
        "meta": {
            "scenario_type": "marker_fallback",
            "primary_marker": "aruco",
            "fallback_marker": "circle",
            "aruco_status": "occluded",
            "circle_status": "active",
        },
        "preset_cases": [
            {
                "name": "5m baseline (circle-corrected pose)",
                "point1": {"x": float(s3_uv1[0, 0]), "y": float(s3_uv1[0, 1])},
                "point2": {"x": float(s3_uv1[1, 0]), "y": float(s3_uv1[1, 1])},
                "expected_distance_m": 5.0,
                "threshold_rel": 0.05,
            },
            {
                "name": "10m baseline (circle-corrected pose)",
                "point1": {"x": float(s3_uv2[0, 0]), "y": float(s3_uv2[0, 1])},
                "point2": {"x": float(s3_uv2[1, 0]), "y": float(s3_uv2[1, 1])},
                "expected_distance_m": 10.0,
                "threshold_rel": 0.05,
            },
        ],
    })

    # --- Scenario 4: water_level_change ---
    s4_camera = base_camera
    s4_pose = base_pose
    s4_wl = 1.5
    s4_plane = Plane.from_height(s4_wl)

    s4_p1a = np.array([-2.5, 10.0, s4_wl], dtype=np.float64)
    s4_p1b = np.array([2.5, 10.0, s4_wl], dtype=np.float64)
    s4_uv1 = _project_world_points(np.vstack([s4_p1a, s4_p1b]), s4_camera, s4_pose)

    s4_p2a = np.array([-5.0, 15.0, s4_wl], dtype=np.float64)
    s4_p2b = np.array([5.0, 15.0, s4_wl], dtype=np.float64)
    s4_uv2 = _project_world_points(np.vstack([s4_p2a, s4_p2b]), s4_camera, s4_pose)

    scenarios.append({
        "id": "water_level_change",
        "name": "水位突变 / Water Level Change",
        "description": "Water level jumps from 0.0m to 1.5m. Points placed on the z=1.5 plane.",
        "camera": s4_camera,
        "pose": s4_pose,
        "plane": s4_plane,
        "water_level_m": s4_wl,
        "meta": {
            "scenario_type": "water_level_change",
            "water_level_m": 1.5,
            "previous_level_m": 0.0,
            "change_type": "sudden_jump",
        },
        "preset_cases": [
            {
                "name": "5m baseline at z=1.5",
                "point1": {"x": float(s4_uv1[0, 0]), "y": float(s4_uv1[0, 1])},
                "point2": {"x": float(s4_uv1[1, 0]), "y": float(s4_uv1[1, 1])},
                "expected_distance_m": 5.0,
                "threshold_rel": 0.05,
            },
            {
                "name": "10m baseline at z=1.5",
                "point1": {"x": float(s4_uv2[0, 0]), "y": float(s4_uv2[0, 1])},
                "point2": {"x": float(s4_uv2[1, 0]), "y": float(s4_uv2[1, 1])},
                "expected_distance_m": 10.0,
                "threshold_rel": 0.05,
            },
        ],
    })

    # --- Scenario 5: pixel_noise ---
    s5_camera = base_camera
    s5_pose = base_pose
    s5_plane = Plane.from_height(0.0)

    s5_p1a = np.array([-2.5, 10.0, 0.0], dtype=np.float64)
    s5_p1b = np.array([2.5, 10.0, 0.0], dtype=np.float64)
    s5_uv1_clean = _project_world_points(np.vstack([s5_p1a, s5_p1b]), s5_camera, s5_pose)
    # Apply fixed ±2px jitter (deterministic seed for reproducibility)
    rng = np.random.default_rng(42)
    s5_noise1 = rng.normal(0.0, 2.0, size=(2, 2))
    s5_uv1 = s5_uv1_clean + s5_noise1

    s5_p2a = np.array([-5.0, 15.0, 0.0], dtype=np.float64)
    s5_p2b = np.array([5.0, 15.0, 0.0], dtype=np.float64)
    s5_uv2_clean = _project_world_points(np.vstack([s5_p2a, s5_p2b]), s5_camera, s5_pose)
    s5_noise2 = rng.normal(0.0, 2.0, size=(2, 2))
    s5_uv2 = s5_uv2_clean + s5_noise2

    scenarios.append({
        "id": "pixel_noise",
        "name": "像素噪声/抖动 / Pixel Noise/Jitter",
        "description": "Standard setup with ±2px Gaussian noise on clicked points, simulating user click imprecision.",
        "camera": s5_camera,
        "pose": s5_pose,
        "plane": s5_plane,
        "water_level_m": 0.0,
        "meta": {
            "scenario_type": "pixel_noise",
            "noise_std_px": 2.0,
            "noise_type": "gaussian",
        },
        "preset_cases": [
            {
                "name": "5m baseline with ±2px noise",
                "point1": {"x": float(s5_uv1[0, 0]), "y": float(s5_uv1[0, 1])},
                "point2": {"x": float(s5_uv1[1, 0]), "y": float(s5_uv1[1, 1])},
                "expected_distance_m": 5.0,
                "threshold_rel": 0.05,
            },
            {
                "name": "10m baseline with ±2px noise",
                "point1": {"x": float(s5_uv2[0, 0]), "y": float(s5_uv2[0, 1])},
                "point2": {"x": float(s5_uv2[1, 0]), "y": float(s5_uv2[1, 1])},
                "expected_distance_m": 10.0,
                "threshold_rel": 0.05,
            },
        ],
    })

    # --- Scenario 6: degenerate_geometry ---
    s6_camera = base_camera
    s6_cam_center = np.array([0.0, 0.0, 0.5], dtype=np.float64)
    s6_target = np.array([0.0, 200.0, 0.3], dtype=np.float64)
    s6_pose = _look_at_pose(s6_cam_center, s6_target)
    s6_plane = Plane.from_height(0.0)

    # Near-horizon pixels — rays nearly parallel to plane
    s6_px1 = {"x": float(s6_camera.width // 2 - 100), "y": 10.0}
    s6_px2 = {"x": float(s6_camera.width // 2 + 100), "y": 10.0}

    scenarios.append({
        "id": "degenerate_geometry",
        "name": "退化几何 / Degenerate Geometry",
        "description": "Camera at z=0.5m looking nearly horizontally. Rays nearly parallel to water — measurement expected to fail or have very low confidence.",
        "camera": s6_camera,
        "pose": s6_pose,
        "plane": s6_plane,
        "water_level_m": 0.0,
        "meta": {
            "scenario_type": "degenerate_geometry",
            "camera_height_m": 0.5,
            "grazing_angle_deg": "~2",
            "failure_reason": "ray_nearly_parallel_to_plane",
        },
        "preset_cases": [
            {
                "name": "near-horizon points (expect failure/low confidence)",
                "point1": s6_px1,
                "point2": s6_px2,
                "expected_distance_m": None,
                "threshold_rel": 0.05,
            },
        ],
    })

    return scenarios


_SCENARIOS = _build_scenarios()
_SCENARIO_MAP: Dict[str, Dict[str, Any]] = {s["id"]: s for s in _SCENARIOS}

_DEGENERATE_CONFIDENCE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PixelPoint(BaseModel):
    x: float = Field(..., description="Pixel x coordinate")
    y: float = Field(..., description="Pixel y coordinate")


class MeasureRequest(BaseModel):
    point1: PixelPoint
    point2: PixelPoint
    water_level_m: Optional[float] = Field(None, description="Override water level (m)")
    scenario_id: Optional[str] = Field(None, description="Use a predefined scenario's camera/pose/plane")


class MeasureResponse(BaseModel):
    success: bool
    distance_m: Optional[float]
    confidence: float
    elapsed_ms: float
    message: str
    diagnostics: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigResponse(BaseModel):
    camera: Dict[str, Any]
    water_level_m: float
    plane: Dict[str, Any]
    image_width: int
    image_height: int


class ScenarioSummary(BaseModel):
    id: str
    name: str
    description: str
    meta: Dict[str, Any]


class PresetCaseResult(BaseModel):
    name: str
    point1: Dict[str, float]
    point2: Dict[str, float]
    expected_distance_m: Optional[float]
    measured_distance_m: Optional[float]
    relative_error: Optional[float]
    threshold_rel: float
    passed: bool
    success: bool
    confidence: float
    message: str


class ScenarioRunResponse(BaseModel):
    scenario_id: str
    scenario_name: str
    overall_pass: bool
    cases: List[PresetCaseResult]
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/config", response_model=ConfigResponse)
def config():
    return ConfigResponse(
        camera=_DEFAULT_CAMERA.to_dict(),
        water_level_m=_DEFAULT_WATER_LEVEL,
        plane=_DEFAULT_PLANE.to_dict(),
        image_width=_DEFAULT_CAMERA.width,
        image_height=_DEFAULT_CAMERA.height,
    )


@app.post("/measure", response_model=MeasureResponse)
def measure(req: MeasureRequest):
    # Resolve scenario overrides
    camera = _DEFAULT_CAMERA
    pose = _DEFAULT_POSE
    water_level = req.water_level_m if req.water_level_m is not None else _DEFAULT_WATER_LEVEL
    scenario_meta: Optional[Dict[str, Any]] = None

    if req.scenario_id is not None:
        scenario = _SCENARIO_MAP.get(req.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found")
        camera = scenario["camera"]
        pose = scenario["pose"]
        water_level = req.water_level_m if req.water_level_m is not None else scenario["water_level_m"]
        scenario_meta = scenario["meta"]

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

    diagnostics = dict(result.diagnostics)
    if scenario_meta is not None:
        diagnostics["scenario_meta"] = scenario_meta

    return MeasureResponse(
        success=result.success,
        distance_m=result.distance_m,
        confidence=result.confidence,
        elapsed_ms=result.elapsed_ms,
        message=result.message,
        diagnostics=diagnostics,
    )


@app.get("/scenarios", response_model=List[ScenarioSummary])
def list_scenarios():
    return [
        ScenarioSummary(
            id=s["id"],
            name=s["name"],
            description=s["description"],
            meta=s["meta"],
        )
        for s in _SCENARIOS
    ]


@app.post("/scenarios/{scenario_id}/run", response_model=ScenarioRunResponse)
def run_scenario(scenario_id: str):
    scenario = _SCENARIO_MAP.get(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")

    camera = scenario["camera"]
    pose = scenario["pose"]
    plane = scenario["plane"]
    cases_results: List[PresetCaseResult] = []
    all_pass = True

    for case in scenario["preset_cases"]:
        pixel_a = (case["point1"]["x"], case["point1"]["y"])
        pixel_b = (case["point2"]["x"], case["point2"]["y"])

        result = measure_distance_between_pixels(
            pixel_a_uv=pixel_a,
            pixel_b_uv=pixel_b,
            camera=camera,
            pose_w2c=pose,
            plane=plane,
            extrinsics_score=1.0,
        )

        expected = case["expected_distance_m"]
        threshold = case["threshold_rel"]

        if expected is None:
            # Degenerate case: pass if system reports failure or very low confidence.
            passed = (not result.success) or (result.confidence < _DEGENERATE_CONFIDENCE_THRESHOLD)
            rel_error = None
        elif result.success and result.distance_m is not None:
            rel_error = abs(result.distance_m - expected) / expected if expected > 0 else 0.0
            passed = rel_error <= threshold
        else:
            rel_error = None
            passed = False

        if not passed:
            all_pass = False

        cases_results.append(PresetCaseResult(
            name=case["name"],
            point1=case["point1"],
            point2=case["point2"],
            expected_distance_m=expected,
            measured_distance_m=result.distance_m,
            relative_error=rel_error,
            threshold_rel=threshold,
            passed=passed,
            success=result.success,
            confidence=result.confidence,
            message=result.message,
        ))

    return ScenarioRunResponse(
        scenario_id=scenario_id,
        scenario_name=scenario["name"],
        overall_pass=all_pass,
        cases=cases_results,
        meta=scenario["meta"],
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
    return HTMLResponse(content="<h1>Water Measurement Demo</h1><p>Frontend not found.</p>")

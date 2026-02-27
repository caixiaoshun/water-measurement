"""FastAPI backend for the water-measurement Web demo.

Endpoints:
    GET  /health   – liveness check
    GET  /config   – current camera & system configuration summary
    POST /measure  – accepts two pixel coordinates, returns distance + diagnostics
    GET  /         – serves the static frontend (index.html)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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
# Request / Response models
# ---------------------------------------------------------------------------

class PixelPoint(BaseModel):
    x: float = Field(..., description="Pixel x coordinate")
    y: float = Field(..., description="Pixel y coordinate")


class MeasureRequest(BaseModel):
    point1: PixelPoint
    point2: PixelPoint
    water_level_m: Optional[float] = Field(None, description="Override water level (m)")


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
    water_level = req.water_level_m if req.water_level_m is not None else _DEFAULT_WATER_LEVEL
    plane = Plane.from_height(water_level)

    pixel_a = (req.point1.x, req.point1.y)
    pixel_b = (req.point2.x, req.point2.y)

    result = measure_distance_between_pixels(
        pixel_a_uv=pixel_a,
        pixel_b_uv=pixel_b,
        camera=_DEFAULT_CAMERA,
        pose_w2c=_DEFAULT_POSE,
        plane=plane,
        extrinsics_score=1.0,
    )

    return MeasureResponse(
        success=result.success,
        distance_m=result.distance_m,
        confidence=result.confidence,
        elapsed_ms=result.elapsed_ms,
        message=result.message,
        diagnostics=result.diagnostics,
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

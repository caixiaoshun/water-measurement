"""Tests for the Web API endpoints (updated for 3D visualization + test case API).

Validates:
    - GET /health returns ok
    - GET /config returns camera configuration
    - POST /measure returns distance + 3D world coordinates
    - GET /test_cases returns all synthetic test cases with 3D scene data
    - POST /test_cases/{id}/run executes a test case and returns result + 3D data
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root and src/ are importable
_project_root = str(Path(__file__).resolve().parents[1])
_src_dir = str(Path(__file__).resolve().parents[1] / "src")
for _p in [_project_root, _src_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi.testclient import TestClient
from web.app import app

EXPECTED_TEST_CASE_IDS = [
    "baseline_5m",
    "baseline_10m",
    "long_range_50m",
    "short_range_2m",
    "water_level_1_5m",
    "pixel_noise_10m",
    "attitude_perturb",
    "barrel_distortion_10m",
    "degenerate_geometry",
]


@pytest.fixture
def client():
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestConfig:
    def test_config_returns_camera_info(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "camera" in data
        assert "water_level_m" in data
        assert "plane" in data
        assert data["image_width"] > 0
        assert data["image_height"] > 0


class TestMeasure:
    def test_measure_returns_3d_world_coords(self, client):
        resp = client.post("/measure", json={
            "point1": {"x": 640, "y": 400},
            "point2": {"x": 800, "y": 400},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "success" in data
        assert "distance_m" in data
        assert "confidence" in data
        assert "elapsed_ms" in data
        assert "camera_center_world" in data
        assert "camera_pose_R" in data
        assert "water_level_m" in data
        # 3D coordinates should be present on success
        if data["success"]:
            assert data["point1_world"] is not None
            assert data["point2_world"] is not None
            assert len(data["camera_center_world"]) == 3
            assert len(data["camera_pose_R"]) == 3  # 3x3 matrix

    def test_measure_with_water_level_override(self, client):
        resp = client.post("/measure", json={
            "point1": {"x": 640, "y": 400},
            "point2": {"x": 800, "y": 400},
            "water_level_m": 1.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["water_level_m"] == 1.5

    def test_measure_no_scenario_id_field(self, client):
        # scenario_id field was removed; sending it should be ignored gracefully
        # (extra fields are ignored by pydantic by default, or may return 422)
        resp = client.post("/measure", json={
            "point1": {"x": 640, "y": 400},
            "point2": {"x": 800, "y": 400},
            "water_level_m": 0.0,
        })
        assert resp.status_code == 200


class TestListTestCases:
    def test_returns_all_expected_test_cases(self, client):
        resp = client.get("/test_cases")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= len(EXPECTED_TEST_CASE_IDS)
        ids = [tc["id"] for tc in data]
        for expected_id in EXPECTED_TEST_CASE_IDS:
            assert expected_id in ids, f"Expected test case '{expected_id}' not found"

    def test_each_test_case_has_required_fields(self, client):
        resp = client.get("/test_cases")
        data = resp.json()
        for tc in data:
            assert "id" in tc
            assert "name" in tc
            assert "description" in tc
            assert "category" in tc
            assert "camera" in tc
            assert "pose" in tc
            assert "plane_height_m" in tc
            assert "point1_world" in tc
            assert "point2_world" in tc
            assert "point1_pixel" in tc
            assert "point2_pixel" in tc
            assert "tags" in tc
            assert len(tc["point1_world"]) == 3
            assert len(tc["point2_world"]) == 3
            assert isinstance(tc["tags"], list)

    def test_pose_data_includes_camera_center(self, client):
        resp = client.get("/test_cases")
        data = resp.json()
        for tc in data:
            pose = tc["pose"]
            assert "R" in pose
            assert "t" in pose
            assert "camera_center_world" in pose
            assert len(pose["camera_center_world"]) == 3
            assert len(pose["R"]) == 3      # 3 rows
            assert len(pose["R"][0]) == 3   # 3 cols

    def test_camera_data_includes_intrinsics(self, client):
        resp = client.get("/test_cases")
        data = resp.json()
        for tc in data:
            cam = tc["camera"]
            assert "K" in cam
            assert "D" in cam
            assert "width" in cam
            assert "height" in cam


class TestRunTestCase:
    def test_baseline_5m_passes(self, client):
        resp = client.post("/test_cases/baseline_5m/run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "baseline_5m"
        assert data["passed"] is True
        mr = data["measure_result"]
        assert mr["success"] is True
        assert mr["distance_m"] is not None
        assert abs(mr["distance_m"] - 5.0) / 5.0 <= 0.05

    def test_run_returns_3d_world_coords(self, client):
        resp = client.post("/test_cases/baseline_5m/run")
        assert resp.status_code == 200
        data = resp.json()
        mr = data["measure_result"]
        assert mr["camera_center_world"] is not None
        assert mr["camera_pose_R"] is not None
        assert mr["point1_world"] is not None
        assert mr["point2_world"] is not None
        assert len(mr["camera_center_world"]) == 3
        assert len(mr["point1_world"]) == 3
        assert len(mr["point2_world"]) == 3

    def test_degenerate_geometry_passes(self, client):
        resp = client.post("/test_cases/degenerate_geometry/run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True
        # Degenerate: expected_distance_m is None, system should fail or low confidence
        mr = data["measure_result"]
        assert not mr["success"] or mr["confidence"] < 0.3

    def test_all_test_cases_return_valid_structure(self, client):
        for cid in EXPECTED_TEST_CASE_IDS:
            resp = client.post(f"/test_cases/{cid}/run")
            assert resp.status_code == 200, f"Test case {cid} returned {resp.status_code}"
            data = resp.json()
            assert "id" in data
            assert "name" in data
            assert "measure_result" in data
            assert "expected_distance_m" in data
            mr = data["measure_result"]
            assert "success" in mr
            assert "confidence" in mr
            assert "elapsed_ms" in mr
            assert "camera_center_world" in mr

    def test_nonexistent_test_case_returns_404(self, client):
        resp = client.post("/test_cases/nonexistent_id/run")
        assert resp.status_code == 404

    def test_water_level_case_uses_correct_plane(self, client):
        resp = client.post("/test_cases/water_level_1_5m/run")
        assert resp.status_code == 200
        data = resp.json()
        mr = data["measure_result"]
        assert mr["water_level_m"] == 1.5
        assert mr["success"] is True
        assert data["passed"] is True

    def test_attitude_perturb_case_runs(self, client):
        resp = client.post("/test_cases/attitude_perturb/run")
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["id"] == "attitude_perturb"

    def test_long_range_50m_passes(self, client):
        resp = client.post("/test_cases/long_range_50m/run")
        assert resp.status_code == 200
        data = resp.json()
        mr = data["measure_result"]
        assert mr["success"] is True
        assert data["passed"] is True
        assert abs(mr["distance_m"] - 50.0) / 50.0 <= 0.05

    def test_scenarios_endpoint_removed(self, client):
        """Old /scenarios endpoint should no longer exist."""
        resp = client.get("/scenarios")
        assert resp.status_code == 404

    def test_old_scenario_run_endpoint_removed(self, client):
        """Old /scenarios/{id}/run endpoint should no longer exist."""
        resp = client.post("/scenarios/init/run")
        assert resp.status_code == 404

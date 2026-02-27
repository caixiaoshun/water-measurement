"""Tests for the Web scenario selector API endpoints.

Validates:
    - GET /scenarios returns all 6 predefined scenarios
    - POST /measure accepts optional scenario_id and returns scenario_meta
    - POST /scenarios/{id}/run executes preset cases and returns pass/fail
    - Backward compatibility: /measure without scenario_id works as before
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

EXPECTED_SCENARIO_IDS = [
    "init",
    "attitude_perturb",
    "aruco_fallback",
    "water_level_change",
    "pixel_noise",
    "degenerate_geometry",
]


@pytest.fixture
def client():
    return TestClient(app)


class TestListScenarios:
    def test_returns_all_six_scenarios(self, client):
        resp = client.get("/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 6
        ids = [s["id"] for s in data]
        for expected_id in EXPECTED_SCENARIO_IDS:
            assert expected_id in ids

    def test_each_scenario_has_required_fields(self, client):
        resp = client.get("/scenarios")
        data = resp.json()
        for s in data:
            assert "id" in s
            assert "name" in s
            assert "description" in s
            assert "meta" in s
            assert isinstance(s["meta"], dict)
            assert len(s["name"]) > 0
            assert len(s["description"]) > 0


class TestMeasureWithScenario:
    def test_measure_without_scenario_backward_compatible(self, client):
        resp = client.post("/measure", json={
            "point1": {"x": 640, "y": 400},
            "point2": {"x": 800, "y": 400},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "scenario_meta" not in data["diagnostics"]

    def test_measure_with_scenario_includes_meta(self, client):
        resp = client.post("/measure", json={
            "point1": {"x": 640, "y": 400},
            "point2": {"x": 800, "y": 400},
            "scenario_id": "init",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "scenario_meta" in data["diagnostics"]
        assert data["diagnostics"]["scenario_meta"]["scenario_type"] == "initialization"

    def test_measure_with_water_level_scenario(self, client):
        resp = client.post("/measure", json={
            "point1": {"x": 640, "y": 400},
            "point2": {"x": 800, "y": 400},
            "scenario_id": "water_level_change",
        })
        assert resp.status_code == 200
        data = resp.json()
        meta = data["diagnostics"]["scenario_meta"]
        assert meta["water_level_m"] == 1.5
        assert meta["change_type"] == "sudden_jump"

    def test_measure_with_invalid_scenario_returns_404(self, client):
        resp = client.post("/measure", json={
            "point1": {"x": 640, "y": 400},
            "point2": {"x": 800, "y": 400},
            "scenario_id": "nonexistent",
        })
        assert resp.status_code == 404


class TestRunScenario:
    def test_init_scenario_passes(self, client):
        resp = client.post("/scenarios/init/run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenario_id"] == "init"
        assert data["overall_pass"] is True
        assert len(data["cases"]) == 2
        for c in data["cases"]:
            assert c["passed"] is True

    def test_degenerate_geometry_passes(self, client):
        resp = client.post("/scenarios/degenerate_geometry/run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_pass"] is True
        case = data["cases"][0]
        assert case["passed"] is True
        # Degenerate: either fails or has very low confidence
        assert not case["success"] or case["confidence"] < 0.3

    def test_all_scenarios_return_valid_structure(self, client):
        for sid in EXPECTED_SCENARIO_IDS:
            resp = client.post(f"/scenarios/{sid}/run")
            assert resp.status_code == 200, f"Scenario {sid} failed"
            data = resp.json()
            assert "scenario_id" in data
            assert "scenario_name" in data
            assert "overall_pass" in data
            assert "cases" in data
            assert "meta" in data
            assert isinstance(data["cases"], list)
            assert len(data["cases"]) >= 1
            for c in data["cases"]:
                assert "name" in c
                assert "passed" in c
                assert "success" in c
                assert "confidence" in c

    def test_run_nonexistent_scenario_returns_404(self, client):
        resp = client.post("/scenarios/nonexistent/run")
        assert resp.status_code == 404

    def test_scenario_meta_matches_definition(self, client):
        resp = client.post("/scenarios/pixel_noise/run")
        data = resp.json()
        assert data["meta"]["scenario_type"] == "pixel_noise"
        assert data["meta"]["noise_std_px"] == 2.0

    def test_attitude_perturb_scenario_runs(self, client):
        resp = client.post("/scenarios/attitude_perturb/run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["meta"]["scenario_type"] == "attitude_perturbation"
        assert data["meta"]["tilt_deg"] == 3.0

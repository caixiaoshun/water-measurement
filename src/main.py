from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

from vision_ranging import (
    CameraParams,
    ExtrinsicsEstimator,
    ExtrinsicsEstimatorConfig,
    Plane,
    Pose,
    detect_markers,
    load_camera_params,
    measure_distance_between_pixels,
    monotonic_ms,
)
from water_level import WaterLevelFusion, create_water_level_source

LOGGER = logging.getLogger("water-measurement")


class FrameSource:
    def __init__(self, source: str) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        self.images: list[Path] = []
        self.idx = 0

        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera id: {source}")
            return

        path = Path(source)
        if path.is_dir():
            files: list[Path] = []
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files.extend(path.glob(pattern))
            self.images = sorted(files)
            if not self.images:
                raise RuntimeError(f"No image files found in directory: {source}")
            return

        if path.exists():
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {source}")
            return

        raise RuntimeError(f"Unsupported input source: {source}")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.cap is not None:
            ok, frame = self.cap.read()
            return bool(ok), frame
        if self.idx >= len(self.images):
            return False, None
        frame = cv2.imread(str(self.images[self.idx]))
        self.idx += 1
        return frame is not None, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class ClickCollector:
    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []
        self.pending = False

    def on_mouse(self, event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        _ = flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x), int(y)))
            if len(self.points) > 2:
                self.points = self.points[-2:]
            self.pending = len(self.points) == 2
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clear()

    def clear(self) -> None:
        self.points.clear()
        self.pending = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a dict")
    return cfg


def setup_logging(cfg: dict[str, Any]) -> None:
    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_file = cfg.get("file")
    if log_file:
        p = Path(str(log_file))
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def parse_camera(cfg: dict[str, Any]) -> CameraParams:
    c = cfg.get("camera", {})
    params_file = c.get("params_file")
    if params_file:
        return load_camera_params(str(params_file))

    required = {"K", "D", "width", "height"}
    missing = required - set(c.keys())
    if missing:
        raise ValueError(f"camera config missing fields: {sorted(missing)}")
    return CameraParams.from_dict(c)


def parse_plane(cfg: dict[str, Any]) -> Plane:
    p = cfg.get("water_plane", {"type": "height", "height_m": 0.0})
    mode = str(p.get("type", "height")).lower()
    if mode == "height":
        return Plane.from_height(float(p.get("height_m", 0.0)))
    if mode == "general":
        return Plane(normal=np.asarray(p["normal"], dtype=np.float64), d=float(p["d"]))
    raise ValueError(f"Unsupported water plane type: {mode}")


def parse_extrinsics_cfg(cfg: dict[str, Any]) -> ExtrinsicsEstimatorConfig:
    e = cfg.get("extrinsics", {})
    return ExtrinsicsEstimatorConfig(
        ransac_reprojection_error_px=float(e.get("ransac_reprojection_error_px", 4.0)),
        ransac_confidence=float(e.get("ransac_confidence", 0.995)),
        ransac_iterations=int(e.get("ransac_iterations", 200)),
        min_inliers=int(e.get("min_inliers", 6)),
        min_inlier_ratio=float(e.get("min_inlier_ratio", 0.5)),
        max_reprojection_error_px=float(e.get("max_reprojection_error_px", 6.0)),
        smoothing_alpha=float(e.get("smoothing_alpha", 0.35)),
        max_rotation_jump_deg=float(e.get("max_rotation_jump_deg", 20.0)),
        max_translation_jump_m=float(e.get("max_translation_jump_m", 2.0)),
    )


def draw_overlay(frame: np.ndarray, clicks: ClickCollector, pose: Optional[Pose], status: str) -> None:
    for p in clicks.points:
        cv2.circle(frame, p, 5, (0, 255, 255), 2)
    if len(clicks.points) == 2:
        cv2.line(frame, clicks.points[0], clicks.points[1], (0, 200, 255), 2)

    pose_text = "pose=available" if pose is not None else "pose=none"
    cv2.putText(frame, pose_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 255, 30), 2)
    cv2.putText(frame, status, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 220, 255), 2)
    cv2.putText(
        frame,
        "L-click: select 2 points | R-click/C: clear | Q: quit",
        (12, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
    )


def run(config_path: str, input_override: Optional[str] = None) -> int:
    cfg = load_config(config_path)
    setup_logging(cfg.get("logging", {}))

    camera = parse_camera(cfg)
    base_plane = parse_plane(cfg)

    water_source = create_water_level_source(cfg.get("water_level", {"type": "static", "height_m": 0.0}))
    water_fusion = WaterLevelFusion(source=water_source, base_plane=base_plane)

    marker_cfg = cfg.get("marker", {})
    estimator = ExtrinsicsEstimator(parse_extrinsics_cfg(cfg))

    source = str(input_override if input_override is not None else cfg.get("input", {}).get("source", "0"))
    frame_source = FrameSource(source)

    runtime = cfg.get("runtime", {})
    parallel_threshold = float(runtime.get("parallel_threshold", 1e-8))
    wait_ms = int(runtime.get("display_wait_ms", 1))
    clear_after_measure = bool(runtime.get("clear_points_after_measure", True))

    output_cfg = cfg.get("output", {})
    save_dir = output_cfg.get("save_annotated_dir")

    window = "water-measurement"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    clicks = ClickCollector()
    cv2.setMouseCallback(window, clicks.on_mouse)

    pose: Optional[Pose] = None
    frame_idx = 0
    status = "waiting for markers"

    while True:
        ok, frame = frame_source.read()
        if not ok or frame is None:
            LOGGER.info("End of stream")
            break

        frame_idx += 1
        loop_start = monotonic_ms()

        try:
            observation = detect_markers(frame, marker_cfg)
        except Exception as exc:
            observation = None
            LOGGER.warning("Marker detection failed: %s", exc)

        update = estimator.estimate(observation, camera, previous_pose=pose)
        pose = update.pose

        status = (
            f"extrinsics={update.quality.status} | score={update.quality.score:.2f} "
            f"| reproj={update.quality.reprojection_error_px:.2f}px"
        )

        if clicks.pending and len(clicks.points) == 2:
            if pose is None:
                LOGGER.warning("Measurement skipped: no valid pose")
                clicks.pending = False
            else:
                plane, level = water_fusion.current_plane(reference_time=utc_now())
                measurement = measure_distance_between_pixels(
                    clicks.points[0],
                    clicks.points[1],
                    camera,
                    pose,
                    plane,
                    extrinsics_score=update.quality.score,
                    parallel_threshold=parallel_threshold,
                )

                payload = {
                    "distance_m": measurement.distance_m,
                    "confidence": measurement.confidence,
                    "elapsed_ms": measurement.elapsed_ms,
                    "message": measurement.message,
                    "extrinsics": update.quality.to_dict(),
                    "plane": plane.to_dict(),
                    "water_level": {
                        "height_m": level.height_m,
                        "timestamp": level.timestamp.isoformat(),
                        "source": level.source,
                    },
                }
                print(json.dumps(payload, ensure_ascii=False))
                LOGGER.info("measurement=%s", json.dumps(payload, ensure_ascii=False))

                if save_dir:
                    out_dir = Path(str(save_dir))
                    out_dir.mkdir(parents=True, exist_ok=True)
                    annotated = frame.copy()
                    draw_overlay(annotated, clicks, pose, status)
                    label = "NA" if measurement.distance_m is None else f"{measurement.distance_m:.3f} m"
                    cv2.putText(annotated, label, clicks.points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.png"), annotated)

                clicks.pending = False
                if clear_after_measure:
                    clicks.clear()

        loop_ms = monotonic_ms() - loop_start
        draw_frame = frame.copy()
        draw_overlay(draw_frame, clicks, pose, f"{status} | frame_ms={loop_ms:.1f}")
        cv2.imshow(window, draw_frame)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("c"), ord("C")):
            clicks.clear()

    frame_source.release()
    cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive monocular visual ranging demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--input", type=str, default=None, help="Override input source")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run(args.config, args.input)


if __name__ == "__main__":
    raise SystemExit(main())

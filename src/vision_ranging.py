from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np


@dataclass
class CameraParams:
    K: np.ndarray
    D: np.ndarray
    width: int
    height: int
    name: str = "unknown"
    model: str = "pinhole"

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(self.D, dtype=np.float64).reshape(-1)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width/height must be positive")
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("Camera focal length must be positive")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraParams":
        return cls(
            K=np.asarray(data["K"], dtype=np.float64),
            D=np.asarray(data.get("D", []), dtype=np.float64),
            width=int(data["width"]),
            height=int(data["height"]),
            name=str(data.get("name", "unknown")),
            model=str(data.get("model", "pinhole")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "width": int(self.width),
            "height": int(self.height),
            "K": self.K.tolist(),
            "D": self.D.tolist(),
        }


@dataclass
class Plane:
    # Plane form: n^T X + d = 0
    normal: np.ndarray
    d: float

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            raise ValueError("Plane normal is near zero")
        self.normal = n / norm
        self.d = float(self.d) / norm

    @classmethod
    def from_height(cls, height_m: float) -> "Plane":
        # z = h  => [0, 0, 1]^T X - h = 0
        return cls(normal=np.array([0.0, 0.0, 1.0], dtype=np.float64), d=-float(height_m))

    def as_height(self, tol: float = 1e-6) -> Optional[float]:
        if abs(self.normal[0]) <= tol and abs(self.normal[1]) <= tol and abs(self.normal[2] - 1.0) <= tol:
            return -self.d
        return None

    def with_height(self, height_m: float) -> "Plane":
        if self.as_height() is not None:
            return Plane.from_height(height_m)
        return Plane(normal=self.normal.copy(), d=-float(height_m))

    def to_dict(self) -> Dict[str, Any]:
        return {"normal": self.normal.tolist(), "d": float(self.d)}


@dataclass
class Pose:
    # World to camera pose: Xc = R * Xw + t
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    @classmethod
    def identity(cls) -> "Pose":
        return cls(R=np.eye(3, dtype=np.float64), t=np.zeros(3, dtype=np.float64))

    def camera_center_world(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class MarkerObservation:
    object_points: np.ndarray
    image_points: np.ndarray
    ids: np.ndarray
    marker_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.object_points = np.asarray(self.object_points, dtype=np.float64).reshape(-1, 3)
        self.image_points = np.asarray(self.image_points, dtype=np.float64).reshape(-1, 2)
        self.ids = np.asarray(self.ids, dtype=np.int32).reshape(-1)
        if self.object_points.shape[0] != self.image_points.shape[0]:
            raise ValueError("object_points and image_points size mismatch")


@dataclass
class ExtrinsicsQuality:
    success: bool
    inlier_count: int
    inlier_ratio: float
    reprojection_error_px: float
    score: float
    status: str
    used_previous_pose: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "inlier_count": int(self.inlier_count),
            "inlier_ratio": float(self.inlier_ratio),
            "reprojection_error_px": float(self.reprojection_error_px),
            "score": float(self.score),
            "status": self.status,
            "used_previous_pose": bool(self.used_previous_pose),
        }


@dataclass
class ExtrinsicsUpdate:
    pose: Optional[Pose]
    quality: ExtrinsicsQuality


@dataclass
class Measurement:
    success: bool
    distance_m: Optional[float]
    confidence: float
    elapsed_ms: float
    message: str
    point1_world: Optional[np.ndarray] = None
    point2_world: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "distance_m": None if self.distance_m is None else float(self.distance_m),
            "confidence": float(self.confidence),
            "elapsed_ms": float(self.elapsed_ms),
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


@dataclass
class ExtrinsicsEstimatorConfig:
    ransac_reprojection_error_px: float = 4.0
    ransac_confidence: float = 0.995
    ransac_iterations: int = 200
    min_inliers: int = 6
    min_inlier_ratio: float = 0.5
    max_reprojection_error_px: float = 6.0
    smoothing_alpha: float = 0.35
    max_rotation_jump_deg: float = 20.0
    max_translation_jump_m: float = 2.0


def monotonic_ms() -> float:
    return cv2.getTickCount() * 1000.0 / cv2.getTickFrequency()


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return arr / norm


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    rel = np.asarray(R1, dtype=np.float64) @ np.asarray(R2, dtype=np.float64).T
    c = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def load_camera_params(path: str) -> CameraParams:
    import json
    from pathlib import Path

    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Camera params file not found: {path}")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported camera params format: {p.suffix}")
    return CameraParams.from_dict(data)


def undistort_points(points_uv: Sequence[Sequence[float]], camera: CameraParams) -> np.ndarray:
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, camera.K, camera.D, P=camera.K)
    return out.reshape(-1, 2)


def pixel_to_camera_ray(point_uv: Sequence[float], camera: CameraParams) -> np.ndarray:
    pt = np.asarray(point_uv, dtype=np.float64).reshape(1, 1, 2)
    uvn = cv2.undistortPoints(pt, camera.K, camera.D).reshape(2)
    return normalize(np.array([uvn[0], uvn[1], 1.0], dtype=np.float64))


def ray_plane_intersection(
    ray_origin_w: np.ndarray,
    ray_dir_w: np.ndarray,
    plane: Plane,
    parallel_threshold: float = 1e-8,
) -> tuple[bool, Optional[np.ndarray], str]:
    origin = np.asarray(ray_origin_w, dtype=np.float64).reshape(3)
    direction = np.asarray(ray_dir_w, dtype=np.float64).reshape(3)
    denom = float(plane.normal @ direction)

    if abs(denom) < parallel_threshold:
        return False, None, "ray_parallel_to_plane"

    lam = -(float(plane.normal @ origin) + float(plane.d)) / denom
    if lam < 0:
        return False, None, "intersection_behind_camera"

    return True, origin + lam * direction, "ok"


def pixel_to_world_on_plane(
    pixel_uv: Sequence[float],
    camera: CameraParams,
    pose_w2c: Pose,
    plane: Plane,
    parallel_threshold: float = 1e-8,
) -> tuple[bool, Optional[np.ndarray], str, float]:
    ray_c = pixel_to_camera_ray(pixel_uv, camera)
    ray_w = pose_w2c.R.T @ ray_c
    cam_center_w = pose_w2c.camera_center_world()

    ok, point_w, status = ray_plane_intersection(cam_center_w, ray_w, plane, parallel_threshold)
    grazing = abs(float(plane.normal @ ray_w))
    return ok, point_w, status, grazing


def measure_distance_between_pixels(
    pixel_a_uv: Sequence[float],
    pixel_b_uv: Sequence[float],
    camera: CameraParams,
    pose_w2c: Pose,
    plane: Plane,
    extrinsics_score: float = 1.0,
    parallel_threshold: float = 1e-8,
) -> Measurement:
    t0 = monotonic_ms()

    ok1, p1_w, s1, g1 = pixel_to_world_on_plane(pixel_a_uv, camera, pose_w2c, plane, parallel_threshold)
    ok2, p2_w, s2, g2 = pixel_to_world_on_plane(pixel_b_uv, camera, pose_w2c, plane, parallel_threshold)

    elapsed = monotonic_ms() - t0
    if not ok1 or not ok2:
        return Measurement(
            success=False,
            distance_m=None,
            confidence=0.0,
            elapsed_ms=elapsed,
            message=s1 if not ok1 else s2,
            point1_world=p1_w,
            point2_world=p2_w,
            diagnostics={
                "point1_status": s1,
                "point2_status": s2,
                "plane": plane.to_dict(),
                "grazing": [g1, g2],
            },
        )

    dist = float(np.linalg.norm(p1_w - p2_w))
    geom_score = clamp(min(g1, g2) / 0.2, 0.0, 1.0)
    conf = clamp(0.6 * geom_score + 0.4 * extrinsics_score, 0.0, 1.0)

    return Measurement(
        success=True,
        distance_m=dist,
        confidence=conf,
        elapsed_ms=elapsed,
        message="ok",
        point1_world=p1_w,
        point2_world=p2_w,
        diagnostics={
            "plane": plane.to_dict(),
            "grazing": [g1, g2],
            "extrinsics_score": float(extrinsics_score),
        },
    )


def _aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module unavailable. Please install opencv-contrib-python.")
    key = name.strip().upper()
    mapping = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(mapping[key])


def _build_aruco_world_map(cfg: Dict[str, Any]) -> Dict[int, np.ndarray]:
    explicit = cfg.get("world_corners_by_id")
    if explicit:
        out: Dict[int, np.ndarray] = {}
        for k, v in explicit.items():
            out[int(k)] = np.asarray(v, dtype=np.float64).reshape(4, 3)
        return out

    rows = int(cfg.get("grid_rows", 0))
    cols = int(cfg.get("grid_cols", 0))
    marker_length = float(cfg.get("marker_length_m", 0.0))
    marker_sep = float(cfg.get("marker_separation_m", 0.0))
    origin = np.asarray(cfg.get("origin_xyz", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    first_id = int(cfg.get("first_marker_id", 0))

    if rows <= 0 or cols <= 0 or marker_length <= 0:
        raise ValueError("ArUco config must define world_corners_by_id or valid grid_rows/grid_cols/marker_length_m")

    step = marker_length + marker_sep
    out: Dict[int, np.ndarray] = {}
    marker_id = first_id
    for r in range(rows):
        for c in range(cols):
            x0 = origin[0] + c * step
            y0 = origin[1] + r * step
            z0 = origin[2]
            out[marker_id] = np.array(
                [
                    [x0, y0, z0],
                    [x0 + marker_length, y0, z0],
                    [x0 + marker_length, y0 + marker_length, z0],
                    [x0, y0 + marker_length, z0],
                ],
                dtype=np.float64,
            )
            marker_id += 1
    return out


def detect_aruco_markers(frame: np.ndarray, cfg: Dict[str, Any]) -> Optional[MarkerObservation]:
    dictionary = _aruco_dictionary(str(cfg.get("dictionary", "DICT_4X4_50")))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    world_map = _build_aruco_world_map(cfg)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    point_ids: list[int] = []

    for corners, marker_id_arr in zip(corners_list, ids.reshape(-1), strict=False):
        marker_id = int(marker_id_arr)
        if marker_id not in world_map:
            continue
        object_points.append(world_map[marker_id])
        image_points.append(np.asarray(corners, dtype=np.float64).reshape(4, 2))
        point_ids.extend([marker_id] * 4)

    if not object_points:
        return None

    return MarkerObservation(
        object_points=np.vstack(object_points),
        image_points=np.vstack(image_points),
        ids=np.asarray(point_ids, dtype=np.int32),
        marker_type="aruco",
        metadata={"detected_marker_count": int(len(object_points))},
    )


def _sort_xy(points: np.ndarray) -> np.ndarray:
    idx = np.lexsort((points[:, 1], points[:, 0]))
    return points[idx]


def detect_circle_markers(frame: np.ndarray, cfg: Dict[str, Any]) -> Optional[MarkerObservation]:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = int(cfg.get("blur_kernel", 5))
    if blur % 2 == 0:
        blur += 1
    gray = cv2.GaussianBlur(gray, (blur, blur), 1.2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(cfg.get("dp", 1.2)),
        minDist=float(cfg.get("min_dist_px", 20.0)),
        param1=float(cfg.get("param1", 120.0)),
        param2=float(cfg.get("param2", 18.0)),
        minRadius=int(cfg.get("min_radius_px", 3)),
        maxRadius=int(cfg.get("max_radius_px", 200)),
    )
    if circles is None:
        return None

    world_points = np.asarray(cfg.get("world_points", []), dtype=np.float64).reshape(-1, 3)
    if world_points.size == 0:
        raise ValueError("circle marker config requires world_points")

    expected_count = int(cfg.get("expected_count", world_points.shape[0]))
    centers = circles[0][:, :2]
    if centers.shape[0] < expected_count:
        return None

    centers_sorted = _sort_xy(centers)[:expected_count]
    order = np.lexsort((world_points[:, 1], world_points[:, 0]))
    world_sorted = world_points[order][:expected_count]

    return MarkerObservation(
        object_points=world_sorted,
        image_points=centers_sorted,
        ids=np.arange(expected_count, dtype=np.int32),
        marker_type="circle",
        metadata={"detected_circle_count": int(centers.shape[0])},
    )


def detect_markers(frame: np.ndarray, cfg: Dict[str, Any]) -> Optional[MarkerObservation]:
    marker_type = str(cfg.get("type", "aruco")).lower()
    if marker_type == "aruco":
        return detect_aruco_markers(frame, cfg)
    if marker_type == "circle":
        return detect_circle_markers(frame, cfg)
    raise ValueError(f"Unsupported marker type: {marker_type}")


class BaseMarkerDetector:
    """Base marker detector abstraction for ordered fallback."""

    name: str = "base"

    def detect(self, frame: np.ndarray, cfg: Dict[str, Any]) -> Optional[MarkerObservation]:
        raise NotImplementedError


class ArucoMarkerDetector(BaseMarkerDetector):
    name = "aruco"

    def detect(self, frame: Any, cfg: Dict[str, Any]) -> Optional[MarkerObservation]:
        if isinstance(frame, dict):
            payload = frame.get("synthetic_aruco_observation")
            if payload is None:
                return None
            if isinstance(payload, MarkerObservation):
                return payload
            return MarkerObservation(
                object_points=np.asarray(payload["object_points"], dtype=np.float64),
                image_points=np.asarray(payload["image_points"], dtype=np.float64),
                ids=np.asarray(payload["ids"], dtype=np.int32),
                marker_type="aruco",
                metadata=dict(payload.get("metadata", {})),
            )
        return detect_aruco_markers(frame, cfg)


class CircleMarkerDetector(BaseMarkerDetector):
    name = "circle"

    def detect(self, frame: Any, cfg: Dict[str, Any]) -> Optional[MarkerObservation]:
        if isinstance(frame, dict):
            payload = frame.get("synthetic_circle_observation")
            if payload is None:
                return None
            if isinstance(payload, MarkerObservation):
                return payload
            return MarkerObservation(
                object_points=np.asarray(payload["object_points"], dtype=np.float64),
                image_points=np.asarray(payload["image_points"], dtype=np.float64),
                ids=np.asarray(payload["ids"], dtype=np.int32),
                marker_type="circle",
                metadata=dict(payload.get("metadata", {})),
            )
        return detect_circle_markers(frame, cfg)


def _marker_cfg_for_type(marker_cfg: Dict[str, Any], marker_type: str) -> Dict[str, Any]:
    cfg = dict(marker_cfg)
    type_specific = marker_cfg.get(marker_type)
    if isinstance(type_specific, dict):
        cfg.update(type_specific)
    cfg["type"] = marker_type
    return cfg


def build_marker_detectors(marker_cfg: Dict[str, Any]) -> list[BaseMarkerDetector]:
    primary_type = str(marker_cfg.get("type", "aruco")).lower()
    fallback_types = [str(x).lower() for x in marker_cfg.get("fallback_types", [])]

    ordered_types = [primary_type] + [t for t in fallback_types if t != primary_type]
    detectors: list[BaseMarkerDetector] = []
    for marker_type in ordered_types:
        if marker_type == "aruco":
            detectors.append(ArucoMarkerDetector())
        elif marker_type == "circle":
            detectors.append(CircleMarkerDetector())
        else:
            raise ValueError(f"Unsupported marker type in detector list: {marker_type}")
    return detectors


def detect_markers_with_fallback(
    frame: np.ndarray,
    marker_cfg: Dict[str, Any],
    detectors: Optional[list[BaseMarkerDetector]] = None,
) -> tuple[Optional[MarkerObservation], Optional[str], list[str]]:
    detector_list = detectors if detectors is not None else build_marker_detectors(marker_cfg)
    attempted: list[str] = []

    for detector in detector_list:
        attempted.append(detector.name)
        cfg = _marker_cfg_for_type(marker_cfg, detector.name)
        obs = detector.detect(frame, cfg)
        if obs is not None:
            return obs, detector.name, attempted

    return None, None, attempted


class ExtrinsicsEstimator:
    def __init__(self, cfg: ExtrinsicsEstimatorConfig) -> None:
        self.cfg = cfg

    def estimate(
        self,
        observation: Optional[MarkerObservation],
        camera: CameraParams,
        previous_pose: Optional[Pose] = None,
    ) -> ExtrinsicsUpdate:
        if observation is None:
            return self._fallback("no_marker_observation", previous_pose)
        if observation.object_points.shape[0] < 4:
            return self._fallback("insufficient_correspondences", previous_pose)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            observation.object_points,
            observation.image_points,
            camera.K,
            camera.D,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self.cfg.ransac_reprojection_error_px,
            confidence=self.cfg.ransac_confidence,
            iterationsCount=self.cfg.ransac_iterations,
        )
        if not ok:
            return self._fallback("pnp_failed", previous_pose)

        R, _ = cv2.Rodrigues(rvec)
        raw_pose = Pose(R=R, t=tvec.reshape(3))

        reproj_error, inlier_count, inlier_ratio = self._reprojection_metrics(
            observation.object_points,
            observation.image_points,
            rvec,
            tvec,
            inliers,
            camera,
        )

        if inlier_count < self.cfg.min_inliers:
            return self._fallback("inliers_below_threshold", previous_pose, reproj_error, inlier_count, inlier_ratio)
        if inlier_ratio < self.cfg.min_inlier_ratio:
            return self._fallback("inlier_ratio_below_threshold", previous_pose, reproj_error, inlier_count, inlier_ratio)
        if reproj_error > self.cfg.max_reprojection_error_px:
            return self._fallback("reprojection_error_too_large", previous_pose, reproj_error, inlier_count, inlier_ratio)

        if previous_pose is not None:
            rot_jump = rotation_angle_deg(raw_pose.R, previous_pose.R)
            trans_jump = float(np.linalg.norm(raw_pose.t - previous_pose.t))
            if rot_jump > self.cfg.max_rotation_jump_deg or trans_jump > self.cfg.max_translation_jump_m:
                return self._fallback("pose_jump_too_large", previous_pose, reproj_error, inlier_count, inlier_ratio)
            pose = self._smooth_pose(previous_pose, raw_pose)
        else:
            pose = raw_pose

        quality = ExtrinsicsQuality(
            success=True,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            reprojection_error_px=reproj_error,
            score=self._quality_score(reproj_error, inlier_ratio),
            status="ok",
            used_previous_pose=False,
        )
        return ExtrinsicsUpdate(pose=pose, quality=quality)

    def _smooth_pose(self, prev: Pose, curr: Pose) -> Pose:
        alpha = clamp(self.cfg.smoothing_alpha, 0.0, 1.0)
        if alpha <= 0.0:
            return prev
        if alpha >= 1.0:
            return curr

        prev_rvec, _ = cv2.Rodrigues(prev.R)
        curr_rvec, _ = cv2.Rodrigues(curr.R)
        rvec = (1.0 - alpha) * prev_rvec + alpha * curr_rvec
        R, _ = cv2.Rodrigues(rvec)
        t = (1.0 - alpha) * prev.t + alpha * curr.t
        return Pose(R=R, t=t)

    def _quality_score(self, reproj_error_px: float, inlier_ratio: float) -> float:
        err_score = float(np.exp(-max(reproj_error_px, 0.0) / max(self.cfg.max_reprojection_error_px, 1e-6)))
        return clamp(0.5 * err_score + 0.5 * inlier_ratio, 0.0, 1.0)

    @staticmethod
    def _reprojection_metrics(
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        inliers: Optional[np.ndarray],
        camera: CameraParams,
    ) -> tuple[float, int, float]:
        proj, _ = cv2.projectPoints(object_points, rvec, tvec, camera.K, camera.D)
        proj = proj.reshape(-1, 2)
        errors = np.linalg.norm(proj - image_points, axis=1)

        if inliers is None or len(inliers) == 0:
            idx = np.arange(object_points.shape[0], dtype=np.int32)
        else:
            idx = inliers.reshape(-1)

        inlier_errors = errors[idx]
        reproj = float(np.mean(inlier_errors)) if inlier_errors.size else float("inf")
        count = int(idx.size)
        ratio = float(count / max(object_points.shape[0], 1))
        return reproj, count, ratio

    def _fallback(
        self,
        status: str,
        previous_pose: Optional[Pose],
        reproj_error: float = float("inf"),
        inlier_count: int = 0,
        inlier_ratio: float = 0.0,
    ) -> ExtrinsicsUpdate:
        quality = ExtrinsicsQuality(
            success=False,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            reprojection_error_px=reproj_error,
            score=0.0,
            status=status,
            used_previous_pose=previous_pose is not None,
        )
        return ExtrinsicsUpdate(pose=previous_pose, quality=quality)


def auto_update_pose(
    frame: np.ndarray,
    camera: CameraParams,
    previous_pose: Optional[Pose],
    marker_cfg: Dict[str, Any],
    estimator: ExtrinsicsEstimator,
    log: Optional[Dict[str, Any]] = None,
    detectors: Optional[list[BaseMarkerDetector]] = None,
) -> tuple[bool, Optional[Pose], ExtrinsicsUpdate]:
    observation, marker_type, attempted = detect_markers_with_fallback(frame, marker_cfg, detectors=detectors)
    update = estimator.estimate(observation, camera, previous_pose=previous_pose)

    if log is not None:
        log.clear()
        log.update(
            {
                "marker_type": marker_type,
                "attempted_detectors": attempted,
                "pose_update_success": bool(update.quality.success),
                "pose_quality": update.quality.score,
                "status": update.quality.status,
                "reprojection_error_px": update.quality.reprojection_error_px,
                "inlier_count": update.quality.inlier_count,
                "inlier_ratio": update.quality.inlier_ratio,
                "used_previous_pose": update.quality.used_previous_pose,
            }
        )

    return bool(update.quality.success), update.pose, update

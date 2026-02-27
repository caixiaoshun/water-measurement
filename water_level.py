from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

from vision_ranging import Plane


@dataclass
class WaterLevelReading:
    height_m: float
    timestamp: datetime
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


class WaterLevelSource(Protocol):
    def get_latest(self, reference_time: Optional[datetime] = None) -> WaterLevelReading:
        ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class StaticWaterLevelSource:
    height_m: float

    def get_latest(self, reference_time: Optional[datetime] = None) -> WaterLevelReading:
        ts = reference_time if reference_time is not None else utc_now()
        return WaterLevelReading(height_m=float(self.height_m), timestamp=ts, source="static")


class InMemoryWaterLevelSource:
    def __init__(self, readings: Iterable[WaterLevelReading], hold_last: bool = True, time_offset_sec: float = 0.0) -> None:
        self.readings: List[WaterLevelReading] = sorted(list(readings), key=lambda x: x.timestamp)
        if not self.readings:
            raise ValueError("Water level readings are empty")
        self.hold_last = hold_last
        self.time_offset_sec = float(time_offset_sec)

    def get_latest(self, reference_time: Optional[datetime] = None) -> WaterLevelReading:
        ref = reference_time if reference_time is not None else utc_now()
        ref = ref + timedelta(seconds=self.time_offset_sec)

        candidates = [r for r in self.readings if r.timestamp <= ref]
        if candidates:
            latest = candidates[-1]
            return WaterLevelReading(height_m=latest.height_m, timestamp=latest.timestamp, source=latest.source)

        if self.hold_last:
            first = self.readings[0]
            return WaterLevelReading(height_m=first.height_m, timestamp=first.timestamp, source=first.source)

        future = [r for r in self.readings if r.timestamp > ref]
        chosen = future[0] if future else self.readings[-1]
        return WaterLevelReading(height_m=chosen.height_m, timestamp=chosen.timestamp, source=chosen.source)


class FileWaterLevelSource(InMemoryWaterLevelSource):
    def __init__(self, file_path: str, hold_last: bool = True, time_offset_sec: float = 0.0) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Water level file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            readings = self._load_csv(path)
        elif suffix in {".json", ".jsonl"}:
            readings = self._load_json(path)
        else:
            raise ValueError(f"Unsupported water level file format: {suffix}")

        super().__init__(readings=readings, hold_last=hold_last, time_offset_sec=time_offset_sec)

    @staticmethod
    def _load_csv(path: Path) -> List[WaterLevelReading]:
        readings: List[WaterLevelReading] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                readings.append(
                    WaterLevelReading(
                        height_m=float(row["height_m"]),
                        timestamp=parse_iso_datetime(row["timestamp"]),
                        source="file_csv",
                    )
                )
        return readings

    @staticmethod
    def _load_json(path: Path) -> List[WaterLevelReading]:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Water level file is empty")

        if path.suffix.lower() == ".jsonl":
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload = payload.get("readings", [])

        readings: List[WaterLevelReading] = []
        for row in payload:
            readings.append(
                WaterLevelReading(
                    height_m=float(row["height_m"]),
                    timestamp=parse_iso_datetime(row["timestamp"]),
                    source="file_json",
                )
            )
        return readings


@dataclass
class WaterLevelFusion:
    source: WaterLevelSource
    base_plane: Plane

    def current_plane(self, reference_time: Optional[datetime] = None) -> tuple[Plane, WaterLevelReading]:
        reading = self.source.get_latest(reference_time=reference_time)
        return self.base_plane.with_height(reading.height_m), reading


def create_water_level_source(cfg: dict) -> WaterLevelSource:
    mode = str(cfg.get("type", "static")).lower()
    if mode == "static":
        return StaticWaterLevelSource(height_m=float(cfg.get("height_m", 0.0)))
    if mode == "file":
        return FileWaterLevelSource(
            file_path=str(cfg["path"]),
            hold_last=bool(cfg.get("hold_last", True)),
            time_offset_sec=float(cfg.get("time_offset_sec", 0.0)),
        )
    raise ValueError(f"Unsupported water level source type: {mode}")

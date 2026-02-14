"""Real weather data from yr.no (Stavanger - Valand station)."""

import json
import math
from pathlib import Path
from typing import Any, NamedTuple

_DATA_FILE = Path(__file__).parent / "sample_temperature.json"


class DayTemperature(NamedTuple):
    min_c: float
    max_c: float


def _load_days() -> list[DayTemperature]:
    with _DATA_FILE.open() as f:
        data: dict[str, Any] = json.load(f)

    months: list[dict[str, Any]] = data["historical"]["months"]

    days: list[DayTemperature] = []
    for month in months:
        for day in month["days"]:
            temp: dict[str, Any] | None = day.get("temperature")
            if temp is None or "min" not in temp:
                continue
            days.append(DayTemperature(min_c=float(temp["min"]), max_c=float(temp["max"])))
    return days


_DAYS: list[DayTemperature] = _load_days()


def external_temp_at_tick(tick: int, tick_duration_s: float) -> float:
    """External temperature derived from real yr.no daily min/max.

    Uses a sinusoidal curve within each day: coldest at 06:00, warmest at 15:00.
    Wraps around the dataset with modulo when ticks exceed the data range.
    """
    elapsed_s = tick * tick_duration_s
    day_index = int(elapsed_s / 86400) % len(_DAYS)
    hour = (elapsed_s % 86400) / 3600

    day = _DAYS[day_index]
    mid = (day.min_c + day.max_c) / 2
    amp = (day.max_c - day.min_c) / 2

    # sin curve: minimum at hour=6, maximum at hour=15
    return mid + amp * math.sin(2 * math.pi * (hour - 6) / 24)

"""Electricity price fetching from hvakosterstrommen.no API."""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TypedDict

import aiohttp

STAVANGER_PRICE_AREA = "NO2"  # Kristiansand / Stavanger region


class PriceEntry(TypedDict):
    NOK_per_kWh: float
    EUR_per_kWh: float
    EXR: float
    time_start: str
    time_end: str


@dataclass
class ElectricityPrice:
    price_nok_per_kwh: float
    valid_from: datetime
    valid_to: datetime


class ElectricityPriceService:
    _cache: list[PriceEntry] | None = None
    _cache_date: date | None = None

    @classmethod
    def _build_url(cls, date: datetime) -> str:
        return f"https://www.hvakosterstrommen.no/api/v1/prices/{date.year:04d}/{date.month:02d}-{date.day:02d}_{STAVANGER_PRICE_AREA}.json"

    @classmethod
    async def get_current_price(cls) -> ElectricityPrice:
        now = datetime.now(UTC)
        today = now.date()

        if cls._cache is None or cls._cache_date != today:
            url = cls._build_url(datetime.combine(today, datetime.min.time()))
            async with aiohttp.ClientSession() as session, session.get(url) as response:
                response.raise_for_status()
                cls._cache = await response.json()
            cls._cache_date = today

        current_hour = now.hour
        if cls._cache is None:
            raise ValueError("No price data available")
        for entry in cls._cache:
            start = datetime.fromisoformat(entry["time_start"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(entry["time_end"].replace("Z", "+00:00"))
            if start.hour <= current_hour < end.hour:
                return ElectricityPrice(
                    price_nok_per_kwh=entry["NOK_per_kWh"],
                    valid_from=start,
                    valid_to=end,
                )

        raise ValueError(f"No price found for hour {current_hour}")

    @classmethod
    async def get_today_prices(cls) -> list[ElectricityPrice]:
        now = datetime.now(UTC)
        today = now.date()

        if cls._cache is None or cls._cache_date != today:
            url = cls._build_url(datetime.combine(today, datetime.min.time()))
            async with aiohttp.ClientSession() as session, session.get(url) as response:
                response.raise_for_status()
                cls._cache = await response.json()
            cls._cache_date = today

        if cls._cache is None:
            return []
        return [
            ElectricityPrice(
                price_nok_per_kwh=entry["NOK_per_kWh"],
                valid_from=datetime.fromisoformat(entry["time_start"].replace("Z", "+00:00")),
                valid_to=datetime.fromisoformat(entry["time_end"].replace("Z", "+00:00")),
            )
            for entry in cls._cache
        ]

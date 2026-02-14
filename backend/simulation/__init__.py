"""Simulation module - sensor data generation."""

from simulation.generator import simulate_tick
from simulation.placement import place_sensors

__all__ = [
    "place_sensors",
    "simulate_tick",
]

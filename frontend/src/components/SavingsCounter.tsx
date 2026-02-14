import { useEffect, useRef, useState } from "react";
import type { MetricsUpdate, RoomMetrics, WastePattern } from "../types";

interface SavingsCounterProps {
  metrics: MetricsUpdate;
  intervalMs?: number;
  electricityPrice?: number;
}

const DEFAULT_PRICE = 1.5;
const CO2_FACTOR = 0.02; // kg CO2 per kWh

function wasteMultiplier(pattern: WastePattern, data: RoomMetrics): number {
  switch (pattern) {
    case "empty_room_heating_on":
      return data.power * 0.8;
    case "rapid_heat_loss":
      return data.power * 0.9;
    case "over_heating":
      return data.power * 0.3;
    case "excessive_ventilation":
      return data.ventilation_power * 0.9;
  }
}

function calculateWasteWatts(rooms: Record<string, RoomMetrics>): number {
  let waste = 0;
  for (const data of Object.values(rooms)) {
    for (const pattern of data.waste_patterns ?? []) {
      waste += wasteMultiplier(pattern, data);
    }
  }
  return waste;
}

export function SavingsCounter({ metrics, intervalMs = 2000, electricityPrice = DEFAULT_PRICE }: SavingsCounterProps) {
  const [savedKwh, setSavedKwh] = useState(0);
  const wasteRate = useRef(0);
  const lastUpdate = useRef(Date.now());

  // Update waste rate when metrics change
  useEffect(() => {
    wasteRate.current = calculateWasteWatts(metrics.rooms);
  }, [metrics]);

  // Accumulate savings over time (simulating that we're preventing waste)
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      const elapsed = (now - lastUpdate.current) / 1000 / 3600; // hours
      const kwhSaved = (wasteRate.current / 1000) * elapsed * 0.5; // assume 50% mitigation
      setSavedKwh((prev) => prev + kwhSaved);
      lastUpdate.current = now;
    }, intervalMs);

    return () => clearInterval(interval);
  }, [intervalMs]);

  const savedNok = savedKwh * electricityPrice;
  const savedCo2 = savedKwh * CO2_FACTOR * 1000; // grams

  return (
    <div className="savings-counter">
      <h3>Session Savings</h3>
      <div className="savings-grid">
        <div className="saving-item">
          <span className="saving-value">{savedKwh.toFixed(3)}</span>
          <span className="saving-unit">kWh</span>
        </div>
        <div className="saving-item">
          <span className="saving-value">{savedNok.toFixed(2)}</span>
          <span className="saving-unit">NOK</span>
        </div>
        <div className="saving-item">
          <span className="saving-value">{savedCo2.toFixed(1)}</span>
          <span className="saving-unit">g CO₂</span>
        </div>
      </div>
    </div>
  );
}

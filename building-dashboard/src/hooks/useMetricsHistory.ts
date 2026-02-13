import { useEffect, useRef } from "react";
import type { MetricsUpdate } from "../types";

export interface MetricsHistory {
  power: number[];
  temperature: number[];
  co2: number[];
  waste: number[];
}

const MAX_HISTORY = 30;

export function useMetricsHistory(metrics: MetricsUpdate): MetricsHistory {
  const history = useRef<MetricsHistory>({
    power: [],
    temperature: [],
    co2: [],
    waste: [],
  });

  useEffect(() => {
    const rooms = Object.values(metrics.rooms);
    if (rooms.length === 0) return;

    const totalPower = rooms.reduce((sum, r) => sum + r.power, 0);
    const avgTemp = rooms.reduce((sum, r) => sum + r.temperature, 0) / rooms.length;
    const avgCo2 = rooms.reduce((sum, r) => sum + r.co2, 0) / rooms.length;
    const wasteCount = rooms.filter((r) => r.waste_patterns.length > 0).length;

    const h = history.current;
    h.power = [...h.power, totalPower].slice(-MAX_HISTORY);
    h.temperature = [...h.temperature, avgTemp].slice(-MAX_HISTORY);
    h.co2 = [...h.co2, avgCo2].slice(-MAX_HISTORY);
    h.waste = [...h.waste, wasteCount].slice(-MAX_HISTORY);
  }, [metrics]);

  return history.current;
}

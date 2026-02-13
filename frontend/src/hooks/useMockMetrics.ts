import { useState, useEffect } from "react";
import type { MetricsUpdate } from "../types";
import { generateMockMetrics } from "../mocks/building";

/**
 * Simulates WebSocket metrics updates with mock data.
 * Replace with real WebSocket connection when backend is ready.
 */
export function useMockMetrics(intervalMs: number = 2000): MetricsUpdate {
  const [metrics, setMetrics] = useState<MetricsUpdate>(() =>
    generateMockMetrics(0)
  );

  useEffect(() => {
    let tick = 0;
    const interval = setInterval(() => {
      tick++;
      setMetrics(generateMockMetrics(tick));
    }, intervalMs);

    return () => clearInterval(interval);
  }, [intervalMs]);

  return metrics;
}

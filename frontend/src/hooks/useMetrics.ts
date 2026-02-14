import { useState, useEffect, useRef, useCallback } from "react";
import type { MetricsUpdate } from "../types";
import { generateMockMetrics } from "../mocks/building";

function getWsUrl(): string {
  const envUrl = import.meta.env.VITE_WS_URL;
  if (envUrl) return envUrl;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

const WS_URL = getWsUrl();

/**
 * Connects to WebSocket for live metrics updates.
 * Falls back to mock data if useMock is true or connection fails.
 */
export function useMetrics(useMock: boolean = true): {
  metrics: MetricsUpdate;
  connected: boolean;
  error: string | null;
} {
  const [metrics, setMetrics] = useState<MetricsUpdate>(() =>
    generateMockMetrics(0)
  );
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const tickRef = useRef(0);

  // Mock metrics simulation
  useEffect(() => {
    if (!useMock) return;

    const interval = setInterval(() => {
      tickRef.current++;
      setMetrics(generateMockMetrics(tickRef.current));
    }, 2000);

    return () => clearInterval(interval);
  }, [useMock]);

  // WebSocket connection
  useEffect(() => {
    if (useMock) return;

    let cancelled = false;
    let reconnectTimer: number | undefined;

    function connect() {
      if (cancelled) return;

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) { ws.close(); return; }
        setConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as MetricsUpdate;
          // Backend sends heating_power + ventilation_power; compute power for the frontend
          for (const room of Object.values(data.rooms)) {
            room.power = (room.heating_power ?? 0) + (room.ventilation_power ?? 0);
          }
          setMetrics(data);
        } catch {
          console.error("Failed to parse metrics:", event.data);
        }
      };

      ws.onerror = () => {
        setError("WebSocket error");
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        if (!cancelled) {
          reconnectTimer = window.setTimeout(connect, 3000);
        }
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [useMock]);

  return { metrics, connected, error };
}

/**
 * Sends a message through the WebSocket connection.
 * Useful for future bidirectional communication.
 */
export function useSendMessage(): (message: unknown) => void {
  const wsRef = useRef<WebSocket | null>(null);

  return useCallback((message: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);
}

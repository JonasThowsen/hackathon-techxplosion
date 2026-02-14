import { useState, useEffect } from "react";
import type { BuildingLayout } from "../types";
import { MOCK_BUILDING } from "../mocks/building";

const API_BASE = import.meta.env.VITE_API_URL || "";

/**
 * Fetches building layout from backend.
 * Falls back to mock data if useMock is true or fetch fails.
 */
export function useBuilding(useMock: boolean = true): {
  building: BuildingLayout | null;
  loading: boolean;
  error: string | null;
} {
  const [building, setBuilding] = useState<BuildingLayout | null>(
    useMock ? MOCK_BUILDING : null
  );
  const [loading, setLoading] = useState(!useMock);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (useMock) return;

    async function fetchBuilding() {
      try {
        let base = API_BASE;
        if (base && !base.startsWith("http")) {
          base = window.location.protocol + "//" + base;
        }
        const res = await fetch(`${base}/building`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setBuilding(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch building");
        setBuilding(MOCK_BUILDING); // Fallback to mock
      } finally {
        setLoading(false);
      }
    }

    fetchBuilding();
  }, [useMock]);

  return { building, loading, error };
}

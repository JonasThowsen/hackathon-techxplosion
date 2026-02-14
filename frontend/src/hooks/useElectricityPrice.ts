import { useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "";

export interface ElectricityPrice {
  price_nok_per_kwh: number;
  valid_from: string;
  valid_to: string;
}

export function useElectricityPrice() {
  const [price, setPrice] = useState<ElectricityPrice | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchPrice() {
      try {
        let base = API_URL;
        if (base && !base.startsWith("http")) {
          base = window.location.protocol + "//" + base;
        }
        const response = await fetch(`${base}/electricity/price`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (!cancelled) {
          setPrice(data);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to fetch price");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchPrice();

    const interval = setInterval(fetchPrice, 60 * 60 * 1000); // refresh hourly

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return { price, loading, error };
}

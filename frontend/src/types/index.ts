// Contract types matching backend API

export interface Point {
  x: number;
  y: number;
}

export interface Room {
  id: string;
  name: string;
  polygon: [number, number][]; // Array of [x, y] coordinate pairs in meters
}

export interface Floor {
  floor_index: number;
  label: string;
  rooms: Room[];
}

export interface BuildingLayout {
  id: string;
  name: string;
  width_m: number;
  height_m: number;
  floors: Floor[];
}

// Metrics types from WebSocket

export type WastePattern =
  | "empty_room_heating_on"
  | "open_window_heating"
  | "appliances_standby"
  | "over_heating"
  | "empty_ventilation";

export interface RoomMetrics {
  temperature: number;
  occupancy: boolean;
  co2: number;
  power: number;
  waste_patterns: WastePattern[];
  heat_flow?: number; // Net heat gain/loss in watts (positive = gaining heat)
}

export interface MetricsUpdate {
  tick: number;
  rooms: Record<string, RoomMetrics>;
}

// UI state types

export type MetricType = "temperature" | "co2" | "power";

export interface MetricConfig {
  label: string;
  unit: string;
  min: number;
  max: number;
}

export const METRIC_CONFIGS: Record<MetricType, MetricConfig> = {
  temperature: { label: "Temperature", unit: "°C", min: 15, max: 30 },
  co2: { label: "CO₂", unit: "ppm", min: 300, max: 1000 },
  power: { label: "Power", unit: "W", min: 0, max: 500 },
};

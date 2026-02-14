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
  | "rapid_heat_loss"
  | "over_heating"
  | "excessive_ventilation";

export type ActionType =
  | "reduce_heating"
  | "boost_heating"
  | "reduce_ventilation"
  | "suspend_heating";

export interface RoomMetrics {
  temperature: number;
  occupancy: boolean;
  co2: number;
  heating_power: number;
  ventilation_power: number;
  power: number; // computed: heating_power + ventilation_power
  waste_patterns?: WastePattern[];
  actions?: ActionType[];
  predicted_temp_30min?: number;
  predicted_temp_1h?: number;
  predicted_temp_2h?: number;
  prediction_uncertainty?: number;
  prediction_warnings?: string[];
  uses_estimated_params?: boolean;
}

export interface HeatFlow {
  from_room: string;
  to_room: string;
  watts: number;
}

export interface MetricsUpdate {
  tick: number;
  rooms: Record<string, RoomMetrics>;
  heat_flows?: HeatFlow[];
  system_enabled?: boolean;
  sun_enabled?: boolean;
  external_temp_c?: number;
  simulated_time?: string;
}

// Thermal estimation types

export interface ThermalRoomParams {
  thermal_mass_j_k: number;
  exterior_conductance_w_k: number | null;
}

export interface ThermalEstimationResult {
  success: boolean;
  message: string;
  rmse: number;
  r_squared: number;
  rooms: Record<string, ThermalRoomParams>;
  conductances: Record<string, number>;
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

import { useState, useMemo, useEffect } from "react";
import { FloorPlan, type SunPosition } from "./FloorPlan";
import { EventFeed } from "./EventFeed";
import { Sparkline } from "./Sparkline";
import { useMetricsHistory } from "../hooks/useMetricsHistory";
import type { BuildingLayout, MetricsUpdate, MetricType, RoomMetrics, WastePattern, ThermalEstimationResult, HeatFlow } from "../types";
import { METRIC_CONFIGS } from "../types";

interface DashboardProps {
  building: BuildingLayout;
  metrics: MetricsUpdate;
  sunPosition?: SunPosition;
  onOpenEditor: () => void;
  connected?: boolean;
  electricityPrice?: number;
}

interface AggregatedMetrics {
  totalPower: number;
  avgTemperature: number;
  avgCo2: number;
  roomCount: number;
  alerts: Alert[];
  energyCost: number;
  co2Emissions: number;
}

interface Alert {
  roomId: string;
  roomName: string;
  pattern: string;
  severity: "warning" | "critical";
  message: string;
  estimatedWaste: number; // Watts
}

const CO2_FACTOR = 0.02; // kg CO2 per kWh (Norway hydro)

function aggregateMetrics(
  building: BuildingLayout,
  metrics: MetricsUpdate,
  electricityPriceNok: number = 1.5
): AggregatedMetrics {
  const rooms = Object.entries(metrics.rooms);
  const roomCount = rooms.length;
  const ELECTRICITY_PRICE = electricityPriceNok;

  if (roomCount === 0) {
    return {
      totalPower: 0,
      avgTemperature: 0,
      avgCo2: 0,
      roomCount: 0,
      alerts: [],
      energyCost: 0,
      co2Emissions: 0,
    };
  }

  let totalPower = 0;
  let totalTemp = 0;
  let totalCo2 = 0;
  const alerts: Alert[] = [];

  // Create room name lookup
  const roomNames: Record<string, string> = {};
  for (const floor of building.floors) {
    for (const room of floor.rooms) {
      roomNames[room.id] = room.name;
    }
  }

  for (const [roomId, data] of rooms) {
    totalPower += data.power;
    totalTemp += data.temperature;
    totalCo2 += data.co2;

    // Convert waste patterns to alerts
    for (const pattern of data.waste_patterns ?? []) {
      alerts.push(createAlert(roomId, roomNames[roomId] || roomId, pattern, data));
    }
  }

  const powerKw = totalPower / 1000;
  const energyCostPerHour = powerKw * ELECTRICITY_PRICE;
  const co2PerHour = powerKw * CO2_FACTOR;

  return {
    totalPower,
    avgTemperature: totalTemp / roomCount,
    avgCo2: totalCo2 / roomCount,
    roomCount,
    alerts,
    energyCost: energyCostPerHour,
    co2Emissions: co2PerHour,
  };
}

function createAlert(
  roomId: string,
  roomName: string,
  pattern: WastePattern,
  data: RoomMetrics
): Alert {
  switch (pattern) {
    case "empty_room_heating_on":
      return {
        roomId,
        roomName,
        pattern,
        severity: "warning",
        message: `Heating running in empty room (${data.temperature.toFixed(1)}°C)`,
        estimatedWaste: data.power * 0.8,
      };
    case "rapid_heat_loss":
      return {
        roomId,
        roomName,
        pattern,
        severity: "critical",
        message: `Rapid heat loss detected (${data.temperature.toFixed(1)}°C, dropping)`,
        estimatedWaste: data.power * 0.9,
      };
    case "over_heating":
      return {
        roomId,
        roomName,
        pattern,
        severity: "warning",
        message: `Excess heating (${data.temperature.toFixed(1)}°C)`,
        estimatedWaste: data.power * 0.3,
      };
    case "excessive_ventilation":
      return {
        roomId,
        roomName,
        pattern,
        severity: "warning",
        message: `Unnecessary ventilation in empty room`,
        estimatedWaste: data.ventilation_power * 0.9,
      };
  }
}

export function Dashboard({ building, metrics, sunPosition, onOpenEditor, connected, electricityPrice = 1.5 }: DashboardProps) {
  const [selectedFloor, setSelectedFloor] = useState(0);
  const [selectedMetric, setSelectedMetric] = useState<MetricType>("temperature");
  const [hoveredRoom, setHoveredRoom] = useState<{
    id: string;
    x: number;
    y: number;
  } | null>(null);
  const [systemEnabled, setSystemEnabled] = useState(true);
  const [sunEnabled, setSunEnabled] = useState(true);
  const [thermalEstimation, setThermalEstimation] = useState<ThermalEstimationResult | null>(null);
  const [showThermalPanel, setShowThermalPanel] = useState(false);

  const API_URL = import.meta.env.VITE_API_URL || "";

  useEffect(() => {
    if (metrics.system_enabled !== undefined) {
      setSystemEnabled(metrics.system_enabled);
    }
    if (metrics.sun_enabled !== undefined) {
      setSunEnabled(metrics.sun_enabled);
    }
  }, [metrics]);

  // Poll for thermal estimation results
  useEffect(() => {
    if (!showThermalPanel) return;
    
    const fetchEstimation = async () => {
      try {
        const res = await fetch(`${API_URL}/thermal/estimation`);
        const data = await res.json();
        if ("success" in data) {
          setThermalEstimation(data as ThermalEstimationResult);
        }
      } catch (e) {
        console.error("Failed to fetch thermal estimation:", e);
      }
    };

    fetchEstimation();
    const interval = setInterval(fetchEstimation, 5000);
    return () => clearInterval(interval);
  }, [showThermalPanel]);

  const handleRunEstimation = async () => {
    try {
      const res = await fetch(`${API_URL}/thermal/estimation/run`, { method: "POST" });
      const data = await res.json();
      if ("success" in data) {
        setThermalEstimation(data as ThermalEstimationResult);
      }
    } catch (e) {
      console.error("Failed to run estimation:", e);
    }
  };

  const handleToggle = async () => {
    try {
      const res = await fetch(`${API_URL}/system/toggle`, { method: "POST" });
      const data = await res.json() as { enabled: boolean };
      setSystemEnabled(data.enabled);
    } catch (e) {
      console.error("Failed to toggle system:", e);
    }
  };

  const handleSunToggle = async () => {
    try {
      const res = await fetch(`${API_URL}/sun/toggle`, { method: "POST" });
      const data = await res.json() as { enabled: boolean };
      setSunEnabled(data.enabled);
    } catch (e) {
      console.error("Failed to toggle sun:", e);
    }
  };

  const floor = building.floors[selectedFloor];
  const history = useMetricsHistory(metrics);
  const aggregated = useMemo(
    () => aggregateMetrics(building, metrics, electricityPrice),
    [building, metrics, electricityPrice]
  );

  const totalWaste = aggregated.alerts.reduce((sum, a) => sum + a.estimatedWaste, 0);
  const wastePercent = aggregated.totalPower > 0
    ? (totalWaste / aggregated.totalPower) * 100
    : 0;

  const handleRoomHover = (roomId: string | null, x: number, y: number) => {
    if (roomId) {
      setHoveredRoom({ id: roomId, x, y });
    } else {
      setHoveredRoom(null);
    }
  };

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="header-left">
          <h1>FlowMetrics</h1>
          <span className="building-name">{building.name}</span>
        </div>
        <div className="header-right">
          <button
            className={`system-toggle ${systemEnabled ? "on" : "off"}`}
            onClick={handleToggle}
            style={{
              padding: "6px 14px",
              borderRadius: "6px",
              border: "none",
              cursor: "pointer",
              fontWeight: 600,
              fontSize: "0.85rem",
              background: systemEnabled ? "#22c55e" : "#ef4444",
              color: "#fff",
            }}
          >
            Control: {systemEnabled ? "ON" : "OFF"}
          </button>
          <button
            onClick={handleSunToggle}
            style={{
              padding: "6px 14px",
              borderRadius: "6px",
              border: "none",
              cursor: "pointer",
              fontWeight: 600,
              fontSize: "0.85rem",
              background: sunEnabled ? "#f59e0b" : "#6b7280",
              color: "#fff",
            }}
          >
            Sun: {sunEnabled ? "ON" : "OFF"}
          </button>
          <span className={`live-indicator ${connected === false ? "disconnected" : ""}`}>
            <span className="live-dot" /> {connected === false ? "Reconnecting..." : "Live"}
          </span>
          <span className="tick">{metrics.simulated_time ? new Date(metrics.simulated_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : `Tick ${metrics.tick}`}</span>
          <button className="editor-btn" onClick={onOpenEditor}>
            Editor
          </button>
        </div>
      </header>
      {!systemEnabled && (
        <div
          style={{
            background: "#fef3cd",
            color: "#856404",
            padding: "8px 16px",
            textAlign: "center",
            fontSize: "0.9rem",
            fontWeight: 500,
          }}
        >
          Control system disabled — monitoring only. Waste patterns detected but not acted upon.
        </div>
      )}

      <div className="dashboard-content">
        {/* Summary Cards */}
        <section className="summary-cards">
          <div className="card energy-card">
            <div className="card-icon">⚡</div>
            <div className="card-content">
              <div className="card-header-row">
                <div className="card-value">{(aggregated.totalPower / 1000).toFixed(2)} kW</div>
                <Sparkline data={history.power} color="#ffc107" />
              </div>
              <div className="card-label">Total Power</div>
              <div className="card-sub">{aggregated.energyCost.toFixed(2)} NOK/hr</div>
            </div>
          </div>

          <div className="card waste-card">
            <div className="card-icon">📉</div>
            <div className="card-content">
              <div className="card-header-row">
                <div className="card-value">{(totalWaste / 1000).toFixed(2)} kW</div>
                <Sparkline data={history.waste} color="#ff6b6b" />
              </div>
              <div className="card-label">Estimated Waste</div>
              <div className="card-sub">{wastePercent.toFixed(1)}% of total</div>
            </div>
          </div>

          <div className="card temp-card">
            <div className="card-icon">🌡️</div>
            <div className="card-content">
              <div className="card-header-row">
                <div className="card-value">{aggregated.avgTemperature.toFixed(1)}°C</div>
                <Sparkline data={history.temperature} color="#4ecdc4" />
              </div>
              <div className="card-label">Avg Temperature</div>
              <div className="card-sub">Across {aggregated.roomCount} rooms</div>
            </div>
          </div>

          <div className="card rooms-card">
            <div className="card-icon">🏠</div>
            <div className="card-content">
              <div className="card-header-row">
                <div className="card-value">{aggregated.roomCount}</div>
              </div>
              <div className="card-label">Active Zones</div>
              <div className="card-sub">{aggregated.alerts.length} with issues</div>
            </div>
          </div>

          <div className="card co2-card">
            <div className="card-icon">🌱</div>
            <div className="card-content">
              <div className="card-header-row">
                <div className="card-value">{aggregated.avgCo2.toFixed(0)} ppm</div>
                <Sparkline data={history.co2} color="#34d399" />
              </div>
              <div className="card-label">Avg CO₂</div>
              <div className="card-sub">{(aggregated.co2Emissions * 1000).toFixed(0)} g/hr CO₂</div>
            </div>
          </div>

          <div className="card external-temp-card">
            <div className="card-icon">🌤️</div>
            <div className="card-content">
              <div className="card-header-row">
                <div className="card-value">{(metrics.external_temp_c ?? 0).toFixed(1)}°C</div>
                <Sparkline data={history.externalTemp} color="#f59e0b" />
              </div>
              <div className="card-label">External Temp</div>
              <div className="card-sub">yr.no Stavanger</div>
            </div>
          </div>
        </section>

        {/* Alerts + Events Row */}
        <section className="info-row">
          <EventFeed metrics={metrics} maxEvents={8} />
          <div className="flow-metrics">
            <h3>Heat Flow Analysis</h3>
            <div className="flow-metrics-grid">
              {Object.entries(metrics.rooms).map(([roomId, data]) => {
                const roomName = building.floors
                  .flatMap(f => f.rooms)
                  .find(r => r.id === roomId)?.name || roomId;
                
                // Get heat flows for this room
                const roomHeatFlows = metrics.heat_flows?.filter(f => f.from_room === roomId || f.to_room === roomId) ?? [];
                const heatToNeighbors = roomHeatFlows.filter(f => f.from_room === roomId).reduce((sum, f) => sum + f.watts, 0);
                const heatFromNeighbors = roomHeatFlows.filter(f => f.to_room === roomId).reduce((sum, f) => sum + f.watts, 0);
                
                // Get estimated exterior heat loss
                const thermalParams = thermalEstimation?.rooms[roomId];
                const extConductance = thermalParams?.exterior_conductance_w_k ?? null;
                const extTemp = metrics.external_temp_c ?? 5;
                const tempDiff = Math.max(0, data.temperature - extTemp);
                const extHeatLoss = extConductance ? extConductance * tempDiff : null;
                
                // Heating input
                const heatingIn = data.heating_power;
                
                // Calculate percentages for visualization
                const totalOut = heatingIn + (heatToNeighbors || 0) + (extHeatLoss || 0);
                
                return (
                  <div key={roomId} className="flow-metric-card">
                    <div className="flow-metric-header">{roomName}</div>
                    <div className="flow-metric-bars">
                      <div className="flow-metric-row">
                        <span className="flow-label">Heating in</span>
                        <div className="flow-bar-track">
                          <div className="flow-bar heating" style={{ width: `${totalOut > 0 ? (heatingIn / totalOut) * 100 : 0}%` }} />
                        </div>
                        <span className="flow-num">{heatingIn.toFixed(0)}W</span>
                      </div>
                      {extHeatLoss !== null && (
                        <div className="flow-metric-row">
                          <span className="flow-label">→ Outside</span>
                          <div className="flow-bar-track">
                            <div className="flow-bar outside" style={{ width: `${totalOut > 0 ? (extHeatLoss / totalOut) * 100 : 0}%` }} />
                          </div>
                          <span className="flow-num">{extHeatLoss.toFixed(0)}W</span>
                        </div>
                      )}
                      {heatToNeighbors > 0 && (
                        <div className="flow-metric-row">
                          <span className="flow-label">→ Neighbors</span>
                          <div className="flow-bar-track">
                            <div className="flow-bar neighbors-out" style={{ width: `${totalOut > 0 ? (heatToNeighbors / totalOut) * 100 : 0}%` }} />
                          </div>
                          <span className="flow-num">{heatToNeighbors.toFixed(0)}W</span>
                        </div>
                      )}
                      {heatFromNeighbors > 0 && (
                        <div className="flow-metric-row">
                          <span className="flow-label">← From neighbors</span>
                          <div className="flow-bar-track">
                            <div className="flow-bar neighbors-in" style={{ width: `${totalOut > 0 ? (heatFromNeighbors / totalOut) * 100 : 0}%` }} />
                          </div>
                          <span className="flow-num">+{heatFromNeighbors.toFixed(0)}W</span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* Alerts Section */}
        <section className="alerts-section">
          <div className="alerts-panel">
            <h3>Alerts {aggregated.alerts.length > 0 && `(${aggregated.alerts.length})`}</h3>
            <div className="alerts-list">
              {aggregated.alerts.length > 0 ? aggregated.alerts.map((alert, i) => (
                <div key={`${alert.roomId}-${i}`} className={`alert-item ${alert.severity}`}>
                  <span className="alert-room">{alert.roomName}</span>
                  <span className="alert-message">{alert.message}</span>
                  <span className="alert-waste">-{(alert.estimatedWaste).toFixed(0)}W</span>
                </div>
              )) : (
                <div className="no-alerts">No alerts</div>
              )}
            </div>
          </div>
        </section>

        {/* Thermal Estimation Panel */}
        <section className="thermal-panel">
          <div className="thermal-header">
            <h2>Thermal Properties</h2>
            <div className="thermal-controls">
              <button
                onClick={() => setShowThermalPanel(!showThermalPanel)}
                style={{
                  padding: "6px 14px",
                  borderRadius: "6px",
                  border: "none",
                  cursor: "pointer",
                  fontWeight: 600,
                  fontSize: "0.85rem",
                  background: showThermalPanel ? "#6366f1" : "#e5e7eb",
                  color: showThermalPanel ? "#fff" : "#374151",
                }}
              >
                {showThermalPanel ? "Hide" : "Show"}
              </button>
              {showThermalPanel && (
                <button
                  onClick={handleRunEstimation}
                  style={{
                    padding: "6px 14px",
                    borderRadius: "6px",
                    border: "none",
                    cursor: "pointer",
                    fontWeight: 600,
                    fontSize: "0.85rem",
                    background: "#10b981",
                    color: "#fff",
                    marginLeft: "8px",
                  }}
                >
                  Re-run Estimation
                </button>
              )}
            </div>
          </div>
          
          {showThermalPanel && (
            <div className="thermal-content">
              {thermalEstimation ? (
                <>
                  <div className="thermal-metrics">
                    <div className="thermal-metric">
                      <span className="metric-label">R² Score:</span>
                      <span className="metric-value">{(thermalEstimation.r_squared * 100).toFixed(1)}%</span>
                    </div>
                    <div className="thermal-metric">
                      <span className="metric-label">RMSE:</span>
                      <span className="metric-value">{thermalEstimation.rmse.toFixed(3)}°C</span>
                    </div>
                  </div>
                  
                  <div className="thermal-rooms">
                    <h3>Room Thermal Properties</h3>
                    <div className="thermal-table">
                      <div className="table-header">
                        <span>Room</span>
                      <span>Thermal Mass (kJ/K)</span>
                      <span>Est. Heat Loss/hr</span>
                    </div>
                    {Object.entries(thermalEstimation.rooms).map(([roomId, params]) => {
                      const roomName = building.floors
                        .flatMap(f => f.rooms)
                        .find(r => r.id === roomId)?.name || roomId;
                      const roomTemp = metrics.rooms[roomId]?.temperature ?? 21;
                      const extTemp = metrics.external_temp_c ?? 5;
                      const tempDiff = Math.max(0, roomTemp - extTemp);
                      const conductance = params.exterior_conductance_w_k ?? 0;
                      const heatLossWatts = conductance * tempDiff;
                      const costPerHour = (heatLossWatts / 1000) * 1.5 * 0.01; // 1.5 NOK/kWh, 0.01 is a factor
                      
                      return (
                        <div key={roomId} className="table-row">
                          <span className="room-name">{roomName}</span>
                          <span>{(params.thermal_mass_j_k / 1000).toFixed(0)}</span>
                          <span className={heatLossWatts > 50 ? "high-loss" : ""}>
                            {heatLossWatts.toFixed(0)}W ({costPerHour.toFixed(2)}kr/h)
                          </span>
                        </div>
                      );
                    })}
                    </div>
                  </div>
                  
                  {Object.keys(thermalEstimation.conductances).length > 0 && (
                    <div className="thermal-conductances">
                      <h3>Wall Conductances</h3>
                      <div className="conductance-bars">
                        {Object.entries(thermalEstimation.conductances)
                          .sort(([, a], [, b]) => b - a)
                          .map(([key, conductance]) => (
                            <div key={key} className="conductance-row">
                              <span className="conductance-rooms">{key.replace(/-/g, " ↔ ")}</span>
                              <div className="conductance-bar-container">
                                <div
                                  className="conductance-bar"
                                  style={{ width: `${Math.min(100, conductance * 2)}%` }}
                                />
                              </div>
                              <span className="conductance-value">{conductance.toFixed(1)} W/K</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="thermal-loading">
                  <p>Collecting data... Need at least 10 ticks of history.</p>
                  <p>Current time: {metrics.simulated_time ? new Date(metrics.simulated_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : `Tick ${metrics.tick}`}</p>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Floor Plan Section */}
        <section className="floorplan-section">
          <div className="floorplan-header">
            <h2>Floor Plan</h2>
            <div className="floorplan-controls">
              <div className="control-group">
                <label>Floor</label>
                <div className="button-group">
                  {building.floors.map((f) => (
                    <button
                      key={f.floor_index}
                      className={selectedFloor === f.floor_index ? "active" : ""}
                      onClick={() => setSelectedFloor(f.floor_index)}
                    >
                      {f.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="control-group">
                <label>Metric</label>
                <div className="button-group">
                  {(Object.keys(METRIC_CONFIGS) as MetricType[]).map((metric) => (
                    <button
                      key={metric}
                      className={selectedMetric === metric ? "active" : ""}
                      onClick={() => setSelectedMetric(metric)}
                    >
                      {METRIC_CONFIGS[metric].label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="floorplan-container">
            <FloorPlan
              floor={floor}
              metrics={metrics.rooms}
              selectedMetric={selectedMetric}
              buildingWidth={building.width_m}
              buildingHeight={building.height_m}
              canvasWidth={900}
              canvasHeight={400}
              sunPosition={sunPosition}
              heatFlows={metrics.heat_flows}
              onRoomHover={handleRoomHover}
            />

            <div className="legend">
              <span className="legend-label">{METRIC_CONFIGS[selectedMetric].label}</span>
              <div className="legend-bar" />
              <div className="legend-values">
                <span>{METRIC_CONFIGS[selectedMetric].min}{METRIC_CONFIGS[selectedMetric].unit}</span>
                <span>{METRIC_CONFIGS[selectedMetric].max}{METRIC_CONFIGS[selectedMetric].unit}</span>
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Tooltip */}
      {hoveredRoom && (
        <RoomTooltip
          roomId={hoveredRoom.id}
          metrics={metrics.rooms[hoveredRoom.id]}
          room={floor.rooms.find((r) => r.id === hoveredRoom.id)}
          x={hoveredRoom.x}
          y={hoveredRoom.y}
          heatFlows={metrics.heat_flows}
          thermalEstimation={thermalEstimation}
          externalTemp={metrics.external_temp_c}
        />
      )}
    </div>
  );
}

interface RoomTooltipProps {
  roomId: string;
  metrics: RoomMetrics | undefined;
  room: { id: string; name: string } | undefined;
  x: number;
  y: number;
  heatFlows?: HeatFlow[];
  thermalEstimation?: ThermalEstimationResult | null;
  externalTemp?: number;
  electricityPrice?: number;
}

function RoomTooltip({ metrics, room, x, y, heatFlows, thermalEstimation, externalTemp, electricityPrice }: RoomTooltipProps) {
  if (!metrics || !room) return null;

  // Find heat flows involving this room
  const roomHeatFlows = heatFlows?.filter(f => f.from_room === room.id || f.to_room === room.id) ?? [];
  const heatGaining = roomHeatFlows.filter(f => f.to_room === room.id).reduce((sum, f) => sum + f.watts, 0);
  const heatLosing = roomHeatFlows.filter(f => f.from_room === room.id).reduce((sum, f) => sum + f.watts, 0);
  
  // Get thermal params for this room
  const thermalParams = thermalEstimation?.rooms[room.id];
  const extConductance = thermalParams?.exterior_conductance_w_k ?? null;
  const thermalMass = thermalParams?.thermal_mass_j_k ?? null;
  
  // Calculate estimated heat loss
  let estimatedHeatLoss = 0;
  let costPerHour = 0;
  if (extConductance && externalTemp !== undefined) {
    const tempDiff = Math.max(0, metrics.temperature - externalTemp);
    estimatedHeatLoss = extConductance * tempDiff;
    costPerHour = (estimatedHeatLoss / 1000) * 1.5; // 1.5 NOK/kWh
  }

  return (
    <div className="tooltip" style={{ left: x + 10, top: y + 10 }}>
      <div className="tooltip-header">{room.name}</div>
      <div className="tooltip-section">
        <div className="tooltip-row">
          <span>Temperature:</span>
          <span>{metrics.temperature.toFixed(1)}°C</span>
        </div>
        <div className="tooltip-row">
          <span>CO₂:</span>
          <span>{metrics.co2.toFixed(0)} ppm</span>
        </div>
        <div className="tooltip-row">
          <span>Heating:</span>
          <span>{metrics.heating_power.toFixed(0)} W</span>
        </div>
        <div className="tooltip-row">
          <span>Ventilation:</span>
          <span>{metrics.ventilation_power.toFixed(0)} W</span>
        </div>
      </div>
      
      {thermalEstimation && (
        <div className="tooltip-section">
          <div className="tooltip-section-title">Estimated Properties</div>
          {thermalMass && (
            <div className="tooltip-row">
              <span>Thermal Mass:</span>
              <span>{(thermalMass / 1000).toFixed(0)} kJ/K</span>
            </div>
          )}
          {extConductance && (
            <div className="tooltip-row">
              <span>Ext. Conductance:</span>
              <span>{extConductance.toFixed(1)} W/K</span>
            </div>
          )}
          {estimatedHeatLoss > 0 && (
            <div className="tooltip-row highlight-loss">
              <span>Est. Heat Loss:</span>
              <span>{estimatedHeatLoss.toFixed(0)} W ({costPerHour.toFixed(2)} kr/h)</span>
            </div>
          )}
        </div>
      )}
      
      {/* Predictions section */}
      {metrics.predicted_temp_30min !== undefined && (
        <div className="tooltip-section">
          <div className="tooltip-section-title">
            Predicted Temperature
            {metrics.uses_estimated_params && <span className="param-badge">estimated</span>}
            {!metrics.uses_estimated_params && <span className="param-badge geometry">geometry</span>}
          </div>
          <div className="tooltip-row">
            <span>30 min:</span>
            <span className="prediction">{metrics.predicted_temp_30min.toFixed(1)}°C</span>
          </div>
          <div className="tooltip-row">
            <span>1 hour:</span>
            <span className="prediction">{metrics.predicted_temp_1h?.toFixed(1) ?? "-"}°C</span>
          </div>
          <div className="tooltip-row">
            <span>2 hours:</span>
            <span className="prediction">{metrics.predicted_temp_2h?.toFixed(1) ?? "-"}°C</span>
          </div>
          {metrics.prediction_uncertainty !== undefined && metrics.prediction_uncertainty > 1.5 && (
            <div className="tooltip-row warning">
              <span>Uncertainty:</span>
              <span>±{metrics.prediction_uncertainty.toFixed(1)}°C</span>
            </div>
          )}
          {metrics.prediction_warnings && metrics.prediction_warnings.length > 0 && (
            <div className="tooltip-row warning">
              <span>Warning:</span>
              <span>{metrics.prediction_warnings.join(", ")}</span>
            </div>
          )}
          {/* Predicted cost */}
          {electricityPrice && metrics.predicted_temp_1h && metrics.predicted_temp_1h < 20.5 && (
            <div className="tooltip-row highlight-cost">
              <span>Est. cost (1h):</span>
              <span>
                {(((metrics.predicted_temp_1h < metrics.temperature ? 500 : 100) / 1000) * electricityPrice * 1).toFixed(2)} kr
              </span>
            </div>
          )}
        </div>
      )}
      
      {roomHeatFlows.length > 0 && (
        <div className="tooltip-section">
          <div className="tooltip-section-title">Heat Flows</div>
          {heatGaining > 0 && (
            <div className="tooltip-row heat-gain">
              <span>From neighbors:</span>
              <span>+{heatGaining.toFixed(0)} W</span>
            </div>
          )}
          {heatLosing > 0 && (
            <div className="tooltip-row heat-loss">
              <span>To neighbors:</span>
              <span>-{heatLosing.toFixed(0)} W</span>
            </div>
          )}
        </div>
      )}
      
      {(metrics.waste_patterns?.length ?? 0) > 0 && (
        <div className="tooltip-waste">
          Issue: {metrics.waste_patterns!.map(p => p.replace(/_/g, " ")).join(", ")}
        </div>
      )}
    </div>
  );
}

import { useState, useMemo, useEffect } from "react";
import { FloorPlan, type SunPosition } from "./FloorPlan";
import { EventFeed } from "./EventFeed";
import { Sparkline } from "./Sparkline";
import { useMetricsHistory } from "../hooks/useMetricsHistory";
import type { BuildingLayout, MetricsUpdate, MetricType, RoomMetrics, WastePattern } from "../types";
import { METRIC_CONFIGS } from "../types";

interface DashboardProps {
  building: BuildingLayout;
  metrics: MetricsUpdate;
  sunPosition?: SunPosition;
  onOpenEditor: () => void;
  connected?: boolean;
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

const ELECTRICITY_PRICE_NOK = 1.5; // NOK per kWh (example)
const CO2_FACTOR = 0.02; // kg CO2 per kWh (Norway hydro)

function aggregateMetrics(
  building: BuildingLayout,
  metrics: MetricsUpdate
): AggregatedMetrics {
  const rooms = Object.entries(metrics.rooms);
  const roomCount = rooms.length;

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
  const energyCostPerHour = powerKw * ELECTRICITY_PRICE_NOK;
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

export function Dashboard({ building, metrics, sunPosition, onOpenEditor, connected }: DashboardProps) {
  const [selectedFloor, setSelectedFloor] = useState(0);
  const [selectedMetric, setSelectedMetric] = useState<MetricType>("temperature");
  const [hoveredRoom, setHoveredRoom] = useState<{
    id: string;
    x: number;
    y: number;
  } | null>(null);
  const [systemEnabled, setSystemEnabled] = useState(true);
  const [sunEnabled, setSunEnabled] = useState(true);

  useEffect(() => {
    if (metrics.system_enabled !== undefined) {
      setSystemEnabled(metrics.system_enabled);
    }
    if (metrics.sun_enabled !== undefined) {
      setSunEnabled(metrics.sun_enabled);
    }
  }, [metrics]);

  const handleToggle = async () => {
    try {
      const res = await fetch("http://localhost:8000/system/toggle", { method: "POST" });
      const data = await res.json() as { enabled: boolean };
      setSystemEnabled(data.enabled);
    } catch (e) {
      console.error("Failed to toggle system:", e);
    }
  };

  const handleSunToggle = async () => {
    try {
      const res = await fetch("http://localhost:8000/sun/toggle", { method: "POST" });
      const data = await res.json() as { enabled: boolean };
      setSunEnabled(data.enabled);
    } catch (e) {
      console.error("Failed to toggle sun:", e);
    }
  };

  const floor = building.floors[selectedFloor];
  const history = useMetricsHistory(metrics);
  const aggregated = useMemo(
    () => aggregateMetrics(building, metrics),
    [building, metrics]
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
          <span className="tick">Tick {metrics.tick}</span>
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
          {aggregated.alerts.length > 0 && (
            <div className="alerts-panel">
              <h3>Alerts ({aggregated.alerts.length})</h3>
              <div className="alerts-list">
                {aggregated.alerts.map((alert, i) => (
                  <div key={`${alert.roomId}-${i}`} className={`alert-item ${alert.severity}`}>
                    <span className="alert-room">{alert.roomName}</span>
                    <span className="alert-message">{alert.message}</span>
                    <span className="alert-waste">-{(alert.estimatedWaste).toFixed(0)}W</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <EventFeed metrics={metrics} maxEvents={8} />
        </section>

        {/* Energy Flow Visualization */}
        <section className="energy-flow">
          <h2>Energy Distribution</h2>
          <div className="flow-bars">
            {Object.entries(metrics.rooms)
              .sort(([, a], [, b]) => b.power - a.power)
              .slice(0, 8)
              .map(([roomId, data]) => {
                const roomName = building.floors
                  .flatMap(f => f.rooms)
                  .find(r => r.id === roomId)?.name || roomId;
                const percent = (data.power / aggregated.totalPower) * 100;
                const hasWaste = (data.waste_patterns?.length ?? 0) > 0;

                return (
                  <div key={roomId} className="flow-bar-row">
                    <span className="flow-room">{roomName}</span>
                    <div className="flow-bar-container">
                      <div
                        className={`flow-bar ${hasWaste ? "waste" : ""}`}
                        style={{ width: `${percent}%` }}
                      />
                    </div>
                    <span className="flow-value">{data.power.toFixed(0)}W</span>
                  </div>
                );
              })}
          </div>
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
        />
      )}
    </div>
  );
}

interface RoomTooltipProps {
  roomId: string;
  metrics: RoomMetrics | undefined;
  room: { name: string } | undefined;
  x: number;
  y: number;
}

function RoomTooltip({ metrics, room, x, y }: RoomTooltipProps) {
  if (!metrics || !room) return null;

  return (
    <div className="tooltip" style={{ left: x + 10, top: y + 10 }}>
      <div className="tooltip-header">{room.name}</div>
      <div className="tooltip-row">
        <span>Temperature:</span>
        <span>{metrics.temperature.toFixed(1)}°C</span>
      </div>
      <div className="tooltip-row">
        <span>CO₂:</span>
        <span>{metrics.co2.toFixed(0)} ppm</span>
      </div>
      <div className="tooltip-row">
        <span>Power:</span>
        <span>{metrics.power.toFixed(0)} W</span>
      </div>
      {(metrics.waste_patterns?.length ?? 0) > 0 && (
        <div className="tooltip-waste">
          Issue: {metrics.waste_patterns!.map(p => p.replace(/_/g, " ")).join(", ")}
        </div>
      )}
    </div>
  );
}

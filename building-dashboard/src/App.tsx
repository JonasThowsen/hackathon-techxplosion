import { useState } from "react";
import { FloorPlan } from "./components/FloorPlan";
import { useMockMetrics } from "./hooks/useMockMetrics";
import { MOCK_BUILDING } from "./mocks/building";
import type { MetricType, RoomMetrics } from "./types";
import { METRIC_CONFIGS } from "./types";
import "./App.css";

function App() {
  const [selectedFloor, setSelectedFloor] = useState(0);
  const [selectedMetric, setSelectedMetric] = useState<MetricType>("temperature");
  const [hoveredRoom, setHoveredRoom] = useState<{
    id: string;
    x: number;
    y: number;
  } | null>(null);

  const metrics = useMockMetrics(2000);
  const building = MOCK_BUILDING;
  const floor = building.floors[selectedFloor];

  const handleRoomHover = (roomId: string | null, x: number, y: number) => {
    if (roomId) {
      setHoveredRoom({ id: roomId, x, y });
    } else {
      setHoveredRoom(null);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>FlowMetrics</h1>
        <span className="building-name">{building.name}</span>
      </header>

      <main className="main">
        <div className="controls">
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

        <div className="grid-container">
          <FloorPlan
            floor={floor}
            metrics={metrics.rooms}
            selectedMetric={selectedMetric}
            buildingWidth={building.width_m}
            buildingHeight={building.height_m}
            canvasWidth={800}
            canvasHeight={540}
            onRoomHover={handleRoomHover}
          />

          <div className="legend">
            <span className="legend-label">
              {METRIC_CONFIGS[selectedMetric].label}
            </span>
            <div className="legend-bar" />
            <div className="legend-values">
              <span>{METRIC_CONFIGS[selectedMetric].min}{METRIC_CONFIGS[selectedMetric].unit}</span>
              <span>{METRIC_CONFIGS[selectedMetric].max}{METRIC_CONFIGS[selectedMetric].unit}</span>
            </div>
          </div>
        </div>

        <div className="info-panel">
          <div className="tick">Tick: {metrics.tick}</div>
          <WasteAlerts rooms={metrics.rooms} />
        </div>
      </main>

      {hoveredRoom && (
        <Tooltip
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

interface TooltipProps {
  roomId: string;
  metrics: RoomMetrics | undefined;
  room: { name: string } | undefined;
  x: number;
  y: number;
}

function Tooltip({ metrics, room, x, y }: TooltipProps) {
  if (!metrics || !room) return null;

  return (
    <div
      className="tooltip"
      style={{
        left: x + 10,
        top: y + 10,
      }}
    >
      <div className="tooltip-header">{room.name}</div>
      <div className="tooltip-row">
        <span>Temperature:</span>
        <span>{metrics.temperature.toFixed(1)}°C</span>
      </div>
      <div className="tooltip-row">
        <span>Occupancy:</span>
        <span>{Math.round(metrics.occupancy * 100)}%</span>
      </div>
      <div className="tooltip-row">
        <span>CO₂:</span>
        <span>{metrics.co2.toFixed(0)} ppm</span>
      </div>
      <div className="tooltip-row">
        <span>Power:</span>
        <span>{metrics.power.toFixed(0)} W</span>
      </div>
      {metrics.waste_patterns.length > 0 && (
        <div className="tooltip-waste">
          ⚠ {metrics.waste_patterns.join(", ")}
        </div>
      )}
    </div>
  );
}

interface WasteAlertsProps {
  rooms: Record<string, RoomMetrics>;
}

function WasteAlerts({ rooms }: WasteAlertsProps) {
  const alerts = Object.entries(rooms)
    .filter(([, m]) => m.waste_patterns.length > 0)
    .map(([id, m]) => ({ id, patterns: m.waste_patterns }));

  if (alerts.length === 0) return null;

  return (
    <div className="waste-alerts">
      <h3>Active Alerts</h3>
      {alerts.map((alert) => (
        <div key={alert.id} className="alert">
          <span className="alert-room">{alert.id}</span>
          <span className="alert-pattern">{alert.patterns.join(", ")}</span>
        </div>
      ))}
    </div>
  );
}

export default App;

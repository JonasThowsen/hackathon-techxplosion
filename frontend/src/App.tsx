import { useState, useEffect } from "react";
import { type SunPosition } from "./components/FloorPlan";
import { FloorPlanEditor } from "./components/FloorPlanEditor";
import { Dashboard } from "./components/Dashboard";
import { useBuilding } from "./hooks/useBuilding";
import { useMetrics } from "./hooks/useMetrics";
import { useElectricityPrice } from "./hooks/useElectricityPrice";
import "./App.css";
import "./components/EditorStyles.css";
import "./components/DashboardStyles.css";

// Set to false to use real backend
const USE_MOCK = false;

// Mock sun position - replace with API call later
function useSunPosition(): SunPosition {
  const [sun, setSun] = useState<SunPosition>({
    azimuth: 135,
    elevation: 45,
    visible: true,
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setSun((prev) => ({
        ...prev,
        azimuth: (prev.azimuth + 0.5) % 360,
        elevation: 30 + 20 * Math.sin((prev.azimuth * Math.PI) / 180),
        visible: true,
      }));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return sun;
}

type AppMode = "dashboard" | "editor";

function App() {
  const [mode, setMode] = useState<AppMode>("dashboard");
  const { building, loading, error: buildingError } = useBuilding(USE_MOCK);
  const { metrics, connected } = useMetrics(USE_MOCK);
  const { price } = useElectricityPrice();
  const sunPosition = useSunPosition();

  if (loading) {
    return <div className="loading">Loading building...</div>;
  }

  if (!building) {
    return <div className="error">Failed to load building: {buildingError}</div>;
  }

  if (mode === "editor") {
    return (
      <>
        <button className="editor-back-btn" onClick={() => setMode("dashboard")}>
          ← Back to Dashboard
        </button>
        <FloorPlanEditor />
      </>
    );
  }

  return (
    <Dashboard
      building={building}
      metrics={metrics}
      sunPosition={sunPosition}
      onOpenEditor={() => setMode("editor")}
      connected={connected}
      electricityPrice={price?.price_nok_per_kwh}
    />
  );
}

export default App;

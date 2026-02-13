import { useState, useEffect } from "react";
import { type SunPosition } from "./components/FloorPlan";
import { FloorPlanEditor } from "./components/FloorPlanEditor";
import { Dashboard } from "./components/Dashboard";
import { useMockMetrics } from "./hooks/useMockMetrics";
import { MOCK_BUILDING } from "./mocks/building";
import "./App.css";
import "./components/EditorStyles.css";
import "./components/DashboardStyles.css";

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
  const metrics = useMockMetrics(2000);
  const sunPosition = useSunPosition();
  const building = MOCK_BUILDING;

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
    />
  );
}

export default App;

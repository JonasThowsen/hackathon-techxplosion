import { useRef, useEffect, useCallback, useState } from "react";
import type { BuildingLayout, Floor, Room } from "../types";

interface EditorState {
  building: BuildingLayout;
  selectedFloor: number;
  selectedRoom: string | null;
  tool: "select" | "draw" | "edit";
  drawingPoints: [number, number][];
  gridSnap: number;
  zoom: number;
  pan: { x: number; y: number };
}

const DEFAULT_BUILDING: BuildingLayout = {
  id: "building-1",
  name: "New Building",
  width_m: 100,
  height_m: 100,
  floors: [
    {
      floor_index: 0,
      label: "Floor 1",
      rooms: [],
    },
  ],
};

const MAX_HISTORY = 50;

export function FloorPlanEditor() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<EditorState>({
    building: DEFAULT_BUILDING,
    selectedFloor: 0,
    selectedRoom: null,
    tool: "draw",
    drawingPoints: [],
    gridSnap: 0.5,
    zoom: 1,
    pan: { x: 0, y: 0 },
  });
  const [isDragging, setIsDragging] = useState(false);
  const [editingRoom, setEditingRoom] = useState<{ name: string; id: string } | null>(null);
  const dragStart = useRef({ x: 0, y: 0 });
  const panStart = useRef({ x: 0, y: 0 });

  // Undo/Redo history
  const [history, setHistory] = useState<BuildingLayout[]>([]);
  const [future, setFuture] = useState<BuildingLayout[]>([]);

  // Push to history (call before making changes)
  const pushHistory = useCallback(() => {
    setHistory((h) => [...h.slice(-MAX_HISTORY + 1), state.building]);
    setFuture([]); // Clear redo stack on new action
  }, [state.building]);

  // Undo
  const undo = useCallback(() => {
    if (history.length === 0) return;
    const prev = history[history.length - 1];
    setHistory((h) => h.slice(0, -1));
    setFuture((f) => [state.building, ...f]);
    setState((s) => ({ ...s, building: prev, selectedRoom: null }));
    setEditingRoom(null);
  }, [history, state.building]);

  // Redo
  const redo = useCallback(() => {
    if (future.length === 0) return;
    const next = future[0];
    setFuture((f) => f.slice(1));
    setHistory((h) => [...h, state.building]);
    setState((s) => ({ ...s, building: next, selectedRoom: null }));
    setEditingRoom(null);
  }, [future, state.building]);

  const canvasWidth = 900;
  const canvasHeight = 600;
  const padding = 40;

  const { building, selectedFloor, selectedRoom, tool, drawingPoints, gridSnap, zoom, pan } = state;
  const floor = building.floors[selectedFloor];

  const baseScale = Math.min(
    (canvasWidth - padding * 2) / building.width_m,
    (canvasHeight - padding * 2) / building.height_m
  );

  const snapToGrid = useCallback(
    (value: number): number => {
      if (gridSnap === 0) return value;
      return Math.round(value / gridSnap) * gridSnap;
    },
    [gridSnap]
  );

  const metersToPixels = useCallback(
    (mx: number, my: number): [number, number] => {
      const cx = canvasWidth / 2;
      const cy = canvasHeight / 2;
      const px = cx + (mx - building.width_m / 2) * baseScale * zoom + pan.x;
      const py = cy + (my - building.height_m / 2) * baseScale * zoom + pan.y;
      return [px, py];
    },
    [baseScale, zoom, pan, building.width_m, building.height_m]
  );

  const pixelsToMeters = useCallback(
    (px: number, py: number): [number, number] => {
      const cx = canvasWidth / 2;
      const cy = canvasHeight / 2;
      const mx = (px - cx - pan.x) / (baseScale * zoom) + building.width_m / 2;
      const my = (py - cy - pan.y) / (baseScale * zoom) + building.height_m / 2;
      return [snapToGrid(mx), snapToGrid(my)];
    },
    [baseScale, zoom, pan, building.width_m, building.height_m, snapToGrid]
  );

  const isPointInPolygon = useCallback(
    (px: number, py: number, polygon: [number, number][]): boolean => {
      let inside = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0], yi = polygon[i][1];
        const xj = polygon[j][0], yj = polygon[j][1];
        if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
          inside = !inside;
        }
      }
      return inside;
    },
    []
  );

  const findRoomAtPoint = useCallback(
    (mx: number, my: number): string | null => {
      for (const room of floor.rooms) {
        if (isPointInPolygon(mx, my, room.polygon)) {
          return room.id;
        }
      }
      return null;
    },
    [floor.rooms, isPointInPolygon]
  );

  // Update building (with history)
  const updateBuilding = useCallback((updates: Partial<BuildingLayout>, saveHistory = true) => {
    if (saveHistory) pushHistory();
    setState((s) => ({
      ...s,
      building: { ...s.building, ...updates },
    }));
  }, [pushHistory]);

  // Update floor (with history)
  const updateFloor = useCallback((floorIndex: number, updates: Partial<Floor>, saveHistory = true) => {
    if (saveHistory) pushHistory();
    setState((s) => ({
      ...s,
      building: {
        ...s.building,
        floors: s.building.floors.map((f, i) =>
          i === floorIndex ? { ...f, ...updates } : f
        ),
      },
    }));
  }, [pushHistory]);

  // Add room
  const addRoom = useCallback((polygon: [number, number][]) => {
    pushHistory();
    const roomCount = floor.rooms.length;
    const newRoom: Room = {
      id: `r-${selectedFloor + 1}${String(roomCount + 1).padStart(2, "0")}`,
      name: `Room ${selectedFloor + 1}${String(roomCount + 1).padStart(2, "0")}`,
      polygon,
    };
    updateFloor(selectedFloor, {
      rooms: [...floor.rooms, newRoom],
    }, false);
    setState((s) => ({ ...s, selectedRoom: newRoom.id, drawingPoints: [] }));
    setEditingRoom({ name: newRoom.name, id: newRoom.id });
  }, [floor.rooms, selectedFloor, updateFloor, pushHistory]);

  // Update room (no history for live edits like name changes)
  const updateRoom = useCallback((roomId: string, updates: Partial<Room>) => {
    updateFloor(selectedFloor, {
      rooms: floor.rooms.map((r) =>
        r.id === roomId ? { ...r, ...updates } : r
      ),
    }, false);
  }, [floor.rooms, selectedFloor, updateFloor]);

  // Delete room
  const deleteRoom = useCallback((roomId: string) => {
    pushHistory();
    updateFloor(selectedFloor, {
      rooms: floor.rooms.filter((r) => r.id !== roomId),
    }, false);
    setState((s) => ({ ...s, selectedRoom: null }));
    setEditingRoom(null);
  }, [floor.rooms, selectedFloor, updateFloor, pushHistory]);

  // Add floor
  const addFloor = useCallback(() => {
    pushHistory();
    const newFloor: Floor = {
      floor_index: building.floors.length,
      label: `Floor ${building.floors.length + 1}`,
      rooms: [],
    };
    updateBuilding({ floors: [...building.floors, newFloor] }, false);
    setState((s) => ({ ...s, selectedFloor: building.floors.length }));
  }, [building.floors, updateBuilding, pushHistory]);

  // Delete floor
  const deleteFloor = useCallback((index: number) => {
    if (building.floors.length <= 1) return;
    pushHistory();
    const newFloors = building.floors
      .filter((_, i) => i !== index)
      .map((f, i) => ({ ...f, floor_index: i }));
    updateBuilding({ floors: newFloors }, false);
    setState((s) => ({
      ...s,
      selectedFloor: Math.min(s.selectedFloor, newFloors.length - 1),
    }));
  }, [building.floors, updateBuilding, pushHistory]);

  // Export JSON
  const exportJSON = useCallback(() => {
    const json = JSON.stringify(building, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${building.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [building]);

  // Copy to clipboard
  const copyToClipboard = useCallback(() => {
    const json = JSON.stringify(building, null, 2);
    navigator.clipboard.writeText(json);
  }, [building]);

  // Import JSON
  const importJSON = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target?.result as string) as BuildingLayout;
        setState((s) => ({
          ...s,
          building: data,
          selectedFloor: 0,
          selectedRoom: null,
          drawingPoints: [],
        }));
      } catch (err) {
        alert("Invalid JSON file");
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  }, []);

  // Canvas click handler
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const [mx, my] = pixelsToMeters(px, py);

      if (tool === "draw") {
        // Check if clicking near first point to close polygon
        if (drawingPoints.length >= 3) {
          const [firstX, firstY] = metersToPixels(drawingPoints[0][0], drawingPoints[0][1]);
          const dist = Math.sqrt((px - firstX) ** 2 + (py - firstY) ** 2);
          if (dist < 15) {
            addRoom(drawingPoints);
            return;
          }
        }
        setState((s) => ({
          ...s,
          drawingPoints: [...s.drawingPoints, [mx, my]],
        }));
      } else if (tool === "select") {
        const roomId = findRoomAtPoint(mx, my);
        setState((s) => ({ ...s, selectedRoom: roomId }));
        if (roomId) {
          const room = floor.rooms.find((r) => r.id === roomId);
          if (room) setEditingRoom({ name: room.name, id: room.id });
        } else {
          setEditingRoom(null);
        }
      }
    },
    [tool, drawingPoints, pixelsToMeters, metersToPixels, addRoom, findRoomAtPoint, floor.rooms]
  );

  // Mouse handlers for pan
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (e.button === 1 || (e.button === 0 && e.altKey)) {
        setIsDragging(true);
        dragStart.current = { x: e.clientX, y: e.clientY };
        panStart.current = { ...pan };
        e.preventDefault();
      }
    },
    [pan]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isDragging) {
        const dx = e.clientX - dragStart.current.x;
        const dy = e.clientY - dragStart.current.y;
        setState((s) => ({
          ...s,
          pan: { x: panStart.current.x + dx, y: panStart.current.y + dy },
        }));
      }
    },
    [isDragging]
  );

  const handleMouseUp = useCallback(() => setIsDragging(false), []);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setState((s) => ({
      ...s,
      zoom: Math.max(0.2, Math.min(10, s.zoom + delta)),
    }));
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (e.key === "Escape") {
        setState((s) => ({ ...s, drawingPoints: [], selectedRoom: null }));
        setEditingRoom(null);
      } else if (e.key === "Delete" && selectedRoom) {
        deleteRoom(selectedRoom);
      } else if (e.key === "d" && !e.ctrlKey && !e.metaKey) {
        setState((s) => ({ ...s, tool: "draw", drawingPoints: [] }));
      } else if (e.key === "s" && !e.ctrlKey && !e.metaKey) {
        setState((s) => ({ ...s, tool: "select" }));
      } else if ((e.ctrlKey || e.metaKey) && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if ((e.ctrlKey || e.metaKey) && (e.key === "y" || (e.key === "z" && e.shiftKey))) {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedRoom, deleteRoom, undo, redo]);

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Background
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Grid
    ctx.strokeStyle = "#2a2a4e";
    ctx.lineWidth = 0.5;

    const gridSizePixels = gridSnap * baseScale * zoom;
    if (gridSizePixels > 5) {
      const [originX, originY] = metersToPixels(0, 0);
      const offsetX = originX % gridSizePixels;
      const offsetY = originY % gridSizePixels;

      for (let x = offsetX; x < canvasWidth; x += gridSizePixels) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasHeight);
        ctx.stroke();
      }
      for (let y = offsetY; y < canvasHeight; y += gridSizePixels) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvasWidth, y);
        ctx.stroke();
      }
    }

    // Building outline
    ctx.strokeStyle = "#4a90d9";
    ctx.lineWidth = 2;
    const [bx1, by1] = metersToPixels(0, 0);
    const [bx2, by2] = metersToPixels(building.width_m, building.height_m);
    ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);

    // Rooms
    for (const room of floor.rooms) {
      const isSelected = room.id === selectedRoom;

      ctx.beginPath();
      const [startX, startY] = metersToPixels(room.polygon[0][0], room.polygon[0][1]);
      ctx.moveTo(startX, startY);
      for (let i = 1; i < room.polygon.length; i++) {
        const [x, y] = metersToPixels(room.polygon[i][0], room.polygon[i][1]);
        ctx.lineTo(x, y);
      }
      ctx.closePath();

      ctx.fillStyle = isSelected ? "rgba(74, 144, 217, 0.4)" : "rgba(100, 150, 200, 0.3)";
      ctx.fill();
      ctx.strokeStyle = isSelected ? "#4a90d9" : "#888";
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.stroke();

      // Room label
      const centerX = room.polygon.reduce((sum, p) => sum + p[0], 0) / room.polygon.length;
      const centerY = room.polygon.reduce((sum, p) => sum + p[1], 0) / room.polygon.length;
      const [labelX, labelY] = metersToPixels(centerX, centerY);

      ctx.fillStyle = "#fff";
      ctx.font = "bold 12px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(room.name, labelX, labelY - 8);
      ctx.font = "10px monospace";
      ctx.fillStyle = "#aaa";
      ctx.fillText(room.id, labelX, labelY + 8);

      // Vertex handles when selected
      if (isSelected) {
        for (const [vx, vy] of room.polygon) {
          const [px, py] = metersToPixels(vx, vy);
          ctx.fillStyle = "#4a90d9";
          ctx.beginPath();
          ctx.arc(px, py, 5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Drawing in progress
    if (drawingPoints.length > 0) {
      ctx.beginPath();
      const [startX, startY] = metersToPixels(drawingPoints[0][0], drawingPoints[0][1]);
      ctx.moveTo(startX, startY);
      for (let i = 1; i < drawingPoints.length; i++) {
        const [x, y] = metersToPixels(drawingPoints[i][0], drawingPoints[i][1]);
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "#4aff88";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Points
      for (let i = 0; i < drawingPoints.length; i++) {
        const [px, py] = metersToPixels(drawingPoints[i][0], drawingPoints[i][1]);
        ctx.fillStyle = i === 0 ? "#ff4a4a" : "#4aff88";
        ctx.beginPath();
        ctx.arc(px, py, i === 0 ? 8 : 5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Closing hint
      if (drawingPoints.length >= 3) {
        ctx.fillStyle = "#ff4a4a";
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        const [hintX, hintY] = metersToPixels(drawingPoints[0][0], drawingPoints[0][1]);
        ctx.fillText("click to close", hintX, hintY - 15);
      }
    }

    // Tool indicator
    ctx.fillStyle = "#4a90d9";
    ctx.font = "bold 12px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`Tool: ${tool.toUpperCase()} | Zoom: ${Math.round(zoom * 100)}%`, 10, 20);
    ctx.fillText(`Grid: ${gridSnap}m | Floor: ${floor.label}`, 10, 36);
  }, [state, metersToPixels, baseScale, floor, building, selectedRoom, drawingPoints, tool, zoom, gridSnap]);

  return (
    <div className="editor-container">
      <div className="editor-sidebar">
        <h2>Floor Plan Editor</h2>

        <section className="editor-section">
          <h3>Building</h3>
          <label>
            Name
            <input
              type="text"
              value={building.name}
              onChange={(e) => updateBuilding({ name: e.target.value })}
            />
          </label>
          <label>
            ID
            <input
              type="text"
              value={building.id}
              onChange={(e) => updateBuilding({ id: e.target.value })}
            />
          </label>
          <div className="input-row">
            <label>
              Width (m)
              <input
                type="number"
                value={building.width_m}
                onChange={(e) => updateBuilding({ width_m: Number(e.target.value) })}
              />
            </label>
            <label>
              Height (m)
              <input
                type="number"
                value={building.height_m}
                onChange={(e) => updateBuilding({ height_m: Number(e.target.value) })}
              />
            </label>
          </div>
        </section>

        <section className="editor-section">
          <h3>Floors</h3>
          <div className="floor-list">
            {building.floors.map((f, i) => (
              <div
                key={i}
                className={`floor-item ${i === selectedFloor ? "active" : ""}`}
                onClick={() => setState((s) => ({ ...s, selectedFloor: i, selectedRoom: null }))}
              >
                <input
                  type="text"
                  value={f.label}
                  onChange={(e) => updateFloor(i, { label: e.target.value })}
                  onClick={(e) => e.stopPropagation()}
                />
                {building.floors.length > 1 && (
                  <button
                    className="delete-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteFloor(i);
                    }}
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
          </div>
          <button onClick={addFloor}>+ Add Floor</button>
        </section>

        <section className="editor-section">
          <h3>Tools</h3>
          <div className="tool-buttons">
            <button
              className={tool === "select" ? "active" : ""}
              onClick={() => setState((s) => ({ ...s, tool: "select" }))}
            >
              Select (S)
            </button>
            <button
              className={tool === "draw" ? "active" : ""}
              onClick={() => setState((s) => ({ ...s, tool: "draw", drawingPoints: [] }))}
            >
              Draw (D)
            </button>
          </div>
          <div className="tool-buttons">
            <button
              onClick={undo}
              disabled={history.length === 0}
              title="Undo (Ctrl+Z)"
            >
              ↩ Undo
            </button>
            <button
              onClick={redo}
              disabled={future.length === 0}
              title="Redo (Ctrl+Y)"
            >
              ↪ Redo
            </button>
          </div>
          <label>
            Grid Snap (m)
            <select
              value={gridSnap}
              onChange={(e) => setState((s) => ({ ...s, gridSnap: Number(e.target.value) }))}
            >
              <option value={0}>Off</option>
              <option value={0.25}>0.25</option>
              <option value={0.5}>0.5</option>
              <option value={1}>1</option>
              <option value={2}>2</option>
            </select>
          </label>
        </section>

        {editingRoom && (
          <section className="editor-section">
            <h3>Selected Room</h3>
            <label>
              Name
              <input
                type="text"
                value={editingRoom.name}
                onChange={(e) => {
                  setEditingRoom({ ...editingRoom, name: e.target.value });
                  updateRoom(editingRoom.id, { name: e.target.value });
                }}
              />
            </label>
            <label>
              ID
              <input
                type="text"
                value={editingRoom.id}
                onChange={(e) => {
                  const oldId = editingRoom.id;
                  const newId = e.target.value;
                  setEditingRoom({ ...editingRoom, id: newId });
                  // Update room with new ID
                  updateFloor(selectedFloor, {
                    rooms: floor.rooms.map((r) =>
                      r.id === oldId ? { ...r, id: newId } : r
                    ),
                  });
                  setState((s) => ({ ...s, selectedRoom: newId }));
                }}
              />
            </label>
            <button className="delete-btn-full" onClick={() => deleteRoom(editingRoom.id)}>
              Delete Room
            </button>
          </section>
        )}

        <section className="editor-section">
          <h3>Rooms ({floor.rooms.length})</h3>
          <div className="room-list">
            {floor.rooms.map((room) => (
              <div
                key={room.id}
                className={`room-item ${room.id === selectedRoom ? "active" : ""}`}
                onClick={() => {
                  setState((s) => ({ ...s, selectedRoom: room.id, tool: "select" }));
                  setEditingRoom({ name: room.name, id: room.id });
                }}
              >
                <span className="room-name">{room.name}</span>
                <span className="room-id">{room.id}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="editor-section">
          <h3>Import / Export</h3>
          <div className="export-buttons">
            <button onClick={exportJSON}>Download JSON</button>
            <button onClick={copyToClipboard}>Copy to Clipboard</button>
            <label className="file-input-label">
              Import JSON
              <input type="file" accept=".json" onChange={importJSON} />
            </label>
          </div>
        </section>

        <section className="editor-section help">
          <h3>Help</h3>
          <ul>
            <li><strong>Draw:</strong> Click to add points, click first point to close</li>
            <li><strong>Pan:</strong> Alt+drag or middle mouse</li>
            <li><strong>Zoom:</strong> Scroll wheel</li>
            <li><strong>Delete:</strong> Select room + Delete key</li>
            <li><strong>Undo:</strong> Ctrl+Z</li>
            <li><strong>Redo:</strong> Ctrl+Y or Ctrl+Shift+Z</li>
            <li><strong>Cancel:</strong> Escape</li>
          </ul>
        </section>
      </div>

      <div className="editor-canvas-container">
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          onClick={handleCanvasClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          style={{ cursor: tool === "draw" ? "crosshair" : isDragging ? "grabbing" : "default" }}
        />
      </div>
    </div>
  );
}

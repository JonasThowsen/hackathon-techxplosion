import { useRef, useEffect, useCallback } from "react";
import type { Floor, RoomMetrics, MetricType } from "../types";
import { METRIC_CONFIGS } from "../types";
import { getMetricColor } from "../utils/colors";

interface FloorPlanProps {
  floor: Floor;
  metrics: Record<string, RoomMetrics>;
  selectedMetric: MetricType;
  buildingWidth: number;
  buildingHeight: number;
  canvasWidth?: number;
  canvasHeight?: number;
  onRoomHover?: (roomId: string | null, x: number, y: number) => void;
  onRoomClick?: (roomId: string) => void;
}

const WALL_THICKNESS = 3;
const OUTER_WALL_THICKNESS = 5;

export function FloorPlan({
  floor,
  metrics,
  selectedMetric,
  buildingWidth,
  buildingHeight,
  canvasWidth = 700,
  canvasHeight = 420,
  onRoomHover,
  onRoomClick,
}: FloorPlanProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Add padding for outer walls
  const padding = 20;
  const innerWidth = canvasWidth - padding * 2;
  const innerHeight = canvasHeight - padding * 2;

  const scaleX = innerWidth / buildingWidth;
  const scaleY = innerHeight / buildingHeight;

  const metersToPixels = useCallback(
    (x: number, y: number): [number, number] => {
      return [padding + x * scaleX, padding + y * scaleY];
    },
    [scaleX, scaleY, padding]
  );

  const pixelsToMeters = useCallback(
    (px: number, py: number): [number, number] => {
      return [(px - padding) / scaleX, (py - padding) / scaleY];
    },
    [scaleX, scaleY, padding]
  );

  const isPointInPolygon = useCallback(
    (px: number, py: number, polygon: [number, number][]): boolean => {
      let inside = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0],
          yi = polygon[i][1];
        const xj = polygon[j][0],
          yj = polygon[j][1];

        if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
          inside = !inside;
        }
      }
      return inside;
    },
    []
  );

  const findRoomAtPixel = useCallback(
    (px: number, py: number): string | null => {
      const [mx, my] = pixelsToMeters(px, py);
      for (const room of floor.rooms) {
        if (isPointInPolygon(mx, my, room.polygon)) {
          return room.id;
        }
      }
      return null;
    },
    [floor.rooms, pixelsToMeters, isPointInPolygon]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Background - blueprint style
    ctx.fillStyle = "#0a1628";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw grid pattern for blueprint effect
    ctx.strokeStyle = "#152238";
    ctx.lineWidth = 0.5;
    const gridSize = 20;
    for (let x = 0; x < canvasWidth; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasHeight);
      ctx.stroke();
    }
    for (let y = 0; y < canvasHeight; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasWidth, y);
      ctx.stroke();
    }

    const config = METRIC_CONFIGS[selectedMetric];

    // Draw rooms - filled with metric color
    for (const room of floor.rooms) {
      const roomMetrics = metrics[room.id];

      // Get fill color based on metric
      let fillColor = "rgba(30, 50, 80, 0.6)";
      if (roomMetrics) {
        const value = roomMetrics[selectedMetric];
        const baseColor = getMetricColor(value, config.min, config.max);
        // Add transparency for floor plan look
        fillColor = baseColor.replace("hsl", "hsla").replace(")", ", 0.6)");
      }

      // Draw room polygon
      ctx.beginPath();
      const [startX, startY] = metersToPixels(room.polygon[0][0], room.polygon[0][1]);
      ctx.moveTo(startX, startY);

      for (let i = 1; i < room.polygon.length; i++) {
        const [x, y] = metersToPixels(room.polygon[i][0], room.polygon[i][1]);
        ctx.lineTo(x, y);
      }
      ctx.closePath();

      // Fill
      ctx.fillStyle = fillColor;
      ctx.fill();
    }

    // Draw interior walls (between rooms)
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = WALL_THICKNESS;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const room of floor.rooms) {
      ctx.beginPath();
      const [startX, startY] = metersToPixels(room.polygon[0][0], room.polygon[0][1]);
      ctx.moveTo(startX, startY);

      for (let i = 1; i < room.polygon.length; i++) {
        const [x, y] = metersToPixels(room.polygon[i][0], room.polygon[i][1]);
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }

    // Draw outer building walls (thicker)
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = OUTER_WALL_THICKNESS;
    const [ox1, oy1] = metersToPixels(0, 0);
    const [ox2, oy2] = metersToPixels(buildingWidth, buildingHeight);
    ctx.strokeRect(ox1, oy1, ox2 - ox1, oy2 - oy1);

    // Draw room labels and values
    for (const room of floor.rooms) {
      const roomMetrics = metrics[room.id];

      // Calculate centroid
      const centerX =
        room.polygon.reduce((sum, p) => sum + p[0], 0) / room.polygon.length;
      const centerY =
        room.polygon.reduce((sum, p) => sum + p[1], 0) / room.polygon.length;
      const [labelX, labelY] = metersToPixels(centerX, centerY);

      // Room name
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(room.name, labelX, labelY - 12);

      // Metric value
      if (roomMetrics) {
        const value = roomMetrics[selectedMetric];
        const displayValue =
          selectedMetric === "occupancy"
            ? `${Math.round(value * 100)}%`
            : `${value.toFixed(1)}${config.unit}`;

        ctx.font = "bold 16px monospace";
        ctx.fillStyle = "#fff";
        ctx.fillText(displayValue, labelX, labelY + 8);

        // Waste indicator
        if (roomMetrics.waste_patterns.length > 0) {
          ctx.fillStyle = "#ff4444";
          ctx.font = "bold 14px sans-serif";
          ctx.fillText("⚠", labelX + 35, labelY - 12);
        }
      }
    }

    // Floor label
    ctx.fillStyle = "#4a90d9";
    ctx.font = "bold 14px monospace";
    ctx.textAlign = "left";
    ctx.fillText(floor.label.toUpperCase(), padding, 14);

    // Scale indicator
    const scaleBarMeters = 5;
    const scaleBarPixels = scaleBarMeters * scaleX;
    const scaleY1 = canvasHeight - 10;
    ctx.strokeStyle = "#4a90d9";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, scaleY1);
    ctx.lineTo(padding + scaleBarPixels, scaleY1);
    ctx.stroke();
    ctx.fillStyle = "#4a90d9";
    ctx.font = "10px monospace";
    ctx.fillText(`${scaleBarMeters}m`, padding + scaleBarPixels + 5, scaleY1 + 3);
  }, [floor, metrics, selectedMetric, canvasWidth, canvasHeight, metersToPixels, buildingWidth, buildingHeight]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onRoomHover) return;
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const roomId = findRoomAtPixel(x, y);
      onRoomHover(roomId, e.clientX, e.clientY);
    },
    [onRoomHover, findRoomAtPixel]
  );

  const handleMouseLeave = useCallback(() => {
    onRoomHover?.(null, 0, 0);
  }, [onRoomHover]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onRoomClick) return;
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const roomId = findRoomAtPixel(x, y);
      if (roomId) onRoomClick(roomId);
    },
    [onRoomClick, findRoomAtPixel]
  );

  return (
    <canvas
      ref={canvasRef}
      width={canvasWidth}
      height={canvasHeight}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      style={{
        cursor: "crosshair",
        borderRadius: "4px",
        border: "1px solid #2a4a6a",
      }}
    />
  );
}

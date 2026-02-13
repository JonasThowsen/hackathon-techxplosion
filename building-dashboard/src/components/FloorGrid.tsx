import { useRef, useEffect, useCallback } from "react";
import type { Floor, RoomMetrics, MetricType } from "../types";
import { METRIC_CONFIGS } from "../types";
import { getMetricColor } from "../utils/colors";

interface FloorGridProps {
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

export function FloorGrid({
  floor,
  metrics,
  selectedMetric,
  buildingWidth,
  buildingHeight,
  canvasWidth = 600,
  canvasHeight = 360,
  onRoomHover,
  onRoomClick,
}: FloorGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Scale factor: meters to pixels
  const scaleX = canvasWidth / buildingWidth;
  const scaleY = canvasHeight / buildingHeight;

  const metersToPixels = useCallback(
    (x: number, y: number): [number, number] => {
      return [x * scaleX, y * scaleY];
    },
    [scaleX, scaleY]
  );

  const pixelsToMeters = useCallback(
    (px: number, py: number): [number, number] => {
      return [px / scaleX, py / scaleY];
    },
    [scaleX, scaleY]
  );

  // Check if point is inside polygon using ray casting
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

  // Find room at given pixel coordinates
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

  // Draw the floor
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    const config = METRIC_CONFIGS[selectedMetric];

    // Draw each room
    for (const room of floor.rooms) {
      const roomMetrics = metrics[room.id];

      // Get color based on metric value
      let fillColor = "#333";
      if (roomMetrics) {
        const value = roomMetrics[selectedMetric];
        fillColor = getMetricColor(value, config.min, config.max);
      }

      // Draw polygon
      ctx.beginPath();
      const [startX, startY] = metersToPixels(room.polygon[0][0], room.polygon[0][1]);
      ctx.moveTo(startX, startY);

      for (let i = 1; i < room.polygon.length; i++) {
        const [x, y] = metersToPixels(room.polygon[i][0], room.polygon[i][1]);
        ctx.lineTo(x, y);
      }
      ctx.closePath();

      ctx.fillStyle = fillColor;
      ctx.fill();

      // Draw border
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw room label
      const centerX =
        room.polygon.reduce((sum, p) => sum + p[0], 0) / room.polygon.length;
      const centerY =
        room.polygon.reduce((sum, p) => sum + p[1], 0) / room.polygon.length;
      const [labelX, labelY] = metersToPixels(centerX, centerY);

      ctx.fillStyle = "#fff";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(room.name, labelX, labelY - 8);

      // Draw metric value
      if (roomMetrics) {
        const value = roomMetrics[selectedMetric];
        const displayValue =
          selectedMetric === "occupancy"
            ? `${Math.round(value * 100)}%`
            : `${value.toFixed(1)}${config.unit}`;
        ctx.font = "bold 14px sans-serif";
        ctx.fillText(displayValue, labelX, labelY + 8);

        // Draw waste indicator
        if (roomMetrics.waste_patterns.length > 0) {
          ctx.fillStyle = "#ff4444";
          ctx.font = "16px sans-serif";
          ctx.fillText("⚠", labelX + 30, labelY - 8);
        }
      }
    }
  }, [floor, metrics, selectedMetric, canvasWidth, canvasHeight, metersToPixels]);

  // Handle mouse events
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
      if (roomId) {
        onRoomClick(roomId);
      }
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
      style={{ cursor: "pointer", borderRadius: "8px" }}
    />
  );
}

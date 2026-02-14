import { useRef, useEffect, useCallback, useState } from "react";
import type { Floor, RoomMetrics, MetricType, HeatFlow } from "../types";
import { METRIC_CONFIGS } from "../types";
import { getMetricColor } from "../utils/colors";

export interface SunPosition {
  azimuth: number;    // Direction in degrees (0=North, 90=East, 180=South, 270=West)
  elevation: number;  // Angle above horizon in degrees (0-90)
  visible: boolean;   // Whether sun is above horizon
}

interface FloorPlanProps {
  floor: Floor;
  metrics: Record<string, RoomMetrics>;
  selectedMetric: MetricType;
  buildingWidth: number;
  buildingHeight: number;
  canvasWidth?: number;
  canvasHeight?: number;
  sunPosition?: SunPosition;
  heatFlows?: HeatFlow[];
  onRoomHover?: (roomId: string | null, x: number, y: number) => void;
  onRoomClick?: (roomId: string) => void;
}

const WALL_THICKNESS = 2;
const OUTER_WALL_THICKNESS = 3;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;
const ZOOM_STEP = 0.2;

export function FloorPlan({
  floor,
  metrics,
  selectedMetric,
  buildingWidth,
  buildingHeight,
  canvasWidth = 800,
  canvasHeight = 540,
  sunPosition,
  heatFlows,
  onRoomHover,
  onRoomClick,
}: FloorPlanProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [animationTick, setAnimationTick] = useState(0);

  // Animate pulse for waste rooms
  useEffect(() => {
    const hasWasteRooms = Object.values(metrics).some(m => (m.waste_patterns?.length ?? 0) > 0);
    if (!hasWasteRooms) return;

    const interval = setInterval(() => {
      setAnimationTick(t => t + 1);
    }, 50);
    return () => clearInterval(interval);
  }, [metrics]);

  // Zoom and pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const panStart = useRef({ x: 0, y: 0 });

  const padding = 20;
  const baseScaleX = (canvasWidth - padding * 2) / buildingWidth;
  const baseScaleY = (canvasHeight - padding * 2) / buildingHeight;
  const baseScale = Math.min(baseScaleX, baseScaleY);

  const metersToPixels = useCallback(
    (mx: number, my: number): [number, number] => {
      const cx = canvasWidth / 2;
      const cy = canvasHeight / 2;
      const px = cx + (mx - buildingWidth / 2) * baseScale * zoom + pan.x;
      const py = cy + (my - buildingHeight / 2) * baseScale * zoom + pan.y;
      return [px, py];
    },
    [baseScale, zoom, pan, canvasWidth, canvasHeight, buildingWidth, buildingHeight]
  );

  const pixelsToMeters = useCallback(
    (px: number, py: number): [number, number] => {
      const cx = canvasWidth / 2;
      const cy = canvasHeight / 2;
      const mx = (px - cx - pan.x) / (baseScale * zoom) + buildingWidth / 2;
      const my = (py - cy - pan.y) / (baseScale * zoom) + buildingHeight / 2;
      return [mx, my];
    },
    [baseScale, zoom, pan, canvasWidth, canvasHeight, buildingWidth, buildingHeight]
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

  // Helper to find shared edge between two rooms
  const findSharedEdge = useCallback((
    roomA: { polygon: [number, number][] },
    roomB: { polygon: [number, number][] }
  ): [number, number, number, number] | null => {
    const vertsA = roomA.polygon;
    const vertsB = roomB.polygon;
    
    for (let i = 0; i < vertsA.length; i++) {
      const a1 = vertsA[i];
      const a2 = vertsA[(i + 1) % vertsA.length];
      
      for (let j = 0; j < vertsB.length; j++) {
        const b1 = vertsB[j];
        const b2 = vertsB[(j + 1) % vertsB.length];
        
        const share1 = (Math.abs(a1[0] - b1[0]) < 0.01 && Math.abs(a1[1] - b1[1]) < 0.01);
        const share2 = (Math.abs(a1[0] - b2[0]) < 0.01 && Math.abs(a1[1] - b2[1]) < 0.01);
        const share3 = (Math.abs(a2[0] - b1[0]) < 0.01 && Math.abs(a2[1] - b1[1]) < 0.01);
        const share4 = (Math.abs(a2[0] - b2[0]) < 0.01 && Math.abs(a2[1] - b2[1]) < 0.01);
        
        if ((share1 && share4) || (share2 && share3)) {
          return [(a1[0] + a2[0]) / 2, (a1[1] + a2[1]) / 2, (b1[0] + b2[0]) / 2, (b1[1] + b2[1]) / 2];
        }
      }
    }
    return null;
  }, []);

  // Zoom handlers
  const handleZoom = useCallback((delta: number, centerX?: number, centerY?: number) => {
    setZoom((prevZoom) => {
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, prevZoom + delta));

      // Zoom toward cursor position if provided
      if (centerX !== undefined && centerY !== undefined) {
        const zoomRatio = newZoom / prevZoom;
        setPan((prevPan) => ({
          x: centerX - (centerX - prevPan.x) * zoomRatio,
          y: centerY - (centerY - prevPan.y) * zoomRatio,
        }));
      }

      return newZoom;
    });
  }, []);

  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const centerX = e.clientX - rect.left - canvasWidth / 2;
      const centerY = e.clientY - rect.top - canvasHeight / 2;
      const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
      handleZoom(delta, centerX, centerY);
    },
    [handleZoom, canvasWidth, canvasHeight]
  );

  // Pan handlers
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 0) { // Left click
      setIsDragging(true);
      dragStart.current = { x: e.clientX, y: e.clientY };
      panStart.current = { ...pan };
    }
  }, [pan]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isDragging) {
        const dx = e.clientX - dragStart.current.x;
        const dy = e.clientY - dragStart.current.y;
        setPan({
          x: panStart.current.x + dx,
          y: panStart.current.y + dy,
        });
      } else if (onRoomHover) {
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const roomId = findRoomAtPixel(x, y);
        onRoomHover(roomId, e.clientX, e.clientY);
      }
    },
    [isDragging, onRoomHover, findRoomAtPixel]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsDragging(false);
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

  // Reset view
  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  // Fit to content
  const fitToContent = useCallback(() => {
    if (floor.rooms.length === 0) return;

    // Find bounding box of all rooms
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const room of floor.rooms) {
      for (const [x, y] of room.polygon) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }

    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    // Calculate zoom to fit content with padding
    const fitZoomX = (canvasWidth - padding * 4) / (contentWidth * baseScale);
    const fitZoomY = (canvasHeight - padding * 4) / (contentHeight * baseScale);
    const fitZoom = Math.min(fitZoomX, fitZoomY, MAX_ZOOM);

    // Calculate pan to center content
    const panX = -(centerX - buildingWidth / 2) * baseScale * fitZoom;
    const panY = -(centerY - buildingHeight / 2) * baseScale * fitZoom;

    setZoom(fitZoom);
    setPan({ x: panX, y: panY });
  }, [floor.rooms, baseScale, canvasWidth, canvasHeight, buildingWidth, buildingHeight, padding]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Background
    ctx.fillStyle = "#0a1628";
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Grid pattern
    ctx.strokeStyle = "#152238";
    ctx.lineWidth = 0.5;
    const gridSpacing = 20;
    for (let x = 0; x < canvasWidth; x += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasHeight);
      ctx.stroke();
    }
    for (let y = 0; y < canvasHeight; y += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasWidth, y);
      ctx.stroke();
    }

    const config = METRIC_CONFIGS[selectedMetric];
    const scaledWall = Math.max(1, WALL_THICKNESS * zoom);
    const scaledOuterWall = Math.max(2, OUTER_WALL_THICKNESS * zoom);

    // Pulse animation for waste rooms
    const pulsePhase = (Date.now() % 2000) / 2000;
    const pulseIntensity = 0.3 + 0.7 * Math.abs(Math.sin(pulsePhase * Math.PI));

    // Draw rooms
    for (const room of floor.rooms) {
      const roomMetrics = metrics[room.id];
      const hasWaste = roomMetrics && (roomMetrics.waste_patterns?.length ?? 0) > 0;

      let fillColor = "rgba(30, 50, 80, 0.6)";
      if (roomMetrics) {
        const value = roomMetrics[selectedMetric];
        const baseColor = getMetricColor(value, config.min, config.max);
        fillColor = baseColor.replace("hsl", "hsla").replace(")", ", 0.6)");
      }

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

      // Pulse effect for waste rooms - color varies by pattern type
      if (hasWaste && roomMetrics) {
        const patterns = roomMetrics.waste_patterns ?? [];
        const hasWindowOpen = patterns.includes("rapid_heat_loss");
        const hasExcessiveVent = patterns.includes("excessive_ventilation");

        if (hasWindowOpen) {
          // Cyan/blue pulse for open window (cold draft)
          ctx.strokeStyle = `rgba(80, 200, 255, ${pulseIntensity * 0.9})`;
        } else if (hasExcessiveVent) {
          // Teal/green pulse for ventilation waste
          ctx.strokeStyle = `rgba(100, 220, 180, ${pulseIntensity * 0.8})`;
        } else {
          // Red pulse for heating waste
          ctx.strokeStyle = `rgba(255, 100, 100, ${pulseIntensity * 0.8})`;
        }
        ctx.lineWidth = 3 * zoom;
        ctx.stroke();
      }
    }

    // Interior walls
    ctx.strokeStyle = "#e0e0e0";
    ctx.lineWidth = scaledWall;
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

    // Outer building walls
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = scaledOuterWall;
    const [ox1, oy1] = metersToPixels(0, 0);
    const [ox2, oy2] = metersToPixels(buildingWidth, buildingHeight);
    ctx.strokeRect(ox1, oy1, ox2 - ox1, oy2 - oy1);

    // Heat flow arrows
    if (heatFlows && heatFlows.length > 0) {
      const floorRoomIds = new Set(floor.rooms.map(r => r.id));
      
      for (const flow of heatFlows) {
        // Only show arrows for rooms on this floor
        if (!floorRoomIds.has(flow.from_room) || !floorRoomIds.has(flow.to_room)) continue;
        
        const fromRoom = floor.rooms.find(r => r.id === flow.from_room);
        const toRoom = floor.rooms.find(r => r.id === flow.to_room);
        if (!fromRoom || !toRoom) continue;
        
        // Get center of each room to determine direction
        const fromCenterX = fromRoom.polygon.reduce((sum, p) => sum + p[0], 0) / fromRoom.polygon.length;
        const fromCenterY = fromRoom.polygon.reduce((sum, p) => sum + p[1], 0) / fromRoom.polygon.length;
        
        // Find the shared edge and determine which point is closer to fromRoom center
        const edge = findSharedEdge(fromRoom, toRoom);
        if (!edge) continue;
        
        const [ex1, ey1, ex2, ey2] = edge;
        
        // Distance from fromRoom center to each edge point
        const dist1 = Math.sqrt((ex1 - fromCenterX) ** 2 + (ey1 - fromCenterY) ** 2);
        const dist2 = Math.sqrt((ex2 - fromCenterX) ** 2 + (ey2 - fromCenterY) ** 2);
        
        // Start from the edge point closer to fromRoom, end at the one closer to toRoom
        let startX, startY, endX, endY;
        if (dist1 < dist2) {
          startX = ex1; startY = ey1;
          endX = ex2; endY = ey2;
        } else {
          startX = ex2; startY = ey2;
          endX = ex1; endY = ey1;
        }
        
        const [px1, py1] = metersToPixels(startX, startY);
        const [px2, py2] = metersToPixels(endX, endY);
        
        // Heat is flowing from fromRoom to toRoom (watts is always positive in backend)
        const intensity = Math.min(1, Math.abs(flow.watts) / 150);
        const color = `rgba(255, 100, 30, ${0.7 + intensity * 0.3})`;
        
        // Draw white outline for visibility
        const lineWidth = Math.max(3, 4 * zoom);
        const headSize = Math.max(10, 12 * zoom);
        
        // White background line
        ctx.beginPath();
        ctx.moveTo(px1, py1);
        ctx.lineTo(px2, py2);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        ctx.lineWidth = lineWidth + 3;
        ctx.lineCap = "round";
        ctx.stroke();
        
        // Colored line
        ctx.beginPath();
        ctx.moveTo(px1, py1);
        ctx.lineTo(px2, py2);
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.stroke();
        
        // Arrow head
        const angle = Math.atan2(py2 - py1, px2 - px1);
        
        // White outline for arrow head
        ctx.beginPath();
        ctx.moveTo(px2, py2);
        ctx.lineTo(
          px2 - headSize * Math.cos(angle - Math.PI / 6),
          py2 - headSize * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          px2 - headSize * Math.cos(angle + Math.PI / 6),
          py2 - headSize * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.fill();
        
        // Colored arrow head
        ctx.beginPath();
        ctx.moveTo(px2, py2);
        ctx.lineTo(
          px2 - (headSize - 2) * Math.cos(angle - Math.PI / 6),
          py2 - (headSize - 2) * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          px2 - (headSize - 2) * Math.cos(angle + Math.PI / 6),
          py2 - (headSize - 2) * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
        
        // Draw watts label
        const midX = (px1 + px2) / 2;
        const midY = (py1 + py2) / 2;
        const labelText = `${flow.watts.toFixed(0)}W`;
        ctx.font = `bold ${Math.max(10, 11 * zoom)}px monospace`;
        const textMetrics = ctx.measureText(labelText);
        const padding = 3;
        
        ctx.fillStyle = "rgba(10, 20, 40, 0.85)";
        ctx.fillRect(
          midX - textMetrics.width / 2 - padding,
          midY - 8 * zoom - padding - 6,
          textMetrics.width + padding * 2,
          14 * zoom
        );
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(
          midX - textMetrics.width / 2 - padding,
          midY - 8 * zoom - padding - 6,
          textMetrics.width + padding * 2,
          14 * zoom
        );
        
        ctx.fillStyle = color;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(labelText, midX, midY - 8 * zoom);
      }
    }

    // Labels (only show if zoomed in enough)
    if (zoom >= 0.8) {
      for (const room of floor.rooms) {
        const roomMetrics = metrics[room.id];
        const centerX = room.polygon.reduce((sum, p) => sum + p[0], 0) / room.polygon.length;
        const centerY = room.polygon.reduce((sum, p) => sum + p[1], 0) / room.polygon.length;
        const [labelX, labelY] = metersToPixels(centerX, centerY);

        // Check if label is visible
        if (labelX < 0 || labelX > canvasWidth || labelY < 0 || labelY > canvasHeight) continue;

        const fontSize = Math.max(9, Math.min(14, 11 * zoom));

        ctx.fillStyle = "#ffffff";
        ctx.font = `bold ${fontSize}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(room.name, labelX, labelY - fontSize);

        if (roomMetrics) {
          const value = roomMetrics[selectedMetric];
          const displayValue = `${value.toFixed(1)}${config.unit}`;

          ctx.font = `bold ${fontSize + 4}px monospace`;
          ctx.fillText(displayValue, labelX, labelY + fontSize * 0.5);

          if ((roomMetrics.waste_patterns?.length ?? 0) > 0) {
            const patterns = roomMetrics.waste_patterns ?? [];
            const hasWindowOpen = patterns.includes("rapid_heat_loss");
            const hasExcessiveVent = patterns.includes("excessive_ventilation");

            let icon = "⚠";
            let iconColor = "#ff4444";
            if (hasWindowOpen) {
              icon = "🪟";
              iconColor = "#50c8ff";
            } else if (hasExcessiveVent) {
              icon = "💨";
              iconColor = "#64dcb4";
            }

            ctx.fillStyle = iconColor;
            ctx.font = `bold ${fontSize}px sans-serif`;
            ctx.fillText(icon, labelX + 30 * zoom, labelY - fontSize);
          }
        }
      }
    }

    // Floor label
    ctx.fillStyle = "#4a90d9";
    ctx.font = "bold 14px monospace";
    ctx.textAlign = "left";
    ctx.fillText(floor.label.toUpperCase(), padding, 14);

    // Zoom indicator
    ctx.fillStyle = "#4a90d9";
    ctx.font = "12px monospace";
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round(zoom * 100)}%`, canvasWidth - padding, 14);

    // Scale bar
    const scaleBarMeters = zoom > 2 ? 2 : zoom > 1 ? 5 : 10;
    const scaleBarPixels = scaleBarMeters * baseScale * zoom;
    const scaleY1 = canvasHeight - 12;
    ctx.strokeStyle = "#4a90d9";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, scaleY1);
    ctx.lineTo(padding + scaleBarPixels, scaleY1);
    ctx.stroke();
    ctx.fillStyle = "#4a90d9";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`${scaleBarMeters}m`, padding + scaleBarPixels + 5, scaleY1 + 3);

    // Sun position indicator
    if (sunPosition && sunPosition.visible) {
      const sunIndicatorX = canvasWidth - 60;
      const sunIndicatorY = canvasHeight - 60;
      const sunRadius = 40;

      // Draw compass circle
      ctx.beginPath();
      ctx.arc(sunIndicatorX, sunIndicatorY, sunRadius, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(10, 22, 40, 0.8)";
      ctx.fill();
      ctx.strokeStyle = "#3a5a7a";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Draw compass directions
      ctx.fillStyle = "#4a6a8a";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("N", sunIndicatorX, sunIndicatorY - sunRadius + 10);
      ctx.fillText("S", sunIndicatorX, sunIndicatorY + sunRadius - 10);
      ctx.fillText("E", sunIndicatorX + sunRadius - 10, sunIndicatorY);
      ctx.fillText("W", sunIndicatorX - sunRadius + 10, sunIndicatorY);

      // Convert azimuth to canvas angle (0=North=up, clockwise)
      // Canvas: 0 = right, so we need to rotate -90 and flip
      const sunAngle = ((sunPosition.azimuth - 90) * Math.PI) / 180;

      // Sun intensity based on elevation (higher = brighter)
      const intensity = Math.min(1, sunPosition.elevation / 60);

      // Draw sun rays (from edge pointing inward)
      const rayLength = sunRadius * 0.7;
      const rayStartX = sunIndicatorX + Math.cos(sunAngle) * (sunRadius - 5);
      const rayStartY = sunIndicatorY + Math.sin(sunAngle) * (sunRadius - 5);
      const rayEndX = sunIndicatorX + Math.cos(sunAngle) * (sunRadius - rayLength);
      const rayEndY = sunIndicatorY + Math.sin(sunAngle) * (sunRadius - rayLength);

      // Draw ray line
      ctx.beginPath();
      ctx.moveTo(rayStartX, rayStartY);
      ctx.lineTo(rayEndX, rayEndY);
      ctx.strokeStyle = `rgba(255, 200, 50, ${0.5 + intensity * 0.5})`;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw sun circle at the edge
      ctx.beginPath();
      ctx.arc(rayStartX, rayStartY, 6, 0, Math.PI * 2);
      const sunGradient = ctx.createRadialGradient(rayStartX, rayStartY, 0, rayStartX, rayStartY, 6);
      sunGradient.addColorStop(0, `rgba(255, 240, 100, ${0.8 + intensity * 0.2})`);
      sunGradient.addColorStop(0.5, `rgba(255, 180, 50, ${0.6 + intensity * 0.3})`);
      sunGradient.addColorStop(1, `rgba(255, 120, 20, ${0.3 + intensity * 0.2})`);
      ctx.fillStyle = sunGradient;
      ctx.fill();

      // Draw arrowhead pointing inward
      const arrowSize = 6;
      const arrowAngle = Math.PI / 6;
      ctx.beginPath();
      ctx.moveTo(rayEndX, rayEndY);
      ctx.lineTo(
        rayEndX + arrowSize * Math.cos(sunAngle + Math.PI - arrowAngle),
        rayEndY + arrowSize * Math.sin(sunAngle + Math.PI - arrowAngle)
      );
      ctx.moveTo(rayEndX, rayEndY);
      ctx.lineTo(
        rayEndX + arrowSize * Math.cos(sunAngle + Math.PI + arrowAngle),
        rayEndY + arrowSize * Math.sin(sunAngle + Math.PI + arrowAngle)
      );
      ctx.strokeStyle = `rgba(255, 200, 50, ${0.5 + intensity * 0.5})`;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw elevation text
      ctx.fillStyle = "#ffc832";
      ctx.font = "bold 10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`${Math.round(sunPosition.elevation)}°`, sunIndicatorX, sunIndicatorY);

      // Draw azimuth label
      ctx.fillStyle = "#8899aa";
      ctx.font = "8px monospace";
      ctx.fillText(`${Math.round(sunPosition.azimuth)}°`, sunIndicatorX, sunIndicatorY + 12);
    }
  }, [floor, metrics, selectedMetric, canvasWidth, canvasHeight, metersToPixels, buildingWidth, buildingHeight, zoom, baseScale, sunPosition, animationTick]);

  return (
    <div className="floor-plan-container">
      <div className="zoom-controls">
        <button onClick={() => handleZoom(ZOOM_STEP)} title="Zoom In">+</button>
        <button onClick={() => handleZoom(-ZOOM_STEP)} title="Zoom Out">−</button>
        <button onClick={resetView} title="Reset View">⟲</button>
        <button onClick={fitToContent} title="Fit to Content">⊡</button>
      </div>
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        style={{
          cursor: isDragging ? "grabbing" : "grab",
          borderRadius: "4px",
          border: "1px solid #2a4a6a",
        }}
      />
    </div>
  );
}

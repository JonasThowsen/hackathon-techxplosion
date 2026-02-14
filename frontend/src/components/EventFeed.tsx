import { useEffect, useState, useRef } from "react";
import type { MetricsUpdate, WastePattern, ActionType } from "../types";

export interface Event {
  id: number;
  time: Date;
  type: "waste" | "resolved" | "info" | "action";
  message: string;
  roomId?: string;
}

interface EventFeedProps {
  metrics: MetricsUpdate;
  maxEvents?: number;
}

let eventId = 0;

export function EventFeed({ metrics, maxEvents = 10 }: EventFeedProps) {
  const [events, setEvents] = useState<Event[]>([]);
  const prevMetrics = useRef<MetricsUpdate | null>(null);

  useEffect(() => {
    if (!prevMetrics.current) {
      const initialEvents: Event[] = [];
      for (const [roomId, data] of Object.entries(metrics.rooms)) {
        for (const pattern of data.waste_patterns ?? []) {
          initialEvents.push({
            id: eventId++,
            time: new Date(),
            type: "waste",
            message: formatWasteMessage(pattern, roomId),
            roomId,
          });
        }

        for (const action of data.actions ?? []) {
          initialEvents.push({
            id: eventId++,
            time: new Date(),
            type: "action",
            message: formatActionMessage(action, roomId),
            roomId,
          });
        }
      }

      if (initialEvents.length > 0) {
        setEvents(initialEvents.slice(0, maxEvents));
      }

      prevMetrics.current = metrics;
      return;
    }

    const newEvents: Event[] = [];
    const prev = prevMetrics.current.rooms;
    const curr = metrics.rooms;

    for (const [roomId, data] of Object.entries(curr)) {
      const prevData = prev[roomId];
      if (!prevData) continue;

      const currPatterns = data.waste_patterns ?? [];
      const prevPatterns = prevData.waste_patterns ?? [];
      const currActions = data.actions ?? [];
      const prevActions = prevData.actions ?? [];

      // New waste pattern detected
      for (const pattern of currPatterns) {
        if (!prevPatterns.includes(pattern)) {
          newEvents.push({
            id: eventId++,
            time: new Date(),
            type: "waste",
            message: formatWasteMessage(pattern, roomId),
            roomId,
          });
        }
      }

      // Waste pattern resolved
      for (const pattern of prevPatterns) {
        if (!currPatterns.includes(pattern)) {
          newEvents.push({
            id: eventId++,
            time: new Date(),
            type: "resolved",
            message: `${roomId}: ${formatPatternName(pattern)} resolved`,
            roomId,
          });
        }
      }

      // New action issued
      for (const action of currActions) {
        if (!prevActions.includes(action)) {
          newEvents.push({
            id: eventId++,
            time: new Date(),
            type: "action",
            message: formatActionMessage(action, roomId),
            roomId,
          });
        }
      }

      // Significant temperature change
      const tempDiff = data.temperature - prevData.temperature;
      if (Math.abs(tempDiff) > 2) {
        newEvents.push({
          id: eventId++,
          time: new Date(),
          type: "info",
          message: `${roomId}: Temperature ${tempDiff > 0 ? "rose" : "dropped"} ${Math.abs(tempDiff).toFixed(1)}°C`,
          roomId,
        });
      }

    }

    if (newEvents.length > 0) {
      setEvents((prev) => [...newEvents, ...prev].slice(0, maxEvents));
    }

    prevMetrics.current = metrics;
  }, [metrics, maxEvents]);

  if (events.length === 0) {
    return (
      <div className="event-feed">
        <h3>Event Feed</h3>
        <div className="event-empty">Monitoring for events...</div>
      </div>
    );
  }

  return (
    <div className="event-feed">
      <h3>Event Feed</h3>
      <div className="event-list">
        {events.map((event) => (
          <div key={event.id} className={`event-item ${event.type}`}>
            <span className="event-time">{formatTime(event.time)}</span>
            <span className="event-icon">{getIcon(event.type)}</span>
            <span className="event-message">{event.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function getIcon(type: Event["type"]): string {
  switch (type) {
    case "waste": return "⚠️";
    case "resolved": return "✅";
    case "action": return "⚡";
    case "info": return "ℹ️";
  }
}

function formatPatternName(pattern: WastePattern): string {
  return pattern.replace(/_/g, " ");
}

function formatWasteMessage(pattern: WastePattern, roomId: string): string {
  switch (pattern) {
    case "empty_room_heating_on":
      return `${roomId}: Heating running in empty room`;
    case "open_window_heating":
      return `${roomId}: Window likely open - heating wasted`;
    case "over_heating":
      return `${roomId}: Excess heating detected`;
    case "excessive_ventilation":
      return `${roomId}: Ventilation running unnecessarily in empty room`;
  }
}

function formatActionMessage(action: ActionType, roomId: string): string {
  switch (action) {
    case "boost_heating":
      return `${roomId}: Boosting heating to recover 21°C baseline`;
    case "reduce_heating":
      return `${roomId}: Lowered heating to hold 21°C baseline`;
    case "reduce_ventilation":
      return `${roomId}: Reducing ventilation in empty room`;
    case "open_window_alert":
      return `${roomId}: Open window detected - reducing heating`;
  }
}

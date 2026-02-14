// Mock data for testing without backend

import type { BuildingLayout, MetricsUpdate, WastePattern, ActionType } from "../types";

// Student housing floor plan - rooms scaled to fill the view
export const MOCK_BUILDING: BuildingLayout = {
  id: "building-1",
  name: "Student Housing A",
  width_m: 24,
  height_m: 12,
  floors: [
    {
      floor_index: 0,
      label: "Floor 1",
      rooms: [
        // Top row of rooms
        {
          id: "r-102",
          name: "Room 102",
          polygon: [[0, 0], [0, 5], [4, 5], [4, 0]],
        },
        {
          id: "r-103",
          name: "Room 103",
          polygon: [[4, 0], [4, 5], [8, 5], [8, 0]],
        },
        {
          id: "r-104",
          name: "Room 104",
          polygon: [[8, 0], [8, 5], [12, 5], [12, 0]],
        },
        {
          id: "r-105",
          name: "Room 105",
          polygon: [[12, 0], [12, 5], [16, 5], [16, 0]],
        },
        {
          id: "r-106",
          name: "Room 106",
          polygon: [[16, 0], [16, 5], [20, 5], [20, 0]],
        },
        {
          id: "r-107",
          name: "Room 107",
          polygon: [[20, 0], [20, 5], [24, 5], [24, 0]],
        },
        // Corridor
        {
          id: "r-corridor",
          name: "Corridor",
          polygon: [[0, 5], [0, 7], [24, 7], [24, 5]],
        },
        // Bottom row of rooms
        {
          id: "r-108",
          name: "Room 108",
          polygon: [[0, 7], [0, 12], [4, 12], [4, 7]],
        },
        {
          id: "r-109",
          name: "Room 109",
          polygon: [[4, 7], [4, 12], [8, 12], [8, 7]],
        },
        {
          id: "r-110",
          name: "Room 110",
          polygon: [[8, 7], [8, 12], [12, 12], [12, 7]],
        },
        {
          id: "r-111",
          name: "Room 111",
          polygon: [[12, 7], [12, 12], [16, 12], [16, 7]],
        },
        {
          id: "r-112",
          name: "Room 112",
          polygon: [[16, 7], [16, 12], [20, 12], [20, 7]],
        },
        {
          id: "r-113",
          name: "Room 113",
          polygon: [[20, 7], [20, 12], [24, 12], [24, 7]],
        },
      ],
    },
  ],
};

/**
 * Generates mock metrics with realistic variation.
 */
export function generateMockMetrics(tick: number): MetricsUpdate {
  const rooms: MetricsUpdate["rooms"] = {};

  const vary = (base: number, amplitude: number) =>
    base + Math.sin(tick * 0.1) * amplitude + (Math.random() - 0.5) * amplitude * 0.5;

  const allRoomIds = MOCK_BUILDING.floors.flatMap((f) => f.rooms.map((r) => r.id));

  for (const id of allRoomIds) {
    const isCorridor = id.includes("corridor");
    const isCommon = id.includes("11");

    let baseTemp = 21;
    let occupied = Math.random() > 0.5;
    let baseCo2 = 450;
    let basePower = 100;
    const wastePatterns: WastePattern[] = [];
    const actions: ActionType[] = [];

    if (isCorridor) {
      baseTemp = 19;
      occupied = false;
      baseCo2 = 380;
      basePower = 30;
    } else if (isCommon) {
      baseTemp = 22;
      occupied = true;
      baseCo2 = 550;
      basePower = 250;
    }

    // Add some waste patterns and corresponding actions
    if (id === "r-103") {
      wastePatterns.push("empty_room_heating_on");
      actions.push("reduce_heating");
      baseTemp = 26;
      occupied = false;
      basePower = 200;
    }
    if (id === "r-108") {
      wastePatterns.push("rapid_heat_loss");
      actions.push("suspend_heating");
      baseTemp = 16;
    }
    if (id === "r-106" && tick % 5 < 3) {
      wastePatterns.push("excessive_ventilation");
      actions.push("reduce_ventilation");
      occupied = false;
      basePower = 180;
    }
    if (id === "r-110") {
      wastePatterns.push("over_heating");
      actions.push("reduce_heating");
      baseTemp = 24;
      basePower = 350;
    }
    if (id === "r-112" && tick % 7 < 2) {
      wastePatterns.push("rapid_heat_loss");
      actions.push("suspend_heating");
      baseTemp = 15;
    }

    const heating_power = Math.max(0, Math.min(400, vary(basePower * 0.7, 20)));
    const ventilation_power = Math.max(0, Math.min(200, vary(basePower * 0.3, 10)));
    rooms[id] = {
      temperature: Math.max(15, Math.min(30, vary(baseTemp, 1.5))),
      occupancy: occupied,
      co2: Math.max(300, Math.min(1000, vary(baseCo2, 50))),
      heating_power,
      ventilation_power,
      power: heating_power + ventilation_power,
      waste_patterns: wastePatterns,
      actions,
    };
  }

  return { tick, rooms };
}

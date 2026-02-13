// Mock data for testing without backend

import type { BuildingLayout, MetricsUpdate } from "../types";

// Realistic student housing floor plan
// 30m x 20m building with central corridor
export const MOCK_BUILDING: BuildingLayout = {
  id: "building-1",
  name: "Student Housing A",
  width_m: 30,
  height_m: 20,
  floors: [
    {
      floor_index: 0,
      label: "Floor 1 - Ground",
      rooms: [
        // Left wing - top row
        {
          id: "r-101",
          name: "Lobby",
          polygon: [
            [0, 0],
            [8, 0],
            [8, 6],
            [0, 6],
          ],
        },
        {
          id: "r-102",
          name: "Office",
          polygon: [
            [8, 0],
            [14, 0],
            [14, 6],
            [8, 6],
          ],
        },
        // Central corridor (top)
        {
          id: "r-corridor-1",
          name: "Corridor",
          polygon: [
            [14, 0],
            [16, 0],
            [16, 20],
            [14, 20],
          ],
        },
        // Right wing - top row
        {
          id: "r-103",
          name: "Meeting",
          polygon: [
            [16, 0],
            [23, 0],
            [23, 6],
            [16, 6],
          ],
        },
        {
          id: "r-104",
          name: "Storage",
          polygon: [
            [23, 0],
            [30, 0],
            [30, 6],
            [23, 6],
          ],
        },
        // Left wing - middle
        {
          id: "r-105",
          name: "Kitchen",
          polygon: [
            [0, 6],
            [8, 6],
            [8, 14],
            [0, 14],
          ],
        },
        {
          id: "r-106",
          name: "Dining",
          polygon: [
            [8, 6],
            [14, 6],
            [14, 14],
            [8, 14],
          ],
        },
        // Right wing - middle
        {
          id: "r-107",
          name: "Lounge",
          polygon: [
            [16, 6],
            [30, 6],
            [30, 14],
            [16, 14],
          ],
        },
        // Bottom row
        {
          id: "r-108",
          name: "Laundry",
          polygon: [
            [0, 14],
            [7, 14],
            [7, 20],
            [0, 20],
          ],
        },
        {
          id: "r-109",
          name: "Utility",
          polygon: [
            [7, 14],
            [14, 14],
            [14, 20],
            [7, 20],
          ],
        },
        {
          id: "r-110",
          name: "Gym",
          polygon: [
            [16, 14],
            [30, 14],
            [30, 20],
            [16, 20],
          ],
        },
      ],
    },
    {
      floor_index: 1,
      label: "Floor 2 - Rooms",
      rooms: [
        // Corridor
        {
          id: "r-corridor-2",
          name: "Corridor",
          polygon: [
            [13, 0],
            [17, 0],
            [17, 20],
            [13, 20],
          ],
        },
        // Left wing rooms
        {
          id: "r-201",
          name: "Room 201",
          polygon: [
            [0, 0],
            [6.5, 0],
            [6.5, 5],
            [0, 5],
          ],
        },
        {
          id: "r-202",
          name: "Room 202",
          polygon: [
            [6.5, 0],
            [13, 0],
            [13, 5],
            [6.5, 5],
          ],
        },
        {
          id: "r-203",
          name: "Room 203",
          polygon: [
            [0, 5],
            [6.5, 5],
            [6.5, 10],
            [0, 10],
          ],
        },
        {
          id: "r-204",
          name: "Room 204",
          polygon: [
            [6.5, 5],
            [13, 5],
            [13, 10],
            [6.5, 10],
          ],
        },
        {
          id: "r-205",
          name: "Room 205",
          polygon: [
            [0, 10],
            [6.5, 10],
            [6.5, 15],
            [0, 15],
          ],
        },
        {
          id: "r-206",
          name: "Room 206",
          polygon: [
            [6.5, 10],
            [13, 10],
            [13, 15],
            [6.5, 15],
          ],
        },
        {
          id: "r-207",
          name: "Bathroom L",
          polygon: [
            [0, 15],
            [13, 15],
            [13, 20],
            [0, 20],
          ],
        },
        // Right wing rooms
        {
          id: "r-208",
          name: "Room 208",
          polygon: [
            [17, 0],
            [23.5, 0],
            [23.5, 5],
            [17, 5],
          ],
        },
        {
          id: "r-209",
          name: "Room 209",
          polygon: [
            [23.5, 0],
            [30, 0],
            [30, 5],
            [23.5, 5],
          ],
        },
        {
          id: "r-210",
          name: "Room 210",
          polygon: [
            [17, 5],
            [23.5, 5],
            [23.5, 10],
            [17, 10],
          ],
        },
        {
          id: "r-211",
          name: "Room 211",
          polygon: [
            [23.5, 5],
            [30, 5],
            [30, 10],
            [23.5, 10],
          ],
        },
        {
          id: "r-212",
          name: "Room 212",
          polygon: [
            [17, 10],
            [23.5, 10],
            [23.5, 15],
            [17, 15],
          ],
        },
        {
          id: "r-213",
          name: "Room 213",
          polygon: [
            [23.5, 10],
            [30, 10],
            [30, 15],
            [23.5, 15],
          ],
        },
        {
          id: "r-214",
          name: "Bathroom R",
          polygon: [
            [17, 15],
            [30, 15],
            [30, 20],
            [17, 20],
          ],
        },
      ],
    },
    {
      floor_index: 2,
      label: "Floor 3 - Rooms",
      rooms: [
        // Corridor
        {
          id: "r-corridor-3",
          name: "Corridor",
          polygon: [
            [13, 0],
            [17, 0],
            [17, 20],
            [13, 20],
          ],
        },
        // Left wing
        {
          id: "r-301",
          name: "Room 301",
          polygon: [
            [0, 0],
            [6.5, 0],
            [6.5, 6.67],
            [0, 6.67],
          ],
        },
        {
          id: "r-302",
          name: "Room 302",
          polygon: [
            [6.5, 0],
            [13, 0],
            [13, 6.67],
            [6.5, 6.67],
          ],
        },
        {
          id: "r-303",
          name: "Room 303",
          polygon: [
            [0, 6.67],
            [6.5, 6.67],
            [6.5, 13.33],
            [0, 13.33],
          ],
        },
        {
          id: "r-304",
          name: "Room 304",
          polygon: [
            [6.5, 6.67],
            [13, 6.67],
            [13, 13.33],
            [6.5, 13.33],
          ],
        },
        {
          id: "r-305",
          name: "Common L",
          polygon: [
            [0, 13.33],
            [13, 13.33],
            [13, 20],
            [0, 20],
          ],
        },
        // Right wing
        {
          id: "r-306",
          name: "Room 306",
          polygon: [
            [17, 0],
            [23.5, 0],
            [23.5, 6.67],
            [17, 6.67],
          ],
        },
        {
          id: "r-307",
          name: "Room 307",
          polygon: [
            [23.5, 0],
            [30, 0],
            [30, 6.67],
            [23.5, 6.67],
          ],
        },
        {
          id: "r-308",
          name: "Room 308",
          polygon: [
            [17, 6.67],
            [23.5, 6.67],
            [23.5, 13.33],
            [17, 13.33],
          ],
        },
        {
          id: "r-309",
          name: "Room 309",
          polygon: [
            [23.5, 6.67],
            [30, 6.67],
            [30, 13.33],
            [23.5, 13.33],
          ],
        },
        {
          id: "r-310",
          name: "Common R",
          polygon: [
            [17, 13.33],
            [30, 13.33],
            [30, 20],
            [17, 20],
          ],
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

  // Generate for all floors
  const allRoomIds = MOCK_BUILDING.floors.flatMap((f) => f.rooms.map((r) => r.id));

  for (const id of allRoomIds) {
    // Corridors are cooler, less occupied
    const isCorridor = id.includes("corridor");
    const isBathroom = id.includes("Bathroom");
    const isCommon = id.includes("Common") || id.includes("Lounge") || id.includes("Gym");

    let baseTemp = 21;
    let baseOccupancy = 0.3;
    let baseCo2 = 450;
    let basePower = 100;
    const wastePatterns: MetricsUpdate["rooms"][string]["waste_patterns"] = [];

    if (isCorridor) {
      baseTemp = 19;
      baseOccupancy = 0.05;
      baseCo2 = 380;
      basePower = 30;
    } else if (isBathroom) {
      baseTemp = 22;
      baseOccupancy = 0.1;
      baseCo2 = 500;
      basePower = 150;
    } else if (isCommon) {
      baseTemp = 22;
      baseOccupancy = 0.5;
      baseCo2 = 550;
      basePower = 250;
    }

    // Add some waste patterns randomly
    if (id === "r-103" || id === "r-301") {
      wastePatterns.push("empty_room_heating_on");
      baseTemp = 25;
      baseOccupancy = 0;
      basePower = 200;
    }
    if (id === "r-201" || id === "r-306") {
      wastePatterns.push("open_window_heating");
      baseTemp = 17;
    }
    if (id === "r-110") {
      baseTemp = 24;
      basePower = 400;
    }

    rooms[id] = {
      temperature: Math.max(15, Math.min(30, vary(baseTemp, 1.5))),
      occupancy: Math.max(0, Math.min(1, vary(baseOccupancy, 0.15))),
      co2: Math.max(300, Math.min(1000, vary(baseCo2, 50))),
      power: Math.max(0, Math.min(500, vary(basePower, 30))),
      waste_patterns: wastePatterns,
    };
  }

  return { tick, rooms };
}

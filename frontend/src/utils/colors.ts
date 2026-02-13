// Heatmap color utilities

/**
 * Interpolates between blue (cold) and red (hot) based on a normalized value.
 * @param value - The value to map (will be clamped to 0-1)
 * @returns CSS color string
 */
export function heatmapColor(value: number): string {
  // Clamp value between 0 and 1
  const t = Math.max(0, Math.min(1, value));

  // Blue (cold) -> Cyan -> Green -> Yellow -> Red (hot)
  // Using HSL for smooth interpolation: hue 240 (blue) to 0 (red)
  const hue = (1 - t) * 240;
  const saturation = 80;
  const lightness = 50;

  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

/**
 * Normalizes a value to 0-1 range based on min/max.
 */
export function normalize(value: number, min: number, max: number): number {
  if (max === min) return 0.5;
  return (value - min) / (max - min);
}

/**
 * Gets a color for a metric value based on its configured range.
 */
export function getMetricColor(
  value: number,
  min: number,
  max: number
): string {
  return heatmapColor(normalize(value, min, max));
}

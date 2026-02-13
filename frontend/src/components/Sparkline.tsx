interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  showTrend?: boolean;
}

export function Sparkline({
  data,
  width = 60,
  height = 20,
  color = "#4a9eff",
  showTrend = true,
}: SparklineProps) {
  if (data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((value, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((value - min) / range) * height;
    return `${x},${y}`;
  }).join(" ");

  const trend = data[data.length - 1] - data[0];
  const trendColor = trend > 0 ? "#ff6b6b" : trend < 0 ? "#4ecdc4" : "#888";

  return (
    <div className="sparkline" style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      <svg width={width} height={height} style={{ overflow: "visible" }}>
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      {showTrend && (
        <span style={{ fontSize: 10, color: trendColor }}>
          {trend > 0 ? "↑" : trend < 0 ? "↓" : "→"}
        </span>
      )}
    </div>
  );
}

"use client";

interface Props {
  latency_ms: number;
  label?: string;
}

export default function LatencyBar({ latency_ms, label }: Props) {
  // Color based on speed: green < 100ms, yellow < 500ms, red > 500ms
  const color =
    latency_ms < 100
      ? "bg-green-500"
      : latency_ms < 500
      ? "bg-yellow-500"
      : "bg-red-500";

  // Bar width (max 300ms for full width, capped)
  const width = Math.min((latency_ms / 1000) * 100, 100);

  return (
    <div className="flex items-center gap-3">
      {label && <span className="text-xs text-gray-500 w-20">{label}</span>}
      <div className="flex-1 bg-gray-800 rounded-full h-2 max-w-48">
        <div
          className={`h-2 rounded-full ${color} transition-all`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 w-16 text-right">
        {latency_ms.toFixed(1)}ms
      </span>
    </div>
  );
}

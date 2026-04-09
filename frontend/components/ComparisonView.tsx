"use client";

import { CompareResponse } from "@/lib/api";
import ScoreBreakdown from "./ScoreBreakdown";
import LatencyBar from "./LatencyBar";
import TechniqueInfoCard from "./TechniqueInfo";

interface Props {
  data: CompareResponse;
}

export default function ComparisonView({ data }: Props) {
  const [left, right] = data.comparisons;

  if (!left || !right) return null;

  return (
    <div className="grid grid-cols-2 gap-6">
      {/* Left technique */}
      <div className="space-y-4">
        <TechniqueInfoCard info={left.technique_info} />
        <LatencyBar latency_ms={left.latency_ms} label="Latency" />

        <div className="space-y-3">
          {left.results.map((result) => (
            <div
              key={`l-${result.document.id}-${result.rank}`}
              className="border border-gray-800 rounded-lg p-3 space-y-2"
            >
              <ScoreBreakdown result={result} />
              <p className="text-sm text-gray-200 leading-relaxed">
                {result.document.content}
              </p>
              <p className="text-xs text-gray-500">
                <span className="font-medium">Why: </span>
                {result.explanation}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Right technique */}
      <div className="space-y-4">
        <TechniqueInfoCard info={right.technique_info} />
        <LatencyBar latency_ms={right.latency_ms} label="Latency" />

        <div className="space-y-3">
          {right.results.map((result) => (
            <div
              key={`r-${result.document.id}-${result.rank}`}
              className="border border-gray-800 rounded-lg p-3 space-y-2"
            >
              <ScoreBreakdown result={result} />
              <p className="text-sm text-gray-200 leading-relaxed">
                {result.document.content}
              </p>
              <p className="text-xs text-gray-500">
                <span className="font-medium">Why: </span>
                {result.explanation}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

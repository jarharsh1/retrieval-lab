"use client";

import { RetrieveResponse } from "@/lib/api";
import ScoreBreakdown from "./ScoreBreakdown";
import LatencyBar from "./LatencyBar";
import TechniqueInfoCard from "./TechniqueInfo";

interface Props {
  data: RetrieveResponse;
}

export default function ResultsPanel({ data }: Props) {
  return (
    <div className="space-y-4">
      <TechniqueInfoCard info={data.technique_info} />
      <LatencyBar latency_ms={data.latency_ms} label="Latency" />

      <div className="space-y-3">
        {data.results.map((result) => (
          <div
            key={`${result.document.id}-${result.rank}`}
            className="border border-gray-800 rounded-lg p-4 space-y-3 hover:border-gray-700 transition-colors"
          >
            <ScoreBreakdown result={result} />

            <p className="text-sm text-gray-200 leading-relaxed">
              {result.document.content}
            </p>

            <div className="bg-gray-900/50 rounded-md p-3">
              <p className="text-xs text-gray-400 leading-relaxed">
                <span className="text-gray-500 font-medium">Why: </span>
                {result.explanation}
              </p>
            </div>

            {Object.keys(result.document.metadata).length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(result.document.metadata).map(([key, val]) => (
                  <span
                    key={key}
                    className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-500"
                  >
                    {key}: {String(val)}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}

        {data.results.length === 0 && (
          <p className="text-sm text-gray-500 text-center py-8">
            No results found.
          </p>
        )}
      </div>
    </div>
  );
}

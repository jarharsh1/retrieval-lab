"use client";

import { RetrievalResult } from "@/lib/api";

interface Props {
  result: RetrievalResult;
}

export default function ScoreBreakdown({ result }: Props) {
  // Score bar: normalize to 0-100% range
  // BM25 scores can be 0-20+, cosine similarity is 0-1
  const maxScore =
    result.metadata.technique === "bm25" ? 15 : 1;
  const percentage = Math.min((result.score / maxScore) * 100, 100);

  const barColor =
    percentage > 70
      ? "bg-green-500"
      : percentage > 40
      ? "bg-yellow-500"
      : "bg-red-500";

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-gray-400">
          Score: {result.score.toFixed(4)}
        </span>
        <span className="text-xs text-gray-500">Rank #{result.rank}</span>
      </div>
      <div className="bg-gray-800 rounded-full h-1.5">
        <div
          className={`h-1.5 rounded-full ${barColor} transition-all`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

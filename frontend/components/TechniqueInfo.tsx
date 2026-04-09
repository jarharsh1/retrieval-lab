"use client";

import { TechniqueInfo as TechniqueInfoType } from "@/lib/api";

interface Props {
  info: TechniqueInfoType;
}

const CATEGORY_BADGES: Record<string, string> = {
  sparse: "bg-green-900/50 text-green-400",
  dense: "bg-green-900/50 text-green-400",
  hybrid: "bg-green-900/50 text-green-400",
  "pre-retrieval": "bg-blue-900/50 text-blue-400",
  "post-retrieval": "bg-orange-900/50 text-orange-400",
  advanced: "bg-purple-900/50 text-purple-400",
};

export default function TechniqueInfoCard({ info }: Props) {
  return (
    <div className="flex items-start gap-3 p-3 bg-gray-800/50 rounded-lg">
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-gray-200">{info.name}</h3>
          <span
            className={`text-xs px-2 py-0.5 rounded-full ${
              CATEGORY_BADGES[info.category] || "bg-gray-700 text-gray-400"
            }`}
          >
            {info.category}
          </span>
        </div>
        <p className="text-xs text-gray-400 mt-1">{info.description}</p>
      </div>
    </div>
  );
}

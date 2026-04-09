"use client";

import { TECHNIQUES } from "@/lib/api";

interface Props {
  selected: string[];
  onToggle: (technique: string) => void;
  compareMode: boolean;
}

const PHASE_LABELS: Record<number, string> = {
  1: "Foundation",
  2: "Pre-Retrieval",
  3: "Post-Retrieval",
  4: "Advanced",
};

const CATEGORY_COLORS: Record<string, string> = {
  sparse: "bg-green-900/50 border-green-700 text-green-300",
  dense: "bg-green-900/50 border-green-700 text-green-300",
  hybrid: "bg-green-900/50 border-green-700 text-green-300",
  "pre-retrieval": "bg-blue-900/50 border-blue-700 text-blue-300",
  "post-retrieval": "bg-orange-900/50 border-orange-700 text-orange-300",
  advanced: "bg-purple-900/50 border-purple-700 text-purple-300",
};

export default function TechniquePicker({
  selected,
  onToggle,
  compareMode,
}: Props) {
  const phases = [1, 2, 3, 4];

  return (
    <div>
      <label className="block text-sm font-medium text-gray-400 mb-2">
        Technique{compareMode ? "s (pick 2)" : ""}
      </label>
      <div className="space-y-3">
        {phases.map((phase) => (
          <div key={phase}>
            <p className="text-xs text-gray-500 mb-1">
              Phase {phase}: {PHASE_LABELS[phase]}
            </p>
            <div className="flex flex-wrap gap-2">
              {TECHNIQUES.filter((t) => t.phase === phase).map((t) => {
                const isSelected = selected.includes(t.id);
                const disabled =
                  compareMode && selected.length >= 2 && !isSelected;

                return (
                  <button
                    key={t.id}
                    onClick={() => !disabled && onToggle(t.id)}
                    disabled={disabled}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                      isSelected
                        ? CATEGORY_COLORS[t.category]
                        : disabled
                        ? "bg-gray-900 border-gray-800 text-gray-600 cursor-not-allowed"
                        : "bg-gray-800 border-gray-700 text-gray-400 hover:bg-gray-700"
                    }`}
                  >
                    {t.name}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

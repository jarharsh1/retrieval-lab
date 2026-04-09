"use client";

const DATASETS = [
  { id: "supply_chain", label: "Supply Chain", icon: "📦" },
  { id: "healthcare", label: "Healthcare", icon: "🏥" },
  { id: "wikipedia", label: "Wikipedia", icon: "📚" },
];

interface Props {
  selected: string;
  onSelect: (dataset: string) => void;
}

export default function DatasetSelector({ selected, onSelect }: Props) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-400 mb-2">
        Dataset
      </label>
      <div className="flex gap-2">
        {DATASETS.map((ds) => (
          <button
            key={ds.id}
            onClick={() => onSelect(ds.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selected === ds.id
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            {ds.icon} {ds.label}
          </button>
        ))}
      </div>
    </div>
  );
}

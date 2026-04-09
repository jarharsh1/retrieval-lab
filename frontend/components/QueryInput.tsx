"use client";

interface Props {
  query: string;
  onChange: (query: string) => void;
  onSubmit: () => void;
  loading: boolean;
  compareMode: boolean;
  onToggleCompare: () => void;
}

const SAMPLE_QUERIES: Record<string, string[]> = {
  supply_chain: [
    "Which suppliers have reliability below 90%?",
    "Find vendors in China",
    "What happens if Taiwan Semiconductor shuts down?",
  ],
  healthcare: [
    "Side effects of metformin 500mg",
    "Treatment options for high blood sugar",
    "Drug interactions between blood thinners and painkillers",
  ],
  wikipedia: [
    "How did the industrial revolution affect farming?",
    "Tall art deco buildings built in New York in the 1930s",
    "Explain photosynthesis",
  ],
};

export default function QueryInput({
  query,
  onChange,
  onSubmit,
  loading,
  compareMode,
  onToggleCompare,
}: Props & { dataset?: string }) {
  const dataset =
    (typeof window !== "undefined" && undefined) || "supply_chain";

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <label className="block text-sm font-medium text-gray-400">
          Query
        </label>
        <button
          onClick={onToggleCompare}
          className={`text-xs px-2 py-0.5 rounded-full transition-colors ${
            compareMode
              ? "bg-purple-600 text-white"
              : "bg-gray-800 text-gray-400 hover:bg-gray-700"
          }`}
        >
          {compareMode ? "Compare Mode ON" : "Compare Mode"}
        </button>
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !loading && onSubmit()}
          placeholder="Enter your search query..."
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={onSubmit}
          disabled={loading || !query.trim()}
          className="px-6 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Searching..." : compareMode ? "Compare" : "Search"}
        </button>
      </div>
    </div>
  );
}

export function SampleQueries({
  dataset,
  onSelect,
}: {
  dataset: string;
  onSelect: (q: string) => void;
}) {
  const queries = SAMPLE_QUERIES[dataset] || SAMPLE_QUERIES.supply_chain;

  return (
    <div>
      <p className="text-xs text-gray-500 mb-1">Try a sample query:</p>
      <div className="flex flex-wrap gap-2">
        {queries.map((q) => (
          <button
            key={q}
            onClick={() => onSelect(q)}
            className="text-xs px-3 py-1 rounded-full bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200 transition-colors"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

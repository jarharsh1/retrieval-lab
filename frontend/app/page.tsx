"use client";

import { useState } from "react";
import DatasetSelector from "@/components/DatasetSelector";
import TechniquePicker from "@/components/TechniquePicker";
import QueryInput, { SampleQueries } from "@/components/QueryInput";
import ResultsPanel from "@/components/ResultsPanel";
import ComparisonView from "@/components/ComparisonView";
import {
  fetchRetrieve,
  fetchCompare,
  RetrieveResponse,
  CompareResponse,
} from "@/lib/api";

export default function Home() {
  const [dataset, setDataset] = useState("supply_chain");
  const [selectedTechniques, setSelectedTechniques] = useState<string[]>([
    "bm25",
  ]);
  const [query, setQuery] = useState("");
  const [compareMode, setCompareMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [singleResult, setSingleResult] = useState<RetrieveResponse | null>(
    null
  );
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(
    null
  );

  const handleToggleTechnique = (technique: string) => {
    if (compareMode) {
      setSelectedTechniques((prev) => {
        if (prev.includes(technique)) {
          return prev.filter((t) => t !== technique);
        }
        if (prev.length >= 2) return prev;
        return [...prev, technique];
      });
    } else {
      setSelectedTechniques([technique]);
    }
  };

  const handleToggleCompare = () => {
    setCompareMode(!compareMode);
    if (!compareMode && selectedTechniques.length < 2) {
      setSelectedTechniques((prev) =>
        prev[0] === "bm25" ? [...prev, "semantic"] : [...prev, "bm25"]
      );
    } else if (compareMode) {
      setSelectedTechniques((prev) => [prev[0]]);
    }
  };

  const handleSubmit = async () => {
    if (!query.trim() || selectedTechniques.length === 0) return;

    setLoading(true);
    setError(null);
    setSingleResult(null);
    setCompareResult(null);

    try {
      if (compareMode && selectedTechniques.length === 2) {
        const data = await fetchCompare(query, dataset, selectedTechniques);
        setCompareResult(data);
      } else {
        const data = await fetchRetrieve(
          query,
          dataset,
          selectedTechniques[0]
        );
        setSingleResult(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-6 lg:col-span-1">
          <DatasetSelector selected={dataset} onSelect={setDataset} />
          <TechniquePicker
            selected={selectedTechniques}
            onToggle={handleToggleTechnique}
            compareMode={compareMode}
          />
        </div>

        <div className="space-y-4 lg:col-span-2">
          <QueryInput
            query={query}
            onChange={setQuery}
            onSubmit={handleSubmit}
            loading={loading}
            compareMode={compareMode}
            onToggleCompare={handleToggleCompare}
          />
          <SampleQueries dataset={dataset} onSelect={setQuery} />
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Results */}
      {singleResult && <ResultsPanel data={singleResult} />}
      {compareResult && <ComparisonView data={compareResult} />}

      {/* Empty state */}
      {!singleResult && !compareResult && !loading && !error && (
        <div className="text-center py-16 space-y-3">
          <p className="text-gray-500 text-lg">
            Pick a dataset, choose a technique, and run a query
          </p>
          <p className="text-gray-600 text-sm">
            Or enable Compare Mode to see two techniques side by side
          </p>
        </div>
      )}
    </div>
  );
}

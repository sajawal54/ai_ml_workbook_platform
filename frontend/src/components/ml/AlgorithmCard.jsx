import { Play, RotateCcw, Loader2 } from "lucide-react";
import AlgorithmResultView from "./AlgorithmResultView";

export default function AlgorithmCard({
  algo,
  params,
  onChangeParam,
  onRun,
  onClearResult,
  result,
  error,
  loading,
}) {
  return (
    <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 flex flex-col justify-between space-y-4 min-w-0">
      <div className="space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <span className="inline-block text-[11px] font-semibold uppercase tracking-wider px-2.5 py-0.5 rounded-full bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
              {algo.badge}
            </span>
            <h4 className="text-base font-bold text-white mt-1.5">
              {algo.name}
            </h4>
            <p className="text-xs text-slate-400 mt-0.5">{algo.description}</p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {result && (
              <button
                type="button"
                onClick={onClearResult}
                title="Clear result"
                className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            )}

            <button
              type="button"
              onClick={onRun}
              disabled={loading}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white transition cursor-pointer"
            >
              {loading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Running...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  <span>Run {algo.shortName}</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Algorithm Hyperparameters */}
        {algo.paramFields?.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 p-3 rounded-xl bg-slate-950/70 border border-slate-800/80">
            {algo.paramFields.map((field) => {
              const val = params[field.key] ?? field.defaultValue ?? "";
              if (field.type === "checkbox") {
                return (
                  <label
                    key={field.key}
                    className="flex items-center gap-2 text-xs text-slate-200 cursor-pointer sm:col-span-2"
                  >
                    <input
                      type="checkbox"
                      checked={Boolean(val)}
                      onChange={(e) =>
                        onChangeParam(field.key, e.target.checked)
                      }
                      className="rounded accent-indigo-500"
                    />
                    <span>{field.label}</span>
                  </label>
                );
              }

              if (field.type === "select") {
                return (
                  <div key={field.key}>
                    <label className="block text-[11px] font-medium text-slate-300 mb-1">
                      {field.label}
                    </label>
                    <select
                      value={val}
                      onChange={(e) => onChangeParam(field.key, e.target.value)}
                      className="w-full rounded-lg bg-slate-900 border border-slate-700 px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500"
                    >
                      {(field.options || []).map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              }

              return (
                <div key={field.key}>
                  <label className="block text-[11px] font-medium text-slate-300 mb-1">
                    {field.label}
                  </label>
                  <input
                    type={field.inputType || "number"}
                    step={field.step || "any"}
                    placeholder={field.placeholder || ""}
                    value={val}
                    onChange={(e) =>
                      onChangeParam(
                        field.key,
                        field.inputType === "text"
                          ? e.target.value
                          : e.target.value === ""
                          ? ""
                          : Number(e.target.value)
                      )
                    }
                    className="w-full rounded-lg bg-slate-900 border border-slate-700 px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-indigo-500"
                  />
                </div>
              );
            })}
          </div>
        )}

        {/* Single Clear Error */}
        {error && (
          <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 px-3.5 py-2.5 text-xs text-rose-300">
            {error}
          </div>
        )}

        {/* Isolated Algorithm Result */}
        <AlgorithmResultView result={result} />
      </div>
    </div>
  );
}
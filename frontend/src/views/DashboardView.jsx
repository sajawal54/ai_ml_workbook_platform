import { Link } from "react-router-dom";
import { Activity, ArrowRight, BrainCircuit, RefreshCw } from "lucide-react";

export default function DashboardView({ apiHealthy, onRetryHealth }) {
  const apiStatus =
    apiHealthy === true
      ? "Connected"
      : apiHealthy === false
      ? "Offline"
      : "Checking";
  const apiStatusClass =
    apiHealthy === true
      ? "text-emerald-400"
      : apiHealthy === false
      ? "text-rose-400"
      : "text-amber-400";

  return (
    <div className="max-w-7xl mx-auto min-w-0 space-y-6">
      <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6 sm:p-8">
        <span className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
          Experimentation Lab
        </span>
        <h1 className="mt-2 text-3xl font-bold text-white">
          AI &amp; ML Dashboard
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-slate-400">
          Prepare a dataset, configure a model, and explore its results in your
          machine learning workspace.
        </p>
        <Link
          to="/machine-learning"
          className="mt-6 inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-500"
        >
          Open Machine Learning
          <ArrowRight className="h-4 w-4" />
        </Link>
      </section>

      <section className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-slate-300">
                Backend connection
              </p>
              <p className={`mt-2 text-xl font-bold ${apiStatusClass}`}>
                {apiStatus}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                FastAPI service on port 8000
              </p>
            </div>
            <Activity className={`h-5 w-5 ${apiStatusClass}`} />
          </div>
          <button
            type="button"
            onClick={onRetryHealth}
            className="mt-5 inline-flex items-center gap-2 text-xs font-semibold text-slate-300 transition hover:text-white"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Check connection again
          </button>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-slate-300">
                Available algorithms
              </p>
              <p className="mt-2 text-xl font-bold text-white">10 models</p>
              <p className="mt-1 text-xs text-slate-500">
                5 supervised and 5 unsupervised
              </p>
            </div>
            <BrainCircuit className="h-5 w-5 text-indigo-400" />
          </div>
          <Link
            to="/machine-learning"
            className="mt-5 inline-flex items-center gap-2 text-xs font-semibold text-indigo-300 transition hover:text-indigo-200"
          >
            Browse algorithms
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </section>
    </div>
  );
}

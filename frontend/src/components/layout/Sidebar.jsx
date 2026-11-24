import { NavLink } from "react-router-dom";
import { LayoutDashboard, BrainCircuit } from "lucide-react";

export default function Sidebar({ apiHealthy }) {
  const linkClass = ({ isActive }) =>
    `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition ${
      isActive
        ? "bg-indigo-600 text-white shadow-sm"
        : "text-slate-300 hover:bg-slate-800 hover:text-white"
    }`;

  return (
    <aside className="w-64 shrink-0 bg-slate-900 border-r border-slate-800 p-4 flex flex-col justify-between min-h-screen">
      <div className="space-y-6">
        <div className="px-2 pt-1 pb-4 border-b border-slate-800">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-indigo-400">
            Experimentation Lab
          </span>
          <h2 className="text-base font-bold text-white mt-0.5">
            AI & ML Workbook
          </h2>
        </div>

        <nav className="space-y-1.5">
          <NavLink to="/" end className={linkClass}>
            <LayoutDashboard className="w-4 h-4 shrink-0" />
            <span>Dashboard</span>
          </NavLink>

          <NavLink to="/machine-learning" className={linkClass}>
            <BrainCircuit className="w-4 h-4 shrink-0" />
            <span>Machine Learning</span>
          </NavLink>
        </nav>
      </div>

      <div className="rounded-xl bg-slate-950/80 border border-slate-800 p-3 space-y-1 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-slate-400">FastAPI (:8000)</span>
          <span
            className={`font-semibold ${
              apiHealthy === true
                ? "text-emerald-400"
                : apiHealthy === false
                ? "text-rose-400"
                : "text-amber-400"
            }`}
          >
            {apiHealthy === true
              ? "Online"
              : apiHealthy === false
              ? "Offline"
              : "Checking"}
          </span>
        </div>
        <p className="text-[11px] text-slate-500">
          Supervised (5) & Unsupervised (5)
        </p>
      </div>
    </aside>
  );
}
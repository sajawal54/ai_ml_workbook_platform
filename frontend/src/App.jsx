import { useCallback, useEffect, useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar";
import DashboardView from "./views/DashboardView";
import MachineLearningView from "./views/MachineLearningView";
import { checkBackendHealth } from "./services/mlApi";

export default function App() {
  const [apiHealthy, setApiHealthy] = useState(null);

  const verifyHealth = useCallback(async () => {
    setApiHealthy(null);
    try {
      await checkBackendHealth();
      setApiHealthy(true);
    } catch {
      setApiHealthy(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    checkBackendHealth()
      .then(() => {
        if (!cancelled) setApiHealthy(true);
      })
      .catch(() => {
        if (!cancelled) setApiHealthy(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <BrowserRouter>
      <div className="min-h-screen w-full bg-slate-950 text-slate-100 flex overflow-x-hidden">
        <Sidebar apiHealthy={apiHealthy} />

        <div className="flex-1 min-w-0 flex flex-col">
          <header className="h-14 px-6 border-b border-slate-800 bg-slate-900/70 flex items-center justify-between shrink-0">
            <span className="text-sm font-semibold text-white">
              AI & ML Experimentation Platform
            </span>
            <span className="text-xs text-slate-400">
              {apiHealthy === true
                ? "API Connected"
                : apiHealthy === false
                ? "API Offline"
                : "Checking API..."}
            </span>
          </header>

          <main className="flex-1 min-w-0 p-6">
            <Routes>
              <Route
                path="/"
                element={
                  <DashboardView
                    apiHealthy={apiHealthy}
                    onRetryHealth={verifyHealth}
                  />
                }
              />
              <Route
                path="/machine-learning"
                element={<MachineLearningView />}
              />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
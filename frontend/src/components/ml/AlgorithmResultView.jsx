export default function AlgorithmResultView({ result }) {
  if (!result) return null;

  const {
    summary,
    metrics = {},
    table_headers = [],
    table_rows = [],
    chart_points = [],
    extra_chart_series = {},
  } = result;

  const metricEntries = Object.entries(metrics);
  const rocCurve = extra_chart_series?.roc_curve || null;

  // Normalize 2D points into SVG viewport coordinates
  const width = 420;
  const height = 190;
  const pad = 24;

  const chartCoordinates = chart_points.map((point) => {
    const x = Number(point.x);
    const y = Number(point.y);
    return {
      x: Number.isFinite(x) ? x : 0,
      y: Number.isFinite(y) ? y : 0,
    };
  });
  const bounds = chartCoordinates.reduce(
    (result, point) => ({
      minX: Math.min(result.minX, point.x),
      maxX: Math.max(result.maxX, point.x),
      minY: Math.min(result.minY, point.y),
      maxY: Math.max(result.maxY, point.y),
    }),
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity }
  );
  const minX = chartCoordinates.length ? bounds.minX : 0;
  const maxX = chartCoordinates.length ? bounds.maxX : 1;
  const minY = chartCoordinates.length ? bounds.minY : 0;
  const maxY = chartCoordinates.length ? bounds.maxY : 1;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const uniqueLabels = [...new Set(chart_points.map((p) => p.label))];
  const palette = [
    "#6366f1",
    "#10b981",
    "#f59e0b",
    "#ec4899",
    "#06b6d4",
    "#f43f5e",
  ];
  const labelColors = new Map(
    uniqueLabels.map((label, index) => [label, palette[index % palette.length]])
  );
  const colorForLabel = (label) => labelColors.get(label) || palette[0];

  const chartStep = Math.max(1, Math.ceil(chart_points.length / 1200));
  const visibleChartPoints = chart_points
    .map((point, index) => ({ point, index }))
    .filter(
      ({ index }) =>
        index % chartStep === 0 || index === chart_points.length - 1
    );

  // ROC curve coordinates inside a square-ish plot area
  const rocSize = 190;
  const rocPad = 26;
  const rocInner = rocSize - rocPad * 2;
  const rocPoints = rocCurve?.points || [];
  const rocPolyline = rocPoints
    .map((pt) => {
      const px = rocPad + (Number(pt.fpr) || 0) * rocInner;
      const py = rocPad + (1 - (Number(pt.tpr) || 0)) * rocInner;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="space-y-4 pt-3 border-t border-slate-800 min-w-0">
      {summary && <p className="text-xs text-slate-300">{summary}</p>}

      {/* Metrics Grid */}
      {metricEntries.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {metricEntries.map(([key, val]) => (
            <div
              key={key}
              className="rounded-xl bg-slate-950/80 border border-slate-800 px-3 py-2"
            >
              <p className="text-[11px] uppercase tracking-wide text-slate-400 truncate">
                {key.replaceAll("_", " ")}
              </p>
              <p className="text-sm font-bold font-mono text-white mt-0.5">
                {String(val)}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Charts: 2D Projection + ROC Curve (when available) */}
      {(chart_points.length > 0 || rocPoints.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {chart_points.length > 0 && (
            <div className="rounded-xl bg-slate-950/80 border border-slate-800 p-3 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  2D Projection View
                </span>
                <div className="flex flex-wrap gap-2">
                  {uniqueLabels.slice(0, 5).map((lbl) => (
                    <span
                      key={lbl}
                      className="inline-flex items-center gap-1 text-[10px] text-slate-300"
                    >
                      <span
                        className="w-2 h-2 rounded-full inline-block"
                        style={{ backgroundColor: colorForLabel(lbl) }}
                      />
                      {lbl}
                    </span>
                  ))}
                </div>
              </div>

              <svg
                viewBox={`0 0 ${width} ${height}`}
                className="w-full h-44 rounded-lg bg-slate-900/70 border border-slate-800/70"
              >
                {visibleChartPoints.map(({ point: pt, index }) => {
                  const point = chartCoordinates[index] || { x: 0, y: 0 };
                  const cx =
                    pad +
                    ((point.x - minX) / spanX) * (width - pad * 2);
                  const cy =
                    height -
                    pad -
                    ((point.y - minY) / spanY) * (height - pad * 2);
                  return (
                    <g key={index}>
                      <circle
                        cx={cx}
                        cy={cy}
                        r={4.5}
                        fill={colorForLabel(pt.label)}
                        fillOpacity={0.9}
                      />
                      <title>{`Row ${pt.row_index}: ${pt.label}`}</title>
                    </g>
                  );
                })}
              </svg>
            </div>
          )}

          {rocPoints.length > 0 && (
            <div className="rounded-xl bg-slate-950/80 border border-slate-800 p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  ROC Curve
                </span>
                <span className="text-[11px] font-mono font-semibold text-emerald-400">
                  AUC: {rocCurve?.auc ?? "—"}
                </span>
              </div>

              <svg
                viewBox={`0 0 ${rocSize} ${rocSize}`}
                className="w-full h-44 rounded-lg bg-slate-900/70 border border-slate-800/70"
              >
                {/* Axes */}
                <line
                  x1={rocPad}
                  y1={rocPad + rocInner}
                  x2={rocPad + rocInner}
                  y2={rocPad + rocInner}
                  stroke="#475569"
                  strokeWidth="1"
                />
                <line
                  x1={rocPad}
                  y1={rocPad}
                  x2={rocPad}
                  y2={rocPad + rocInner}
                  stroke="#475569"
                  strokeWidth="1"
                />
                {/* Random baseline */}
                <line
                  x1={rocPad}
                  y1={rocPad + rocInner}
                  x2={rocPad + rocInner}
                  y2={rocPad}
                  stroke="#64748b"
                  strokeDasharray="3 3"
                  strokeWidth="1"
                />
                {/* ROC Curve */}
                <polyline
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="2.2"
                  points={rocPolyline}
                />
                <text
                  x={rocPad + rocInner / 2}
                  y={rocSize - 6}
                  textAnchor="middle"
                  className="fill-slate-400 text-[9px]"
                >
                  FPR
                </text>
                <text
                  x={9}
                  y={rocPad + rocInner / 2}
                  textAnchor="middle"
                  transform={`rotate(-90 9 ${rocPad + rocInner / 2})`}
                  className="fill-slate-400 text-[9px]"
                >
                  TPR
                </text>
              </svg>
            </div>
          )}
        </div>
      )}

      {/* Results Table */}
      {table_headers.length > 0 && table_rows.length > 0 && (
        <div className="overflow-x-auto max-h-52 rounded-xl border border-slate-800 bg-slate-950/80">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="sticky top-0 bg-slate-900 border-b border-slate-800">
              <tr>
                {table_headers.map((h, i) => (
                  <th
                    key={i}
                    className="py-2 px-2.5 font-semibold text-slate-300 whitespace-nowrap"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {table_rows.map((row, rIdx) => (
                <tr key={rIdx} className="hover:bg-slate-900/50">
                  {row.map((cell, cIdx) => (
                    <td
                      key={cIdx}
                      className="py-1.5 px-2.5 font-mono text-slate-200 whitespace-nowrap"
                    >
                      {cell === null || cell === undefined ? "-" : String(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
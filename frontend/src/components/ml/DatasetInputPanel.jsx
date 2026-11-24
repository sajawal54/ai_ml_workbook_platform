import { Upload, Plus, Trash2, AlertTriangle } from "lucide-react";

const MAX_CATEGORICAL_VALUES = 10;
const MAX_CATEGORY_TEXT_LENGTH = 80;
const TEXT_COLUMN_HINTS = new Set([
  "address",
  "comment",
  "description",
  "email",
  "free_text",
  "name",
  "note",
  "phone",
  "review",
  "street",
  "text",
  "url",
]);

function isIdentifierColumn(column) {
  const normalized = column
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
  return (
    ["id", "identifier", "uuid"].includes(normalized) ||
    normalized.endsWith("_id") ||
    normalized.endsWith("_identifier") ||
    normalized.endsWith("_uuid")
  );
}

function isFreeTextColumn(column) {
  const normalized = column
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
  return normalized.split("_").some((part) => TEXT_COLUMN_HINTS.has(part));
}

function detectDelimiter(text) {
  const counts = { ",": 0, ";": 0, "\t": 0 };
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') {
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (!quoted && (char === "\n" || char === "\r")) {
      break;
    } else if (!quoted && Object.prototype.hasOwnProperty.call(counts, char)) {
      counts[char] += 1;
    }
  }
  return Object.entries(counts).reduce(
    (best, entry) => (entry[1] > best[1] ? entry : best),
    [",", 0]
  )[0];
}

function parseCsvRecords(text, delimiter) {
  const records = [];
  let row = [];
  let field = "";
  let quoted = false;
  let afterQuote = false;

  const pushField = () => {
    row.push(field.trim());
    field = "";
  };
  const pushRecord = () => {
    pushField();
    if (row.length > 1 || row.some((cell) => cell !== "")) {
      records.push(row);
    }
    row = [];
    afterQuote = false;
  };

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
        afterQuote = true;
      } else {
        field += char;
      }
    } else if (afterQuote) {
      if (char === delimiter) {
        pushField();
        afterQuote = false;
      } else if (char === "\n" || char === "\r") {
        pushRecord();
        if (char === "\r" && text[i + 1] === "\n") i += 1;
      } else if (!/\s/.test(char)) {
        throw new Error("CSV has unexpected text after a quoted field.");
      }
    } else if (char === '"') {
      if (field.trim() !== "") {
        throw new Error("CSV contains a quote inside an unquoted value.");
      }
      field = "";
      quoted = true;
    } else if (char === delimiter) {
      pushField();
    } else if (char === "\n" || char === "\r") {
      pushRecord();
      if (char === "\r" && text[i + 1] === "\n") i += 1;
    } else {
      field += char;
    }
  }

  if (quoted) throw new Error("CSV contains an unclosed quoted field.");
  if (field !== "" || row.length > 0 || afterQuote) pushRecord();
  return records;
}

function parseCsvText(text) {
  const normalizedText = text.replace(/^\uFEFF/, "");
  const records = parseCsvRecords(
    normalizedText,
    detectDelimiter(normalizedText)
  );
  if (records.length < 2) {
    throw new Error("CSV must have a header row and at least one data row.");
  }

  const columns = records[0].map(
    (column, idx) => column || `col_${idx + 1}`
  );
  if (new Set(columns).size !== columns.length) {
    throw new Error("CSV column names must be unique.");
  }
  if (new Set(columns).size !== columns.length) {
    throw new Error("CSV column names must be unique.");
  }
  const rows = records.slice(1).map((record, rowIdx) => {
    if (record.length > columns.length) {
      throw new Error(
        `CSV row ${rowIdx + 2} has ${record.length} values; expected at most ${columns.length}.`
      );
    }
    return columns.map((_, idx) => {
      const raw = record[idx] ?? "";
      if (["", "na", "nan", "null", "none"].includes(raw.toLowerCase())) {
        return null;
      }
      const numeric = Number(raw);
      return Number.isFinite(numeric) ? numeric : raw;
    });
  });

  return { columns, rows };
}

function getFeatureKind(rows, colIdx, column) {
  const values = rows
    .map((row) => row[colIdx])
    .filter(
      (value) =>
        value !== null &&
        value !== undefined &&
        String(value).trim() !== "" &&
        !["na", "nan", "null", "none"].includes(
          String(value).trim().toLowerCase()
        )
    );
  if (values.length === 0) return "empty";
  if (isIdentifierColumn(column)) return "text";

  const hasNonFiniteNumber = values.some((value) => {
    const numeric = typeof value === "number" ? value : Number(value);
    return !Number.isNaN(numeric) && !Number.isFinite(numeric);
  });
  if (hasNonFiniteNumber) return "text";
  if (values.every((value) => {
    const numeric = typeof value === "number" ? value : Number(value);
    return Number.isFinite(numeric);
  })) {
    return "numeric";
  }

  const categories = values.map((value) => String(value).trim());
  const uniqueCount = new Set(categories).size;
  const mostlyUnique = uniqueCount >= 5 && uniqueCount / categories.length > 0.8;
  const longText = categories.some(
    (category) => category.length > MAX_CATEGORY_TEXT_LENGTH
  );
  if (
    uniqueCount > MAX_CATEGORICAL_VALUES ||
    mostlyUnique ||
    longText ||
    isFreeTextColumn(column)
  ) {
    return "text";
  }
  return "categorical";
}

export default function DatasetInputPanel({
  dataset,
  onChangeDataset,
  featureColumns,
  onChangeFeatureColumns,
  targetColumn,
  onChangeTargetColumn,
  taskType,
  onChangeTaskType,
  testSize,
  onChangeTestSize,
  scaleFeatures,
  onChangeScaleFeatures,
  onCsvError,
}) {
  const columnKinds = dataset.columns.map((column, idx) =>
    getFeatureKind(dataset.rows, idx, column)
  );
  const unsupportedColumns = dataset.columns.filter(
    (_, idx) => columnKinds[idx] === "text"
  );
  const categoricalColumns = dataset.columns.filter(
    (_, idx) => columnKinds[idx] === "categorical"
  );
  const selectedUnsupportedFeatures = featureColumns.filter((col) => {
    const idx = dataset.columns.indexOf(col);
    return idx >= 0 && columnKinds[idx] === "text";
  });

  const handleCsvFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseCsvText(text);
      onCsvError("");
      onChangeDataset(parsed);

      const suggestedTarget = parsed.columns[parsed.columns.length - 1] || "";
      const cleanFeatures = parsed.columns.filter((column, idx) => {
        const kind = getFeatureKind(parsed.rows, idx, column);
        return (
          column !== suggestedTarget &&
          (kind === "numeric" || kind === "categorical")
        );
      });
      if (
        parsed.columns.length === 1 &&
        ["numeric", "categorical"].includes(
          getFeatureKind(parsed.rows, 0, parsed.columns[0])
        )
      ) {
        cleanFeatures.push(suggestedTarget);
      }
      onChangeFeatureColumns(cleanFeatures);
      onChangeTargetColumn(suggestedTarget);
    } catch (err) {
      onCsvError(err.message || "Failed to parse CSV file.");
    } finally {
      e.target.value = "";
    }
  };

  const handleCellChange = (rowIdx, colIdx, value) => {
    const nextRows = dataset.rows.map((r, rI) =>
      rI === rowIdx
        ? r.map((cell, cI) => {
            if (cI !== colIdx) return cell;
            if (value.trim() === "") return null;
            const num = Number(value);
            return Number.isNaN(num) ? value : num;
          })
        : r
    );
    onChangeDataset({ ...dataset, rows: nextRows });
  };

  const handleColumnNameChange = (colIdx, newName) => {
    const oldName = dataset.columns[colIdx];
    const cleanName = newName || `col_${colIdx + 1}`;
    const nextCols = dataset.columns.map((c, idx) =>
      idx === colIdx ? cleanName : c
    );
    onChangeDataset({ ...dataset, columns: nextCols });
    onChangeFeatureColumns(
      featureColumns.map((f) => (f === oldName ? cleanName : f))
    );
    if (targetColumn === oldName) {
      onChangeTargetColumn(cleanName);
    }
  };

  const handleAddRow = () => {
    const nextRow = dataset.columns.map(() => 0);
    onChangeDataset({ ...dataset, rows: [...dataset.rows, nextRow] });
  };

  const handleDeleteRow = (rowIdx) => {
    if (dataset.rows.length <= 4) return;
    onChangeDataset({
      ...dataset,
      rows: dataset.rows.filter((_, idx) => idx !== rowIdx),
    });
  };

  const handleAddColumn = () => {
    const newColName = `feature_${dataset.columns.length + 1}`;
    onChangeDataset({
      columns: [...dataset.columns, newColName],
      rows: dataset.rows.map((r) => [...r, 0]),
    });
    onChangeFeatureColumns([...featureColumns, newColName]);
  };

  const toggleFeatureCol = (col) => {
    if (featureColumns.includes(col)) {
      if (featureColumns.length > 1) {
        onChangeFeatureColumns(featureColumns.filter((item) => item !== col));
      }
    } else {
      onChangeFeatureColumns([...featureColumns, col]);
    }
  };

  return (
    <section className="rounded-2xl bg-slate-900 border border-slate-800 p-5 space-y-5 min-w-0">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
            Step 1 • Dataset & Column Setup
          </span>
          <h3 className="text-base font-bold text-white mt-0.5">
            Upload CSV or Edit Table Values
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Numeric and low-cardinality categorical features are supported; high-cardinality text and ID columns are excluded.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="cursor-pointer inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition">
            <Upload className="w-3.5 h-3.5" />
            <span>Upload CSV</span>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={handleCsvFile}
              className="hidden"
            />
          </label>

          <button
            type="button"
            onClick={handleAddRow}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Row
          </button>

          <button
            type="button"
            onClick={handleAddColumn}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Column
          </button>
        </div>
      </div>

      {/* Feature Data Notice / Warning */}
      <div className="rounded-xl bg-amber-500/10 border border-amber-500/30 px-4 py-3 text-xs text-amber-200 flex items-start gap-2.5">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-semibold text-amber-300">
            Features support numbers or categories with up to 10 values.
          </p>
          {selectedUnsupportedFeatures.length > 0 ? (
            <p>
              Remove high-cardinality text/ID feature(s):{" "}
              <span className="font-mono font-semibold text-white">
                {selectedUnsupportedFeatures.join(", ")}
              </span>
              .
            </p>
          ) : unsupportedColumns.length > 0 ? (
            <p>
              High-cardinality text/ID columns are not encoded:{" "}
              <span className="font-mono text-amber-100">
                {unsupportedColumns.join(", ")}
              </span>
              . Low-cardinality categories are one-hot encoded.
            </p>
          ) : categoricalColumns.length > 0 ? (
            <p className="text-amber-200/80">
              Low-cardinality categorical features are one-hot encoded.
            </p>
          ) : (
            <p className="text-amber-200/80">
              All current feature columns look numeric.
            </p>
          )}
        </div>
      </div>

      {/* Feature, Target & Preprocessing Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 p-4 rounded-xl bg-slate-950/70 border border-slate-800">
        <div className="lg:col-span-5">
          <p className="text-xs font-semibold text-white">
            Input Columns (Features)
          </p>
          <p className="text-[11px] text-slate-400 mb-2">
            Select numeric or low-cardinality categorical columns (unselect the target).
          </p>
          <div className="flex flex-wrap gap-1.5">
            {dataset.columns.map((col, idx) => {
              const active = featureColumns.includes(col);
              const kind = columnKinds[idx];
              return (
                <button
                  key={col}
                  type="button"
                  onClick={() => toggleFeatureCol(col)}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border transition cursor-pointer ${
                    active
                      ? "bg-indigo-600/25 border-indigo-400 text-indigo-200"
                      : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <span>{col}</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                      kind === "numeric"
                        ? "bg-emerald-500/15 text-emerald-300"
                        : kind === "categorical"
                        ? "bg-sky-500/15 text-sky-300"
                        : "bg-amber-500/15 text-amber-300"
                    }`}
                  >
                    {kind === "numeric"
                      ? "num"
                      : kind === "categorical"
                      ? "cat"
                      : "text"}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="lg:col-span-4 space-y-3">
          <div>
            <label className="block text-xs font-semibold text-white mb-1">
              Target Column (Supervised)
            </label>
            <select
              value={targetColumn}
              onChange={(e) => onChangeTargetColumn(e.target.value)}
              className="w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
            >
              {dataset.columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-white mb-1">
              Target Mode
            </label>
            <select
              value={taskType}
              onChange={(e) => onChangeTaskType(e.target.value)}
              className="w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
            >
              <option value="auto">Auto Detect</option>
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
        </div>

        <div className="lg:col-span-3 space-y-3">
          <div>
            <div className="flex justify-between text-xs text-white mb-1">
              <span className="font-semibold">Test Split</span>
              <span className="font-mono text-indigo-400">
                {Math.round(testSize * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={testSize}
              onChange={(e) => onChangeTestSize(Number(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
          </div>

          <label className="flex items-center gap-2 text-xs text-slate-200 cursor-pointer pt-1">
            <input
              type="checkbox"
              checked={scaleFeatures}
              onChange={(e) => onChangeScaleFeatures(e.target.checked)}
              className="rounded accent-indigo-500"
            />
            <span>Standardize features (StandardScaler)</span>
          </label>
        </div>
      </div>

      {/* Editable Data Table */}
      <div className="overflow-x-auto max-h-64 rounded-xl border border-slate-800 bg-slate-950/70">
        <table className="w-full text-left border-collapse text-xs">
          <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 z-10">
            <tr>
              <th className="py-2 px-3 text-slate-400 font-semibold w-12">
                Row
              </th>
              {dataset.columns.map((col, colIdx) => (
                <th key={colIdx} className="py-2 px-2">
                  <input
                    type="text"
                    value={col}
                    onChange={(e) =>
                      handleColumnNameChange(colIdx, e.target.value)
                    }
                    className="w-full min-w-24 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs font-semibold text-indigo-300 focus:outline-none focus:border-indigo-500"
                  />
                </th>
              ))}
              <th className="py-2 px-2 w-10 text-center text-slate-400">
                Del
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {dataset.rows.map((row, rowIdx) => (
              <tr key={rowIdx} className="hover:bg-slate-900/40">
                <td className="py-1.5 px-3 text-slate-500 font-mono">
                  {rowIdx + 1}
                </td>
                {row.map((cell, colIdx) => (
                  <td key={colIdx} className="py-1.5 px-2">
                    <input
                      type="text"
                      value={
                        cell === null || cell === undefined ? "" : String(cell)
                      }
                      onChange={(e) =>
                        handleCellChange(rowIdx, colIdx, e.target.value)
                      }
                      className="w-full min-w-20 bg-slate-900 border border-slate-800 focus:border-indigo-500 rounded px-2 py-1 text-xs font-mono text-slate-100 focus:outline-none"
                    />
                  </td>
                ))}
                <td className="py-1.5 px-2 text-center">
                  <button
                    type="button"
                    onClick={() => handleDeleteRow(rowIdx)}
                    disabled={dataset.rows.length <= 4}
                    title="Delete row"
                    className="p-1 text-slate-500 hover:text-rose-400 disabled:opacity-30 transition cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
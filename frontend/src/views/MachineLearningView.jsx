import { useRef, useState } from "react";
import { PlaySquare, Loader2 } from "lucide-react";
import DatasetInputPanel from "../components/ml/DatasetInputPanel";
import AlgorithmCard from "../components/ml/AlgorithmCard";
import {
  checkBackendHealth,
  runSupervisedAlgorithm,
  runUnsupervisedAlgorithm,
} from "../services/mlApi";

const DEFAULT_DATASET = {
  columns: ["feature_1", "feature_2", "feature_3", "target"],
  rows: [
    [1.2, 3.4, 5.0, 0],
    [1.5, 3.1, 4.8, 0],
    [2.0, 2.9, 5.2, 0],
    [2.3, 3.5, 4.9, 0],
    [6.1, 7.8, 8.5, 1],
    [6.5, 8.2, 9.0, 1],
    [7.0, 7.5, 8.8, 1],
    [7.2, 8.0, 9.2, 1],
  ],
};

const SUPERVISED_ALGOS = [
  {
    id: "logistic_regression",
    name: "Logistic Regression",
    shortName: "Logistic Reg",
    badge: "Classification",
    description: "Linear model for binary or multi-class classification.",
    paramFields: [
      { key: "C", label: "Inverse Regularization (C)", defaultValue: 1.0, step: "0.1" },
      { key: "max_iter", label: "Max Iterations", defaultValue: 2000, step: "100" },
    ],
  },
  {
    id: "random_forest_supervised",
    name: "Random Forest",
    shortName: "Random Forest",
    badge: "Classification / Regression",
    description: "Ensemble of decision trees for classification or continuous targets.",
    paramFields: [
      { key: "n_estimators", label: "Trees (n_estimators)", defaultValue: 150, step: "10" },
      {
        key: "max_depth",
        label: "Max Depth (blank = None)",
        inputType: "text",
        defaultValue: "",
        placeholder: "e.g. 5 or blank",
      },
    ],
  },
  {
    id: "svm_supervised",
    name: "Support Vector Machine (SVM)",
    shortName: "SVM",
    badge: "SVC / SVR",
    description: "Margin-based model using linear, RBF, or polynomial kernels.",
    paramFields: [
      { key: "C", label: "Penalty (C)", defaultValue: 1.0, step: "0.1" },
      {
        key: "kernel",
        label: "Kernel",
        type: "select",
        defaultValue: "rbf",
        options: [
          { value: "rbf", label: "RBF (Gaussian)" },
          { value: "linear", label: "Linear" },
          { value: "poly", label: "Polynomial" },
        ],
      },
    ],
  },
  {
    id: "knn_supervised",
    name: "K-Nearest Neighbors (KNN)",
    shortName: "KNN",
    badge: "Classification / Regression",
    description: "Predicts from the k closest neighbors in feature space.",
    paramFields: [
      { key: "n_neighbors", label: "Neighbors (k)", defaultValue: 5, step: "1" },
      {
        key: "weights",
        label: "Weighting",
        type: "select",
        defaultValue: "uniform",
        options: [
          { value: "uniform", label: "Uniform" },
          { value: "distance", label: "Distance" },
        ],
      },
    ],
  },
  {
    id: "linear_ridge_regression",
    name: "Linear / Ridge Regression",
    shortName: "Linear / Ridge",
    badge: "Regression",
    description: "Ordinary least squares or L2-regularized Ridge regression.",
    paramFields: [
      { key: "alpha", label: "Ridge Alpha", defaultValue: 1.0, step: "0.1" },
      { key: "use_ridge", label: "Use Ridge (L2)", type: "checkbox", defaultValue: true },
    ],
  },
];

const UNSUPERVISED_ALGOS = [
  {
    id: "kmeans",
    name: "K-Means Clustering",
    shortName: "K-Means",
    badge: "Clustering",
    description: "Partitions rows into k clusters around centroids.",
    paramFields: [
      { key: "n_clusters", label: "Clusters (k)", defaultValue: 3, step: "1" },
      { key: "max_iter", label: "Max Iterations", defaultValue: 300, step: "50" },
    ],
  },
  {
    id: "dbscan",
    name: "DBSCAN",
    shortName: "DBSCAN",
    badge: "Density Clustering",
    description: "Finds dense clusters and marks sparse points as noise (-1).",
    paramFields: [
      { key: "eps", label: "Radius (eps)", defaultValue: 0.9, step: "0.1" },
      { key: "min_samples", label: "Min Samples", defaultValue: 3, step: "1" },
    ],
  },
  {
    id: "hierarchical",
    name: "Hierarchical Clustering",
    shortName: "Hierarchical",
    badge: "Clustering",
    description: "Agglomerative clustering with configurable linkage.",
    paramFields: [
      { key: "n_clusters", label: "Clusters (k)", defaultValue: 3, step: "1" },
      {
        key: "linkage",
        label: "Linkage",
        type: "select",
        defaultValue: "ward",
        options: [
          { value: "ward", label: "Ward" },
          { value: "complete", label: "Complete" },
          { value: "average", label: "Average" },
        ],
      },
    ],
  },
  {
    id: "pca",
    name: "Principal Component Analysis (PCA)",
    shortName: "PCA",
    badge: "Dimensionality Reduction",
    description: "Projects numeric features onto top variance components.",
    paramFields: [
      { key: "n_components", label: "Components", defaultValue: 2, step: "1" },
    ],
  },
  {
    id: "isolation_forest",
    name: "Isolation Forest",
    shortName: "Isolation Forest",
    badge: "Anomaly Detection",
    description: "Flags unusual rows/anomalies using isolation trees.",
    paramFields: [
      { key: "n_estimators", label: "Trees (n_estimators)", defaultValue: 100, step: "10" },
      {
        key: "contamination",
        label: "Contamination ('auto' or 0.05-0.5)",
        inputType: "text",
        defaultValue: "auto",
        placeholder: "auto or 0.1",
      },
    ],
  },
];

function buildInitialParams(algos) {
  const map = {};
  for (const algo of algos) {
    map[algo.id] = {};
    for (const f of algo.paramFields) {
      map[algo.id][f.key] = f.defaultValue;
    }
  }
  return map;
}

function getTargetInfo(dataset, targetColumn) {
  const targetIndex = dataset.columns.indexOf(targetColumn);
  const values = dataset.rows
    .map((row) => row[targetIndex])
    .filter(
      (value) =>
        value !== null &&
        value !== undefined &&
        !["", "na", "nan", "null", "none"].includes(
          String(value).trim().toLowerCase()
        )
    );
  const numericValues = values.map((value) =>
    typeof value === "number" ? value : Number(value)
  );
  const isNumeric =
    values.length > 0 && numericValues.every(Number.isFinite);
  const classes = new Map();
  for (const value of values) {
    const label = String(value);
    classes.set(label, (classes.get(label) || 0) + 1);
  }
  const minClassCount = [...classes.values()].reduce(
    (minimum, count) => Math.min(minimum, count),
    Infinity
  );
  const uniqueNumericValues = isNumeric ? [...new Set(numericValues)] : [];
  const autoTask =
    isNumeric &&
    uniqueNumericValues.length >= 2 &&
    uniqueNumericValues.length <= 10 &&
    uniqueNumericValues.every(Number.isInteger)
      ? "classification"
      : isNumeric
      ? "regression"
      : "classification";

  return {
    values,
    isNumeric,
    autoTask,
    classCount: classes.size,
    minClassCount,
  };
}

function getModelTask(algo, taskType, targetInfo) {
  if (taskType !== "auto") return taskType;
  if (algo.id === "linear_ridge_regression" && targetInfo.isNumeric) {
    return "regression";
  }
  return targetInfo.autoTask;
}

function getModelIncompatibility(algo, resolvedTask, targetInfo) {
  if (resolvedTask === "regression" && !targetInfo.isNumeric) {
    return "regression requires a numeric target";
  }
  if (algo.id === "logistic_regression" && resolvedTask !== "classification") {
    return "Logistic Regression supports classification only";
  }
  if (
    algo.id === "linear_ridge_regression" &&
    resolvedTask !== "regression"
  ) {
    return "Linear / Ridge Regression supports regression only";
  }
  return "";
}

function validateRunInputs(dataset, featureColumns, targetColumn, category) {
  if (dataset.rows.length < 4) {
    return "Add at least 4 data rows before running an algorithm.";
  }
  if (featureColumns.length === 0) {
    return "Select at least one input feature.";
  }
  if (new Set(featureColumns).size !== featureColumns.length) {
    return "Input feature columns must be unique.";
  }

  const missingFeatures = featureColumns.filter(
    (column) => !dataset.columns.includes(column)
  );
  if (missingFeatures.length > 0) {
    return `Feature column '${missingFeatures[0]}' is no longer in the dataset.`;
  }

  if (category === "supervised") {
    if (!dataset.columns.includes(targetColumn)) {
      return "Select a target column that exists in the dataset.";
    }
    if (featureColumns.includes(targetColumn)) {
      return "Unselect the target column from Input Features.";
    }
    const targetIndex = dataset.columns.indexOf(targetColumn);
    const validTargets = dataset.rows
      .map((row) => row[targetIndex])
      .filter((value) => {
        return (
          value !== null &&
          value !== undefined &&
          !["", "na", "nan", "null", "none"].includes(
            String(value).trim().toLowerCase()
          )
        );
      });
    if (validTargets.length < 5) {
      return "At least 5 rows with target values are required for supervised models.";
    }
    const targetIsNumeric = validTargets.every((value) => {
      const numeric = typeof value === "number" ? value : Number(value);
      return !Number.isNaN(numeric);
    });
    if (
      targetIsNumeric &&
      validTargets.some((value) => !Number.isFinite(Number(value)))
    ) {
      return "Numeric target values must be finite numbers.";
    }
  }

  for (const column of featureColumns) {
    const columnIndex = dataset.columns.indexOf(column);
    const values = dataset.rows
      .map((row) => row[columnIndex])
      .filter(
        (value) =>
          value !== null &&
          value !== undefined &&
          !["", "na", "nan", "null", "none"].includes(
            String(value).trim().toLowerCase()
          )
      );
    if (values.length === 0) {
      return `Feature column '${column}' has no numeric values.`;
    }
    if (
      values.some((value) => {
        const numeric = typeof value === "number" ? value : Number(value);
        return !Number.isFinite(numeric);
      })
    ) {
      return `Feature column '${column}' contains non-numeric or non-finite values.`;
    }
  }

  return "";
}

export default function MachineLearningView() {
  const [activeTab, setActiveTab] = useState("supervised");
  const [dataset, setDataset] = useState(DEFAULT_DATASET);
  const requestVersions = useRef({});
  const runInputVersion = useRef(0);
  const [featureColumns, setFeatureColumns] = useState(
    DEFAULT_DATASET.columns.filter((c) => c !== "target")
  );
  const [targetColumn, setTargetColumn] = useState("target");
  const [taskType, setTaskType] = useState("auto");
  const [testSize, setTestSize] = useState(0.25);
  const [scaleFeatures, setScaleFeatures] = useState(true);
  const [csvError, setCsvError] = useState("");
  const [runAllError, setRunAllError] = useState("");
  const [runAllNote, setRunAllNote] = useState("");
  const [runningAll, setRunningAll] = useState(false);

  const [paramsByAlgo, setParamsByAlgo] = useState(() =>
    buildInitialParams([...SUPERVISED_ALGOS, ...UNSUPERVISED_ALGOS])
  );
  const [resultsByAlgo, setResultsByAlgo] = useState({});
  const [errorsByAlgo, setErrorsByAlgo] = useState({});
  const [loadingByAlgo, setLoadingByAlgo] = useState({});

  const clearRunState = () => {
    runInputVersion.current += 1;
    for (const algoId of Object.keys(requestVersions.current)) {
      requestVersions.current[algoId] += 1;
    }
    setResultsByAlgo({});
    setErrorsByAlgo({});
    setLoadingByAlgo({});
    setCsvError("");
    setRunAllError("");
    setRunAllNote("");
  };

  const clearAlgorithmState = (algoId) => {
    requestVersions.current[algoId] =
      (requestVersions.current[algoId] || 0) + 1;
    setResultsByAlgo((prev) => {
      const next = { ...prev };
      delete next[algoId];
      return next;
    });
    setErrorsByAlgo((prev) => {
      const next = { ...prev };
      delete next[algoId];
      return next;
    });
    setLoadingByAlgo((prev) => ({ ...prev, [algoId]: false }));
  };

  const handleDatasetChange = (nextDataset) => {
    clearRunState();
    setDataset(nextDataset);
  };

  const handleParamChange = (algoId, key, value) => {
    clearAlgorithmState(algoId);
    setParamsByAlgo((prev) => ({
      ...prev,
      [algoId]: {
        ...(prev[algoId] || {}),
        [key]: value,
      },
    }));
  };

  const handleRunAlgorithm = async (algo, category) => {
    setRunAllError("");
    const requestVersion = (requestVersions.current[algo.id] || 0) + 1;
    requestVersions.current[algo.id] = requestVersion;
    setLoadingByAlgo((prev) => ({ ...prev, [algo.id]: true }));
    setErrorsByAlgo((prev) => ({ ...prev, [algo.id]: "" }));
    setResultsByAlgo((prev) => {
      const next = { ...prev };
      delete next[algo.id];
      return next;
    });

    try {
      const response =
        category === "supervised"
          ? await runSupervisedAlgorithm({
              algorithm: algo.id,
              dataset,
              feature_columns: featureColumns,
              target_column: targetColumn,
              task_type: taskType,
              test_size: testSize,
              scale_features: scaleFeatures,
              params: paramsByAlgo[algo.id] || {},
            })
          : await runUnsupervisedAlgorithm({
              algorithm: algo.id,
              dataset,
              feature_columns: featureColumns,
              scale_features: scaleFeatures,
              params: paramsByAlgo[algo.id] || {},
            });

      if (requestVersions.current[algo.id] === requestVersion) {
        setResultsByAlgo((prev) => ({ ...prev, [algo.id]: response }));
      }
    } catch (err) {
      if (requestVersions.current[algo.id] === requestVersion) {
        setErrorsByAlgo((prev) => ({
          ...prev,
          [algo.id]: err.message || "Failed to run algorithm.",
        }));
      }
    } finally {
      if (requestVersions.current[algo.id] === requestVersion) {
        setLoadingByAlgo((prev) => ({ ...prev, [algo.id]: false }));
      }
    }
  };

  const activeAlgos =
    activeTab === "supervised" ? SUPERVISED_ALGOS : UNSUPERVISED_ALGOS;

  const handleRunAll = async () => {
    const inputError = validateRunInputs(
      dataset,
      featureColumns,
      targetColumn,
      activeTab
    );
    if (inputError) {
      setRunAllError(inputError);
      setRunAllNote("");
      return;
    }

    let algorithmsToRun = activeAlgos;
    let skippedAlgorithms = [];
    let taskTypesToRun = [];
    if (activeTab === "supervised") {
      const targetInfo = getTargetInfo(dataset, targetColumn);
      const plannedAlgorithms = activeAlgos.map((algo) => ({
        algo,
        task: getModelTask(algo, taskType, targetInfo),
      }));
      const compatibleAlgorithms = [];
      for (const planned of plannedAlgorithms) {
        const reason = getModelIncompatibility(
          planned.algo,
          planned.task,
          targetInfo
        );
        if (reason) {
          skippedAlgorithms.push({ ...planned, reason });
        } else {
          compatibleAlgorithms.push(planned);
        }
      }

      const hasClassificationModel = compatibleAlgorithms.some(
        ({ task }) => task === "classification"
      );
      if (hasClassificationModel && targetInfo.classCount < 2) {
        setRunAllError("Classification requires at least two distinct target classes.");
        setRunAllNote("");
        return;
      }
      if (
        hasClassificationModel &&
        targetInfo.minClassCount < 2
      ) {
        setRunAllError("Each classification target class needs at least two rows.");
        setRunAllNote("");
        return;
      }

      algorithmsToRun = compatibleAlgorithms.map(({ algo }) => algo);
      taskTypesToRun = compatibleAlgorithms.map(({ task }) => task);
    }

    const skippedNames = skippedAlgorithms
      .map(({ algo }) => algo.shortName)
      .join(", ");
    const skipReasons = [...new Set(skippedAlgorithms.map(({ reason }) => reason))];
    const runAllNotes = [];
    if (skippedAlgorithms.length) {
      runAllNotes.push(
        `Skipped ${skippedNames}: ${skipReasons.join("; ")}.${
          algorithmsToRun.length
            ? ` Running ${algorithmsToRun.length} compatible model(s).`
            : " No compatible models to run."
        }`
      );
    }
    if (new Set(taskTypesToRun).size > 1) {
      runAllNotes.push(
        "Auto mode uses both classification and regression; compare metrics within each task."
      );
    }
    setRunAllNote(runAllNotes.join(" "));
    setRunAllError("");
    if (algorithmsToRun.length === 0) return;

    const inputVersion = runInputVersion.current;
    setRunningAll(true);
    try {
      await checkBackendHealth();
      if (runInputVersion.current !== inputVersion) return;
      await Promise.all(
        algorithmsToRun.map((algo) => handleRunAlgorithm(algo, activeTab))
      );
    } catch (error) {
      setRunAllNote("");
      setRunAllError(
        error.message || "Unable to run the selected algorithms."
      );
    } finally {
      setRunningAll(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto min-w-0 space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="min-w-0">
          <span className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
            Machine Learning Workspace
          </span>
          <h2 className="text-2xl font-bold text-white mt-0.5">
            Supervised (5) & Unsupervised (5) Models
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Upload a CSV or edit rows in Step 1, then run individual models or Run All 5 in Step 2.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={handleRunAll}
            disabled={runningAll}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white transition cursor-pointer"
          >
            {runningAll ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Running compatible models...</span>
              </>
            ) : (
              <>
                <PlaySquare className="w-3.5 h-3.5" />
                <span>Run All 5</span>
              </>
            )}
          </button>

          <div className="inline-flex p-1 rounded-xl bg-slate-900 border border-slate-800">
            <button
              type="button"
              disabled={runningAll}
              onClick={() => setActiveTab("supervised")}
              className={`px-4 py-2 rounded-lg text-xs font-semibold transition cursor-pointer ${
                activeTab === "supervised"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Supervised (5)
            </button>
            <button
              type="button"
              disabled={runningAll}
              onClick={() => setActiveTab("unsupervised")}
              className={`px-4 py-2 rounded-lg text-xs font-semibold transition cursor-pointer ${
                activeTab === "unsupervised"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Unsupervised (5)
            </button>
          </div>
        </div>
      </div>

      {csvError && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 px-4 py-3 text-xs text-rose-300">
          {csvError}
        </div>
      )}

      {runAllError && (
        <div
          className="rounded-xl bg-rose-500/10 border border-rose-500/30 px-4 py-3 text-xs text-rose-300"
          role="alert"
        >
          {runAllError}
        </div>
      )}

      {runAllNote && (
        <p className="text-xs text-amber-300" role="status">
          {runAllNote}
        </p>
      )}

      <DatasetInputPanel
        dataset={dataset}
        onChangeDataset={handleDatasetChange}
        featureColumns={featureColumns}
        onChangeFeatureColumns={(columns) => {
          clearRunState();
          setFeatureColumns(columns);
        }}
        targetColumn={targetColumn}
        onChangeTargetColumn={(column) => {
          clearRunState();
          setFeatureColumns((previous) =>
            previous.filter((feature) => feature !== column)
          );
          setTargetColumn(column);
        }}
        taskType={taskType}
        onChangeTaskType={(task) => {
          clearRunState();
          setTaskType(task);
        }}
        testSize={testSize}
        onChangeTestSize={(size) => {
          clearRunState();
          setTestSize(size);
        }}
        scaleFeatures={scaleFeatures}
        onChangeScaleFeatures={(scale) => {
          clearRunState();
          setScaleFeatures(scale);
        }}
        onCsvError={setCsvError}
      />

      <section className="space-y-4 min-w-0">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
            Step 2 • Algorithms
          </span>
          <h3 className="text-lg font-bold text-white mt-0.5">
            {activeTab === "supervised"
              ? "5 Supervised Algorithms"
              : "5 Unsupervised Algorithms"}
          </h3>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5 min-w-0">
          {activeAlgos.map((algo) => (
            <AlgorithmCard
              key={algo.id}
              algo={algo}
              params={paramsByAlgo[algo.id] || {}}
              onChangeParam={(key, val) => handleParamChange(algo.id, key, val)}
              onRun={() => {
                setRunAllNote("");
                handleRunAlgorithm(algo, activeTab);
              }}
              onClearResult={() =>
                clearAlgorithmState(algo.id)
              }
              result={resultsByAlgo[algo.id] || null}
              error={errorsByAlgo[algo.id] || ""}
              loading={Boolean(loadingByAlgo[algo.id])}
            />
          ))}
        </div>
      </section>
    </div>
  );
}
export const API_BASE_URL = "http://127.0.0.1:8000";

async function parseErrorResponse(res) {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    if (Array.isArray(data?.detail)) {
      const unique = [
        ...new Set(data.detail.map((item) => item.msg || String(item))),
      ];
      return unique.slice(0, 3).join(" | ");
    }
    return `Request failed with status ${res.status}.`;
  } catch {
    return `Request failed with status ${res.status}.`;
  }
}

async function safeFetch(url, options) {
  try {
    return await fetch(url, options);
  } catch {
    throw new Error(
      `Cannot reach the backend at ${API_BASE_URL}. Check that FastAPI is running on port 8000 and that the frontend origin is allowed.`
    );
  }
}

function serializePayload(payload) {
  return JSON.stringify(payload, (_key, value) => {
    if (typeof value === "number" && !Number.isFinite(value)) {
      throw new Error("Numeric inputs must be finite numbers.");
    }
    return value;
  });
}

async function parseJsonResponse(res) {
  try {
    return await res.json();
  } catch {
    throw new Error(`Backend returned an invalid response (HTTP ${res.status}).`);
  }
}

export async function checkBackendHealth(baseUrl = API_BASE_URL) {
  const res = await safeFetch(`${baseUrl}/health`);
  if (!res.ok) {
    throw new Error("Backend health check failed.");
  }
  return parseJsonResponse(res);
}

export async function runSupervisedAlgorithm(payload, baseUrl = API_BASE_URL) {
  const res = await safeFetch(`${baseUrl}/api/ml/supervised`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: serializePayload(payload),
  });
  if (!res.ok) {
    throw new Error(await parseErrorResponse(res));
  }
  return parseJsonResponse(res);
}

export async function runUnsupervisedAlgorithm(payload, baseUrl = API_BASE_URL) {
  const res = await safeFetch(`${baseUrl}/api/ml/unsupervised`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: serializePayload(payload),
  });
  if (!res.ok) {
    throw new Error(await parseErrorResponse(res));
  }
  return parseJsonResponse(res);
}
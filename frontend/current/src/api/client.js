const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch { /* keep statusText */ }
    throw new Error(`API ${res.status}: ${detail}`)
  }
  return res.json()
}

function qs(params) {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v != null && v !== '') p.set(k, v)
  }
  const s = p.toString()
  return s ? `?${s}` : ''
}

export const api = {
  meta: () => request('/meta'),
  graphYears: () => request('/graph/years'),
  graphByYear: (year) => request(`/graph/${year}`),
  graphLocate: (symbol) => request(`/graph/locate/${encodeURIComponent(symbol)}`),

  modelMetrics: (run) => request(`/model/metrics${qs({ run })}`),
  modelEvalLog: (run) => request(`/model/eval-log${qs({ run })}`),
  modelExperiments: () => request('/model/experiments'),
  predictionsTest: (run, limit = 0) => request(`/predictions/test${qs({ run, limit })}`),
  predictionsFuture: (run, limit = 0) => request(`/predictions/future${qs({ run, limit })}`),

  companySearch: (q) => request(`/company/search${qs({ q })}`),
  companyDetail: (symbol, run) => request(`/company/${encodeURIComponent(symbol)}${qs({ run })}`),

  inference: (run, year, top = 50) => request(`/inference${qs({ run, year, top })}`),
}

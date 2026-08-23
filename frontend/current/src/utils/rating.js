export const RATING_ORDER = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

export const RATING_COLORS = {
  AAA: '#1a9850',
  AA: '#66bd63',
  A: '#a6d96a',
  BBB: '#fee08b',
  BB: '#fdae61',
  B: '#f46d43',
  CCC: '#d73027',
  D: '#a50026',
}

export function ratingColor(rating) {
  return RATING_COLORS[rating] || '#8a9099'
}

export function probToRating(p) {
  if (p == null) return '—'
  if (p < 0.01) return 'AAA'
  if (p < 0.05) return 'AA'
  if (p < 0.1) return 'A'
  if (p < 0.2) return 'BBB'
  if (p < 0.3) return 'BB'
  if (p < 0.4) return 'B'
  if (p < 0.6) return 'CCC'
  return 'D'
}

export function fmtProb(p, digits = 2) {
  if (p == null) return '—'
  return `${(p * 100).toFixed(digits)}%`
}

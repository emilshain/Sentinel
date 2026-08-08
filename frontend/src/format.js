// Shared numeric formatting for the dashboard.
//
// g2() mimics printf's "%.2g": render a number to 2 significant figures with no
// trailing zeros and no exponent for the value ranges this UI deals with
// (~1e-4 .. ~1e3). Use it for every displayed *measured* quantity — confidence,
// risk score, runtime, z-scores, anomaly scores — so rounding is consistent.
//
// Do NOT use it for exact counts or identifiers (vote tallies, row indices,
// sample counts, class labels): 2-sig-fig rounding would distort them.
export function g2(x) {
  if (typeof x !== 'number' || !Number.isFinite(x)) return '—'
  if (x === 0) return '0'
  // toPrecision(2) gives 2 significant figures but may emit exponential form
  // (e.g. "1.0e+2") and trailing zeros; Number(...).toString() normalizes both.
  return Number(x.toPrecision(2)).toString()
}

/**
 * TensorBoard-style exponential moving average with debias correction.
 *
 * weight ∈ [0, 1) controls smoothing strength: 0 = no smoothing (returns raw),
 * 0.6 (default) = mild, 0.9+ = aggressive. The debias term `1 - weight^k`
 * prevents the early-step pull toward zero that a naive EMA exhibits.
 */
export const smoothEma = (values: ReadonlyArray<number>, weight: number): number[] => {
  if (weight <= 0 || values.length === 0) {
    return values.slice();
  }
  const w = Math.min(weight, 0.999);
  const out = new Array<number>(values.length);
  let last = 0;
  let debiasWeight = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      out[i] = Number.NaN;
      continue;
    }
    last = last * w + (1 - w) * v;
    debiasWeight = debiasWeight * w + (1 - w);
    out[i] = last / debiasWeight;
  }
  return out;
};

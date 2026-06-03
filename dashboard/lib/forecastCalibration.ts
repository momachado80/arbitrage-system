/**
 * Forecast calibration — métricas puras para validar um modelo probabilístico
 * ANTES de qualquer capital. A pergunta central: as probabilidades do modelo são
 * bem calibradas contra os resultados reais? (e, num passo seguinte, batem o mercado?)
 *
 * Brier score (menor = melhor; 0.25 = chute 50/50; 0 = perfeito).
 * Log loss (menor = melhor; penaliza confiança errada).
 * Calibração por bins: nos eventos previstos a ~p, a frequência real foi ~p?
 *
 * Sem rede, sem I/O, sem execução. Apenas estatística sobre (predição, resultado).
 */

export interface ForecastSample {
  /** Probabilidade prevista pelo modelo para o evento (0..1). */
  predicted: number;
  /** Resultado realizado: 1 se o evento ocorreu, 0 se não. */
  outcome: 0 | 1;
}

/** Brier score = média de (predicted − outcome)². */
export function brierScore(samples: ForecastSample[]): number | null {
  if (samples.length === 0) return null;
  let acc = 0;
  for (const s of samples) acc += (s.predicted - s.outcome) ** 2;
  return acc / samples.length;
}

/** Log loss = −média de [y·ln p + (1−y)·ln(1−p)], com clamp para evitar ln(0). */
export function logLoss(samples: ForecastSample[], eps = 1e-12): number | null {
  if (samples.length === 0) return null;
  let acc = 0;
  for (const s of samples) {
    const p = Math.min(1 - eps, Math.max(eps, s.predicted));
    acc += s.outcome === 1 ? -Math.log(p) : -Math.log(1 - p);
  }
  return acc / samples.length;
}

export interface CalibrationBin {
  lo: number;
  hi: number;
  count: number;
  meanPredicted: number | null;
  empiricalRate: number | null;
}

/**
 * Agrupa predições em `nBins` faixas iguais de [0,1] e compara a média prevista
 * com a frequência empírica. Bem calibrado ⇔ meanPredicted ≈ empiricalRate em cada bin.
 */
export function calibrationBins(samples: ForecastSample[], nBins = 10): CalibrationBin[] {
  const bins: CalibrationBin[] = [];
  for (let b = 0; b < nBins; b++) {
    const lo = b / nBins;
    const hi = (b + 1) / nBins;
    const inBin = samples.filter(s => (b === nBins - 1 ? s.predicted >= lo && s.predicted <= hi : s.predicted >= lo && s.predicted < hi));
    const count = inBin.length;
    bins.push({
      lo,
      hi,
      count,
      meanPredicted: count > 0 ? inBin.reduce((a, s) => a + s.predicted, 0) / count : null,
      empiricalRate: count > 0 ? inBin.reduce((a, s) => a + s.outcome, 0) / count : null,
    });
  }
  return bins;
}

/**
 * Brier do "modelo de referência" que sempre prevê a base rate (frequência média
 * de ocorrência). Se o modelo não bate isto, ele não sabe nada útil.
 */
export function baseRateBrier(samples: ForecastSample[]): number | null {
  if (samples.length === 0) return null;
  const base = samples.reduce((a, s) => a + s.outcome, 0) / samples.length;
  let acc = 0;
  for (const s of samples) acc += (base - s.outcome) ** 2;
  return acc / samples.length;
}

/**
 * Brier Skill Score vs base rate: 1 − brier(modelo)/brier(baseRate).
 * > 0 ⇒ o modelo bate a frequência ingênua; ≤ 0 ⇒ não agrega valor.
 */
export function brierSkillScore(samples: ForecastSample[]): number | null {
  const model = brierScore(samples);
  const base = baseRateBrier(samples);
  if (model === null || base === null || base === 0) return null;
  return 1 - model / base;
}

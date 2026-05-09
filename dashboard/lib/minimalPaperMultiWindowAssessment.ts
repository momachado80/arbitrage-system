/**
 * Multi-Window Assessment — compara janela imediata (segundos, se existir) com janela
 * standard (observedWindow, ~horas). Detecta se o sinal melhora, degrada ou mantém-se
 * entre horizontes temporais sem alterar a colecção de dados existente.
 */

import type {
  MinimalPaperEntry,
  MinimalPaperMarketLite,
  MinimalPaperObservationalOutcomeLabel,
} from "./minimalPaperExecutionProbe";

function r4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

export interface MinimalPaperAdditionalObservedWindow {
  windowLabel: string;
  observedAt: string;
  gapFromEntryMs: number;
  marketsLiteAfter: MinimalPaperMarketLite[];
  maxAbsPriceDeltaAcrossComponents: number | null;
  observationalOutcomeLabel: MinimalPaperObservationalOutcomeLabel;
  outcomeNotes: string[];
}

interface WindowSnapshot {
  label: string;
  gapMs: number;
  spreadDeltaMean: number | null;
  priceMeanDelta: number | null;
  maxAbs: number | null;
  outcomeLabel: string;
}

function snapshotWindow(
  label: string,
  gapMs: number,
  before: MinimalPaperMarketLite[],
  after: MinimalPaperMarketLite[],
  outcomeLabel: string,
): WindowSnapshot {
  const afterMap = new Map(after.map(m => [m.id, m]));
  const sd: number[] = [];
  const pd: number[] = [];
  let maxAbs = 0;
  let any = false;
  for (const b of before) {
    const a = afterMap.get(b.id);
    if (!a) continue;
    sd.push(r4(a.spread - b.spread));
    if (b.prices.length === a.prices.length) {
      for (let i = 0; i < b.prices.length; i++) {
        const d = Math.abs(a.prices[i]! - b.prices[i]!);
        pd.push(r4(d));
        maxAbs = Math.max(maxAbs, d);
        any = true;
      }
    }
  }
  return {
    label,
    gapMs,
    spreadDeltaMean: sd.length > 0 ? r4(sd.reduce((a, b) => a + b, 0) / sd.length) : null,
    priceMeanDelta: pd.length > 0 ? r4(pd.reduce((a, b) => a + b, 0) / pd.length) : null,
    maxAbs: any ? r4(maxAbs) : null,
    outcomeLabel,
  };
}

export type MultiWindowVerdict =
  | "insufficient_multi_window_data"
  | "consistent_neutral_across_windows"
  | "delayed_positive_pattern"
  | "delayed_negative_pattern"
  | "immediate_positive_fading"
  | "unstable_across_windows"
  | "single_window_only";

export interface MultiWindowEpisodeRow {
  entryId: string;
  windows: WindowSnapshot[];
  tiltDirection: "positive" | "negative" | "neutral" | "n/a";
}

export interface MultiWindowAssessmentDigest {
  readDisclaimer: string;
  totalEntriesWithMultiWindow: number;
  totalEntriesSingleWindow: number;
  multiWindowVerdict: MultiWindowVerdict;
  whichWindowShowsMorePositiveSignal: string | null;
  shortVsMediumVsLongSummary: string;
  repeatedNeutralAcrossAllWindows: boolean;
  delayedPositivePattern: boolean;
  unstableAcrossWindows: boolean;
  episodeSample: MultiWindowEpisodeRow[];
  supportingReasons: string[];
  blockingReasons: string[];
}

export function buildMultiWindowAssessment(
  entries: readonly MinimalPaperEntry[],
): MultiWindowAssessmentDigest {
  const closed = entries.filter(e => e.observedWindow);
  const multiRows: MultiWindowEpisodeRow[] = [];
  let totalMulti = 0;
  let totalSingle = 0;

  for (const e of closed) {
    const ow = e.observedWindow!;
    if (
      ow.observationalOutcomeLabel === "insufficient_data" ||
      ow.observationalOutcomeLabel === "component_missing_in_followup"
    ) continue;

    const owGap = e.paperEntryAt ? Date.now() - Date.parse(e.paperEntryAt) : 0;
    const standardWin = snapshotWindow(
      "standard", owGap, e.entrySnapshot.marketsLite, ow.marketsLiteAfter, ow.observationalOutcomeLabel,
    );

    const addWindows: MinimalPaperAdditionalObservedWindow[] =
      (e as unknown as Record<string, unknown>).additionalObservedWindows as MinimalPaperAdditionalObservedWindow[] ?? [];

    const windows: WindowSnapshot[] = [];
    for (const aw of addWindows) {
      windows.push(snapshotWindow(
        aw.windowLabel, aw.gapFromEntryMs, e.entrySnapshot.marketsLite,
        aw.marketsLiteAfter, aw.observationalOutcomeLabel,
      ));
    }
    windows.push(standardWin);
    windows.sort((a, b) => a.gapMs - b.gapMs);

    if (windows.length > 1) totalMulti++;
    else totalSingle++;

    let tiltDirection: "positive" | "negative" | "neutral" | "n/a" = "n/a";
    if (windows.length >= 2) {
      const first = windows[0]!;
      const last = windows[windows.length - 1]!;
      const firstTilt = first.spreadDeltaMean != null ? -(first.spreadDeltaMean) + (first.priceMeanDelta ?? 0) : null;
      const lastTilt = last.spreadDeltaMean != null ? -(last.spreadDeltaMean) + (last.priceMeanDelta ?? 0) : null;
      if (firstTilt != null && lastTilt != null) {
        const delta = lastTilt - firstTilt;
        if (delta > 0.001) tiltDirection = "positive";
        else if (delta < -0.001) tiltDirection = "negative";
        else tiltDirection = "neutral";
      }
    }
    multiRows.push({ entryId: e.id, windows, tiltDirection });
  }

  const supportingReasons: string[] = [];
  const blockingReasons: string[] = [];

  if (totalMulti === 0 && totalSingle === 0) {
    return {
      readDisclaimer: "Comparação entre janelas temporais (imediata vs standard). Sem episódios elegíveis.",
      totalEntriesWithMultiWindow: 0,
      totalEntriesSingleWindow: 0,
      multiWindowVerdict: "insufficient_multi_window_data",
      whichWindowShowsMorePositiveSignal: null,
      shortVsMediumVsLongSummary: "Sem dados.",
      repeatedNeutralAcrossAllWindows: false,
      delayedPositivePattern: false,
      unstableAcrossWindows: false,
      episodeSample: [],
      supportingReasons: [],
      blockingReasons: ["Nenhum episódio fechado elegível."],
    };
  }

  if (totalMulti === 0) {
    return {
      readDisclaimer: "Entradas existentes só têm janela standard. Entradas futuras terão janela imediata para comparação.",
      totalEntriesWithMultiWindow: 0,
      totalEntriesSingleWindow: totalSingle,
      multiWindowVerdict: "single_window_only",
      whichWindowShowsMorePositiveSignal: null,
      shortVsMediumVsLongSummary: `${totalSingle} episódio(s) com janela standard apenas; sem comparação multi-window possível.`,
      repeatedNeutralAcrossAllWindows: false,
      delayedPositivePattern: false,
      unstableAcrossWindows: false,
      episodeSample: multiRows.slice(0, 20),
      supportingReasons: ["Entradas futuras incluirão janela imediata para comparação multi-window."],
      blockingReasons: ["Entradas históricas não têm janela imediata."],
    };
  }

  const multiOnly = multiRows.filter(r => r.windows.length > 1);
  const allNeutral = multiOnly.every(r => r.tiltDirection === "neutral");
  const posCount = multiOnly.filter(r => r.tiltDirection === "positive").length;
  const negCount = multiOnly.filter(r => r.tiltDirection === "negative").length;
  const delayedPos = posCount > negCount && posCount >= 2;
  const unstable = posCount > 0 && negCount > 0 && Math.abs(posCount - negCount) <= 1;

  let bestWindow: string | null = null;
  const windowTiltSums = new Map<string, { sum: number; n: number }>();
  for (const r of multiOnly) {
    for (const w of r.windows) {
      const t = w.spreadDeltaMean != null ? -(w.spreadDeltaMean) + (w.priceMeanDelta ?? 0) : 0;
      const e = windowTiltSums.get(w.label) ?? { sum: 0, n: 0 };
      e.sum += t;
      e.n++;
      windowTiltSums.set(w.label, e);
    }
  }
  let bestAvg = -Infinity;
  for (const [label, { sum, n }] of Array.from(windowTiltSums.entries())) {
    const avg = n > 0 ? sum / n : 0;
    if (avg > bestAvg) { bestAvg = avg; bestWindow = label; }
  }

  let verdict: MultiWindowVerdict;
  if (allNeutral) {
    verdict = "consistent_neutral_across_windows";
    supportingReasons.push("Todas as comparações multi-window neutras entre janelas.");
  } else if (delayedPos) {
    verdict = "delayed_positive_pattern";
    supportingReasons.push(
      `Padrão delayed-positive: ${posCount} episódio(s) com tilt que melhora entre janela imediata e standard.`,
    );
  } else if (negCount > posCount && negCount >= 2) {
    verdict = "delayed_negative_pattern";
    blockingReasons.push(`Tilt negativo entre janelas: ${negCount} vs ${posCount} positivo.`);
  } else if (posCount > 0 && !delayedPos && multiOnly.some(r => {
    const immW = r.windows.find(w => w.label === "immediate");
    const stdW = r.windows.find(w => w.label === "standard");
    if (!immW || !stdW) return false;
    const immT = immW.spreadDeltaMean != null ? -(immW.spreadDeltaMean) : 0;
    const stdT = stdW.spreadDeltaMean != null ? -(stdW.spreadDeltaMean) : 0;
    return immT > stdT + 0.001;
  })) {
    verdict = "immediate_positive_fading";
    supportingReasons.push("Sinal positivo imediato que se dissipa na janela standard.");
  } else if (unstable) {
    verdict = "unstable_across_windows";
    blockingReasons.push("Sinal instável: tilt oscila entre janelas.");
  } else {
    verdict = "consistent_neutral_across_windows";
    supportingReasons.push("Sem padrão claro entre janelas.");
  }

  const labels = Array.from(windowTiltSums.keys()).sort();
  const summaryParts = labels.map(l => {
    const e = windowTiltSums.get(l)!;
    return `${l}:avgTilt=${e.n > 0 ? r4(e.sum / e.n) : 0}(n=${e.n})`;
  });

  return {
    readDisclaimer: "Comparação entre janelas temporais. delayed_positive ≠ edge; immediate_positive_fading sugere ruído de curto prazo.",
    totalEntriesWithMultiWindow: totalMulti,
    totalEntriesSingleWindow: totalSingle,
    multiWindowVerdict: verdict,
    whichWindowShowsMorePositiveSignal: bestWindow,
    shortVsMediumVsLongSummary: summaryParts.join(" | ") || "Sem dados multi-window.",
    repeatedNeutralAcrossAllWindows: allNeutral && totalMulti > 0,
    delayedPositivePattern: delayedPos,
    unstableAcrossWindows: unstable,
    episodeSample: multiRows.slice(0, 20),
    supportingReasons,
    blockingReasons,
  };
}

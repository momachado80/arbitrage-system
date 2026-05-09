/**
 * Agregados por dia UTC para oportunidades no ciclo paper (memória; reinício zera).
 */

export interface PaperDayOpportunityMetrics {
  opportunitiesSeen: number;
  opportunitiesExecutable: number;
  cycles: number;
}

const buckets = new Map<string, PaperDayOpportunityMetrics>();

function utcDayKey(ts: number): string {
  return new Date(ts).toISOString().slice(0, 10);
}

export function recordPaperCycleOpportunityMetrics(
  opportunitiesSeen: number,
  opportunitiesExecutable: number,
  ts: number = Date.now()
): void {
  const key = utcDayKey(ts);
  const prev = buckets.get(key) ?? {
    opportunitiesSeen: 0,
    opportunitiesExecutable: 0,
    cycles: 0,
  };
  buckets.set(key, {
    opportunitiesSeen: prev.opportunitiesSeen + opportunitiesSeen,
    opportunitiesExecutable: prev.opportunitiesExecutable + opportunitiesExecutable,
    cycles: prev.cycles + 1,
  });
}

export function getPaperOpportunityMetricsToday(
  ts: number = Date.now()
): PaperDayOpportunityMetrics | null {
  return buckets.get(utcDayKey(ts)) ?? null;
}

export function getPaperOpportunityMetricsByDay(): Record<string, PaperDayOpportunityMetrics> {
  return Object.fromEntries(buckets.entries());
}

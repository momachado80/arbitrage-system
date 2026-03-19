/**
 * Shadow Runtime Consistency Diagnostics — writer/reader separation e mutações.
 * Rastreia se economia, heartbeat e funnel mutam no mesmo processo.
 */

const processStartMs = Date.now();

const economicMutationByProfile = new Map<string, { count: number; lastAt: string }>();
let totalEconomicMutations = 0;
let lastEconomicMutationAt: string | null = null;

const heartbeatMutationByProfile = new Map<string, string>();
let totalHeartbeatMutations = 0;
let lastHeartbeatMutationAt: string | null = null;

const funnelMutationByProfile = new Map<string, { count: number; lastAt: string }>();
let totalFunnelMutations = 0;
let lastFunnelMutationAt: string | null = null;

export function recordEconomicMutation(profileId: string): void {
  totalEconomicMutations++;
  lastEconomicMutationAt = new Date().toISOString();
  const prev = economicMutationByProfile.get(profileId);
  economicMutationByProfile.set(profileId, {
    count: (prev?.count ?? 0) + 1,
    lastAt: lastEconomicMutationAt,
  });
}

export function recordHeartbeatMutation(profileId: string): void {
  totalHeartbeatMutations++;
  lastHeartbeatMutationAt = new Date().toISOString();
  heartbeatMutationByProfile.set(profileId, lastHeartbeatMutationAt);
}

export function recordFunnelMutation(profileId: string): void {
  totalFunnelMutations++;
  lastFunnelMutationAt = new Date().toISOString();
  const prev = funnelMutationByProfile.get(profileId);
  funnelMutationByProfile.set(profileId, {
    count: (prev?.count ?? 0) + 1,
    lastAt: lastFunnelMutationAt,
  });
}

export interface ShadowRuntimeConsistencyDebug {
  runtimeInstanceId: string;
  processUptimeMs: number;
  auditServedAt: string;
  lastEconomicMutationAt: string | null;
  lastHeartbeatMutationAt: string | null;
  lastFunnelMutationAt: string | null;
  economicMutationsCount: number;
  heartbeatMutationsCount: number;
  funnelMutationsCount: number;
  writerReaderConsistencyStatus: "CONSISTENT" | "ECONOMICS_WITHOUT_HEARTBEAT" | "ECONOMICS_WITHOUT_FUNNEL" | "READER_ONLY" | "INCONCLUSIVE";
  summary: string;
  byProfile: Record<
    string,
    {
      lastCloseMutationAt: string | null;
      lastHeartbeatAt: string | null;
      lastFunnelCounterMutationAt: string | null;
      economicMutationsCount: number;
      funnelMutationsCount: number;
      economicHistoryChangedSinceBoot: boolean;
      funnelChangedSinceBoot: boolean;
    }
  >;
}

function deriveConsistencyStatus(
  economicCount: number,
  heartbeatCount: number,
  funnelCount: number,
  inMemoryClosedCount: number
): ShadowRuntimeConsistencyDebug["writerReaderConsistencyStatus"] {
  if (economicCount > 0 && heartbeatCount === 0) {
    return "ECONOMICS_WITHOUT_HEARTBEAT";
  }
  if (economicCount > 0 && funnelCount === 0) {
    return "ECONOMICS_WITHOUT_FUNNEL";
  }
  if (inMemoryClosedCount > 0 && economicCount === 0) {
    return "READER_ONLY";
  }
  if (economicCount > 0 && heartbeatCount > 0 && funnelCount > 0) {
    return "CONSISTENT";
  }
  return "INCONCLUSIVE";
}

export function getShadowRuntimeConsistencyDebug(params: {
  profiles: Array<{
    profileId: string;
    closedTrades: Array<{ status: string; closedAt: string | null }>;
    lastCycleProcessedAt: string | null;
  }>;
}): ShadowRuntimeConsistencyDebug {
  const inMemoryClosedCount = params.profiles.reduce(
    (s, p) => s + p.closedTrades.filter((t) => t.status === "closed" && t.closedAt).length,
    0
  );

  const status = deriveConsistencyStatus(
    totalEconomicMutations,
    totalHeartbeatMutations,
    totalFunnelMutations,
    inMemoryClosedCount
  );

  let summary: string;
  switch (status) {
    case "ECONOMICS_WITHOUT_HEARTBEAT":
      summary =
        "Closes observados neste processo mas heartbeat nunca mutou. Indica close path sem updateProfileHeartbeat ou writer/reader em processos distintos.";
      break;
    case "ECONOMICS_WITHOUT_FUNNEL":
      summary =
        "Closes observados mas funnel nunca mutou. Indica close fora de runCycle ou record* não alcançados.";
      break;
    case "READER_ONLY":
      summary =
        "Histórico econômico em memória mas nenhum close neste processo. Leitura de reidratação ou instância diferente do writer.";
      break;
    case "CONSISTENT":
      summary = "Economia, heartbeat e funnel mutaram neste processo. Writer e reader coerentes.";
      break;
    default:
      summary = "Dados insuficientes. Aguardar ciclos ou verificar multi-instância.";
  }

  const allProfileIds = new Set([
    ...economicMutationByProfile.keys(),
    ...heartbeatMutationByProfile.keys(),
    ...funnelMutationByProfile.keys(),
    ...params.profiles.map((p) => p.profileId),
  ]);

  const byProfile: ShadowRuntimeConsistencyDebug["byProfile"] = {};
  for (const pid of Array.from(allProfileIds)) {
    const profile = params.profiles.find((p) => p.profileId === pid);
    const closedCount = profile
      ? profile.closedTrades.filter((t) => t.status === "closed" && t.closedAt).length
      : 0;
    const economic = economicMutationByProfile.get(pid);
    const heartbeat = heartbeatMutationByProfile.get(pid);
    const funnel = funnelMutationByProfile.get(pid);

    byProfile[pid] = {
      lastCloseMutationAt: economic?.lastAt ?? null,
      lastHeartbeatAt: heartbeat ?? null,
      lastFunnelCounterMutationAt: funnel?.lastAt ?? null,
      economicMutationsCount: economic?.count ?? 0,
      funnelMutationsCount: funnel?.count ?? 0,
      economicHistoryChangedSinceBoot: closedCount > 0,
      funnelChangedSinceBoot: (funnel?.count ?? 0) > 0,
    };
  }

  return {
    runtimeInstanceId: `node-${process.pid}-${Date.now().toString(36)}`,
    processUptimeMs: Date.now() - processStartMs,
    auditServedAt: new Date().toISOString(),
    lastEconomicMutationAt,
    lastHeartbeatMutationAt,
    lastFunnelMutationAt,
    economicMutationsCount: totalEconomicMutations,
    heartbeatMutationsCount: totalHeartbeatMutations,
    funnelMutationsCount: totalFunnelMutations,
    writerReaderConsistencyStatus: status,
    summary,
    byProfile,
  };
}

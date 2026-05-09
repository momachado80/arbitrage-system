import { NextResponse } from "next/server";
import { ensureCatalogPocketProbe, getCatalogPocketHealth } from "@/lib/catalogPocketProbe";
import {
  ensurePocketEconomicsProbe,
  getPocketEconomicsHealth,
} from "@/lib/pocketEconomicsProbe";
import {
  ensurePocketExecutionProbe,
  getPocketExecutionHealth,
} from "@/lib/pocketExecutionProbe";
import {
  ensureMinimalPaperExecutionProbe,
  getMinimalPaperExecutionHealth,
} from "@/lib/minimalPaperExecutionProbe";
import { buildPaperPersistedStateHygieneHealth } from "@/lib/paperProbePersistedState";

export const dynamic = "force-dynamic";

/**
 * Health check — schedulers idempotentes; sem I/O pesado por defeito.
 * Higiene de ficheiros .paper (3× read+parse) só com ?hygiene=1 ou HEALTHZ_INCLUDE_HYGIENE=1
 * para evitar timeouts em proxies (ex.: Railway 502 no primeiro request).
 */
export async function GET(request: Request) {
  const t0 = Date.now();
  const url = new URL(request.url);
  const includeHygiene =
    process.env.HEALTHZ_INCLUDE_HYGIENE === "1" || url.searchParams.get("hygiene") === "1";

  ensureCatalogPocketProbe();
  ensurePocketEconomicsProbe();
  ensurePocketExecutionProbe();
  ensureMinimalPaperExecutionProbe();
  const tAfterEnsure = Date.now();

  const uptimeSec = process.uptime();
  const processStartMs = Date.now() - uptimeSec * 1000;
  const pocket = getCatalogPocketHealth();
  const pocketEconomics = getPocketEconomicsHealth();
  const pocketExecution = getPocketExecutionHealth();
  const minimalPaperExecution = getMinimalPaperExecutionHealth();

  let paperPersistedHygiene: ReturnType<typeof buildPaperPersistedStateHygieneHealth> | null = null;
  if (includeHygiene) {
    paperPersistedHygiene = buildPaperPersistedStateHygieneHealth();
  }

  const body: Record<string, unknown> = {
    ok: true,
    uptime: Math.round(uptimeSec * 1000) / 1000,
    processStartTime: new Date(processStartMs).toISOString(),
    healthzHygieneIncluded: includeHygiene,
    ...pocket,
    ...pocketEconomics,
    ...pocketExecution,
    ...minimalPaperExecution,
  };
  if (paperPersistedHygiene) {
    Object.assign(body, paperPersistedHygiene);
  }

  const tEnd = Date.now();
  console.log(
    `[healthz] totalMs=${tEnd - t0} ensuresMs=${tAfterEnsure - t0} hygieneMs=${includeHygiene ? tEnd - tAfterEnsure : 0}`,
  );

  return NextResponse.json(body);
}

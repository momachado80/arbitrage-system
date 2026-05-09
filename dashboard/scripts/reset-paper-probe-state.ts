/**
 * Remove apenas ficheiros de estado persistidos em .paper (ou paths via env) para os probes
 * pocket-economics, pocket-execution, minimal-paper-execution. Exige --confirm; nunca corre automaticamente.
 *
 * Uso: npm run reset-probe-state -- --probe=minimal-paper-execution --confirm
 *      npm run reset-probe-state -- --probe=all --confirm
 */

import fs from "fs";
import {
  isPaperPersistedProbeId,
  PAPER_PERSISTED_PROBE_IDS,
  resolvePaperPersistedStatePath,
  type PaperPersistedProbeId,
} from "../lib/paperProbePersistedState";

function parseArgs(argv: string[]): { confirm: boolean; probes: PaperPersistedProbeId[] } {
  const confirm = argv.includes("--confirm");
  const probeVals: string[] = [];
  for (const a of argv) {
    if (a.startsWith("--probe=")) probeVals.push(a.slice("--probe=".length).trim());
  }
  if (probeVals.length === 0) {
    return { confirm: false, probes: [] };
  }
  if (probeVals.includes("all")) {
    if (probeVals.length !== 1) {
      throw new Error("With --probe=all, pass exactly one --probe=all (no other --probe).");
    }
    return { confirm, probes: [...PAPER_PERSISTED_PROBE_IDS] };
  }
  const probes: PaperPersistedProbeId[] = [];
  for (const p of probeVals) {
    if (!isPaperPersistedProbeId(p)) {
      throw new Error(`Unknown probe "${p}". Expected one of: ${PAPER_PERSISTED_PROBE_IDS.join(", ")}, all`);
    }
    probes.push(p);
  }
  const seen = new Set<PaperPersistedProbeId>();
  const uniq: PaperPersistedProbeId[] = [];
  for (const p of probes) {
    if (seen.has(p)) continue;
    seen.add(p);
    uniq.push(p);
  }
  return { confirm, probes: uniq };
}

function main(): void {
  let parsed: { confirm: boolean; probes: PaperPersistedProbeId[] };
  try {
    parsed = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error(e instanceof Error ? e.message : e);
    process.exit(1);
    return;
  }

  if (!parsed.confirm || parsed.probes.length === 0) {
    console.error(
      "Uso: npm run reset-probe-state -- --probe=<pocket-economics|pocket-execution|minimal-paper-execution|all> --confirm",
    );
    console.error("Só remove ficheiros de estado persistidos (.paper ou POCKET_* / MINIMAL_PAPER_* paths).");
    console.error("Reinicia o servidor depois para limpar estado em memória.");
    process.exit(1);
    return;
  }

  const cwd = process.cwd();
  for (const probe of parsed.probes) {
    const fp = resolvePaperPersistedStatePath(probe, cwd);
    if (fs.existsSync(fp)) {
      fs.unlinkSync(fp);
      console.log(`[reset-probe-state] removed: ${fp}`);
    } else {
      console.log(`[reset-probe-state] skip (missing): ${fp}`);
    }
  }
  console.log("[reset-probe-state] done. Restart the dashboard process before relying on clean probes.");
}

main();

#!/usr/bin/env node
/**
 * Snapshot operacional do /api/probe/system-ladder (uso diário no terminal).
 *
 * PORT: LADDER_STATUS_PORT (default 3091)
 * URL completa: LADDER_STATUS_URL (sobrepõe host/porta), ex. http://127.0.0.1:3091
 * Sem linhas por camada: LADDER_STATUS_LAYERS=0 ou flag --no-layers
 */

const showLayers = !process.argv.includes("--no-layers") && process.env.LADDER_STATUS_LAYERS !== "0";

function baseUrl() {
  const u = process.env.LADDER_STATUS_URL?.trim();
  if (u) return u.replace(/\/$/, "");
  const port = Number(process.env.LADDER_STATUS_PORT?.trim()) || 3091;
  return `http://127.0.0.1:${port}`;
}

function layerLine(name, L) {
  if (!L || typeof L !== "object") return `${name}: (ausente)`;
  const st = L.scanStatus ?? "?";
  const pv = L.primaryVerdict ?? "?";
  let extra = "";
  if (name === "pocket-execution" && L.executionPromotionVerdict != null) {
    extra = ` | execPromo=${L.executionPromotionVerdict}`;
  }
  if (name === "minimal-paper-execution" && L.gateOverallExecutionPromotionVerdict != null) {
    extra = ` | gate=${L.gateOverallExecutionPromotionVerdict}`;
  }
  return `${name}: status=${st} | primary=${pv}${extra}`;
}

async function main() {
  const base = baseUrl();
  const url = `${base}/api/probe/system-ladder`;
  let res;
  try {
    res = await fetch(url, { headers: { Accept: "application/json" } });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`[ladder-status] Não foi possível contactar o servidor (${url}).`);
    console.error(`[ladder-status] ${msg}`);
    console.error("[ladder-status] Confirma que o dashboard está a correr (ex.: PORT=3091 npm run start).");
    process.exit(1);
    return;
  }

  const text = await res.text();
  if (!res.ok) {
    console.error(`[ladder-status] HTTP ${res.status} em ${url}`);
    console.error(text.slice(0, 800));
    process.exit(1);
    return;
  }

  let d;
  try {
    d = JSON.parse(text);
  } catch {
    console.error("[ladder-status] Resposta não é JSON válido.");
    console.error(text.slice(0, 400));
    process.exit(1);
    return;
  }

  const es = d.executiveSummary ?? {};
  const tc = d.temporalConsistency ?? {};
  const tr = d.ladderTrajectoryAssessment ?? {};
  const layers = d.layers ?? {};

  console.log(`=== system-ladder · ${d.computedAt ?? "?"} ===`);
  console.log(`URL: ${url}`);
  console.log("");
  console.log(`headline: ${es.headline ?? "(n/d)"}`);
  console.log(`currentStage: ${es.currentStage ?? "(n/d)"}`);
  console.log(`mainConstraint: ${es.mainConstraint ?? "(n/d)"}`);
  console.log(`nextBestAction: ${es.nextBestAction ?? "(n/d)"}`);
  console.log(`confidenceNote: ${es.confidenceNote ?? "(n/d)"}`);
  console.log("");
  console.log(`ladderTrajectoryAssessment: ${tr.verdict ?? "(n/d)"}`);
  if (Array.isArray(tr.notes) && tr.notes.length > 0) {
    console.log(`  notes: ${tr.notes.join(" | ")}`);
  }
  console.log(`temporalConsistency: ${tc.overallTemporalConsistencyVerdict ?? "(n/d)"}`);

  if (showLayers) {
    console.log("");
    console.log("--- layers ---");
    console.log(layerLine("catalog-pocket", layers.catalogPocket));
    console.log(layerLine("pocket-economics", layers.pocketEconomics));
    console.log(layerLine("pocket-execution", layers.pocketExecution));
    console.log(layerLine("minimal-paper-execution", layers.minimalPaperExecution));
  }
}

main();

/**
 * Microcapital Readiness Gate — CLI script.
 *
 * Usage:
 *   PAPER_STATE_DIR=$PWD/.paper npm run readiness:microcapital
 *
 * Prints a JSON dossier and an executive summary. Exits:
 *   0  → APPROVED_FOR_SHADOW_READINESS or APPROVED_WITH_REMARKS
 *   1  → any BLOCKED_* verdict
 *
 * Strict policy: shadow_only_no_api_submit_v1.
 * NEVER submits orders, NEVER signs, NEVER touches wallet/signer/private key.
 */

import { composeMicrocapitalReadinessDossier } from "../../lib/readiness/composeMicrocapitalReadinessDossier";
import { isBlockedVerdict } from "../../lib/readiness/microcapitalReadinessDossier";
import { renderMicrocapitalReadinessExecutiveSummary } from "../../lib/readiness/renderMicrocapitalReadinessExecutiveSummary";

function main(): void {
  const dossier = composeMicrocapitalReadinessDossier();
  const summary = renderMicrocapitalReadinessExecutiveSummary(dossier);

  process.stdout.write(JSON.stringify(dossier, null, 2));
  process.stdout.write("\n");
  process.stderr.write(summary);
  process.stderr.write("\n");

  if (isBlockedVerdict(dossier.finalVerdict)) {
    process.exit(1);
  }
  process.exit(0);
}

main();

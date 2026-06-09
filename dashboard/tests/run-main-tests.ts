import { getRegistry } from "./_assert";

import "./gamma1823789MarketsResponseNormalize.test";
import "./marketSuitabilityGate.test";
import "./liveMarketDiscoveryRanking.test";
import "./preflightSingleTrack553856.test";
import "./genericSingleTrackReadOnlyMarkout.test";
import "./singleTrackMarkoutSampler.test";
import "./observationWindowSuitability.test";
import "./marketUniverseQuality.test";
import "./catalystObservationSchedule.test";
import "./catalystSchedulePicker.test";
import "./catalystPlanTimeMarketGate.test";
import "./mechanicalEdgeCensus.test";
import "./mechanicalEdgeCensusBook.test";
import "./resolutionConvergence.test";
import "./poissonGoalsModel.test";
import "./soccerEloModel.test";
import "./forecastCalibration.test";
import "./rollingElo.test";
import "./mecPersistence.test";
import "./catalystObservationOrchestrator.test";
import "./shadowClosedTradePersistence.test";
import "./universeQualityGate.test";
import "./postEventReversionHypothesis.test";
import "./postEventReversionPlanReader.test";

async function main(): Promise<void> {
  const registry = getRegistry();
  let total = 0;
  let failed = 0;
  const failures: Array<{ file: string; name: string; err: unknown }> = [];

  for (const [file, cases] of registry.entries()) {
    process.stdout.write(`\n=== ${file} ===\n`);
    for (const c of cases) {
      total++;
      try {
        await c.fn();
        process.stdout.write(`  PASS  ${c.name}\n`);
      } catch (err) {
        failed++;
        failures.push({ file, name: c.name, err });
        process.stdout.write(`  FAIL  ${c.name}\n`);
        if (err instanceof Error) process.stdout.write(`        ${err.message}\n`);
      }
    }
  }

  process.stdout.write(`\nTotal: ${total}, Passed: ${total - failed}, Failed: ${failed}\n`);
  if (failed > 0) {
    process.exit(1);
  }
  process.exit(0);
}

main().catch(err => {
  console.error("[main tests] fatal:", err);
  process.exit(1);
});

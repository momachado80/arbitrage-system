/**
 * Microcapital Readiness Gate — test runner.
 *
 * Imports each .test.ts file (which registers cases via tests/readiness/_assert.ts),
 * then executes every case sequentially. Exits 1 on any failure.
 *
 * Paper/shadow only. Policy: shadow_only_no_api_submit_v1.
 */

import { getRegistry } from "./_assert";

import "./microcapitalReadinessDossier.test";
import "./executionRealismHarness.test";
import "./microcapitalFailureModes.test";
import "./prohibitedExecutionTerms.test";

async function main(): Promise<void> {
  const registry = getRegistry();
  let total = 0;
  let failed = 0;
  const failures: Array<{ file: string; name: string; err: unknown }> = [];

  const entries = Array.from(registry.entries());
  for (const [file, cases] of entries) {
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
        if (err instanceof Error) {
          process.stdout.write(`        ${err.message}\n`);
        }
      }
    }
  }

  process.stdout.write(`\nTotal: ${total}, Passed: ${total - failed}, Failed: ${failed}\n`);
  if (failed > 0) {
    process.stdout.write("\nFailures:\n");
    for (const f of failures) {
      process.stdout.write(`  - [${f.file}] ${f.name}\n`);
      if (f.err instanceof Error && f.err.stack) {
        process.stdout.write(`    ${f.err.stack.split("\n").slice(0, 3).join("\n    ")}\n`);
      }
    }
    process.exit(1);
  }
  process.exit(0);
}

main().catch(err => {
  console.error("[readiness tests] runner error:", err);
  process.exit(1);
});

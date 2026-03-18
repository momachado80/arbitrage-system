#!/usr/bin/env node
/**
 * Full check: runs preflight, deploy-check, snapshot, brief in sequence.
 *
 * Usage: npm run shadow:full-check  (from dashboard/)
 */

import { execSync } from "child_process";
import * as path from "path";

const SCRIPTS = [
  { name: "preflight", cmd: "shadow:preflight" },
  { name: "deploy-check", cmd: "shadow:deploy-check" },
  { name: "snapshot", cmd: "shadow:snapshot" },
  { name: "brief", cmd: "shadow:brief" },
  { name: "judge", cmd: "shadow:judge" },
] as const;

function run(): void {
  const cwd = process.cwd();
  const isDashboard = path.basename(cwd) === "dashboard";
  const workDir = isDashboard ? cwd : path.join(cwd, "dashboard");

  process.chdir(workDir);

  console.log("\n=== SHADOW FULL CHECK ===\n");

  for (const { name, cmd } of SCRIPTS) {
    console.log(`--- ${name} ---`);
    try {
      execSync(`npm run ${cmd}`, {
        stdio: "inherit",
        cwd: workDir,
      });
    } catch (e) {
      const code = (e as { status?: number }).status ?? 1;
      process.stderr.write(`\nFull check failed at step: ${name}\n`);
      process.exit(code);
    }
    console.log("");
  }

  console.log("=== FULL CHECK COMPLETE ===\n");
}

run();

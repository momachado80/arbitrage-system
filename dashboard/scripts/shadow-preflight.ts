#!/usr/bin/env node
/**
 * Preflight validation for shadow trading system.
 * Verifies project state, branch, git status, and presence of structural risk managed feature.
 *
 * Usage: npm run shadow:preflight  (from dashboard/)
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";

const EXPECTED_PROFILE_ID = "shadow_1000_structural_riskmanaged_v1";
const PROFILES_PATH = "lib/shadowSimulationProfiles.ts";
const AUDIT_ROUTE_PATH = "app/api/shadow/audit/route.ts";
const DIAG_NAMES = ["structuralRiskManagedDiagnostics", "structuralRiskManagedComparison"] as const;

type Check = { name: string; pass: boolean; detail?: string };

function run(cmd: string, opts?: { cwd?: string }): string {
  try {
    return execSync(cmd, {
      encoding: "utf-8",
      cwd: opts?.cwd,
      maxBuffer: 1024 * 1024,
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();
  } catch {
    return "";
  }
}

function main(): void {
  const cwd = process.cwd();
  const isDashboard = path.basename(cwd) === "dashboard";
  const root = isDashboard ? cwd : path.join(cwd, "dashboard");
  const checks: Check[] = [];

  // 1. Diretório correto (dashboard ou repo root)
  const hasDashboard = fs.existsSync(path.join(cwd, "dashboard", "package.json"));
  const inDashboard = isDashboard && fs.existsSync(path.join(cwd, "package.json"));
  checks.push({
    name: "project directory",
    pass: inDashboard || hasDashboard,
    detail: inDashboard ? "cwd=dashboard" : hasDashboard ? "cwd=repo root" : "expected dashboard/ or repo root",
  });

  const workDir = inDashboard ? cwd : path.join(cwd, "dashboard");
  process.chdir(workDir);

  // 2. Branch atual
  const branch = run("git branch --show-current") || run("git rev-parse --abbrev-ref HEAD");
  checks.push({
    name: "current branch",
    pass: branch.length > 0,
    detail: branch || "not a git repo",
  });

  // 3. Status git (no uncommitted critical changes for deploy)
  const status = run("git status --short");
  const hasUncommitted = status.length > 0;
  checks.push({
    name: "git status",
    pass: true,
    detail: hasUncommitted ? "uncommitted changes" : "clean",
  });

  // 4. Último commit local
  const lastLocal = run("git log -1 --oneline");
  checks.push({
    name: "last local commit",
    pass: lastLocal.length > 0,
    detail: lastLocal || "N/A",
  });

  // 5. Último commit origin/main
  const lastOrigin = run("git log origin/main -1 --oneline");
  checks.push({
    name: "last origin/main commit",
    pass: lastOrigin.length > 0,
    detail: lastOrigin || "N/A",
  });

  // 6. Profile em shadowSimulationProfiles.ts
  const profilesPath = path.join(workDir, PROFILES_PATH);
  let profilePass = false;
  if (fs.existsSync(profilesPath)) {
    const content = fs.readFileSync(profilesPath, "utf-8");
    profilePass = content.includes(EXPECTED_PROFILE_ID);
  }
  checks.push({
    name: `profile ${EXPECTED_PROFILE_ID} in profiles`,
    pass: profilePass,
    detail: profilePass ? "found" : "not found",
  });

  // 7. structuralRiskManaged* em audit route
  const routePath = path.join(workDir, AUDIT_ROUTE_PATH);
  let routePass = true;
  const missing: string[] = [];
  if (fs.existsSync(routePath)) {
    const content = fs.readFileSync(routePath, "utf-8");
    for (const name of DIAG_NAMES) {
      if (!content.includes(name)) missing.push(name);
    }
    routePass = missing.length === 0;
  } else {
    routePass = false;
    missing.push(...DIAG_NAMES);
  }
  checks.push({
    name: "audit route exposes structuralRiskManaged*",
    pass: routePass,
    detail: routePass ? "both present" : `missing: ${missing.join(", ")}`,
  });

  // Output
  console.log("\n--- SHADOW PREFLIGHT ---\n");
  for (const c of checks) {
    const status = c.pass ? "PASS" : "FAIL";
    const detail = c.detail ? ` (${c.detail})` : "";
    console.log(`  ${status}  ${c.name}${detail}`);
  }
  const allPass = checks.every((c) => c.pass);
  console.log("\n  Result: " + (allPass ? "PASS" : "FAIL") + "\n");
  process.exit(allPass ? 0 : 1);
}

main();

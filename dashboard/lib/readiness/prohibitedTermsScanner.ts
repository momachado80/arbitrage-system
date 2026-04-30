/**
 * Microcapital Readiness Gate — prohibited terms scanner.
 *
 * Walks lib/**, scripts/**, app/api/**, workers/** (if present) and reports any
 * appearance of forbidden execution-related terms outside the explicit whitelist.
 *
 * Read-only. No side effects. No execution.
 */

import fs from "fs";
import path from "path";

import {
  PROHIBITED_EXECUTION_TERMS,
  PROHIBITED_TERMS_WHITELIST,
} from "./microcapitalReadinessThresholds";

export interface ProhibitedTermFinding {
  file: string;
  line: number;
  text: string;
  matchedTerm: string;
}

export interface ProhibitedTermsScanResult {
  scannedFileCount: number;
  findings: ProhibitedTermFinding[];
  passed: boolean;
}

const ROOT_DIRS = ["lib", "scripts", "app/api", "workers"] as const;

function isWhitelisted(absoluteFile: string): boolean {
  const normalized = absoluteFile.replace(/\\/g, "/");
  return PROHIBITED_TERMS_WHITELIST.some(suffix =>
    normalized.endsWith(suffix.replace(/\\/g, "/")),
  );
}

function walkTs(dir: string, acc: string[]): void {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name === "node_modules" || e.name === ".next") continue;
      walkTs(full, acc);
    } else if (e.isFile()) {
      if (e.name.endsWith(".ts") || e.name.endsWith(".tsx")) acc.push(full);
    }
  }
}

export function scanProhibitedTerms(projectRoot: string): ProhibitedTermsScanResult {
  const files: string[] = [];
  for (const rel of ROOT_DIRS) {
    const abs = path.join(projectRoot, rel);
    if (fs.existsSync(abs)) walkTs(abs, files);
  }

  const findings: ProhibitedTermFinding[] = [];
  for (const file of files) {
    if (isWhitelisted(file)) continue;
    let content: string;
    try {
      content = fs.readFileSync(file, "utf8");
    } catch {
      continue;
    }
    const lines = content.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      for (const term of PROHIBITED_EXECUTION_TERMS) {
        if (line.includes(term)) {
          findings.push({
            file,
            line: i + 1,
            text: line.trim(),
            matchedTerm: term,
          });
        }
      }
    }
  }

  return {
    scannedFileCount: files.length,
    findings,
    passed: findings.length === 0,
  };
}

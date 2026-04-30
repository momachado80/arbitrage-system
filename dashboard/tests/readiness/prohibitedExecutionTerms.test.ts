/**
 * Tests — prohibited execution terms scanner.
 *
 * Verifies that the scanner walks lib/**, scripts/**, app/api/** (and workers/**
 * if present), respects the whitelist, and reports findings with file + line.
 *
 * Paper/shadow only. Policy: shadow_only_no_api_submit_v1.
 */

import fs from "fs";
import os from "os";
import path from "path";

import { scanProhibitedTerms } from "../../lib/readiness/prohibitedTermsScanner";

import { assertEqual, assertIncludes, assertTrue, describe, test } from "./_assert";

function makeTempProject(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "readiness-scan-"));
  fs.mkdirSync(path.join(root, "lib"), { recursive: true });
  fs.mkdirSync(path.join(root, "scripts"), { recursive: true });
  fs.mkdirSync(path.join(root, "app", "api"), { recursive: true });
  return root;
}

describe("tests/readiness/prohibitedExecutionTerms.test.ts", () => {
  test("scanner passes on a clean project", () => {
    const root = makeTempProject();
    fs.writeFileSync(
      path.join(root, "lib", "clean.ts"),
      "export const x = 1;\nexport function audit(): void {}\n",
    );
    const result = scanProhibitedTerms(root);
    assertEqual(result.passed, true, "clean project passes");
    assertEqual(result.findings.length, 0, "no findings");
  });

  test("scanner detects 'wallet' in lib/", () => {
    const root = makeTempProject();
    fs.writeFileSync(
      path.join(root, "lib", "bad.ts"),
      "export const wallet = 1;\n",
    );
    const r = scanProhibitedTerms(root);
    assertEqual(r.passed, false, "should fail");
    assertTrue(r.findings.length >= 1, "at least one finding");
    const f = r.findings[0];
    assertEqual(f.matchedTerm, "wallet", "matched term reported");
    assertIncludes(f.file, "bad.ts", "file path reported");
    assertEqual(f.line, 1, "line number reported");
  });

  test("scanner detects 'submitOrder' inside scripts/", () => {
    const root = makeTempProject();
    fs.writeFileSync(
      path.join(root, "scripts", "x.ts"),
      "// fake\nfunction submitOrder() {}\n",
    );
    const r = scanProhibitedTerms(root);
    assertEqual(r.passed, false, "must fail on submitOrder");
    assertTrue(
      r.findings.some(f => f.matchedTerm === "submitOrder"),
      "matched submitOrder",
    );
  });

  test("scanner detects 'apiSubmitAllowed: true' inside app/api/", () => {
    const root = makeTempProject();
    const dir = path.join(root, "app", "api", "thing");
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(
      path.join(dir, "route.ts"),
      "export const cfg = { apiSubmitAllowed: true };\n",
    );
    const r = scanProhibitedTerms(root);
    assertTrue(
      r.findings.some(f => f.matchedTerm === "apiSubmitAllowed: true"),
      "apiSubmitAllowed:true detected",
    );
  });

  test("scanner respects whitelisted files", () => {
    const root = makeTempProject();
    fs.mkdirSync(path.join(root, "scripts"), { recursive: true });
    // Whitelisted file (existing core scanner)
    fs.writeFileSync(
      path.join(root, "scripts", "assertCrossVenueAnchor1823789ShadowOnly.ts"),
      "// terms allowed here: wallet, signer, privateKey\n",
    );
    const r = scanProhibitedTerms(root);
    assertEqual(r.passed, true, "whitelisted file does not produce findings");
  });

  test("scanner walks workers/ if present", () => {
    const root = makeTempProject();
    fs.mkdirSync(path.join(root, "workers"), { recursive: true });
    fs.writeFileSync(
      path.join(root, "workers", "w.ts"),
      "const signer = null;\n",
    );
    const r = scanProhibitedTerms(root);
    assertTrue(
      r.findings.some(f => f.matchedTerm === "signer"),
      "workers/ scanned",
    );
  });

  test("scan against the actual dashboard root passes (whitelist consistent)", () => {
    const dashboardRoot = path.resolve(__dirname, "..", "..");
    const r = scanProhibitedTerms(dashboardRoot);
    if (!r.passed) {
      const sample = r.findings
        .slice(0, 5)
        .map(f => `${f.file}:${f.line} [${f.matchedTerm}] ${f.text}`)
        .join("\n  ");
      throw new Error(
        `dashboard scan should pass — got ${r.findings.length} findings:\n  ${sample}`,
      );
    }
    assertEqual(r.passed, true, "dashboard scan must pass");
  });
});

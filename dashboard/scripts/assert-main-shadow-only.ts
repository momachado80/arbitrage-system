/**
 * Auditoria shadow-only minimal para o pacote dashboard (paper/shadow/read-only).
 * Varredura estática sobre lib/, scripts/, app/. Não execução, não redes.
 */

import fs from "fs";
import path from "path";

const AUDIT_SCRIPT_BASENAME = "assert-main-shadow-only.ts";
/** Lista termos bloqueados como strings; ignorar para evitar falsos positivos no scan amplo. */
const SKIP_AUDIT_ENUM_BASENAMES = new Set([
  AUDIT_SCRIPT_BASENAME,
  "assertCrossVenueAnchor1823789ShadowOnly.ts",
]);

const SKIP_DIR_NAMES = new Set([
  "node_modules",
  ".next",
  ".git",
  "dist",
  "coverage",
  ".paper",
]);

const SCAN_TOP_LEVEL = ["lib", "scripts", "app"] as const;

function isUnderTestsDir(absPath: string): boolean {
  const parts = absPath.split(path.sep);
  return parts.includes("tests");
}

/** Palavras / identificadores (fronteira de palavra). */
const WORD_PATTERNS: ReadonlyArray<{ name: string; re: RegExp }> = [
  { name: "createOrder", re: /\bcreateOrder\b/ },
  { name: "submitOrder", re: /\bsubmitOrder\b/ },
  { name: "postOrder", re: /\bpostOrder\b/ },
  { name: "placeOrder", re: /\bplaceOrder\b/ },
  { name: "signOrder", re: /\bsignOrder\b/ },
  { name: "privateKey", re: /\bprivateKey\b/ },
  { name: "signer", re: /\bsigner\b/ },
  { name: "wallet", re: /\bwallet\b/ },
  { name: "signTypedData", re: /\bsignTypedData\b/ },
  { name: "sendTransaction", re: /\bsendTransaction\b/ },
  { name: "broadcast", re: /\bbroadcast\b/ },
  { name: "seedPhrase", re: /\bseedPhrase\b/ },
  { name: "mnemonic", re: /\bmnemonic\b/i },
];

const TOKEN_PATTERNS: ReadonlyArray<{ name: string; re: RegExp }> = [
  { name: "PRIVATE_KEY", re: /PRIVATE_KEY/ },
  { name: "MNEMONIC", re: /\bMNEMONIC\b/ },
  { name: "SEED_PHRASE", re: /\bSEED_PHRASE\b/ },
];

const APPROVE_ONLY = /\bapprove\b/;

const PHRASE_INCLUDES = ["seed phrase"];

interface Finding {
  file: string;
  lineNum: number;
  lineText: string;
  matched: string;
}

function shouldScanFile(absPath: string): boolean {
  if (SKIP_AUDIT_ENUM_BASENAMES.has(path.basename(absPath))) return false;
  if (isUnderTestsDir(absPath)) return false;
  const ext = path.extname(absPath);
  if (!(ext === ".ts" || ext === ".tsx" || ext === ".js" || ext === ".jsx" || ext === ".mjs")) {
    return false;
  }
  return true;
}

function collectFilesRecursive(dir: string, out: string[]): void {
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return;
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    if (e.name.startsWith(".")) continue;
    if (SKIP_DIR_NAMES.has(e.name)) continue;
    const full = path.join(dir, e.name);
    if (e.isDirectory()) collectFilesRecursive(full, out);
    else if (e.isFile() && shouldScanFile(full)) out.push(full);
  }
}

function scanLine(line: string, lineNum: number, fileRel: string, findings: Finding[]): void {
  if (line.length > 5000) return;
  for (const { name, re } of WORD_PATTERNS) {
    if (re.test(line)) findings.push({ file: fileRel, lineNum, lineText: line.trim(), matched: name });
  }
  for (const { name, re } of TOKEN_PATTERNS) {
    if (re.test(line)) findings.push({ file: fileRel, lineNum, lineText: line.trim(), matched: name });
  }
  if (APPROVE_ONLY.test(line)) {
    findings.push({ file: fileRel, lineNum, lineText: line.trim(), matched: "approve" });
  }
  for (const p of PHRASE_INCLUDES) {
    if (line.includes(p))
      findings.push({ file: fileRel, lineNum, lineText: line.trim(), matched: `phrase:${p}` });
  }
}

function main(): void {
  const cwd = process.cwd();
  const root = cwd;
  const files: string[] = [];
  for (const seg of SCAN_TOP_LEVEL) {
    collectFilesRecursive(path.join(root, seg), files);
  }
  files.sort();

  console.log("[main-shadow-only] arquivos verificados (lib/scripts/app, sem tests/, sem artefactos):");
  for (const f of files) {
    console.log("  -", path.relative(root, f));
  }

  const findings: Finding[] = [];
  for (const abs of files) {
    const rel = path.relative(root, abs);
    const raw = fs.readFileSync(abs, "utf8");
    const lines = raw.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      scanLine(lines[i], i + 1, rel, findings);
    }
  }

  console.log(`[main-shadow-only] achados suspeitos: ${findings.length}`);
  if (findings.length > 0) {
    console.log("[main-shadow-only] detalhes:");
    for (const f of findings) {
      console.log(
        `  SHADOW_ONLY_FAIL\t${f.file}:${f.lineNum}\t[${f.matched}]\t${f.lineText.slice(0, 200)}`,
      );
    }
    console.log("SHADOW_ONLY_FAIL");
    process.exit(1);
  }

  console.log("SHADOW_ONLY_PASS");
}

main();

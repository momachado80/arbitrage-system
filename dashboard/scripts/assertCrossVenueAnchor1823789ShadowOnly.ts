/**
 * Auditoria estática shadow-only para a trilha cross_venue_anchor_1823789.
 *
 * Falha (exit 1) se encontrar:
 *   - Termos sensíveis associados a submissão real / chaves / RPC de escrita.
 *   - Flags hardcoded `realSubmission: true`, `wouldSubmit: true`,
 *     `shadowRealSubmission: true`.
 *
 * Aceita explicitamente:
 *   - `realSubmission: false`, `wouldSubmit: false`, `shadowRealSubmission: false`
 *   - `shadowSubmissionPolicy: "shadow_only_no_api_submit_v1"`
 *
 * O próprio script é incluído na lista de "arquivos verificados", mas a varredura
 * de termos proibidos pula a si mesmo para que sua tabela de termos não cause
 * falso positivo.
 */

import fs from "fs";
import path from "path";

const FORBIDDEN_TERMS: ReadonlyArray<string> = [
  "createOrder",
  "submitOrder",
  "signOrder",
  "postOrder",
  "placeOrder",
  "privateKey",
  "PRIVATE_KEY",
  "apiSecret",
  "wallet",
  "signer",
  "orderbook submit",
  "clob submit",
  "trade submit",
  "executableOrder",
  "broadcast",
  "sendTransaction",
  "signTypedData",
  "approve",
  "MNEMONIC",
  "SEED_PHRASE",
  "POLYGON_RPC_WRITE",
  "RPC_WRITE",
  "WRITE_RPC",
];

const FORBIDDEN_TRUE_FLAGS: ReadonlyArray<{ label: string; pattern: RegExp }> = [
  { label: "realSubmission_true", pattern: /\brealSubmission\s*:\s*true\b/ },
  { label: "wouldSubmit_true", pattern: /\bwouldSubmit\s*:\s*true\b/ },
  { label: "shadowRealSubmission_true", pattern: /\bshadowRealSubmission\s*:\s*true\b/ },
];

const SELF_BASENAME = "assertCrossVenueAnchor1823789ShadowOnly.ts";

interface Finding {
  file: string;
  line: number;
  text: string;
  matched: string;
}

function listLibFiles(): string[] {
  const dir = path.resolve(__dirname, "..", "lib");
  return fs
    .readdirSync(dir)
    .filter(f => f.startsWith("crossVenueAnchor1823789") && f.endsWith(".ts"))
    .map(f => path.join(dir, f))
    .sort();
}

function listScriptFiles(): string[] {
  const dir = path.resolve(__dirname);
  const targets = ["runCrossVenueAnchor1823789PaperExecution.ts", SELF_BASENAME];
  return targets.map(f => path.join(dir, f)).filter(p => fs.existsSync(p));
}

function scanFile(file: string, isSelf: boolean): Finding[] {
  const findings: Finding[] = [];
  if (isSelf) return findings;
  const content = fs.readFileSync(file, "utf8");
  const lines = content.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    for (const term of FORBIDDEN_TERMS) {
      if (line.includes(term)) {
        findings.push({ file, line: i + 1, text: line, matched: `term:${term}` });
      }
    }
    for (const { label, pattern } of FORBIDDEN_TRUE_FLAGS) {
      if (pattern.test(line)) {
        findings.push({ file, line: i + 1, text: line, matched: `flag:${label}` });
      }
    }
  }
  return findings;
}

function main(): void {
  const files = [...listLibFiles(), ...listScriptFiles()];
  const findings: Finding[] = [];

  console.log("[shadow-only-audit] arquivos verificados:");
  for (const f of files) {
    console.log("  -", path.relative(process.cwd(), f));
  }

  for (const f of files) {
    const isSelf = path.basename(f) === SELF_BASENAME;
    findings.push(...scanFile(f, isSelf));
  }

  console.log(`[shadow-only-audit] achados suspeitos: ${findings.length}`);
  if (findings.length > 0) {
    console.log("[shadow-only-audit] linhas suspeitas:");
    for (const f of findings) {
      console.log(
        `  ${path.relative(process.cwd(), f.file)}:${f.line} [${f.matched}] ${f.text.trim()}`,
      );
    }
    console.log("SHADOW_ONLY_FAIL");
    process.exit(1);
  }

  console.log("SHADOW_ONLY_PASS");
}

main();

#!/usr/bin/env node
/**
 * Export / import do bundle da trilha observacional (4 JSONs sob PAPER_STATE_DIR).
 * Manter alinhado com lib/paperStateDir.ts → PAPER_TRAIL_FILENAMES.
 *
 * Export:  node scripts/paper-trail-bundle.mjs export [--out DIR] [--dry-run]
 * Import:  node scripts/paper-trail-bundle.mjs import --from DIR [--to DIR] [--dry-run] [--confirm]
 *
 * Diretório fonte (export) / alvo (import): PAPER_STATE_DIR ou <cwd>/.paper
 */

import fs from "fs";
import path from "path";

const TRAIL_FILES = [
  "pocket-economics-state.json",
  "pocket-execution-state.json",
  "minimal-paper-execution-state.json",
  "system-ladder-history.json",
];

const MANIFEST = "paper-trail-manifest.json";

function resolvePaperStateDir(cwd) {
  const raw = process.env.PAPER_STATE_DIR?.trim();
  if (raw && raw.length > 0) return path.resolve(raw);
  return path.join(cwd, ".paper");
}

function usage(code = 0) {
  const msg = `
paper-trail-bundle — export/import conservador da trilha (sem alterar formato dos JSONs)

  export [--out DIR] [--dry-run]
      Copia os 4 ficheiros existentes para DIR (default: ./paper-trail-export-<timestamp>)
      e grava ${MANIFEST}.

  import --from BUNDLE_DIR [--to TARGET_DIR] [--dry-run] [--confirm]
      Copia do bundle para TARGET_DIR (default: PAPER_STATE_DIR ou ./.paper).
      Escrita real exige --confirm (proteção contra overwrite acidental).

Variáveis: PAPER_STATE_DIR (opcional; senão <cwd>/.paper)
`;
  console.error(msg.trim());
  process.exit(code);
}

function parseArgs(argv) {
  const flags = new Set();
  const kv = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--dry-run") flags.add("dryRun");
    else if (a === "--confirm") flags.add("confirm");
    else if (a.startsWith("--out=")) kv.out = a.slice("--out=".length);
    else if (a.startsWith("--from=")) kv.from = a.slice("--from=".length);
    else if (a.startsWith("--to=")) kv.to = a.slice("--to=".length);
    else if (a === "--out") kv.out = argv[++i];
    else if (a === "--from") kv.from = argv[++i];
    else if (a === "--to") kv.to = argv[++i];
    else if (a === "--help" || a === "-h") kv.help = true;
  }
  return { flags, kv };
}

function statSafe(fp) {
  try {
    return fs.statSync(fp);
  } catch {
    return null;
  }
}

function summarizeFile(absPath) {
  const st = statSafe(absPath);
  if (!st || !st.isFile()) {
    return { present: false, bytes: 0, path: absPath };
  }
  return { present: true, bytes: st.size, path: absPath, mtimeMs: st.mtimeMs };
}

function cmdExport(cwd, dryRun, outDirArg) {
  const sourceDir = resolvePaperStateDir(cwd);
  const outDir =
    outDirArg?.trim() ||
    path.join(cwd, `paper-trail-export-${new Date().toISOString().replace(/[:.]/g, "-")}`);

  const entries = TRAIL_FILES.map((name) => {
    const abs = path.join(sourceDir, name);
    const s = summarizeFile(abs);
    return { name, ...s };
  });

  const manifest = {
    version: 1,
    exportedAt: new Date().toISOString(),
    sourcePaperStateDir: sourceDir,
    files: entries.map((e) => ({
      name: e.name,
      present: e.present,
      bytes: e.bytes,
    })),
  };

  console.log("[paper-trail] export summary");
  console.log(`  source: ${sourceDir}`);
  console.log(`  dest:   ${outDir}`);
  for (const e of entries) {
    console.log(`  - ${e.name}: ${e.present ? `${e.bytes} bytes` : "(ausente — não copiado)"}`);
  }

  if (dryRun) {
    console.log("[paper-trail] dry-run: nada escrito.");
    return;
  }

  fs.mkdirSync(outDir, { recursive: true });
  let copied = 0;
  for (const e of entries) {
    if (!e.present) continue;
    const dest = path.join(outDir, e.name);
    fs.copyFileSync(e.path, dest);
    copied += 1;
    console.log(`[paper-trail] copied ${e.name}`);
  }
  fs.writeFileSync(path.join(outDir, MANIFEST), JSON.stringify(manifest, null, 2), "utf8");
  console.log(`[paper-trail] wrote ${MANIFEST} (${copied} ficheiro(s) copiado(s)).`);
}

function readBundleManifest(bundleDir) {
  const mp = path.join(bundleDir, MANIFEST);
  const st = statSafe(mp);
  if (st?.isFile()) {
    try {
      return JSON.parse(fs.readFileSync(mp, "utf8"));
    } catch {
      return null;
    }
  }
  return null;
}

function cmdImport(cwd, dryRun, fromDir, toDirArg, confirmed) {
  const bundleDir = path.resolve(cwd, fromDir.trim());
  const bSt = statSafe(bundleDir);
  if (!bSt?.isDirectory()) {
    console.error(`[paper-trail] --from não é um diretório: ${bundleDir}`);
    process.exit(1);
  }

  const targetDir = toDirArg?.trim()
    ? path.resolve(cwd, toDirArg.trim())
    : resolvePaperStateDir(cwd);

  const manifest = readBundleManifest(bundleDir);

  const toCopy = [];
  for (const name of TRAIL_FILES) {
    const src = path.join(bundleDir, name);
    const s = summarizeFile(src);
    if (!s.present) continue;
    const dest = path.join(targetDir, name);
    const destSt = statSafe(dest);
    toCopy.push({
      name,
      src: s.path,
      bytes: s.bytes,
      dest,
      destExists: !!(destSt && destSt.isFile()),
      destBytes: destSt?.isFile() ? destSt.size : 0,
    });
  }

  if (toCopy.length === 0) {
    console.error(
      "[paper-trail] Nenhum dos 4 ficheiros encontrado no bundle. Esperados (se existirem):",
      TRAIL_FILES.join(", "),
    );
    process.exit(1);
  }

  console.log("[paper-trail] import summary");
  if (manifest?.exportedAt) {
    console.log(`  manifest exportedAt: ${manifest.exportedAt}`);
    if (manifest.sourcePaperStateDir) {
      console.log(`  manifest sourcePaperStateDir: ${manifest.sourcePaperStateDir}`);
    }
  } else {
    console.log(`  (sem ${MANIFEST} legível — a importar só por ficheiros presentes)`);
  }
  console.log(`  bundle: ${bundleDir}`);
  console.log(`  alvo:   ${targetDir}`);
  for (const row of toCopy) {
    const ow = row.destExists ? `SOBRESCREVE (${row.destBytes} bytes → ${row.bytes} bytes)` : "novo";
    console.log(`  - ${row.name}: ${ow}`);
  }

  const missingInBundle = TRAIL_FILES.filter(
    (n) => !toCopy.some((t) => t.name === n),
  );
  if (missingInBundle.length) {
    console.log(`  (omitidos no bundle, alvo não tocado: ${missingInBundle.join(", ")})`);
  }

  if (dryRun) {
    console.log("[paper-trail] dry-run: nada escrito.");
    return;
  }

  if (!confirmed) {
    console.error(
      "[paper-trail] Import bloqueado: ficheiros no alvo seriam sobrescritos ou criados. " +
        "Rever o resumo acima e correr de novo com --confirm",
    );
    process.exit(1);
  }

  fs.mkdirSync(targetDir, { recursive: true });
  for (const row of toCopy) {
    fs.copyFileSync(row.src, row.dest);
    console.log(`[paper-trail] wrote ${row.dest}`);
  }
  console.log(
    "[paper-trail] import concluído. Reiniciar o processo do dashboard no ambiente alvo antes de confiar no estado em memória.",
  );
}

const argv = process.argv.slice(2);
const cmd = argv[0];
const rest = argv.slice(1);
const { flags, kv } = parseArgs(rest);

if (kv.help || !cmd) usage(1);

const cwd = process.cwd();
const dryRun = flags.has("dryRun");

try {
  if (cmd === "export") {
    cmdExport(cwd, dryRun, kv.out);
  } else if (cmd === "import") {
    if (!kv.from?.trim()) {
      console.error("[paper-trail] import exige --from <BUNDLE_DIR>");
      usage(1);
    }
    cmdImport(cwd, dryRun, kv.from, kv.to, flags.has("confirm"));
  } else {
    console.error(`[paper-trail] comando desconhecido: ${cmd}`);
    usage(1);
  }
} catch (e) {
  console.error("[paper-trail] erro:", e instanceof Error ? e.message : e);
  process.exit(1);
}

#!/usr/bin/env node
/**
 * Runtime validation of Railway volume and shadow persistence.
 * Run from repo root: node dashboard/scripts/validate-persistence-runtime.js
 * Or from dashboard/: node scripts/validate-persistence-runtime.js
 * In Railway: railway run node dashboard/scripts/validate-persistence-runtime.js
 */
const fs = require("fs");
const path = require("path");

const out = (label, value) => console.log(`[${label}]`, typeof value === "object" ? JSON.stringify(value, null, 2) : value);

out("SHADOW_PERSISTENCE_PATH", process.env.SHADOW_PERSISTENCE_PATH ?? "(not set)");
out("DATA_PATH", process.env.DATA_PATH ?? "(not set)");
out("RAILWAY_VOLUME_MOUNT_PATH", process.env.RAILWAY_VOLUME_MOUNT_PATH ?? "(not set)");
out("process.cwd()", process.cwd());

const base = process.env.SHADOW_PERSISTENCE_PATH || process.env.DATA_PATH || process.cwd();
const filePath = path.join(base, "shadow-closed-trades.json");
out("Resolved persistence path", filePath);

const checks = [];

const dataExists = fs.existsSync("/data");
checks.push({ check: "/data exists", result: dataExists });

let dataIsDir = false;
if (dataExists) {
  try {
    dataIsDir = fs.statSync("/data").isDirectory();
    checks.push({ check: "/data is directory", result: dataIsDir });
  } catch (e) {
    checks.push({ check: "/data stat", result: String(e.message) });
  }
}

let writeReadOk = false;
if (dataExists && dataIsDir) {
  const testFile = "/data/.persistence-validation-" + Date.now();
  try {
    fs.writeFileSync(testFile, "ok", "utf-8");
    const read = fs.readFileSync(testFile, "utf-8");
    fs.unlinkSync(testFile);
    writeReadOk = read === "ok";
    checks.push({ check: "write/read /data", result: writeReadOk });
  } catch (e) {
    checks.push({ check: "write/read /data", result: String(e.message) });
  }
} else {
  checks.push({ check: "write/read /data", result: "skipped (no /data)" });
}

let listData = [];
if (dataExists && dataIsDir) {
  try {
    listData = fs.readdirSync("/data");
    checks.push({ check: "/data contents", result: listData });
  } catch (e) {
    checks.push({ check: "/data readdir", result: String(e.message) });
  }
}

const shadowFileExists = fs.existsSync(filePath);
checks.push({ check: "shadow-closed-trades.json exists", result: shadowFileExists, path: filePath });

let filePreview = null;
if (shadowFileExists) {
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const parsed = JSON.parse(raw);
    filePreview = {
      schemaVersion: parsed.schemaVersion,
      savedAt: parsed.savedAt,
      profileIds: Object.keys(parsed.byProfile ?? {}),
      totalTrades: Object.values(parsed.byProfile ?? {}).reduce((s, arr) => s + (Array.isArray(arr) ? arr.length : 0), 0),
    };
  } catch (e) {
    filePreview = { error: String(e.message) };
  }
}
checks.push({ check: "file preview", result: filePreview });

out("---", "CHECKS");
checks.forEach((c) => console.log(JSON.stringify(c)));

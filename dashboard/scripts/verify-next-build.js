/**
 * Ensures App Router API/page modules exist on disk for every path in app-paths-manifest.json.
 * Next resolves handlers via require() of paths like `.next/server/app/api/.../route.js`;
 * if the manifest references files that were never emitted (stale/partial .next), runtime fails with MODULE_NOT_FOUND.
 */

const fs = require("fs");
const path = require("path");

function verifyNextBuildArtifacts() {
  const root = path.join(__dirname, "..");
  const serverDir = path.join(root, ".next", "server");
  const manifestPath = path.join(serverDir, "app-paths-manifest.json");

  if (!fs.existsSync(path.join(root, ".next", "BUILD_ID"))) {
    console.error(
      "[verify-next-build] Missing .next/BUILD_ID — run `npm run build` in the project root (same cwd as package.json)."
    );
    process.exit(1);
  }

  if (!fs.existsSync(manifestPath)) {
    console.error(
      "[verify-next-build] Missing .next/server/app-paths-manifest.json — run `npm run build`."
    );
    process.exit(1);
  }

  let manifest;
  try {
    manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  } catch (e) {
    console.error("[verify-next-build] Invalid app-paths-manifest.json:", e.message);
    process.exit(1);
  }

  const missing = [];
  for (const [route, rel] of Object.entries(manifest)) {
    if (typeof rel !== "string" || rel.length === 0) continue;
    const full = path.join(serverDir, rel);
    if (!fs.existsSync(full)) {
      missing.push({ route, rel });
    }
  }

  if (missing.length > 0) {
    console.error(
      "[verify-next-build] Next build output does not match app-paths-manifest.json (incomplete .next/server)."
    );
    console.error("  Run a clean build: rm -rf .next && npm run build");
    console.error("  Missing files (showing up to 20):");
    for (const m of missing.slice(0, 20)) {
      console.error(`    ${m.route} -> ${m.rel}`);
    }
    if (missing.length > 20) {
      console.error(`    ... and ${missing.length - 20} more`);
    }
    process.exit(1);
  }

  console.log(
    `[verify-next-build] OK — ${Object.keys(manifest).length} app paths present under .next/server`
  );
}

module.exports = { verifyNextBuildArtifacts };

if (require.main === module) {
  verifyNextBuildArtifacts();
  const domain = process.env.RAILWAY_PUBLIC_DOMAIN;
  if (domain) {
    console.log(`[DASHBOARD_URL] https://${domain}`);
  }
}

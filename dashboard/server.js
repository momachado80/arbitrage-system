const { createServer } = require("http");
const { parse } = require("url");
const next = require("next");
const { verifyNextBuildArtifacts } = require("./scripts/verify-next-build.js");

/** Fail fast if .next/server is missing per-route modules (avoids obscure MODULE_NOT_FOUND inside Next). */
verifyNextBuildArtifacts();

const port = parseInt(process.env.PORT || "3000", 10);
const hostname = "0.0.0.0";
const app = next({ dev: false, hostname, port });
const handle = app.getRequestHandler();

function normalizePathname(pathname) {
  if (!pathname || pathname === "/") return "/";
  return pathname.length > 1 && pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
}

app.prepare().then(() => {
  console.log("[BOOT] Next app.prepare() done; binding HTTP server");
  createServer(async (req, res) => {
    try {
      const parsed = parse(req.url, true);
      const pathname = normalizePathname(parsed.pathname || "/");
      // Isolated from App Router: Railway healthcheck must not depend on Next routing / first-compile.
      if (
        (req.method === "GET" || req.method === "HEAD") &&
        pathname === "/api/live"
      ) {
        console.log("[live] hit", req.method, pathname);
        res.statusCode = 200;
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        if (req.method === "HEAD") {
          res.end();
        } else {
          res.end(
            JSON.stringify({
              ok: true,
              live: true,
              ts: new Date().toISOString(),
            })
          );
        }
        console.log("[live] response sent", req.method);
        return;
      }
      await handle(req, res, parsed);
    } catch (err) {
      console.error("Error:", req.url, err);
      res.statusCode = 500;
      res.end("internal server error");
    }
  })
    .once("error", (err) => {
      console.error(err);
      process.exit(1);
    })
    .listen(port, hostname, () => {
      console.log(`> Ready on http://${hostname}:${port}`);
      const domain = process.env.RAILWAY_PUBLIC_DOMAIN;
      if (domain) console.log(`[DASHBOARD_URL] https://${domain}`);
    });
});

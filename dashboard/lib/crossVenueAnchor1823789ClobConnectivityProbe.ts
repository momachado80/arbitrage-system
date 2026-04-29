/**
 * Probe dedicado de conectividade CLOB para o track ativo 1823789.
 * Executa no mesmo runtime do worker paper-exec para capturar falhas de rede reais desse ambiente.
 */

const GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets";
const CLOB_BOOK_URL = "https://clob.polymarket.com/book";
const DEFAULT_TIMEOUT_MS = 8_000;

type HttpKind =
  | "ok"
  | "timeout"
  | "dns_resolution_failed"
  | "network_error"
  | "http_error"
  | "empty_body"
  | "json_parse_failed";

type BookStructure = "empty" | "asks_only" | "bids_only" | "two_sided" | "unusable";

export interface ClobConnectivityProbeAttempt {
  probeVersion: "cross-venue-anchor-1823789-clob-connectivity-v1";
  isoTimestamp: string;
  marketId: "1823789";
  targetTokenRole: "yes";
  attempt: number;
  tokenId: string;
  httpStatus: number | null;
  httpKind: HttpKind;
  errorDetail: string;
  fetchDurationMs: number;
  dnsResolutionFailed: boolean;
  timeoutOccurred: boolean;
  bodyEmpty: boolean;
  jsonParseFailed: boolean;
  bookStructure: BookStructure;
}

function parseJsonStringArray(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(v => String(v)).filter(Boolean);
  if (typeof value !== "string") return [];
  try {
    const parsed = JSON.parse(value) as unknown;
    return Array.isArray(parsed) ? parsed.map(v => String(v)).filter(Boolean) : [];
  } catch {
    return [];
  }
}

async function resolveYesTokenFromGamma(): Promise<string> {
  const url = `${GAMMA_MARKETS_URL}?id=1823789`;
  const res = await fetch(url, {
    signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`gamma_http_${res.status}`);
  const json = (await res.json()) as unknown;
  const row =
    Array.isArray(json) && json.length > 0 && json[0] && typeof json[0] === "object"
      ? (json[0] as Record<string, unknown>)
      : null;
  if (!row) throw new Error("gamma_payload_invalid");
  const outcomes = parseJsonStringArray(row.outcomes);
  const tokenIds = parseJsonStringArray(row.clobTokenIds);
  if (tokenIds.length === 0) throw new Error("gamma_missing_clob_token_ids");
  const yesIdx = outcomes.findIndex(o => /^yes$/i.test(o.trim()));
  if (yesIdx >= 0 && yesIdx < tokenIds.length) return tokenIds[yesIdx]!;
  return tokenIds[0]!;
}

function deriveBookStructure(raw: unknown): BookStructure {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return "unusable";
  const r = raw as Record<string, unknown>;
  const bids = Array.isArray(r.bids) ? r.bids : [];
  const asks = Array.isArray(r.asks) ? r.asks : [];
  if (bids.length > 0 && asks.length > 0) return "two_sided";
  if (bids.length > 0 && asks.length === 0) return "bids_only";
  if (bids.length === 0 && asks.length > 0) return "asks_only";
  return "empty";
}

function networkFailureKind(err: unknown): {
  httpKind: HttpKind;
  detail: string;
  dnsResolutionFailed: boolean;
  timeoutOccurred: boolean;
} {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();
  const timeoutOccurred =
    lower.includes("timeout") || lower.includes("aborterror") || lower.includes("timed out");
  const dnsResolutionFailed =
    lower.includes("enotfound") ||
    lower.includes("eai_again") ||
    lower.includes("dns") ||
    lower.includes("getaddrinfo");
  if (timeoutOccurred) {
    return { httpKind: "timeout", detail: msg || "timeout", dnsResolutionFailed, timeoutOccurred: true };
  }
  if (dnsResolutionFailed) {
    return {
      httpKind: "dns_resolution_failed",
      detail: msg || "dns_resolution_failed",
      dnsResolutionFailed: true,
      timeoutOccurred,
    };
  }
  return { httpKind: "network_error", detail: msg || "network_error", dnsResolutionFailed, timeoutOccurred };
}

export async function runCrossVenueAnchor1823789ClobConnectivityProbe(
  attempts: number = 3,
): Promise<ClobConnectivityProbeAttempt[]> {
  const yesToken = await resolveYesTokenFromGamma();
  const out: ClobConnectivityProbeAttempt[] = [];

  for (let attempt = 1; attempt <= Math.max(1, attempts); attempt++) {
    const isoTimestamp = new Date().toISOString();
    const t0 = Date.now();
    let httpStatus: number | null = null;
    let httpKind: HttpKind = "network_error";
    let errorDetail = "";
    let bodyEmpty = false;
    let jsonParseFailed = false;
    let dnsResolutionFailed = false;
    let timeoutOccurred = false;
    let bookStructure: BookStructure = "unusable";

    try {
      const res = await fetch(`${CLOB_BOOK_URL}?token_id=${encodeURIComponent(yesToken)}`, {
        signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
        headers: { Accept: "application/json" },
      });
      httpStatus = res.status;
      if (!res.ok) {
        httpKind = "http_error";
        errorDetail = `http_${res.status}`;
      } else {
        const body = await res.text();
        if (!body.trim()) {
          bodyEmpty = true;
          httpKind = "empty_body";
          errorDetail = "empty_body";
        } else {
          try {
            const parsed = JSON.parse(body) as unknown;
            httpKind = "ok";
            bookStructure = deriveBookStructure(parsed);
            errorDetail = "";
          } catch {
            jsonParseFailed = true;
            httpKind = "json_parse_failed";
            errorDetail = "json_parse_failed";
          }
        }
      }
    } catch (e) {
      const nf = networkFailureKind(e);
      httpKind = nf.httpKind;
      errorDetail = nf.detail;
      dnsResolutionFailed = nf.dnsResolutionFailed;
      timeoutOccurred = nf.timeoutOccurred;
    }

    out.push({
      probeVersion: "cross-venue-anchor-1823789-clob-connectivity-v1",
      isoTimestamp,
      marketId: "1823789",
      targetTokenRole: "yes",
      attempt,
      tokenId: yesToken,
      httpStatus,
      httpKind,
      errorDetail,
      fetchDurationMs: Date.now() - t0,
      dnsResolutionFailed,
      timeoutOccurred,
      bodyEmpty,
      jsonParseFailed,
      bookStructure,
    });
  }

  return out;
}

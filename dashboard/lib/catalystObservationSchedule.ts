/**
 * Agenda read-only de catalisadores para Observation Window Scout.
 * Funções puras (sem rede). Sem execução real, capital ou `.paper`.
 *
 * Regra-mãe: mercado primeiro. Catalisador depois.
 */

export type CatalystType = "NBA_TEAM_FUTURE" | "NHL_TEAM_FUTURE" | "FIFA_WORLD_CUP_FUTURE" | "UNKNOWN_OR_LOW_SIGNAL";

export type CatalystLeague = "NBA" | "NHL" | "FIFA" | "UNKNOWN";

export type CatalystPriority = "HIGH" | "MEDIUM" | "LOW";

export type ObservationWindowType =
  | "PRE_EVENT_60M"
  | "PRE_EVENT_15M"
  | "EVENT_START"
  | "MID_EVENT_ESTIMATE"
  | "POST_EVENT_15M"
  | "POST_EVENT_60M";

export type CatalystReadinessVerdict =
  | "HAS_NEAR_CATALYST"
  | "NO_NEAR_CATALYST"
  | "NEEDS_SCHEDULE_SOURCE"
  | "LOW_SIGNAL_MARKET";

export interface InferCatalystProfileInput {
  marketId?: string | null;
  question?: string | null;
  slug?: string | null;
  category?: string | null;
  tags?: unknown;
  endDate?: string | null;
  nowIso?: string | null;
}

export interface CatalystProfile {
  catalystType: CatalystType;
  subject: string;
  league: CatalystLeague;
  catalystPriority: CatalystPriority;
  reasons: string[];
}

export interface CatalystEventScheduleInput {
  marketId: string;
  label: string | null;
  catalystType: CatalystType;
  eventName: string;
  eventStartUtc: string;
  eventEndUtc?: string | null;
  source: string;
  confidence: "high" | "medium" | "low";
}

export interface ObservationWindowSlot {
  windowType: ObservationWindowType;
  runAtUtc: string;
  reason: string;
}

export type ScheduleFetchStatus = "ok" | "failed" | "not_applicable";

export interface EvaluateCatalystObservationReadinessInput {
  nowIso: string;
  horizonHours: number;
  profile: CatalystProfile;
  /** Início UTC do próximo evento identificado (se existir). */
  nextEventStartUtc?: string | null;
  scheduleFetchStatus: ScheduleFetchStatus;
}

export interface CatalystObservationReadiness {
  catalystReadinessVerdict: CatalystReadinessVerdict;
  nextObservationWindowUtc: string | null;
  reasons: string[];
}

const MS_MIN = 60_000;
const MS_HOUR = 60 * MS_MIN;

const LOW_SIGNAL_PATTERNS: RegExp[] = [
  /\bgta\b/i,
  /\bbitcoin\b.*\$?\s*1\s*m/i,
  /\bjesus\s+christ\b/i,
  /\b(second\s+coming)\b/i,
];

function tagsToString(tags: unknown): string {
  if (tags == null) return "";
  if (typeof tags === "string") return tags;
  if (Array.isArray(tags)) return tags.map(t => String(t)).join(" ");
  try {
    return JSON.stringify(tags);
  } catch {
    return "";
  }
}

function composeText(input: InferCatalystProfileInput): string {
  return [input.question, input.slug, input.category, tagsToString(input.tags)].filter(Boolean).join(" ");
}

function pushReason(list: string[], r: string): void {
  if (!list.includes(r)) list.push(r);
}

/**
 * Infere perfil de catalisador a partir do texto do mercado (read-only, sem rede).
 */
export function inferCatalystProfileFromMarket(input: InferCatalystProfileInput): CatalystProfile {
  const reasons: string[] = [];
  const text = composeText(input);

  for (const rx of LOW_SIGNAL_PATTERNS) {
    if (rx.test(text)) {
      pushReason(reasons, "low_signal_or_meme_keywords_detected");
      return {
        catalystType: "UNKNOWN_OR_LOW_SIGNAL",
        subject: "",
        league: "UNKNOWN",
        catalystPriority: "LOW",
        reasons,
      };
    }
  }

  const nbaRx = /will\s+the\s+(.+?)\s+win\s+(?:the\s+)?(\d{4})\s+nba\s+finals/i;
  const nhlRx = /will\s+the\s+(.+?)\s+win\s+(?:the\s+)?(\d{4})\s+nhl\s+stanley\s+cup/i;
  const fifaRx = /will\s+(.+?)\s+win\s+(?:the\s+)?(\d{4})\s+fifa\s+world\s+cup/i;

  const nbaM = text.match(nbaRx);
  if (nbaM?.[1]) {
    const subject = nbaM[1]!.trim();
    pushReason(reasons, "matched_nba_finals_future_pattern");
    return {
      catalystType: "NBA_TEAM_FUTURE",
      subject,
      league: "NBA",
      catalystPriority: "HIGH",
      reasons,
    };
  }

  const nhlM = text.match(nhlRx);
  if (nhlM?.[1]) {
    const subject = nhlM[1]!.trim();
    pushReason(reasons, "matched_nhl_stanley_cup_future_pattern");
    return {
      catalystType: "NHL_TEAM_FUTURE",
      subject,
      league: "NHL",
      catalystPriority: "HIGH",
      reasons,
    };
  }

  const fifaM = text.match(fifaRx);
  if (fifaM?.[1]) {
    const subject = fifaM[1]!.trim();
    pushReason(reasons, "matched_fifa_world_cup_future_pattern");
    return {
      catalystType: "FIFA_WORLD_CUP_FUTURE",
      subject,
      league: "FIFA",
      catalystPriority: "MEDIUM",
      reasons,
    };
  }

  pushReason(reasons, "no_high_signal_catalyst_pattern");
  return {
    catalystType: "UNKNOWN_OR_LOW_SIGNAL",
    subject: "",
    league: "UNKNOWN",
    catalystPriority: "LOW",
    reasons,
  };
}

function addMinutesIso(startIso: string, deltaMinutes: number): string {
  const d = new Date(startIso.trim());
  const t = d.getTime();
  if (!Number.isFinite(t)) return startIso;
  return new Date(t + deltaMinutes * MS_MIN).toISOString();
}

/**
 * Constrói janelas de observação padrão relativas ao início do evento (UTC).
 * Inclui T-60, T-15, T+0, T+45 (mid-estimate), T+120 (pós aproximado via POST_EVENT_15M),
 * T+240 (pós tardio via POST_EVENT_60M), conforme especificação do readiness pack.
 */
export function buildObservationWindowsForCatalyst(event: CatalystEventScheduleInput): ObservationWindowSlot[] {
  const start = event.eventStartUtc.trim();
  const slots: ObservationWindowSlot[] = [
    {
      windowType: "PRE_EVENT_60M",
      runAtUtc: addMinutesIso(start, -60),
      reason: "pre_event_minus_60m",
    },
    {
      windowType: "PRE_EVENT_15M",
      runAtUtc: addMinutesIso(start, -15),
      reason: "pre_event_minus_15m",
    },
    {
      windowType: "EVENT_START",
      runAtUtc: start,
      reason: "event_start_t0",
    },
    {
      windowType: "MID_EVENT_ESTIMATE",
      runAtUtc: addMinutesIso(start, 45),
      reason: "mid_event_estimate_plus_45m_nba_nhl_style",
    },
    {
      windowType: "POST_EVENT_15M",
      runAtUtc: addMinutesIso(start, 120),
      reason: "post_event_approx_plus_120m",
    },
    {
      windowType: "POST_EVENT_60M",
      runAtUtc: addMinutesIso(start, 240),
      reason: "post_event_late_plus_240m",
    },
  ];
  return slots;
}

function parsedUtc(iso: string): Date | null {
  const d = new Date(iso.trim());
  return Number.isFinite(d.getTime()) ? d : null;
}

function earliestFutureWindowUtc(now: Date, windows: ObservationWindowSlot[]): string | null {
  let best: string | null = null;
  let bestT = Infinity;
  for (const w of windows) {
    const dt = parsedUtc(w.runAtUtc);
    if (!dt) continue;
    if (dt.getTime() >= now.getTime() && dt.getTime() < bestT) {
      bestT = dt.getTime();
      best = dt.toISOString();
    }
  }
  return best;
}

/**
 * Avalia se há catalisador próximo suficiente para justificar scout agendado (função pura).
 */
export function evaluateCatalystObservationReadiness(
  input: EvaluateCatalystObservationReadinessInput,
): CatalystObservationReadiness {
  const reasons: string[] = [];
  const push = (x: string): void => pushReason(reasons, x);

  const now = parsedUtc(input.nowIso) ?? new Date();
  const horizonEnd = now.getTime() + input.horizonHours * MS_HOUR;

  if (input.profile.catalystType === "UNKNOWN_OR_LOW_SIGNAL") {
    push("profile_unknown_or_low_signal");
    return {
      catalystReadinessVerdict: "LOW_SIGNAL_MARKET",
      nextObservationWindowUtc: null,
      reasons,
    };
  }

  const leagueNeedsSchedule = input.profile.league === "NBA" || input.profile.league === "NHL";

  if (leagueNeedsSchedule && input.scheduleFetchStatus === "failed") {
    push("schedule_public_fetch_failed");
    return {
      catalystReadinessVerdict: "NEEDS_SCHEDULE_SOURCE",
      nextObservationWindowUtc: null,
      reasons,
    };
  }

  if (leagueNeedsSchedule && input.scheduleFetchStatus === "ok" && !input.nextEventStartUtc) {
    push("no_matching_game_found_inside_horizon_scan");
    return {
      catalystReadinessVerdict: "NO_NEAR_CATALYST",
      nextObservationWindowUtc: null,
      reasons,
    };
  }

  if (input.profile.league === "FIFA") {
    if (input.scheduleFetchStatus === "not_applicable" || !input.nextEventStartUtc) {
      push("fifa_world_cup_requires_external_fixture_feed_not_wired_here");
      return {
        catalystReadinessVerdict: "NO_NEAR_CATALYST",
        nextObservationWindowUtc: null,
        reasons,
      };
    }
  }

  const evtIso = input.nextEventStartUtc?.trim();
  if (!evtIso) {
    push("missing_next_event_start_after_resolution");
    return {
      catalystReadinessVerdict: "NEEDS_SCHEDULE_SOURCE",
      nextObservationWindowUtc: null,
      reasons,
    };
  }

  const evt = parsedUtc(evtIso);
  if (!evt) {
    push("invalid_next_event_start_iso");
    return {
      catalystReadinessVerdict: "NEEDS_SCHEDULE_SOURCE",
      nextObservationWindowUtc: null,
      reasons,
    };
  }

  if (evt.getTime() > horizonEnd) {
    push("next_event_beyond_horizon");
    return {
      catalystReadinessVerdict: "NO_NEAR_CATALYST",
      nextObservationWindowUtc: null,
      reasons,
    };
  }

  if (evt.getTime() < now.getTime()) {
    push("next_event_already_started_relative_to_now");
    const windows = buildObservationWindowsForCatalyst({
      marketId: "_synthetic_",
      label: null,
      catalystType: input.profile.catalystType,
      eventName: "resolved_next_event",
      eventStartUtc: evt.toISOString(),
      source: "plan",
      confidence: "medium",
    });
    const nextWin = earliestFutureWindowUtc(now, windows);
    return {
      catalystReadinessVerdict: nextWin ? "HAS_NEAR_CATALYST" : "NO_NEAR_CATALYST",
      nextObservationWindowUtc: nextWin,
      reasons: [...reasons, nextWin ? "using_future_relative_windows_after_event_start" : "all_standard_windows_elapsed"],
    };
  }

  const windows = buildObservationWindowsForCatalyst({
    marketId: "_synthetic_",
    label: null,
    catalystType: input.profile.catalystType,
    eventName: "resolved_next_event",
    eventStartUtc: evt.toISOString(),
    source: "plan",
    confidence: "medium",
  });

  const nextWin = earliestFutureWindowUtc(now, windows);
  push("next_event_within_horizon");
  return {
    catalystReadinessVerdict: nextWin ? "HAS_NEAR_CATALYST" : "NO_NEAR_CATALYST",
    nextObservationWindowUtc: nextWin,
    reasons,
  };
}

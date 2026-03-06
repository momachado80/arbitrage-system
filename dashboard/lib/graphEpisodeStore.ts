import type { GraphEpisode } from "./graphEpisodeTracker";
import { GraphEpisodeTracker } from "./graphEpisodeTracker";
import type { GraphOpportunity } from "./graphArbitrageEngine";

const MAX_ACTIVE = 500;
const MAX_CLOSED = 5000;

let closedEpisodes: GraphEpisode[] = [];
let lastUpdateMs = 0;

function onEpisodeClose(ep: GraphEpisode): void {
  closedEpisodes.push(ep);
  if (closedEpisodes.length > MAX_CLOSED) {
    closedEpisodes = closedEpisodes.slice(-MAX_CLOSED);
  }
}

const tracker = new GraphEpisodeTracker({ onClose: onEpisodeClose });

export function updateGraphEpisodes(opportunities: GraphOpportunity[]): void {
  try {
    tracker.update(opportunities);
    lastUpdateMs = Date.now();

    if (tracker.getActiveCount() > MAX_ACTIVE) {
      console.warn(
        `[GraphEpisodeStore] Active episodes (${tracker.getActiveCount()}) exceeds max (${MAX_ACTIVE})`
      );
    }
  } catch (err) {
    console.error("[GraphEpisodeStore] Update failed:", err);
  }
}

export function getActiveGraphEpisodes(): GraphEpisode[] {
  return tracker.getActive();
}

export function getRecentClosedGraphEpisodes(limit = 50): GraphEpisode[] {
  const start = Math.max(0, closedEpisodes.length - limit);
  return closedEpisodes.slice(start);
}

export interface GraphEpisodeSummary {
  activeEpisodes: number;
  closedEpisodes: number;
  avgActiveDurationMs: number;
  longestRecentEpisodeMs: number;
  lastUpdate: string | null;
  totalEpisodesTracked: number;
}

export function getGraphEpisodeSummary(): GraphEpisodeSummary {
  const active = tracker.getActive();
  const activeCount = active.length;
  const closedCount = closedEpisodes.length;

  let avgDuration = 0;
  if (activeCount > 0) {
    const total = active.reduce((s, ep) => s + ep.durationMs, 0);
    avgDuration = total / activeCount;
  }

  let longest = 0;
  const recentClosed = getRecentClosedGraphEpisodes(100);
  for (const ep of recentClosed) {
    if (ep.durationMs > longest) longest = ep.durationMs;
  }
  for (const ep of active) {
    if (ep.durationMs > longest) longest = ep.durationMs;
  }

  return {
    activeEpisodes: activeCount,
    closedEpisodes: closedCount,
    avgActiveDurationMs: Math.round(avgDuration),
    longestRecentEpisodeMs: longest,
    lastUpdate: lastUpdateMs > 0 ? new Date(lastUpdateMs).toISOString() : null,
    totalEpisodesTracked: activeCount + closedCount,
  };
}

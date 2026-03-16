/**
 * Experiment Ledger — in-memory experiment history with durable persistence.
 * Rehydrates on startup. Does not auto-activate or mutate historical records.
 */

import type { ExperimentLedgerEntry } from "./experimentLedgerPersistence";
import {
  persistExperimentLedger,
  restoreExperimentLedger,
} from "./experimentLedgerPersistence";

let ledgerEntries: ExperimentLedgerEntry[] = [];
let rehydrated = false;

/**
 * Rehydrate ledger from disk. Call once at startup.
 */
export function rehydrateExperimentLedger(): void {
  if (rehydrated) return;
  ledgerEntries = restoreExperimentLedger();
  rehydrated = true;
}

/**
 * Get all ledger entries. Call rehydrateExperimentLedger first.
 */
export function getExperimentLedgerEntries(): ExperimentLedgerEntry[] {
  return [...ledgerEntries];
}

/**
 * Add a new entry (manual operation; not called by auto-activation).
 * Persists after add.
 */
export function addExperimentLedgerEntry(entry: ExperimentLedgerEntry): void {
  if (ledgerEntries.some((e) => e.experimentId === entry.experimentId)) return;
  ledgerEntries.push(entry);
  persistExperimentLedger(ledgerEntries);
}

/**
 * Update an existing entry. Does not mutate history; only updates status/decision for active entries.
 */
export function updateExperimentLedgerEntry(
  experimentId: string,
  updates: Partial<ExperimentLedgerEntry>
): void {
  const idx = ledgerEntries.findIndex((e) => e.experimentId === experimentId);
  if (idx < 0) return;
  ledgerEntries[idx] = { ...ledgerEntries[idx], ...updates };
  persistExperimentLedger(ledgerEntries);
}

/**
 * Seed ledger with initial entries. Idempotent: only adds if experimentId not present.
 */
export function seedExperimentLedgerIfEmpty(entries: ExperimentLedgerEntry[]): void {
  rehydrateExperimentLedger();
  const existing = new Set(ledgerEntries.map((e) => e.experimentId));
  let added = 0;
  for (const e of entries) {
    if (!existing.has(e.experimentId)) {
      ledgerEntries.push(e);
      existing.add(e.experimentId);
      added++;
    }
  }
  if (added > 0) {
    persistExperimentLedger(ledgerEntries);
    console.log(`[ExperimentLedger] Seeded ${added} historical entries`);
  }
}

export type { ExperimentLedgerEntry };

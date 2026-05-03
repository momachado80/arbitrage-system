/**
 * Minimal test harness (readiness-style) for dashboard/unit tests.
 */

export interface TestCase {
  name: string;
  fn: () => void | Promise<void>;
}

const REGISTRY: Map<string, TestCase[]> = new Map();
let CURRENT_FILE: string | null = null;

export function describe(file: string, body: () => void): void {
  CURRENT_FILE = file;
  REGISTRY.set(file, []);
  body();
  CURRENT_FILE = null;
}

export function test(name: string, fn: () => void | Promise<void>): void {
  if (!CURRENT_FILE) throw new Error("test() called outside describe()");
  const list = REGISTRY.get(CURRENT_FILE);
  if (!list) throw new Error(`describe() not initialized for ${CURRENT_FILE}`);
  list.push({ name, fn });
}

export function assertEqual<T>(actual: T, expected: T, msg: string): void {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) {
    throw new Error(`assertEqual failed: ${msg}\n  expected: ${e}\n  actual:   ${a}`);
  }
}

export function assertIncludes(haystack: string, needle: string, msg: string): void {
  if (!haystack.includes(needle)) {
    throw new Error(`assertIncludes failed: ${msg}`);
  }
}

export function assertTrue(cond: unknown, msg: string): void {
  if (!cond) throw new Error(`assertTrue failed: ${msg}`);
}

export function assertThrows(fn: () => unknown, msg: string): void {
  let threw = false;
  try {
    fn();
  } catch {
    threw = true;
  }
  if (!threw) throw new Error(`assertThrows failed: ${msg}`);
}

export function getRegistry(): ReadonlyMap<string, ReadonlyArray<TestCase>> {
  return REGISTRY;
}

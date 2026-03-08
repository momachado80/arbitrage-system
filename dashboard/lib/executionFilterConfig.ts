/**
 * Execution Filter Config — configurable pre-dispatch quality gates.
 * Easily adjustable for testing; prefer config over hardcoding.
 */

export const MIN_EXECUTABLE_EXPECTED_VALUE_USD =
  typeof process !== "undefined" && typeof process.env?.MIN_EEV_USD === "string"
    ? parseFloat(process.env.MIN_EEV_USD) || 0.01
    : 0.01;

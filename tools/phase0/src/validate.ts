// Checks run on every extraction result before a human reviews it.
// Nothing here changes data; it only attaches issues for the review screen.

import { crossCheckDims, parseMmPair, type ExtractedPlan, type ExtractedSpace, type TextToken } from "@planparse/shared";

export type Severity = "error" | "warning" | "info";
export interface Issue { severity: Severity; spaceCode: string | null; rule: string; message: string }

export function validatePlan(plan: Pick<ExtractedPlan, "spaces" | "wings">, tokens: TextToken[]): Issue[] {
  const issues: Issue[] = [];
  const tokenById = new Map(tokens.map((t) => [t.id, t]));
  const codes = new Map<string, number>();
  const wingCodes = new Set(plan.wings.map((w) => w.code));
  const label = (s: ExtractedSpace) => s.code ?? s.name;

  for (const s of plan.spaces) {
    if (s.code) codes.set(s.code, (codes.get(s.code) ?? 0) + 1);

    // 1. every value must trace back to something read from the drawing
    const missing = s.sourceTokenIds.filter((id) => !tokenById.has(id));
    if (missing.length) issues.push({ severity: "error", spaceCode: label(s), rule: "unknown-source", message: `cites snippet ids that do not exist: ${missing.join(", ")}` });
    // Coded units with no text behind them need a look; unlabelled shared spaces (lobby, passages) are normal.
    if (s.sourceTokenIds.length === 0) issues.push({ severity: s.code && s.type !== "loft" ? "warning" : "info", spaceCode: label(s), rule: "ai-only", message: "no OCR/PDF snippet supports this space (read by AI from the image only)" });

    // 2. printed dimensions must appear in a cited snippet (catches invented numbers)
    if (s.dimsMmText) {
      const mm = parseMmPair(s.dimsMmText);
      const cited = s.sourceTokenIds.map((id) => tokenById.get(id)?.text ?? "").join(" ");
      if (mm && !(cited.includes(String(mm.a)) && cited.includes(String(mm.b)))) {
        issues.push({ severity: "warning", spaceCode: label(s), rule: "dims-not-in-ocr", message: `'${s.dimsMmText}' not found in cited snippets; check against the drawing` });
      }
    }
    if (s.dimsFtText) {
      // every number in the ft-in text should appear in what it cites (catches sizes built from dimension lines)
      const cited = s.sourceTokenIds.map((id) => tokenById.get(id)?.text ?? "").join(" ");
      const citedNums = new Set(cited.match(/\d+/g) ?? []);
      const missingNums = (s.dimsFtText.match(/\d+/g) ?? []).filter((n) => !citedNums.has(n));
      if (missingNums.length) {
        // PDF text is exact, so a miss there means the size was probably computed → warning.
        // OCR garbles ft-in text often, so for OCR sources the AI reading the image is normal → info.
        const exact = s.sourceTokenIds.length > 0 && s.sourceTokenIds.every((id) => tokenById.get(id)?.source === "pdf");
        issues.push({
          severity: exact ? "warning" : "info", spaceCode: label(s), rule: "dims-not-in-ocr",
          message: `'${s.dimsFtText}': ${missingNums.join(", ")} not in cited snippets; ${exact ? "may be computed, check against the drawing" : "read from the image (OCR could not confirm)"}`,
        });
      }
    }

    // 3. mm vs feet-inches cross-check
    const chk = crossCheckDims(s.dimsMmText, s.dimsFtText);
    if (chk.status === "mismatch") issues.push({ severity: "warning", spaceCode: label(s), rule: "dims-mismatch", message: `mm and ft values disagree: ${chk.detail}` });

    // 4. references
    if (s.wingCode && !wingCodes.has(s.wingCode)) issues.push({ severity: "error", spaceCode: label(s), rule: "unknown-wing", message: `wing '${s.wingCode}' is not defined` });
    if (s.parentCode && !plan.spaces.some((p) => p.code === s.parentCode)) issues.push({ severity: "error", spaceCode: label(s), rule: "unknown-parent", message: `parent '${s.parentCode}' not found` });
    if (s.confidence < 0.6) issues.push({ severity: "info", spaceCode: label(s), rule: "low-confidence", message: `confidence ${s.confidence.toFixed(2)}` });
    if (s.isRectangular === false && s.dimsMmText) issues.push({ severity: "info", spaceCode: label(s), rule: "non-rectangular", message: "area from width×depth is only an estimate" });
  }

  // 5. duplicate codes
  for (const [code, n] of codes) if (n > 1) issues.push({ severity: "error", spaceCode: code, rule: "duplicate-code", message: `appears ${n} times` });

  // 6. gaps in numbered series (SHOP-19..44 missing 31 → probably missed)
  const series = new Map<string, Set<number>>();
  for (const s of plan.spaces) {
    const all = s.mergedCodes.length ? s.mergedCodes : s.code ? [s.code] : [];
    for (const c of all) {
      const m = c.match(/^([A-Z]+)-(\d+)$/);
      if (!m) continue;
      if (!series.has(m[1]!)) series.set(m[1]!, new Set());
      series.get(m[1]!)!.add(Number(m[2]));
    }
  }
  for (const [prefix, nums] of series) {
    if (nums.size < 3) continue;
    const sorted = [...nums].sort((a, b) => a - b);
    const gaps: number[] = [];
    for (let n = sorted[0]!; n <= sorted[sorted.length - 1]!; n++) if (!nums.has(n)) gaps.push(n);
    if (gaps.length) issues.push({ severity: "warning", spaceCode: null, rule: "series-gap", message: `${prefix} numbers missing: ${gaps.join(", ")} (missed, or on another floor)` });
  }
  return issues;
}

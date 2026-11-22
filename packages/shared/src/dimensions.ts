// Parsing and cross-checking of dimension strings printed on plans.
// Plans often print the same size twice: millimetres ("3000X5440") and
// feet-inches ("9'-10\"X17'-10\""). Comparing the two catches OCR mistakes.

const MM_PER_FOOT = 304.8;
const MM_PER_INCH = 25.4;

/** Normalise the quote/x characters OCR tends to mangle. */
export function normaliseDimText(s: string): string {
  return s
    .replace(/[‘’`´′]/g, "'")
    .replace(/[“”″]/g, '"')
    .replace(/''/g, '"')
    .replace(/[×✕✖*]/g, "X")
    .replace(/\s+/g, " ")
    .trim();
}

export interface MmDims { a: number; b: number }

/** "3000X5440" -> {a:3000, b:5440}. Returns null if not a mm pair. */
export function parseMmPair(raw: string): MmDims | null {
  const m = normaliseDimText(raw).match(/(?<!\d)(\d{3,5})\s*[xX]\s*(\d{3,5})(?!\d)/);
  if (!m) return null;
  return { a: Number(m[1]), b: Number(m[2]) };
}

/** One feet-inches length: 9'-10", 13', 16'7", 12'-4 1/2" -> millimetres. */
export function parseFeetInches(raw: string): number | null {
  const s = normaliseDimText(raw);
  const m = s.match(/(\d{1,3})\s*'\s*(?:-?\s*(\d{1,2})(?:\s+(\d)\/(\d))?\s*"?)?/);
  if (!m) return null;
  const feet = Number(m[1]);
  const inches = m[2] ? Number(m[2]) : 0;
  const frac = m[3] && m[4] ? Number(m[3]) / Number(m[4]) : 0;
  if (inches >= 12) return null;
  return Math.round(feet * MM_PER_FOOT + (inches + frac) * MM_PER_INCH);
}

/** "(9'-10\"X17'-10\")" -> {a, b} in mm. */
export function parseFtPair(raw: string): MmDims | null {
  const s = normaliseDimText(raw).replace(/[()]/g, "");
  const parts = s.split(/\s*X\s*/i);
  if (parts.length !== 2) return null;
  const a = parseFeetInches(parts[0]!);
  const b = parseFeetInches(parts[1]!);
  return a != null && b != null ? { a, b } : null;
}

/** Feet-inches lengths agree with mm if within 2.5% or 75 mm (drawings round). */
function close(mm: number, fromFt: number): boolean {
  return Math.abs(mm - fromFt) <= Math.max(75, mm * 0.025);
}

export type DimCheck =
  | { status: "ok" }
  | { status: "mismatch"; detail: string }
  | { status: "unchecked"; detail: string };

export function crossCheckDims(mmText: string | null, ftText: string | null): DimCheck {
  if (!mmText || !ftText) return { status: "unchecked", detail: "only one unit system printed" };
  const mm = parseMmPair(mmText);
  const ft = parseFtPair(ftText);
  if (!mm || !ft) return { status: "unchecked", detail: `could not parse '${mmText}' / '${ftText}'` };
  const okA = close(mm.a, ft.a);
  const okB = close(mm.b, ft.b);
  if (okA && okB) return { status: "ok" };
  const bad = [
    !okA ? `${mm.a}mm vs ${ft.a}mm` : null,
    !okB ? `${mm.b}mm vs ${ft.b}mm` : null,
  ].filter(Boolean).join(", ");
  return { status: "mismatch", detail: bad };
}

/** Area in m² from a mm pair; only meaningful for rectangles. */
export function areaSqm(mmText: string | null): number | null {
  const mm = mmText ? parseMmPair(mmText) : null;
  return mm ? Math.round((mm.a * mm.b) / 10_000) / 100 : null;
}

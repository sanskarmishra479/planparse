// Scores an extraction against a hand-labelled answer key ("ground truth").

import { readFile } from "node:fs/promises";
import { z } from "zod";
import { normaliseDimText, parseFtPair, parseMmPair, type ExtractedSpace } from "@planparse/shared";

export const TruthSchema = z.object({
  plan: z.string(),
  units: z.array(z.object({
    code: z.string(),
    type: z.string(),
    category: z.string().nullable(),
    dimsMm: z.string().nullable(),
    dimsFt: z.string().nullable(),
    hasLoft: z.boolean(),
  })),
  /** shared spaces: how many of each type the plan really has */
  amenities: z.record(z.string(), z.number()),
});
export type Truth = z.infer<typeof TruthSchema>;

export async function loadTruth(path: string): Promise<Truth> {
  return TruthSchema.parse(JSON.parse(await readFile(path, "utf8")));
}

/** SHOP 33,34 / SHOP-34,33 / mergedCodes → "SHOP-33,34" */
export function canonicalCode(s: Pick<ExtractedSpace, "code" | "mergedCodes">): string | null {
  if (s.mergedCodes.length > 1) {
    const prefix = s.mergedCodes[0]!.split("-")[0];
    const nums = s.mergedCodes.map((c) => Number(c.split("-")[1])).sort((a, b) => a - b);
    return `${prefix}-${nums.join(",")}`;
  }
  if (!s.code) return null;
  const m = s.code.toUpperCase().replace(/\s+/g, "").match(/^([A-Z]+)-?([\d,]+)$/);
  if (!m) return s.code.toUpperCase();
  return `${m[1]}-${m[2]!.split(",").map(Number).sort((a, b) => a - b).join(",")}`;
}

const sameMm = (a: string | null, b: string | null) => {
  const x = a ? parseMmPair(a) : null, y = b ? parseMmPair(b) : null;
  return !!x && !!y && x.a === y.a && x.b === y.b;
};
const sameFt = (a: string | null, b: string | null) => {
  const x = a ? parseFtPair(a) : null, y = b ? parseFtPair(b) : null;
  if (x && y) return Math.abs(x.a - y.a) < 13 && Math.abs(x.b - y.b) < 13;
  return !!a && !!b && normaliseDimText(a) === normaliseDimText(b);
};

export interface Score {
  unitsExpected: number;
  unitsFound: number;        // correct codes found
  unitsExtra: string[];      // codes not on the plan (invented or misread)
  unitsMissing: string[];
  typeCorrect: number;
  categoryCorrect: number;
  dimsMmCorrect: number;
  dimsFtCorrect: number;
  loftCorrect: number;
  amenityRecall: Record<string, { expected: number; found: number }>;
  unitRecall: number;
  unitPrecision: number;
  fieldAccuracy: number;     // over found units, all 5 fields
}

export function score(spaces: ExtractedSpace[], truth: Truth): Score {
  const unitTypes = new Set(truth.units.map((u) => u.type));
  const predicted = new Map<string, ExtractedSpace>();
  for (const s of spaces) {
    if (!unitTypes.has(s.type) && !s.code) continue;
    const c = canonicalCode(s);
    if (c && !predicted.has(c) && s.type !== "loft") predicted.set(c, s);
  }
  const lofts = spaces.filter((s) => s.type === "loft" && s.parentCode).map((s) => canonicalCode({ code: s.parentCode, mergedCodes: [] }));

  let found = 0, typeOk = 0, catOk = 0, mmOk = 0, ftOk = 0, loftOk = 0;
  const missing: string[] = [];
  for (const u of truth.units) {
    const p = predicted.get(u.code);
    if (!p) { missing.push(u.code); continue; }
    found++;
    if (p.type === u.type) typeOk++;
    if ((p.category ?? null)?.toLowerCase() === (u.category ?? null)?.toLowerCase()) catOk++;
    if (u.dimsMm ? sameMm(p.dimsMmText, u.dimsMm) : !p.dimsMmText) mmOk++;
    if (u.dimsFt ? sameFt(p.dimsFtText, u.dimsFt) : !p.dimsFtText) ftOk++;
    if (lofts.includes(u.code) === u.hasLoft) loftOk++;
  }
  const truthCodes = new Set(truth.units.map((u) => u.code));
  const unitPrefixes = new Set(truth.units.map((u) => u.code.split("-")[0]));
  const extra = [...predicted.keys()].filter((c) => !truthCodes.has(c) && unitPrefixes.has(c.split("-")[0]));

  const amenityRecall: Score["amenityRecall"] = {};
  for (const [type, expected] of Object.entries(truth.amenities)) {
    const n = spaces.filter((s) => s.type === type).length;
    amenityRecall[type] = { expected, found: Math.min(n, expected) };
  }

  return {
    unitsExpected: truth.units.length, unitsFound: found, unitsExtra: extra, unitsMissing: missing,
    typeCorrect: typeOk, categoryCorrect: catOk, dimsMmCorrect: mmOk, dimsFtCorrect: ftOk, loftCorrect: loftOk,
    amenityRecall,
    unitRecall: found / truth.units.length,
    unitPrecision: found / Math.max(1, found + extra.length),
    fieldAccuracy: found ? (typeOk + catOk + mmOk + ftOk + loftOk) / (found * 5) : 0,
  };
}

// Rule-based understanding of text snippets: classify each snippet, then
// build a "rules only" result (no AI) by attaching the nearest dimension
// snippets to each unit code. This is the baseline we compare the AI against.

import { normaliseDimText, parseMmPair, type ExtractedSpace, type SpaceType, type TextToken } from "@planparse/shared";

export type TokenKind =
  | "unit_code" | "unit_keyword" | "number" | "category"
  | "dim_mm" | "dim_ft" | "amenity" | "other";

export interface Classified extends TextToken { kind: TokenKind; norm: string }

const UNIT_PREFIX: Record<string, { prefix: string; type: SpaceType }> = {
  SHOP: { prefix: "SHOP", type: "shop" },
  STORE: { prefix: "STORE", type: "store" },
  SHOWROOM: { prefix: "SHOWROOM", type: "showroom" },
  KIOSK: { prefix: "KIOSK", type: "kiosk" },
  OFFICE: { prefix: "OFFICE", type: "office" },
  FLAT: { prefix: "FLAT", type: "apartment" },
  UNIT: { prefix: "UNIT", type: "shop" },
};
const CATEGORY_WORDS = ["ECONOMY", "PREMIUM", "DELUXE", "ANCHOR", "STANDARD", "LUXURY"];
const AMENITY_WORDS: [RegExp, SpaceType][] = [
  [/LADIES\s*TOILET/i, "toilet_ladies"], [/GENTS\s*TOILET/i, "toilet_gents"],
  [/\bTOILET\b/i, "toilet"], [/\bW\.?\s*C\b/i, "wc"], [/URINAL/i, "urinal"], [/\bBATH\b/i, "bath"],
  [/FIRE\s*LIFT/i, "fire_lift"], [/\bLIFT\b/i, "lift"], [/ESCALATOR/i, "escalator"],
  [/STAIR/i, "staircase"], [/\bLOBBY\b/i, "lobby"], [/ATRI/i, "atrium"],
  [/PASSAGE|CORRIDOR/i, "passage"], [/EMERGENCY\s*EXIT/i, "emergency_exit"],
  [/ENTRANCE|ENTRY/i, "entrance"], [/\bRAMP\b/i, "ramp"], [/DUCT/i, "fire_duct"],
  [/\bLOFT\b/i, "loft"], [/KITCHEN/i, "kitchen"], [/BEDR(OO)?M|BDRM/i, "bedroom"],
  [/LIVING/i, "living_room"], [/DINING/i, "dining_room"], [/GARAGE/i, "garage"],
  [/PANTRY/i, "pantry"], [/LAUNDRY/i, "laundry"], [/CLOSET/i, "closet"], [/OFFICE/i, "office"],
  [/FOYER/i, "foyer"], [/PATIO/i, "patio"], [/PORCH/i, "porch"], [/FAMILY/i, "family_room"],
];

/** Fix the OCR confusions we see most on plans. */
function fixOcr(s: string): string {
  return s
    .replace(/SH[O0]R\b|SH0P|5HOP/gi, "SHOP")
    .replace(/SHOWR[O0D]{2,3}M?|SHOWRODD|HOWROOM/gi, "SHOWROOM")
    .replace(/ST[O0]RE(?=\s*-?\s*\d)/gi, "STORE");
}

export function classify(tokens: TextToken[]): Classified[] {
  return tokens.map((t) => {
    const text = fixOcr(t.text);
    const up = text.toUpperCase();
    let kind: TokenKind = "other";
    let norm = text;

    const code = up.match(/\b(SHOP|STORE|SHOWROOM|KIOSK|OFFICE|FLAT|UNIT)\s*[-–.]?\s*(\d{1,4}(?:\s*[,&]\s*\d{1,4})*)\b/);
    if (code) {
      kind = "unit_code";
      norm = `${code[1]}-${code[2]!.replace(/\s+/g, "").replace(/&/g, ",")}`;
    } else if (/^(SHOP|STORE|SHOWROOM|KIOSK|UNIT)S?$/.test(up.trim())) {
      kind = "unit_keyword";
      norm = up.trim().replace(/S$/, "") === "STORE" ? "STORE" : up.trim();
    } else if (parseMmPair(text)) {
      const p = parseMmPair(text)!;
      kind = "dim_mm";
      norm = `${p.a}X${p.b}`;
    } else if (/\d\s*'/.test(normaliseDimText(text))) {
      kind = "dim_ft";
      norm = normaliseDimText(text);
    } else if (/^\d{1,3}$/.test(up.trim())) {
      kind = "number";
      norm = up.trim();
    } else if (CATEGORY_WORDS.includes(up.trim())) {
      kind = "category";
      norm = up.trim().toLowerCase();
    } else if (AMENITY_WORDS.some(([re]) => re.test(up))) {
      kind = "amenity";
    }
    return { ...t, text, kind, norm };
  });
}

const cx = (t: TextToken) => t.box[0] + t.box[2] / 2;
const cy = (t: TextToken) => t.box[1] + t.box[3] / 2;

/** Nearest token of a kind that sits below (or level with) `from`, within a radius. */
function nearestBelow(from: TextToken, pool: Classified[], kind: TokenKind, used: Set<string>, maxLines = 5) {
  const lineH = Math.max(from.box[3], 6);
  let best: Classified | undefined;
  let bestD = Infinity;
  for (const t of pool) {
    if (t.kind !== kind || used.has(t.id)) continue;
    const dy = cy(t) - cy(from);
    const dx = Math.abs(cx(t) - cx(from));
    if (dy < -lineH * 0.5 || dy > lineH * maxLines || dx > Math.max(from.box[2], t.box[2]) * 0.8) continue;
    const d = Math.hypot(dx, dy);
    if (d < bestD) { bestD = d; best = t; }
  }
  return best;
}

/** Rules-only extraction: codes + nearest dims + category above. */
export function rulesBaseline(tokens: Classified[]): ExtractedSpace[] {
  const used = new Set<string>();
  const spaces: ExtractedSpace[] = [];

  // "SHOWROOM" on one line and "13" on the next → one code
  const codes: { code: string; tokens: Classified[] }[] = [];
  for (const t of tokens) {
    if (t.kind === "unit_code") codes.push({ code: t.norm, tokens: [t] });
    if (t.kind === "unit_keyword") {
      const num = nearestBelow(t, tokens, "number", used, 2.5);
      if (num) { used.add(num.id); codes.push({ code: `${t.norm}-${num.norm}`, tokens: [t, num] }); }
    }
  }

  for (const { code, tokens: src } of codes) {
    const anchor = src[src.length - 1]!;
    const mm = nearestBelow(anchor, tokens, "dim_mm", used);
    if (mm) used.add(mm.id);
    const ft = nearestBelow(mm ?? anchor, tokens, "dim_ft", used, 3);
    if (ft) used.add(ft.id);
    // category word is usually just above the code
    const cat = tokens.find((t) => t.kind === "category" && !used.has(t.id)
      && cy(src[0]!) - cy(t) > 0 && cy(src[0]!) - cy(t) < src[0]!.box[3] * 2.5
      && Math.abs(cx(t) - cx(src[0]!)) < src[0]!.box[2]);
    if (cat) used.add(cat.id);

    const [prefix, nums = ""] = code.split("-");
    const numList = nums.split(",").filter(Boolean);
    const info = UNIT_PREFIX[prefix!] ?? { prefix: prefix!, type: "other" as SpaceType };
    const sources = [...src, mm, ft, cat].filter((x): x is Classified => !!x);
    const xs = sources.map((s) => s.box);
    const x0 = Math.min(...xs.map((b) => b[0])), y0 = Math.min(...xs.map((b) => b[1]));
    const x1 = Math.max(...xs.map((b) => b[0] + b[2])), y1 = Math.max(...xs.map((b) => b[1] + b[3]));

    spaces.push({
      code,
      name: `${cat ? cat.norm + " " : ""}${info.prefix.toLowerCase()} ${nums}`.trim(),
      type: info.type,
      category: cat?.norm ?? null,
      wingCode: null,
      parentCode: null,
      mergedCodes: numList.length > 1 ? numList.map((n) => `${info.prefix}-${n}`) : [],
      dimsMmText: mm?.norm ?? null,
      dimsFtText: ft?.norm ?? null,
      widthText: null,
      isRectangular: null,
      approxBox: [x0, y0, x1 - x0, y1 - y0],
      sourceTokenIds: sources.map((s) => s.id),
      confidence: Math.min(...sources.map((s) => s.confidence)),
      notes: "rules-only",
    });
  }

  // amenities: one space per amenity snippet (no grouping)
  for (const t of tokens) {
    if (t.kind !== "amenity" || used.has(t.id)) continue;
    const type = AMENITY_WORDS.find(([re]) => re.test(t.text.toUpperCase()))![1];
    spaces.push({
      code: null, name: t.text, type, category: null, wingCode: null, parentCode: null,
      mergedCodes: [], dimsMmText: null, dimsFtText: null, widthText: null, isRectangular: null,
      approxBox: [...t.box], sourceTokenIds: [t.id], confidence: t.confidence, notes: "rules-only",
    });
  }
  return spaces;
}

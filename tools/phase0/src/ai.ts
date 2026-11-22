// AI step. Two connections, same code (both speak the OpenAI chat API):
//   - "openrouter": any allow-listed model via OpenRouter (ids like "anthropic/claude-opus-5.5")
//   - "openai":     OpenAI directly with an OpenAI key (ids like "openai/gpt-6.2-sol")
//
// Speed: the slow part is the model *writing* output, so
//   1. the plan is split into 6 calls that run at the same time:
//        overview | units (left) | units (right) | indoor circulation | entrances+outdoor | rooms
//   2. the model writes a compact format (short keys, no boxes/notes); code expands it
//   3. reasoning effort is set low (configurable)
// Wings are assigned to spaces by position in code, so all calls agree.
//
// Rules enforced here:
//   - only allow-listed model vendors (no Chinese models, never "auto" routing)
//   - OpenRouter providers that keep/train on data are refused
//   - every answer is validated with Zod; one retry with the errors if it fails

import OpenAI from "openai";
import sharp from "sharp";
import { z } from "zod";
import {
  LEASABLE_TYPES, SPACE_TYPE_GROUPS, SpaceTypeSchema,
  type ExtractedPlan, type ExtractedSpace, type TextToken,
} from "@planparse/shared";

const ALLOWED_VENDORS = ["anthropic/", "openai/", "google/"];
const MAX_IMAGE_EDGE = 2400;

export function assertAllowedModel(model: string): void {
  if (!ALLOWED_VENDORS.some((v) => model.startsWith(v))) {
    throw new Error(`Model '${model}' is not allowed. Allowed vendors: ${ALLOWED_VENDORS.join(", ")}`);
  }
}

export type AiConnection = "openrouter" | "openai";
export type Effort = "minimal" | "low" | "medium" | "high";

export interface CallStats { part: string; ms: number; inputTokens: number; outputTokens: number; reasoningTokens: number; attempts: number }

export interface AiResult {
  plan: ExtractedPlan;
  model: string;
  provider: string | null;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  costUsd: number | null;
  attempts: number;
  ms: number;          // wall time (calls run in parallel)
  calls: CallStats[];
  effort: Effort | null;
}

// ---------- compact wire format (what the model writes) ----------

const WireSpace = z.object({
  c: z.string().nullable().describe("code, e.g. SHOP-37, SHOWROOM-3, STAIRCASE-2; null if none printed"),
  n: z.string().describe("short name, e.g. 'Economy Shop 37', 'Gents Toilet'"),
  t: SpaceTypeSchema,
  g: z.string().nullable().describe("printed grade/category: economy, premium, deluxe…"),
  p: z.string().nullable().describe("parent code for sub-spaces (loft in a showroom)"),
  m: z.array(z.string()).describe("merged unit codes, e.g. ['SHOP-33','SHOP-34']; else []"),
  mm: z.string().nullable().describe("mm dims exactly as printed, e.g. 4025X6370"),
  ft: z.string().nullable().describe("ft-in dims exactly as printed, e.g. 13'-3\"X21'"),
  wd: z.string().nullable().describe("printed width for passages, e.g. 7'"),
  s: z.array(z.string()).describe("snippet ids used"),
  b: z.array(z.number()).describe("rough [x,y,w,h] of the space ONLY when s is empty; otherwise []"),
  k: z.number().describe("confidence 0..1"),
});
type WireSpace = z.infer<typeof WireSpace>;

const WireSpaces = z.object({ spaces: z.array(WireSpace) });

const WireOverview = z.object({
  drawingTitle: z.string().nullable(),
  building: z.object({
    suggestedCode: z.string(), name: z.string().nullable(), type: z.string(), northDirection: z.string().nullable(),
  }),
  floor: z.object({
    level: z.number().nullable().describe("entrance/ground level = 0 (a house's main floor is 0), floors above 1, 2…, basements -1, -2…"),
    name: z.string().nullable(),
  }),
  wings: z.array(z.object({
    code: z.string(), name: z.string(), source: z.enum(["labelled", "inferred"]), reason: z.string(),
    box: z.array(z.number()).describe("[x,y,w,h] area of the image this wing covers"),
  })),
  externalFeatures: z.array(z.object({ kind: z.string(), text: z.string() })),
  unreadable: z.array(z.string()),
});

interface Part<T> { name: string; schema: z.ZodType<T>; task: string }

const PART_OVERVIEW: Part<z.infer<typeof WireOverview>> = {
  name: "overview",
  schema: WireOverview,
  task: `TASK: return only the overview: drawing title, building, floor, wings, external features, unreadable items.
Wings: only for BIG buildings (mall, shopping centre, office complex, hospital, campus, large commercial or multi-block buildings) or when the plan itself labels wings/blocks.
- Labelled wings/blocks: use them (source "labelled").
- Big building without labels: infer 3-5 wings from the layout (source "inferred"). Name them by compass direction using the drawn north arrow (e.g. "North Wing", "East Wing", "Central"), not "left/right".
- Houses, flats, villas and other small buildings: return wings [] (no wings).
Give each wing a box [x,y,w,h] covering its area; boxes should together cover the building without overlapping much.
floor.level: entrance/ground level is 0 (a house's main floor is 0); floors above are 1, 2…; basements -1, -2….
building.suggestedCode: short, uppercase, human-friendly (e.g. "MALL-01", "RES-LAKEPOINT").
externalFeatures: roads, directions ("Towards Bus Stand"), retaining walls, parking, landscape.`,
};

const EVERY = "List EVERY item individually, even small or repeated ones; never summarise several items as one entry.";

/** Leasable units, split into two halves of the image so the two calls write in parallel. */
function unitsPart(half: "left" | "right", W: number): Part<z.infer<typeof WireSpaces>> {
  const mid = Math.round(W / 2);
  return {
    name: `units-${half}`,
    schema: WireSpaces,
    task: `TASK: return only LEASABLE UNITS (shops, showrooms, stores, kiosks, offices, flats, restaurants…) whose code label is in the ${half.toUpperCase()} half of the image (label centre x ${half === "left" ? "<" : ">="} ${mid} px), plus each one's sub-spaces (every LOFT/mezzanine as its own entry with t "loft", p = parent unit code). ${EVERY} Do not return shared spaces (lobby, passages, stairs, lifts, toilets, ducts, exits).`,
  };
}

const NOT_OTHERS = "Do not return leasable units, lofts, toilets, rooms or garages; other tasks return those.";

const PART_CIRCULATION: Part<z.infer<typeof WireSpaces>> = {
  name: "circulation",
  schema: WireSpaces,
  task: `TASK: return only INDOOR CIRCULATION: lobby, atrium, foyer, every passage/corridor/hall (printed width in wd), every staircase (internal), lift, fire lift, escalator, travelator. ${EVERY} ${NOT_OTHERS} Do not return entrances, exits, ramps, outside steps or outdoor spaces (another task returns those).`,
};

const PART_OUTDOOR: Part<z.infer<typeof WireSpaces>> = {
  name: "entrances-outdoor",
  schema: WireSpaces,
  task: `TASK: return only ENTRANCES, EXITS and OUTDOOR spaces: every entrance and emergency exit, every ramp, entrance/outside steps, every otla/verandah segment, porches, porticos, patios, decks, balconies, drop-offs, open parking, open spaces. ${EVERY} ${NOT_OTHERS} Do not return lobbies, passages, internal staircases or lifts.`,
};

const PART_ROOMS: Part<z.infer<typeof WireSpaces>> = {
  name: "rooms",
  schema: WireSpaces,
  task: `TASK: return only ROOMS that are not leasable units: ladies/gents/accessible toilet blocks, every W.C. cubicle, urinal, bath, washbasin area, every fire duct (t "fire_duct") and other shaft (t "shaft"), electrical/pump/fire/HVAC/storage/security rooms, garages. For a house or flat plan, also every room (kitchen, bedrooms, living, dining, bath, shower, closet, pantry, laundry, mud room, office, garage…). ${EVERY} Do not return shops/showrooms/stores, their lofts, or circulation/outdoor spaces (lobby, passages, halls, stairs, entrances, patios, decks, porches — another task returns those).`,
};

const COMMON_RULES = `You extract structured data from architectural drawings (floor plans, mall plans, building plans).

You get the drawing image and the text snippets OCR or the PDF text layer already read (id|text|x,y,w,h|confidence|rotation). Snippets may contain OCR mistakes ("SHOR-27" = "SHOP-27", truncated numbers, garbled quotes); trust the image when they disagree.

Rules for spaces:
- c: the plan's own code, normalised PREFIX-NUMBER ("SHOP-37", "SHOWROOM-3", "STORE-4", "STAIRCASE-2"). Merged units like "SHOP - 33,34": c "SHOP-33,34", m ["SHOP-33","SHOP-34"].
- mm / ft: only when a size label is printed for that space (e.g. "4025X6370" or "13'-6\\" X 15'-6\\"" written inside the room). Copy it exactly. Never compute sizes, never combine dimension-line values, never "fix" numbers; if no size label is printed, use null.
- s: ids of the snippets you used; [] if read only from the image (then lower k and give a rough box in b).
- t must be one of: ${JSON.stringify(SPACE_TYPE_GROUPS)}. Use "other" if nothing fits and keep the label in n.
- Ignore title blocks, legends, notes, 3D renderings and sheet borders as spaces.
- Keep n short. Output JSON only.`;

// ---------- main ----------

export async function extractWithAi(args: {
  png: Buffer; width: number; height: number; tokens: TextToken[]; model: string; apiKey: string;
  connection: AiConnection; effort: Effort | null; parallel: boolean;
}): Promise<AiResult> {
  assertAllowedModel(args.model);
  const direct = args.connection === "openai";
  if (direct && !args.model.startsWith("openai/")) {
    throw new Error(`AI_PROVIDER=openai can only run OpenAI models (openai/...), got "${args.model}"`);
  }
  const client = new OpenAI({
    apiKey: args.apiKey,
    ...(direct ? {} : { baseURL: "https://openrouter.ai/api/v1", defaultHeaders: { "X-Title": "planparse" } }),
    timeout: 15 * 60 * 1000,
  });

  // Downscale big pages; snippet boxes are rescaled to match what the model sees.
  const k = Math.min(1, MAX_IMAGE_EDGE / Math.max(args.width, args.height));
  const W = Math.round(args.width * k), H = Math.round(args.height * k);
  const img = await sharp(args.png).resize(W, H).jpeg({ quality: 88 }).toBuffer();
  const scaled = args.tokens.map((t) => ({ ...t, box: t.box.map((v) => Math.round(v * k)) as TextToken["box"] }));
  const tokenLines = scaled
    .map((t) => `${t.id}|${t.text.replace(/\|/g, "/")}|${t.box.join(",")}|${t.confidence.toFixed(2)}|r${t.rotation}`)
    .join("\n");

  const structured = direct || (await supportsStructuredOutputs(args.model));
  const ctx: CallCtx = {
    client, direct, structured, effort: args.effort, apiModel: direct ? args.model.slice("openai/".length) : args.model,
    // Same prefix for every call (system + image + snippets) so providers can cache it.
    prefix: [
      { role: "system", content: COMMON_RULES },
      {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: `data:image/jpeg;base64,${img.toString("base64")}` } },
          { type: "text", text: `Image size: ${W}x${H} px.\nSnippets (${scaled.length}):\n${tokenLines}` },
        ],
      },
    ],
  };

  const started = Date.now();
  let overview: z.infer<typeof WireOverview>;
  let wire: WireSpace[];
  let calls: CallResult<unknown>[];

  if (args.parallel) {
    const [o, ...rest] = await Promise.all([
      callPart(ctx, PART_OVERVIEW),
      callPart(ctx, unitsPart("left", W)),
      callPart(ctx, unitsPart("right", W)),
      callPart(ctx, PART_CIRCULATION),
      callPart(ctx, PART_OUTDOOR),
      callPart(ctx, PART_ROOMS),
    ]);
    overview = o.data; wire = rest.flatMap((r) => r.data.spaces); calls = [o, ...rest];
  } else {
    const all: Part<z.infer<typeof WireOverview> & z.infer<typeof WireSpaces>> = {
      name: "all",
      schema: WireOverview.extend(WireSpaces.shape),
      task: `${PART_OVERVIEW.task}\n\nAlso return EVERY space: leasable units with their lofts, and all shared/service spaces.`,
    };
    const r = await callPart(ctx, all);
    overview = r.data; wire = r.data.spaces; calls = [r];
  }

  const plan = expand(overview, wire, scaled, 1 / k);
  const price = await listPrice(args.model);
  const sum = (f: (c: CallResult<unknown>) => number) => calls.reduce((a, c) => a + f(c), 0);
  const inputTokens = sum((c) => c.stats.inputTokens), outputTokens = sum((c) => c.stats.outputTokens);
  const routerCost = calls.every((c) => c.cost != null) ? sum((c) => c.cost!) : null;
  return {
    plan, model: args.model, provider: direct ? "openai (direct)" : calls[0]!.provider,
    inputTokens, outputTokens, reasoningTokens: sum((c) => c.stats.reasoningTokens),
    // OpenAI does not return cost: estimate from list prices
    costUsd: direct ? (price ? inputTokens * price.input + outputTokens * price.output : null) : routerCost,
    attempts: Math.max(...calls.map((c) => c.stats.attempts)), ms: Date.now() - started,
    calls: calls.map((c) => c.stats), effort: args.effort,
  };
}

// ---------- one call ----------

interface CallCtx {
  client: OpenAI; direct: boolean; structured: boolean; effort: Effort | null; apiModel: string;
  prefix: OpenAI.Chat.ChatCompletionMessageParam[];
}
interface CallResult<T> { data: T; stats: CallStats; cost: number | null; provider: string | null }

async function callPart<T>(ctx: CallCtx, part: Part<T>): Promise<CallResult<T>> {
  const schema = strictJsonSchema(z.toJSONSchema(part.schema, { target: "draft-7" }) as JsonObj);
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    ...ctx.prefix,
    { role: "user", content: (ctx.structured ? "" : `Return ONLY JSON matching this schema:\n${JSON.stringify(schema)}\n\n`) + part.task },
  ];
  const started = Date.now();
  const stats: CallStats = { part: part.name, ms: 0, inputTokens: 0, outputTokens: 0, reasoningTokens: 0, attempts: 0 };
  let cost: number | null = 0, provider: string | null = null, lastError = "";
  let effort = ctx.effort;

  for (let attempt = 1; attempt <= 3; attempt++) {
    stats.attempts = attempt;
    const body = {
      model: ctx.apiModel,
      messages,
      ...(ctx.direct ? { max_completion_tokens: 32000 } : { max_tokens: 32000 }),
      ...(ctx.structured ? { response_format: { type: "json_schema", json_schema: { name: part.name, strict: true, schema } } } : {}),
      ...(effort ? (ctx.direct ? { reasoning_effort: effort } : { reasoning: { effort } }) : {}),
      ...(ctx.direct ? {} : {
        provider: { require_parameters: ctx.structured, data_collection: "deny", allow_fallbacks: true },
        usage: { include: true },
      }),
    };
    let res: OpenAI.Chat.ChatCompletion & { provider?: string; usage?: { cost?: number; completion_tokens_details?: { reasoning_tokens?: number } } };
    try {
      res = (await ctx.client.chat.completions.create(body as never)) as typeof res;
    } catch (e) {
      // Model does not accept this reasoning setting → retry once without it.
      if (effort && e instanceof OpenAI.BadRequestError && /reasoning/i.test(e.message)) {
        console.warn(`   [${part.name}] model rejected reasoning effort "${effort}", retrying without it`);
        effort = null;
        continue;
      }
      throw e;
    }
    stats.inputTokens += res.usage?.prompt_tokens ?? 0;
    stats.outputTokens += res.usage?.completion_tokens ?? 0;
    stats.reasoningTokens += res.usage?.completion_tokens_details?.reasoning_tokens ?? 0;
    if (cost != null) cost = res.usage?.cost != null ? cost + res.usage.cost : null;
    provider = res.provider ?? provider;

    const choice = res.choices[0];
    const text = choice?.message?.content ?? "";
    if (choice?.finish_reason === "length") {
      lastError = "Output was cut off (too long). Keep n short; keep every space.";
    } else {
      try {
        const parsed = part.schema.safeParse(JSON.parse(stripFences(text)));
        if (parsed.success) {
          stats.ms = Date.now() - started;
          return { data: parsed.data, stats, cost, provider };
        }
        lastError = z.prettifyError(parsed.error).slice(0, 4000);
      } catch (e) {
        lastError = `Not valid JSON: ${(e as Error).message}`;
      }
    }
    messages.push({ role: "assistant", content: text.slice(0, 20000) });
    messages.push({ role: "user", content: `Your previous answer was invalid:\n${lastError}\nReturn the complete corrected JSON.` });
  }
  throw new Error(`[${part.name}] AI output invalid after retries: ${lastError}`);
}

// ---------- expand compact output into the full plan ----------

function expand(o: z.infer<typeof WireOverview>, wire: WireSpace[], tokens: TextToken[], back: number): ExtractedPlan {
  const byId = new Map(tokens.map((t) => [t.id, t]));
  const all: ExtractedSpace[] = wire.map((w) => {
    const boxes = w.s.map((id) => byId.get(id)?.box).filter((b): b is TextToken["box"] => !!b);
    // position: from the snippets it was built from, else the AI's rough box
    const box = boxes.length ? unionBox(boxes) : w.b.length === 4 ? (w.b as [number, number, number, number]) : null;
    return {
      code: w.c, name: w.n, type: w.t, category: w.g, wingCode: box ? wingFor(box, o.wings) : null,
      parentCode: w.p, mergedCodes: w.m, dimsMmText: w.mm, dimsFtText: w.ft, widthText: w.wd,
      isRectangular: null,
      approxBox: box ? box.map((v) => Math.round(v * back)) : [],
      sourceTokenIds: w.s, confidence: w.k, notes: null,
    };
  });
  const spaces = dedupe(all);
  // a loft belongs to the same wing as its parent unit
  for (const s of spaces) {
    if (s.parentCode) s.wingCode = spaces.find((p) => p.code === s.parentCode)?.wingCode ?? s.wingCode;
  }
  return {
    drawingTitle: o.drawingTitle, building: o.building, floor: o.floor,
    wings: o.wings.map((w) => ({ ...w, box: w.box.map((v) => Math.round(v * back)) })),
    spaces, externalFeatures: o.externalFeatures, unreadable: o.unreadable,
  };
}

function unionBox(boxes: TextToken["box"][]): [number, number, number, number] {
  const x0 = Math.min(...boxes.map((b) => b[0])), y0 = Math.min(...boxes.map((b) => b[1]));
  const x1 = Math.max(...boxes.map((b) => b[0] + b[2])), y1 = Math.max(...boxes.map((b) => b[1] + b[3]));
  return [x0, y0, x1 - x0, y1 - y0];
}

/** Wing whose area contains the space's centre (nearest wing if none does). No wings → null. */
function wingFor(box: number[], wings: { code: string; box: number[] }[]): string | null {
  if (!wings.length) return null;
  const cx = box[0]! + box[2]! / 2, cy = box[1]! + box[3]! / 2;
  let best: string | null = null, bestD = Infinity;
  for (const w of wings) {
    const [x = 0, y = 0, ww = 0, hh = 0] = w.box;
    const dx = Math.max(x - cx, 0, cx - (x + ww)), dy = Math.max(y - cy, 0, cy - (y + hh));
    const d = Math.hypot(dx, dy);
    if (d < bestD) { bestD = d; best = w.code; }
  }
  return best;
}

/**
 * Parallel calls can both return a borderline space. Drop a space if an earlier one
 *  - has the same code (and same kind: unit / loft / shared type), or
 *  - has no code, the same name, and sits in the same place (two different
 *    "Bedroom"s in different places are both kept).
 */
function dedupe(spaces: ExtractedSpace[]): ExtractedSpace[] {
  const kept: ExtractedSpace[] = [];
  const codeKeys = new Set<string>();
  const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  for (const s of spaces) {
    if (s.code) {
      const key = `${s.code}|${s.type === "loft" ? "loft" : LEASABLE_TYPES.has(s.type as never) ? "unit" : s.type}`;
      if (codeKeys.has(key)) continue;
      codeKeys.add(key);
      kept.push(s);
      continue;
    }
    const dup = kept.some((k) => !k.code && norm(k.name) === norm(s.name) && samePlace(k, s));
    if (!dup) kept.push(s);
  }
  return kept;
}

function samePlace(a: ExtractedSpace, b: ExtractedSpace): boolean {
  if (a.sourceTokenIds.some((id) => b.sourceTokenIds.includes(id))) return true;
  if (a.approxBox.length !== 4 || b.approxBox.length !== 4) return true; // no position → trust the name
  const [ax, ay, aw, ah] = a.approxBox as [number, number, number, number];
  const [bx, by, bw, bh] = b.approxBox as [number, number, number, number];
  const ix = Math.max(0, Math.min(ax + aw, bx + bw) - Math.max(ax, bx));
  const iy = Math.max(0, Math.min(ay + ah, by + bh) - Math.max(ay, by));
  return (ix * iy) / (Math.min(aw * ah, bw * bh) || 1) > 0.3;
}

// ---------- helpers ----------

type ModelInfo = { id: string; supported_parameters?: string[]; pricing?: { prompt: string; completion: string } };
let modelList: Promise<ModelInfo[]> | null = null;
/** OpenRouter's public model list (no key needed): capabilities + list prices. */
function models(): Promise<ModelInfo[]> {
  modelList ??= fetch("https://openrouter.ai/api/v1/models")
    .then((r) => r.json() as Promise<{ data: ModelInfo[] }>)
    .then((j) => j.data)
    .catch(() => []);
  return modelList;
}
async function supportsStructuredOutputs(model: string): Promise<boolean> {
  return (await models()).find((m) => m.id === model)?.supported_parameters?.includes("structured_outputs") ?? false;
}
/** USD per token from list prices, to estimate direct OpenAI cost. */
async function listPrice(model: string): Promise<{ input: number; output: number } | null> {
  const p = (await models()).find((m) => m.id === model)?.pricing;
  return p ? { input: Number(p.prompt), output: Number(p.completion) } : null;
}

function stripFences(s: string): string {
  return s.trim().replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "");
}

type JsonObj = { [k: string]: unknown };
/** Strict JSON-schema mode needs additionalProperties:false and every property required. */
function strictJsonSchema(node: JsonObj): JsonObj {
  const walk = (n: unknown): unknown => {
    if (Array.isArray(n)) return n.map(walk);
    if (!n || typeof n !== "object") return n;
    const o: JsonObj = {};
    for (const [key, v] of Object.entries(n as JsonObj)) if (key !== "$schema") o[key] = walk(v);
    if (o.type === "object" && o.properties) {
      o.additionalProperties = false;
      o.required = Object.keys(o.properties as JsonObj);
    }
    return o;
  };
  return walk(node) as JsonObj;
}

// Phase 0 runner.
//   pnpm phase0 <file> [--page N] [--truth answer-key.json] [--model vendor/model] [--no-ai] [--no-rotations]
// Writes everything to out/<file-name>[-pN]/ : tokens.json, rules.json, ai.json, issues.json, report.md

import { mkdir, writeFile } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { config as loadEnv } from "dotenv";
import { areaSqm, type ExtractedSpace, type TextToken } from "@planparse/shared";
import { loadPages, describePdf } from "./ingest.js";
import { ocrPage, overlap } from "./ocr.js";
import { classify, rulesBaseline } from "./parse.js";
import { extractWithAi, assertAllowedModel, type AiResult, type AiConnection, type Effort } from "./ai.js";
import { validatePlan, type Issue } from "./validate.js";
import { loadTruth, score, type Score } from "./evaluate.js";

// .env lives at the repo root; pnpm runs us from tools/phase0.
const ROOT = resolve(import.meta.dirname, "../../..");
loadEnv({ path: resolve(ROOT, ".env") });

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    page: { type: "string" },
    truth: { type: "string" },
    model: { type: "string" },
    "no-ai": { type: "boolean", default: false },
    "no-rotations": { type: "boolean", default: false },
    "list-pages": { type: "boolean", default: false },
    effort: { type: "string" },                       // minimal | low | medium | high | none
    single: { type: "boolean", default: false },     // one AI call instead of 5 parallel calls
  },
});

const fileArg = positionals[0];
if (!fileArg) {
  console.error("usage: pnpm phase0 <plan.png|plan.pdf> [--page N] [--truth key.json] [--model vendor/model] [--no-ai] [--no-rotations] [--list-pages]");
  process.exit(1);
}
const file = resolve(process.env.INIT_CWD ?? process.cwd(), fileArg);
const truthPath = values.truth ? resolve(process.env.INIT_CWD ?? process.cwd(), values.truth) : null;

if (values["list-pages"]) {
  for (const p of await describePdf(file)) console.log(`page ${p.page}: ${p.textItems} text items — ${p.sample.slice(0, 120)}`);
  process.exit(0);
}

const pageNo = values.page ? Number(values.page) : undefined;
const outDir = resolve(ROOT, "out", basename(file, extname(file)) + (pageNo ? `-p${pageNo}` : ""));
await mkdir(outDir, { recursive: true });
const t = (ms: number) => `${(ms / 1000).toFixed(1)}s`;
const timings: Record<string, number> = {};
const runStart = Date.now();

// 1. load
let start = Date.now();
const [page] = await loadPages(file, pageNo ? [pageNo] : [1]);
if (!page) throw new Error("no page");
await writeFile(resolve(outDir, "page.png"), page.png);
timings.load = Date.now() - start;
console.log(`1. loaded ${page.kind} ${page.width}x${page.height} (${t(Date.now() - start)})`);

// 2. text: PDF text layer if it is a CAD PDF, else OCR
start = Date.now();
let tokens: TextToken[];
if (page.kind === "pdf-vector") {
  tokens = page.pdfTokens;
  console.log(`2. read ${tokens.length} snippets from the PDF text layer (no OCR needed) (${t(Date.now() - start)})`);
} else {
  const ocr = await ocrPage(page.png, page.width, page.height, { rotations: !values["no-rotations"] });
  // pdf-mixed: keep the exact PDF text, add OCR only where the PDF has none
  const fromPdf = page.pdfTokens;
  const extra = ocr.filter((o) => !fromPdf.some((p) => overlap(p.box, o.box) > 0.3));
  tokens = [...fromPdf, ...extra];
  console.log(`2. OCR found ${ocr.length} snippets${fromPdf.length ? `, plus ${fromPdf.length} from the PDF text layer` : ""} (${t(Date.now() - start)})`);
}
timings.text = Date.now() - start;
const classified = classify(tokens);
await writeFile(resolve(outDir, "tokens.json"), JSON.stringify(classified, null, 1));

// 3. rules-only baseline
const rules = rulesBaseline(classified);
const rulesIssues = validatePlan({ spaces: rules, wings: [] }, tokens);
await writeFile(resolve(outDir, "rules.json"), JSON.stringify(rules, null, 1));
console.log(`3. rules-only: ${rules.filter((s) => s.code).length} coded units, ${rules.filter((s) => !s.code).length} other spaces`);

// 4. AI
let ai: AiResult | null = null;
let aiIssues: Issue[] = [];
let aiError: string | null = null;
// AI_PROVIDER picks the connection: "openrouter" (default) or "openai" (direct OpenAI key).
const connection = (process.env.AI_PROVIDER || "openrouter") as AiConnection;
if (connection !== "openrouter" && connection !== "openai") {
  throw new Error(`AI_PROVIDER must be "openrouter" or "openai", got "${connection}"`);
}
const keyName = connection === "openai" ? "OPENAI_API_KEY" : "OPENROUTER_API_KEY";
const apiKey = process.env[keyName];
const model = values.model
  ?? (connection === "openai"
    ? process.env.OPENAI_MODEL || "openai/gpt-6.2-sol"
    : process.env.OPENROUTER_MODEL || "anthropic/claude-opus-5.5");
if (values["no-ai"]) {
  console.log("4. AI skipped (--no-ai)");
} else if (!apiKey) {
  console.log(`4. AI skipped: ${keyName} is not set in .env`);
} else {
  assertAllowedModel(model);
  console.log(`4. AI: calling ${model} via ${connection === "openai" ? "OpenAI (direct)" : "OpenRouter"}…`);
  try {
    const effortArg = values.effort ?? process.env.AI_REASONING_EFFORT ?? "low";
    ai = await extractWithAi({
      png: page.png, width: page.width, height: page.height, tokens, model, apiKey, connection,
      effort: effortArg === "none" ? null : (effortArg as Effort),
      parallel: !values.single,
    });
    aiIssues = validatePlan(ai.plan, tokens);
    await writeFile(resolve(outDir, "ai.json"), JSON.stringify(ai, null, 1));
    await writeFile(resolve(outDir, "issues.json"), JSON.stringify(aiIssues, null, 1));
    console.log(`   ${ai.plan.spaces.length} spaces, ${ai.plan.wings.length} wings, ${ai.inputTokens} in / ${ai.outputTokens} out tokens, ` +
      `cost ${ai.costUsd != null ? "$" + ai.costUsd.toFixed(4) : "n/a"}, ${t(ai.ms)}, provider ${ai.provider ?? "?"}`);
  } catch (e) {
    aiError = (e as Error).message;
    console.log(`   AI failed: ${aiError}`);
  }
}

timings.ai = ai?.ms ?? 0;
timings.total = Date.now() - runStart;

// 5. score against answer key
let rulesScore: Score | null = null, aiScore: Score | null = null;
if (truthPath) {
  const truth = await loadTruth(truthPath);
  rulesScore = score(rules, truth);
  if (ai) aiScore = score(ai.plan.spaces, truth);
}

// 6. report
const report = buildReport();
await writeFile(resolve(outDir, "report.md"), report);
console.log(`\n${report}\nwritten to ${outDir}`);

function buildReport(): string {
  const L: string[] = [`# Phase 0 report: ${basename(file)}${pageNo ? ` page ${pageNo}` : ""}`, ""];
  L.push(`- Input: ${page!.kind}, ${page!.width}x${page!.height}px`);
  L.push(`- Text source: ${page!.kind === "pdf-vector" ? "PDF text layer" : page!.kind === "pdf-mixed" ? "PDF text layer + Tesseract OCR" : "Tesseract OCR"}, ${tokens.length} snippets`);
  if (ai) {
    L.push(`- AI: ${ai.model} (provider ${ai.provider ?? "?"}, effort ${ai.effort ?? "default"}), ${ai.inputTokens} in / ${ai.outputTokens} out tokens (${ai.reasoningTokens} reasoning), cost ${ai.costUsd != null ? "$" + ai.costUsd.toFixed(4) : "n/a"}, wall time ${t(ai.ms)}`);
    for (const c of ai.calls) L.push(`  - call "${c.part}": ${t(c.ms)}, ${c.inputTokens} in / ${c.outputTokens} out (${c.reasoningTokens} reasoning), attempts ${c.attempts}`);
  }
  L.push(`- Timings: ${Object.entries(timings).map(([k, v]) => `${k} ${t(v)}`).join(", ")}`);
  if (aiError) L.push(`- AI failed: ${aiError}`);
  L.push("");

  if (rulesScore) {
    L.push("## Accuracy against the answer key", "");
    L.push("| Metric | OCR + rules | OCR + AI |", "|---|---|---|");
    const row = (name: string, f: (s: Score) => string) => L.push(`| ${name} | ${f(rulesScore!)} | ${aiScore ? f(aiScore) : "—"} |`);
    row("Units found (recall)", (s) => `${s.unitsFound}/${s.unitsExpected} (${pct(s.unitRecall)})`);
    row("Wrong/extra unit codes", (s) => `${s.unitsExtra.length}${s.unitsExtra.length ? ": " + s.unitsExtra.join(", ") : ""}`);
    row("Type correct", (s) => frac(s.typeCorrect, s.unitsFound));
    row("Category correct", (s) => frac(s.categoryCorrect, s.unitsFound));
    row("mm dimensions correct", (s) => frac(s.dimsMmCorrect, s.unitsFound));
    row("ft dimensions correct", (s) => frac(s.dimsFtCorrect, s.unitsFound));
    row("Loft linked correctly", (s) => frac(s.loftCorrect, s.unitsFound));
    row("All fields (found units)", (s) => pct(s.fieldAccuracy));
    const amen = (s: Score) => {
      const e = Object.values(s.amenityRecall).reduce((a, b) => a + b.expected, 0);
      const f = Object.values(s.amenityRecall).reduce((a, b) => a + b.found, 0);
      return `${f}/${e} (${pct(f / e)})`;
    };
    row("Shared spaces found", amen);
    L.push("");
    if (aiScore?.unitsMissing.length) L.push(`AI missed: ${aiScore.unitsMissing.join(", ")}`, "");
    if (aiScore) {
      L.push("Shared spaces by type (AI): " + Object.entries(aiScore.amenityRecall).map(([k, v]) => `${k} ${v.found}/${v.expected}`).join(", "), "");
    }
  }

  const spaces: ExtractedSpace[] = ai ? ai.plan.spaces : rules;
  if (ai) {
    L.push("## Building / floor / wings", "");
    L.push(`- Drawing title: ${ai.plan.drawingTitle ?? "—"}`);
    L.push(`- Building: ${ai.plan.building.suggestedCode} — ${ai.plan.building.name ?? "unnamed"} (${ai.plan.building.type}), north: ${ai.plan.building.northDirection ?? "?"}`);
    L.push(`- Floor: ${ai.plan.floor.name ?? "?"} (level ${ai.plan.floor.level ?? "?"})`);
    for (const w of ai.plan.wings) L.push(`- Wing ${w.code} "${w.name}" [${w.source}] — ${w.reason}`);
    if (ai.plan.externalFeatures.length) L.push(`- External: ${ai.plan.externalFeatures.map((f) => `${f.kind}: ${f.text}`).join("; ")}`);
    if (ai.plan.unreadable.length) L.push(`- Unreadable: ${ai.plan.unreadable.join("; ")}`);
    L.push("");
  }

  const issues = ai ? aiIssues : rulesIssues;
  L.push(`## Validation issues (${ai ? "AI" : "rules"} result): ${issues.filter((i) => i.severity === "error").length} errors, ${issues.filter((i) => i.severity === "warning").length} warnings`, "");
  for (const i of issues.filter((x) => x.severity !== "info").slice(0, 60)) L.push(`- **${i.severity}** ${i.spaceCode ?? ""} [${i.rule}] ${i.message}`);
  L.push("");

  L.push(`## Extracted spaces (${spaces.length}, ${ai ? "AI" : "rules-only"})`, "");
  L.push("| Code | Name | Type | Cat. | Wing | Parent | mm | ft | m² | Src | Conf |", "|---|---|---|---|---|---|---|---|---|---|---|");
  for (const s of spaces) {
    L.push(`| ${s.code ?? ""} | ${s.name} | ${s.type} | ${s.category ?? ""} | ${s.wingCode ?? ""} | ${s.parentCode ?? ""} | ${s.dimsMmText ?? ""} | ${(s.dimsFtText ?? "").replace(/\|/g, "/")} | ${areaSqm(s.dimsMmText) ?? ""} | ${s.sourceTokenIds.length} | ${s.confidence.toFixed(2)} |`);
  }
  return L.join("\n");
}

function pct(x: number) { return `${Math.round(x * 100)}%`; }
function frac(a: number, b: number) { return b ? `${a}/${b} (${pct(a / b)})` : "—"; }

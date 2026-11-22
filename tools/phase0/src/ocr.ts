// OCR with Tesseract.js (open source, runs locally). Plans have small text,
// coloured fills and rotated labels, so we:
//   1. upscale + grayscale + boost contrast
//   2. use "sparse text" mode (labels are scattered, not paragraphs)
//   3. also OCR the image rotated 90° and 270° to catch vertical labels

import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import sharp from "sharp";
import { createWorker, PSM, type Worker } from "tesseract.js";
import type { TextToken } from "@planparse/shared";

const TARGET_LONG_EDGE = Number(process.env.OCR_LONG_EDGE ?? 5400); // px after upscaling
const MIN_CONF = 0.35;         // drop very low-confidence lines
const MIN_CONF_ROTATED = 0.6;  // rotated passes produce more junk; be stricter

type Box = [number, number, number, number];

export interface OcrOptions { rotations: boolean }

export async function ocrPage(png: Buffer, width: number, height: number, opts: OcrOptions): Promise<TextToken[]> {
  const scale = Math.min(3, Math.max(1, TARGET_LONG_EDGE / Math.max(width, height)));
  const W = Math.round(width * scale);
  const H = Math.round(height * scale);

  const prepared = await sharp(png)
    .flatten({ background: "#ffffff" })
    .resize(W, H, { kernel: "lanczos3" })
    .grayscale()
    .normalise()
    .sharpen()
    .png()
    .toBuffer();

  const cachePath = resolve(process.cwd(), ".cache/tesseract");
  await mkdir(cachePath, { recursive: true });

  // The three passes (upright, 90°, 270°) run at the same time, one worker each.
  const images = [prepared];
  if (opts.rotations) {
    images.push(
      await sharp(prepared).rotate(90).png().toBuffer(),
      await sharp(prepared).rotate(270).png().toBuffer(),
    );
  }
  const [upright = [], r90 = [], r270 = []] = await Promise.all(images.map(async (img) => {
    const worker = await createWorker("eng", 1, { cachePath });
    try {
      await worker.setParameters({ tessedit_pageseg_mode: PSM.SPARSE_TEXT, preserve_interword_spaces: "1" });
      return await readLines(worker, img);
    } finally {
      await worker.terminate();
    }
  }));

  const tokens: TextToken[] = [];
  const push = (text: string, box: Box, conf: number, rotation: number) => {
    // back to page-pixel coordinates
    const b = box.map((v) => Math.round(v / scale)) as Box;
    tokens.push({ id: `t${tokens.length}`, text, box: b, confidence: conf, rotation, source: "ocr" });
  };
  for (const l of upright) if (l.conf >= MIN_CONF) push(l.text, l.box, l.conf, 0);
  // 90° clockwise: rotated(rx,ry) -> original(x = ry, y = H - rx)
  for (const l of r90) {
    const [rx, ry, rw, rh] = l.box;
    const box: Box = [ry, H - (rx + rw), rh, rw];
    if (keepRotated(l, box, tokens, scale)) push(l.text, box, l.conf, 90);
  }
  // 270° clockwise: rotated(rx,ry) -> original(x = W - ry, y = rx)
  for (const l of r270) {
    const [rx, ry, rw, rh] = l.box;
    const box: Box = [W - (ry + rh), rx, rh, rw];
    if (keepRotated(l, box, tokens, scale)) push(l.text, box, l.conf, 270);
  }
  return tokens;
}

interface Line { text: string; box: Box; conf: number }

async function readLines(worker: Worker, img: Buffer): Promise<Line[]> {
  const { data } = await worker.recognize(img, {}, { blocks: true, text: false, hocr: false, tsv: false });
  const out: Line[] = [];
  for (const block of data.blocks ?? []) {
    for (const para of block.paragraphs) {
      for (const line of para.lines) {
        // Tesseract often glues neighbouring labels into one line
        // ("SHOP-29 SHOP-28"). Split wherever the gap between words is
        // wider than the text height.
        let group: typeof line.words = [];
        const flush = () => {
          const words = group.filter((w) => /[A-Za-z0-9]/.test(w.text));
          group = [];
          if (!words.length) return;
          const text = words.map((w) => w.text).join(" ").trim();
          const x0 = Math.min(...words.map((w) => w.bbox.x0));
          const y0 = Math.min(...words.map((w) => w.bbox.y0));
          const x1 = Math.max(...words.map((w) => w.bbox.x1));
          const y1 = Math.max(...words.map((w) => w.bbox.y1));
          const conf = words.reduce((s, w) => s + w.confidence, 0) / words.length / 100;
          out.push({ text, box: [x0, y0, x1 - x0, y1 - y0], conf });
        };
        for (const w of line.words) {
          const prev = group[group.length - 1];
          const h = Math.max(w.bbox.y1 - w.bbox.y0, 1);
          // junk separators ("§", "|", "J") that Tesseract reads from wall lines
          const isSeparator = /^[§|¦]+$/.test(w.text) || /^(J|Jf|jj|j§|§f|f§|\|\|)$/.test(w.text);
          if (isSeparator) { flush(); continue; }
          if (prev && w.bbox.x0 - prev.bbox.x1 > h * 1.2) flush();
          group.push(w);
        }
        flush();
      }
    }
  }
  return out;
}

/** Keep a rotated-pass line only if it is confident, real-looking, and not already found upright. */
function keepRotated(l: Line, boxScaled: Box, existing: TextToken[], scale: number): boolean {
  if (l.conf < MIN_CONF_ROTATED) return false;
  if (l.text.replace(/[^A-Za-z0-9]/g, "").length < 3) return false;
  const b = boxScaled.map((v) => v / scale) as Box;
  return !existing.some((t) => overlap(t.box, b) > 0.3);
}

/** Intersection over the smaller box's area. */
export function overlap(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const small = Math.min(a[2] * a[3], b[2] * b[3]) || 1;
  return inter / small;
}

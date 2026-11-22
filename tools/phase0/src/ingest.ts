// Turn an input file (image or PDF) into page images, plus the PDF's own
// text layer when it has one. CAD-exported PDFs carry real text, which is
// exact and free; scanned PDFs and images go through OCR instead.

import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import sharp from "sharp";
import type { TextToken } from "@planparse/shared";

export interface PageInput {
  index: number;          // 1-based page number
  png: Buffer;            // rendered page, white background
  width: number;
  height: number;
  pdfTokens: TextToken[]; // empty for images and scanned PDFs
  /** pdf-vector: real text, no OCR. pdf-mixed: some text + a big embedded picture → OCR too. pdf-raster: scan. */
  kind: "image" | "pdf-vector" | "pdf-mixed" | "pdf-raster";
}

const IMAGE_EXT = new Set([".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"]);
/** A page with at least this many real text snippets is treated as a CAD/vector PDF. */
const VECTOR_TEXT_MIN = 15;
/** Pages with an embedded picture and fewer text snippets than this are OCR-ed as well
 *  (e.g. a scanned plan inside a PDF with only a typed header/footer). */
const MIXED_TEXT_MAX = 150;
/** Render PDFs so the long edge is about this many pixels. */
const PDF_TARGET_LONG_EDGE = 4000;

export async function loadPages(file: string, pages?: number[]): Promise<PageInput[]> {
  const ext = extname(file).toLowerCase();
  if (IMAGE_EXT.has(ext)) return [await loadImage(file)];
  if (ext === ".pdf") return loadPdf(file, pages);
  throw new Error(`Unsupported file type: ${ext}`);
}

async function loadImage(file: string): Promise<PageInput> {
  const { data, info } = await sharp(await readFile(file))
    .rotate() // respect EXIF orientation
    .flatten({ background: "#ffffff" })
    .png()
    .toBuffer({ resolveWithObject: true });
  return { index: 1, png: data, width: info.width, height: info.height, pdfTokens: [], kind: "image" };
}

async function loadPdf(file: string, wanted?: number[]): Promise<PageInput[]> {
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const task = pdfjs.getDocument({
    data: new Uint8Array(await readFile(file)),
    verbosity: 0,
  });
  const doc = await task.promise;

  const pageNumbers = wanted?.length ? wanted : Array.from({ length: doc.numPages }, (_, i) => i + 1);
  const out: PageInput[] = [];

  for (const n of pageNumbers) {
    if (n < 1 || n > doc.numPages) throw new Error(`Page ${n} out of range (1..${doc.numPages})`);
    const page = await doc.getPage(n);
    const base = page.getViewport({ scale: 1 });
    const scale = Math.min(4, PDF_TARGET_LONG_EDGE / Math.max(base.width, base.height));
    const viewport = page.getViewport({ scale });
    const width = Math.ceil(viewport.width);
    const height = Math.ceil(viewport.height);

    // Render with the canvas implementation pdf.js bundles for Node.
    const factory = (doc as unknown as { canvasFactory: CanvasFactory }).canvasFactory;
    const { canvas, context } = factory.create(width, height);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    await page.render({ canvas, canvasContext: context, viewport } as never).promise;
    const png = canvas.toBuffer("image/png");

    // Text layer → tokens in rendered-pixel coordinates.
    const content = await page.getTextContent();
    const tokens: TextToken[] = [];
    for (const item of content.items) {
      if (!("str" in item) || !item.str.trim()) continue;
      const tx = pdfjs.Util.transform(viewport.transform, item.transform) as number[];
      const [a = 1, b = 0, c = 0, d = 1, e = 0, f = 0] = tx;
      const fontH = Math.hypot(c, d);
      const angle = Math.round((Math.atan2(b, a) * 180) / Math.PI);
      const rotation = ((Math.round(angle / 90) * 90) % 360 + 360) % 360;
      const len = item.width * scale;
      // Approximate an axis-aligned box from the baseline origin.
      let box: [number, number, number, number];
      if (rotation === 0) box = [e, f - fontH, len, fontH];
      else if (rotation === 270) box = [e - fontH, f - len, fontH, len];   // text runs upward
      else if (rotation === 90) box = [e, f, fontH, len];
      else box = [e - len, f, len, fontH];
      tokens.push({
        id: `p${tokens.length}`,
        text: item.str.trim(),
        box: box.map((v) => Math.round(v)) as [number, number, number, number],
        confidence: 1,
        rotation,
        source: "pdf",
      });
    }

    const ops = await page.getOperatorList();
    const hasImage = ops.fnArray.some((fn) => fn === pdfjs.OPS.paintImageXObject || fn === pdfjs.OPS.paintInlineImageXObject);
    const kind: PageInput["kind"] =
      tokens.length < VECTOR_TEXT_MIN ? "pdf-raster"
      : hasImage && tokens.length < MIXED_TEXT_MAX ? "pdf-mixed"
      : "pdf-vector";
    out.push({ index: n, png, width, height, pdfTokens: tokens, kind });
    page.cleanup();
  }
  await task.destroy();
  return out;
}

interface CanvasFactory {
  create(w: number, h: number): {
    canvas: { toBuffer(mime: "image/png"): Buffer };
    context: { fillStyle: string; fillRect(x: number, y: number, w: number, h: number): void };
  };
}

/** Quick per-page summary without rendering, used to pick pages from big PDF sets. */
export async function describePdf(file: string): Promise<{ page: number; textItems: number; sample: string }[]> {
  const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
  const task = pdfjs.getDocument({ data: new Uint8Array(await readFile(file)), verbosity: 0 });
  const doc = await task.promise;
  const res = [];
  for (let n = 1; n <= doc.numPages; n++) {
    const page = await doc.getPage(n);
    const content = await page.getTextContent();
    const strs = content.items.flatMap((i) => ("str" in i && i.str.trim() ? [i.str.trim()] : []));
    res.push({ page: n, textItems: strs.length, sample: strs.slice(0, 12).join(" | ") });
    page.cleanup();
  }
  await task.destroy();
  return res;
}

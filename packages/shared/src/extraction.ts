import { z } from "zod";
import { SPACE_TYPES } from "./spaceTypes.js";

/**
 * What the AI must return for one plan.
 * Every value should be traceable to text snippets (sourceTokenIds) found by
 * OCR or read from the PDF. Values with no source get flagged for review.
 *
 * Kept deliberately flat and nullable (no optional fields) so it works with
 * strict JSON-schema output modes.
 */

export const SpaceTypeSchema = z.enum(SPACE_TYPES as [string, ...string[]]);

export const ExtractedWingSchema = z.object({
  code: z.string().describe("Short code, e.g. 'N', 'W', 'A'"),
  name: z.string().describe("e.g. 'North Wing', 'A Wing'"),
  source: z.enum(["labelled", "inferred"]),
  reason: z.string().describe("Why this wing exists / how it was inferred"),
  box: z.array(z.number()).describe("[x, y, w, h] image-pixel area the wing covers; spaces are assigned to wings by position"),
});

export const ExtractedSpaceSchema = z.object({
  code: z.string().nullable().describe("Normalised id, e.g. 'SHOP-37', 'SHOWROOM-3', 'STORE-4', 'STAIRCASE-2'. null if the plan gives none"),
  name: z.string().describe("Human label, e.g. 'Economy Shop 37', 'Gents Toilet', '7 ft wide passage'"),
  type: SpaceTypeSchema,
  category: z.string().nullable().describe("Grade/category printed on the plan, e.g. 'economy', 'premium', 'deluxe'"),
  wingCode: z.string().nullable(),
  parentCode: z.string().nullable().describe("For sub-spaces (loft inside a showroom, wc inside a toilet): the parent's code"),
  mergedCodes: z.array(z.string()).describe("For merged units like 'SHOP - 33,34': ['SHOP-33','SHOP-34']; else []"),
  dimsMmText: z.string().nullable().describe("Exactly as printed, e.g. '4025X6370'"),
  dimsFtText: z.string().nullable().describe("Exactly as printed, e.g. \"13'-3\\\"X21'\""),
  widthText: z.string().nullable().describe("For passages/corridors: printed width, e.g. \"7'\""),
  isRectangular: z.boolean().nullable().describe("false if the drawn shape is clearly not a rectangle"),
  approxBox: z.array(z.number()).describe("[x, y, w, h] in image pixels, approximate; [] if unknown"),
  sourceTokenIds: z.array(z.string()).describe("ids of the text snippets this space was built from"),
  confidence: z.number().describe("0..1"),
  notes: z.string().nullable(),
});

export const ExtractedPlanSchema = z.object({
  drawingTitle: z.string().nullable(),
  building: z.object({
    suggestedCode: z.string().describe("Short, human-friendly, e.g. 'MALL-KOP-01'"),
    name: z.string().nullable(),
    type: z.string().describe("mall | office | residential | hospital | mixed | other"),
    northDirection: z.string().nullable().describe("e.g. 'up-left' as drawn on the compass"),
  }),
  floor: z.object({
    level: z.number().nullable().describe("-1 basement, 0 ground, 1 first ..."),
    name: z.string().nullable(),
  }),
  wings: z.array(ExtractedWingSchema),
  spaces: z.array(ExtractedSpaceSchema),
  externalFeatures: z.array(z.object({
    kind: z.string().describe("road | landmark_direction | retaining_wall | parking | landscape | other"),
    text: z.string(),
  })),
  unreadable: z.array(z.string()).describe("Things the model could see but not read reliably"),
});

export type ExtractedWing = z.infer<typeof ExtractedWingSchema>;
export type ExtractedSpace = z.infer<typeof ExtractedSpaceSchema>;
export type ExtractedPlan = z.infer<typeof ExtractedPlanSchema>;

/** One text snippet found on the drawing (from OCR or the PDF text layer). */
export const TextTokenSchema = z.object({
  id: z.string(),
  text: z.string(),
  box: z.tuple([z.number(), z.number(), z.number(), z.number()]), // x, y, w, h in page pixels
  confidence: z.number(), // 0..1 (1 for PDF text)
  rotation: z.number(), // 0 | 90 | 270 degrees the text was found at
  source: z.enum(["ocr", "pdf"]),
});
export type TextToken = z.infer<typeof TextTokenSchema>;

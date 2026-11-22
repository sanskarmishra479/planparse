// Downloads the public sample plans into samples/downloaded/ (git-ignored).
// They are not committed because they belong to their publishers.
//   pnpm samples

import { mkdir, writeFile, access } from "node:fs/promises";
import { resolve } from "node:path";

const SAMPLES: { file: string; url: string; note: string }[] = [
  {
    file: "lake-point.pdf",
    url: "https://cloud.chiefarchitect.com/1/samples/projects/lake-point/lake-point.pdf",
    note: "Chief Architect demo plan set (CAD PDF, page 6 = main floor)",
  },
  {
    file: "maricopa-sample.pdf",
    url: "https://www.maricopa.gov/DocumentCenter/View/7469/Sample-Floor-Plan-PDF",
    note: "Maricopa County sample floor plan (PDF with an embedded plan picture)",
  },
  {
    file: "wiki-sample-floorplan.jpg",
    url: "https://commons.wikimedia.org/wiki/Special:FilePath/Sample_Floorplan.jpg",
    note: "Wikimedia Commons Sample_Floorplan.jpg (house plan)",
  },
  {
    file: "whitehouse-1900.jpg",
    url: "https://commons.wikimedia.org/wiki/Special:FilePath/State_floor_plan_-_White_House_-_1900.jpg",
    note: "White House main floor plan, 1900 (public domain scan)",
  },
];

const dir = resolve(import.meta.dirname, "../../../samples/downloaded");
await mkdir(dir, { recursive: true });

for (const s of SAMPLES) {
  const dest = resolve(dir, s.file);
  if (await access(dest).then(() => true, () => false)) {
    console.log(`✓ ${s.file} (already downloaded)`);
    continue;
  }
  const res = await fetch(s.url, { headers: { "User-Agent": "planparse-sample-fetch" }, redirect: "follow" });
  if (!res.ok) {
    console.error(`✗ ${s.file}: HTTP ${res.status} from ${s.url}`);
    continue;
  }
  await writeFile(dest, Buffer.from(await res.arrayBuffer()));
  console.log(`↓ ${s.file} — ${s.note}`);
}

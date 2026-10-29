import { promises as fs } from "fs";
import path from "path";

const dataDir = path.join(process.cwd(), "public", "data");

export async function readJson<T>(fileName: string, fallback: T): Promise<T> {
  try {
    const file = await fs.readFile(path.join(dataDir, fileName), "utf8");
    return JSON.parse(file) as T;
  } catch {
    return fallback;
  }
}

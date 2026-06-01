import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";

const csvPath = path.resolve("../rag_backup/medium-english-50mb.csv");

if (!fs.existsSync(csvPath)) {
  console.error("CSV file not found at:", csvPath);
  process.exit(1);
}

const fileContent = fs.readFileSync(csvPath, "utf8");

const records = parse(fileContent, {
  columns: true,
  skip_empty_lines: true
});

console.log("Total articles:", records.length);

console.log("\nFirst article preview:");
console.log({
  title: records[0].title,
  authors: records[0].authors,
  tags: records[0].tags,
  text_preview: records[0].text?.slice(0, 300)
});
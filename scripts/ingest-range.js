import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { parse } from "csv-parse/sync";
import { Pinecone } from "@pinecone-database/pinecone";

dotenv.config({ path: ".env.local" });

const CSV_PATH = path.resolve("../rag_backup/medium-english-50mb.csv");

const CHUNK_SIZE_CHARS = 2000;
const OVERLAP_CHARS = 400;
const BATCH_SIZE = 50;

const startArticle = Number(process.argv[2]);
const numArticles = Number(process.argv[3]);

if (!Number.isInteger(startArticle) || startArticle < 0) {
  console.error("Usage: node scripts/ingest-range.js <startArticle> <numArticles>");
  console.error("Example: node scripts/ingest-range.js 100 500");
  process.exit(1);
}

if (!Number.isInteger(numArticles) || numArticles <= 0) {
  console.error("Usage: node scripts/ingest-range.js <startArticle> <numArticles>");
  console.error("Example: node scripts/ingest-range.js 100 500");
  process.exit(1);
}

if (!process.env.LLMOD_API_KEY) {
  console.error("Missing LLMOD_API_KEY in .env.local");
  process.exit(1);
}

if (!process.env.LLMOD_BASE_URL) {
  console.error("Missing LLMOD_BASE_URL in .env.local");
  process.exit(1);
}

if (!process.env.PINECONE_API_KEY) {
  console.error("Missing PINECONE_API_KEY in .env.local");
  process.exit(1);
}

if (!process.env.PINECONE_INDEX_NAME) {
  console.error("Missing PINECONE_INDEX_NAME in .env.local");
  process.exit(1);
}

function chunkText(text) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + CHUNK_SIZE_CHARS, text.length);
    const chunk = text.slice(start, end).trim();

    if (chunk.length > 0) {
      chunks.push(chunk);
    }

    if (end === text.length) break;
    start = end - OVERLAP_CHARS;
  }

  return chunks;
}

async function embedText(text) {
  const response = await fetch(`${process.env.LLMOD_BASE_URL}/v1/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.LLMOD_API_KEY}`,
    },
    body: JSON.stringify({
      model: "4UHRUIN-text-embedding-3-small",
      input: text,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    console.error("Embedding request failed:", data);
    throw new Error("Embedding request failed");
  }

  return data.data[0].embedding;
}

async function upsertBatch(index, vectors, batchNumber) {
  if (vectors.length === 0) return;

  await index.upsert({
    records: vectors,
  });

  console.log(`Upserted batch ${batchNumber} with ${vectors.length} vectors`);
}

async function main() {
  if (!fs.existsSync(CSV_PATH)) {
    console.error("CSV file not found at:", CSV_PATH);
    process.exit(1);
  }

  const fileContent = fs.readFileSync(CSV_PATH, "utf8");
  const records = parse(fileContent, {
    columns: true,
    skip_empty_lines: true,
  });

  const endArticleExclusive = Math.min(startArticle + numArticles, records.length);
  const selectedArticles = records.slice(startArticle, endArticleExclusive);

  console.log(`Loaded ${records.length} articles.`);
  console.log(`Using articles ${startArticle} to ${endArticleExclusive - 1}.`);
  console.log(`Selected ${selectedArticles.length} articles.`);

  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

  let batch = [];
  let batchNumber = 1;
  let totalVectors = 0;

  for (let articleIndex = 0; articleIndex < selectedArticles.length; articleIndex++) {
    const article = selectedArticles[articleIndex];
    const originalArticleIndex = startArticle + articleIndex;

    const articleId = String(originalArticleIndex);
    const title = article.title || "";
    const authors = article.authors || "";
    const url = article.url || "";
    const timestamp = article.timestamp || "";
    const tags = article.tags || "";
    const text = article.text || "";

    const chunks = chunkText(text);

    console.log(
      `Article ${articleIndex + 1}/${selectedArticles.length} ` +
      `(global ${originalArticleIndex}): "${title}" -> ${chunks.length} chunks`
    );

    for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
      const chunk = chunks[chunkIndex];
      const embedding = await embedText(chunk);

      batch.push({
        id: `${articleId}-${chunkIndex}`,
        values: embedding,
        metadata: {
          article_id: articleId,
          title,
          authors,
          url,
          timestamp,
          tags,
          chunk,
          chunk_index: chunkIndex,
        },
      });

      if (batch.length >= BATCH_SIZE) {
        await upsertBatch(index, batch, batchNumber);
        totalVectors += batch.length;
        batch = [];
        batchNumber++;
      }
    }
  }

  if (batch.length > 0) {
    await upsertBatch(index, batch, batchNumber);
    totalVectors += batch.length;
  }

  console.log(`Done. Inserted/updated ${totalVectors} vectors in Pinecone.`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
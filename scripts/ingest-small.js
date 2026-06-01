import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import { parse } from "csv-parse/sync";
import { Pinecone } from "@pinecone-database/pinecone";

dotenv.config({ path: ".env.local" });

const CSV_PATH = path.resolve("../rag_backup/medium-english-50mb.csv");

const CHUNK_SIZE_CHARS = 2000;
const OVERLAP_CHARS = 400;
const MAX_ARTICLES = 3;

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

  const selectedArticles = records.slice(0, MAX_ARTICLES);
  console.log(`Loaded ${records.length} articles. Using first ${selectedArticles.length}.`);

  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

  const vectors = [];

  for (let articleIndex = 0; articleIndex < selectedArticles.length; articleIndex++) {
    const article = selectedArticles[articleIndex];

    const articleId = String(articleIndex);
    const title = article.title || "";
    const authors = article.authors || "";
    const url = article.url || "";
    const timestamp = article.timestamp || "";
    const tags = article.tags || "";
    const text = article.text || "";

    const chunks = chunkText(text);

    console.log(`Article ${articleIndex + 1}/${selectedArticles.length}: "${title}" -> ${chunks.length} chunks`);

    for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
      const chunk = chunks[chunkIndex];

      const embedding = await embedText(chunk);

      vectors.push({
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
    }
  }

  console.log(`Upserting ${vectors.length} vectors to Pinecone...`);

  await index.upsert({
  records: vectors,
});

  console.log("Done. Vectors inserted into Pinecone.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
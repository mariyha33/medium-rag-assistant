import { Pinecone } from "@pinecone-database/pinecone";

const SYSTEM_PROMPT = `You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data." Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful. If the user asks for a specific output format, such as "return only the titles", "list exactly 3", or "return only...", you must follow that format exactly and must not add explanations or extra text.`;
const TOP_K = 20;
const MAX_CONTEXT_ITEMS = 7;

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

export async function POST(request) {
  const body = await request.json();
  const question = body.question;

  if (!question) {
    return Response.json(
      { error: "Missing question field" },
      { status: 400 }
    );
  }

  const questionEmbedding = await embedText(question);

  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

  const searchResults = await index.query({
    vector: questionEmbedding,
    topK: TOP_K,
    includeMetadata: true,
  });

  const uniqueArticles = new Map();

for (const match of searchResults.matches || []) {
  const articleId = match.metadata?.article_id || "";

  if (!uniqueArticles.has(articleId)) {
    uniqueArticles.set(articleId, {
      article_id: articleId,
      title: match.metadata?.title || "",
      authors: match.metadata?.authors || "",
      url: match.metadata?.url || "",
      chunk: match.metadata?.chunk || "",
      score: match.score || 0,
    });
  }
}

const context = Array.from(uniqueArticles.values()).slice(0, MAX_CONTEXT_ITEMS);


  const contextText = context
    .map((item, i) => {
      return `Context chunk ${i + 1}
Article ID: ${item.article_id}
Title: ${item.title}
Authors: ${item.authors}
URL: ${item.url}
Score: ${item.score}
Passage:
${item.chunk}`;
    })
    .join("\n\n---\n\n");

  const userPrompt = `
Question:
${question}

Retrieved Medium articles context:
${contextText || "No relevant context was retrieved."}

Answer the question using only the retrieved context above.
`;

  const response = await fetch(`${process.env.LLMOD_BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.LLMOD_API_KEY}`,
    },
    body: JSON.stringify({
      model: "4UHRUIN-gpt-5-mini",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    return Response.json(
      {
        error: "LLMod API request failed",
        details: data,
      },
      { status: 500 }
    );
  }

  return Response.json({
    response: data.choices?.[0]?.message?.content || "No response from model.",
    context,
    Augmented_prompt: {
      System: SYSTEM_PROMPT,
      User: userPrompt,
    },
  });
}
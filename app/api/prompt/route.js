import { Pinecone } from "@pinecone-database/pinecone";

const UNKNOWN_RESPONSE = "I don't know based on the provided Medium articles data.";

const TOP_K = 30;
const MAX_CONTEXT_ITEMS = 8;

const SYSTEM_PROMPT = [
  "You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you.",
  "",
  "You must not use external knowledge, the open internet, or information that is not explicitly contained in the retrieved context.",
  "",
  `If the retrieved context chunks and metadata do not contain enough information to answer the user's question, respond exactly and only with: "${UNKNOWN_RESPONSE}"`,
  "",
  "Do not add explanations after that sentence when the answer is unknown.",
  "",
  "When answering, use only the retrieved context chunks and metadata.",
  "",
  "Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.",
  "",
  "For topic-listing questions, such as 'List exactly 3 articles about education', use the retrieved article titles and metadata to return distinct article titles related to the requested topic. If at least the requested number of distinct relevant article titles appear in the retrieved context, return exactly that number of titles and do not answer unknown.",
  "If the user asks for a specific output format, such as 'return only the titles', 'list exactly 3', or 'return only...', follow that format exactly and do not add extra text.",
  "",
  "For recommendation questions, recommend one article only and justify the recommendation using evidence from the retrieved context."
].join("\n");

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJsonWithRetry(fetchFunction, label, maxAttempts = 3) {
  let lastError;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const response = await fetchFunction();
      const rawText = await response.text();

      if (!response.ok) {
        throw new Error(`${label} failed with status ${response.status}: ${rawText}`);
      }

      try {
        return JSON.parse(rawText);
      } catch {
        throw new Error(`${label} returned non-JSON response: ${rawText}`);
      }
    } catch (error) {
      lastError = error;
      console.warn(`${label} attempt ${attempt}/${maxAttempts} failed:`, error.message);

      if (attempt < maxAttempts) {
        await sleep(1000 * attempt);
      }
    }
  }

  throw lastError;
}

async function embedText(text) {
  const data = await fetchJsonWithRetry(
    () =>
      fetch(`${process.env.LLMOD_BASE_URL}/v1/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.LLMOD_API_KEY}`,
        },
        body: JSON.stringify({
          model: "4UHRUIN-text-embedding-3-small",
          input: text,
        }),
      }),
    "Embedding request"
  );

  return data.data[0].embedding;
}

export async function POST(request) {
  try {
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

    const data = await fetchJsonWithRetry(
      () =>
        fetch(`${process.env.LLMOD_BASE_URL}/v1/chat/completions`, {
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
        }),
      "Chat request"
    );

    const modelResponse =
      data.choices?.[0]?.message?.content?.trim() || "No response from model.";

    const finalResponse = modelResponse.startsWith(UNKNOWN_RESPONSE)
      ? UNKNOWN_RESPONSE
      : modelResponse;

    const responseContext = context.map((item) => ({
      article_id: item.article_id,
      title: item.title,
      chunk: item.chunk,
      score: item.score,
    }));

    return Response.json({
      response: finalResponse,
      context: responseContext,
      Augmented_prompt: {
        System: SYSTEM_PROMPT,
        User: userPrompt,
      },
    });
  } catch (error) {
    console.error("Prompt API failed:", error);

    return Response.json(
      {
        error: "Prompt API failed",
        details: error.message,
      },
      { status: 500 }
    );
  }
}
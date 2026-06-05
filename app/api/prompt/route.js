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
  "",
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

function getListingTopic(question) {
  const topicMatch = question.match(/about\s+(.+?)(?:\.|,|\?|$)/i);

  if (
    topicMatch &&
    question.toLowerCase().includes("list") &&
    question.toLowerCase().includes("article")
  ) {
    return topicMatch[1].trim().toLowerCase();
  }

  return null;
}

function buildRetrievalQuery(question) {
  const topic = getListingTopic(question);

  if (topic) {
    const topicExpansions = {
      education:
        "education school schools university universities student students teacher teachers teaching learning classroom college graduate academic campus degree curriculum",
      writing:
        "writing writer writers author authors essay blog blogging headline publishing articles readers",
      habits:
        "habits habit behavior routine consistency self improvement productivity atomic habits"
    };

    return topicExpansions[topic] || topic;
  }

  return question;
}

function countKeywordHits(text, keywords) {
  let hits = 0;

  for (const keyword of keywords) {
    if (text.includes(keyword)) {
      hits += 1;
    }
  }

  return hits;
}

function topicRelevanceScore(item, topic) {
  const titleText = String(item.title || "").toLowerCase();
  const chunkText = String(item.chunk || "").toLowerCase();
  const tagsText = String(item.tags || "").toLowerCase();

  const topicKeywords = {
    education: [
      "education",
      "school",
      "schools",
      "university",
      "universities",
      "student",
      "students",
      "teacher",
      "teachers",
      "teaching",
      "learning",
      "classroom",
      "college",
      "graduate",
      "academic",
      "campus",
      "degree",
      "curriculum"
    ],
    writing: [
      "writing",
      "writer",
      "writers",
      "author",
      "authors",
      "essay",
      "blog",
      "blogging",
      "headline",
      "publishing",
      "articles",
      "readers"
    ],
    habits: [
      "habit",
      "habits",
      "routine",
      "behavior",
      "consistency",
      "self improvement",
      "productivity",
      "atomic habits"
    ]
  };

  const keywords = topicKeywords[topic] || [topic];

  const titleHits = countKeywordHits(titleText, keywords);
  const tagsHits = countKeywordHits(tagsText, keywords);
  const chunkHits = countKeywordHits(chunkText, keywords);

  return titleHits * 5 + tagsHits * 3 + chunkHits;
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

    const retrievalQuery = buildRetrievalQuery(question);
    const questionEmbedding = await embedText(retrievalQuery);

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
          tags: match.metadata?.tags || "",
          chunk: match.metadata?.chunk || "",
          score: match.score || 0,
        });
      }
    }

    let context = Array.from(uniqueArticles.values());

    const listingTopic = getListingTopic(question);

    if (listingTopic) {
      const rerankedContext = context
        .map((item) => ({
          ...item,
          topic_score: topicRelevanceScore(item, listingTopic),
        }))
        .filter((item) => item.topic_score > 0)
        .sort((a, b) => {
          if (b.topic_score !== a.topic_score) {
            return b.topic_score - a.topic_score;
          }

          return b.score - a.score;
        });

      if (rerankedContext.length >= 3) {
        context = rerankedContext;
      }
    }

    context = context.slice(0, MAX_CONTEXT_ITEMS);

    const contextText = context
      .map((item, i) => {
        return `Context chunk ${i + 1}
Article ID: ${item.article_id}
Title: ${item.title}
Authors: ${item.authors}
URL: ${item.url}
Tags: ${item.tags}
Score: ${item.score}
Passage:
${item.chunk}`;
      })
      .join("\n\n---\n\n");

    const listingInstruction = listingTopic
      ? `This is a topic-listing question about "${listingTopic}". Use the retrieved context to choose distinct article titles that are clearly related to this topic. If the user asked to return only the titles, do not add explanations or numbering unless necessary.`
      : "";

    const userPrompt = `
Question:
${question}

${listingInstruction}

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
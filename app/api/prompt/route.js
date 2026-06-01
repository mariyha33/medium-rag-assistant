const SYSTEM_PROMPT = `You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “I don't know based on the provided Medium articles data.” Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.`;

export async function POST(request) {
  const body = await request.json();
  const question = body.question;

  const userPrompt = `
Question:
${question}

Context:
No Medium article context was retrieved yet. This is a temporary test before connecting Pinecone.
`;

  const response = await fetch(`${process.env.LLMOD_BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.LLMOD_API_KEY}`
    },
    body: JSON.stringify({
      model: "4UHRUIN-gpt-5-mini",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt }
      ]
    })
  });

  const data = await response.json();

  if (!response.ok) {
    return Response.json(
      {
        error: "LLMod API request failed",
        details: data
      },
      { status: 500 }
    );
  }

  return Response.json({
    response: data.choices?.[0]?.message?.content || "No response from model.",
    context: [],
    Augmented_prompt: {
      System: SYSTEM_PROMPT,
      User: userPrompt
    }
  });
}
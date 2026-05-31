export async function POST(request) {
  const body = await request.json();
  const question = body.question;

  return Response.json({
    response: `This is a temporary response. Your question was: ${question}`,
    context: [],
    Augmented_prompt: {
      System: "Temporary system prompt",
      User: question
    }
  });
}
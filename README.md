# Medium RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) assistant for Medium articles.

The assistant answers questions only based on the retrieved Medium article context from the provided dataset. It does not rely on external knowledge.

## Live Deployment

Live URL:

https://medium-rag-assistant-rose.vercel.app

GitHub URL:

https://github.com/mariyha33/medium-rag-assistant

## API Endpoints

### POST `/api/prompt`

Used to query the RAG assistant.

Input:

```json
{
  "question": "Your question here"
}
```

Output:

```json
{
  "response": "Final natural language answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Sample article title",
      "chunk": "Retrieved article chunk",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "The system prompt used to query the chat model",
    "User": "The user prompt used to query the chat model"
  }
}
```

### GET `/api/stats`

Returns the current RAG configuration.

Output:

```json
{
  "chunk_size": 500,
  "overlap_ratio": 0.2,
  "top_k": 30
}
```
## RAG Configuration

- Chunk size: 2000 characters, approximately 500 tokens
- Overlap: 400 characters, approximately 20%
- Top-k retrieval: 30
- Final context items sent to the model: up to 8 distinct articles
- Vector database: Pinecone
- Deployment platform: Vercel

We retrieve `top_k=30` candidate chunks from Pinecone, deduplicate the results by `article_id`, rerank the retrieved articles when needed for topic-listing questions, and pass up to 8 distinct articles to the model. This helps reduce noise and cost while keeping enough relevant context for the required question types.

## Models

- Embeddings model: `4UHRUIN-text-embedding-3-small`
- Chat model: `4UHRUIN-gpt-5-mini`

## Notes

- The Pinecone index should remain active until grading is complete.
- API keys and environment variables are not included in this repository.
- The app uses environment variables for LLMod and Pinecone credentials.
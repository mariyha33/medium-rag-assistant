export async function GET() {
  return Response.json({
    chunk_size: 2000,
    overlap_ratio: 0.2,
    top_k: 20
  });
}
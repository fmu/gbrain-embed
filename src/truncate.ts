/**
 * Qwen3-Embedding is a Matryoshka model: the first N coordinates of its native
 * vector, L2-renormalized, are a valid N-dimensional embedding.
 *
 * gbrain applies exactly this rule in `fitMatryoshkaPrefix` for recipes that
 * declare `matryoshka`, but the `llama-server` recipe does not (its width is
 * whatever the launched GGUF emits), so the supervisor does it at the wire.
 * The rule is mirrored here rather than imported: the supervisor must not
 * depend on the gbrain install's internal module layout.
 */
export function truncateAndNormalize(vector: number[], dimensions: number): number[] {
  if (!Array.isArray(vector) || vector.length === 0) {
    throw new Error("embedding is empty");
  }
  if (!Number.isInteger(dimensions) || dimensions <= 0) {
    throw new Error(`invalid embedding dimensions ${dimensions}`);
  }
  if (vector.length < dimensions) {
    throw new Error(`embedding has ${vector.length} dimensions, cannot truncate to ${dimensions}`);
  }
  if (vector.length === dimensions) return vector;
  const prefix = vector.slice(0, dimensions);
  let sum = 0;
  for (const value of prefix) sum += value * value;
  const norm = Math.sqrt(sum);
  // gbrain's guard refuses a zero/non-finite vector; returning it unnormalized
  // keeps the refusal at the layer that owns the message.
  if (!Number.isFinite(norm) || norm === 0) return prefix;
  return prefix.map((value) => value / norm);
}

/** OpenAI's base64 embedding encoding is little-endian Float32 bytes. */
export function encodeEmbeddingBase64(vector: number[]): string {
  const buffer = Buffer.alloc(vector.length * 4);
  for (let i = 0; i < vector.length; i += 1) buffer.writeFloatLE(vector[i], i * 4);
  return buffer.toString("base64");
}

export function decodeEmbeddingBase64(encoded: string): number[] {
  const buffer = Buffer.from(encoded, "base64");
  if (buffer.length % 4 !== 0) throw new Error("base64 embedding is not a multiple of 4 bytes");
  const out: number[] = new Array(buffer.length / 4);
  for (let i = 0; i < out.length; i += 1) out[i] = buffer.readFloatLE(i * 4);
  return out;
}

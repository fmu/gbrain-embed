import { describe, expect, test } from "bun:test";
import { decodeEmbeddingBase64, encodeEmbeddingBase64, truncateAndNormalize } from "../src/truncate.ts";

describe("truncateAndNormalize", () => {
  test("takes the Matryoshka prefix and renormalizes it", () => {
    const vector = new Array(2560).fill(0);
    vector[0] = 3;
    vector[1] = 4;
    const out = truncateAndNormalize(vector, 2);
    expect(out).toEqual([0.6, 0.8]);
  });

  test("returns the vector unchanged when it is already the target width", () => {
    const vector = [0.6, 0.8];
    expect(truncateAndNormalize(vector, 2)).toBe(vector);
  });

  test("refuses a vector narrower than the requested width", () => {
    expect(() => truncateAndNormalize([1, 2], 3)).toThrow(/cannot truncate/);
  });

  test("returns a zero prefix unnormalized so the caller can refuse it", () => {
    expect(truncateAndNormalize([0, 0, 1], 2)).toEqual([0, 0]);
  });

  test("rejects invalid widths", () => {
    expect(() => truncateAndNormalize([1, 2], 0)).toThrow(/invalid embedding dimensions/);
    expect(() => truncateAndNormalize([1, 2], 1.5)).toThrow(/invalid embedding dimensions/);
  });
});

describe("base64 embeddings", () => {
  test("round-trips little-endian float32 values", () => {
    const vector = [0.25, -1.5, 3.0];
    const encoded = encodeEmbeddingBase64(vector);
    const decoded = decodeEmbeddingBase64(encoded);
    expect(decoded).toHaveLength(3);
    expect(decoded[0]).toBeCloseTo(0.25, 6);
    expect(decoded[1]).toBeCloseTo(-1.5, 6);
    expect(decoded[2]).toBeCloseTo(3.0, 6);
  });
});

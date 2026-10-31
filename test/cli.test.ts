import { describe, expect, test } from "bun:test";
import { main } from "../src/cli.ts";

describe("cli", () => {
  test("exports the command entry point", () => {
    expect(typeof main).toBe("function");
  });
});

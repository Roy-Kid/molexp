import { describe, expect, it } from "@rstest/core";
import { splitChatAnswer } from "./chat-answer-parse";

describe("splitChatAnswer", () => {
  it("extracts molplot fences", () => {
    const text = `Hello\n\n\`\`\`molplot title="Rg vs N"\n{"mark":"line","data":{"values":[]}}\n\`\`\`\n\nArchive?`;
    const segs = splitChatAnswer(text);
    expect(segs).toHaveLength(3);
    expect(segs[0]).toMatchObject({ kind: "markdown" });
    expect(segs[1]).toMatchObject({ kind: "molplot", title: "Rg vs N" });
    if (segs[1].kind === "molplot") {
      expect(segs[1].spec.mark).toBe("line");
    }
    expect(segs[2]).toMatchObject({ kind: "markdown" });
  });

  it("keeps plain markdown", () => {
    const segs = splitChatAnswer("just text");
    expect(segs).toEqual([{ kind: "markdown", text: "just text" }]);
  });
});

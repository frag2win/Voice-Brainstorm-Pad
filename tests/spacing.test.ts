import { describe, it, expect } from "vitest";
import { formatDictationText } from "../src/renderer/spacing";

describe("formatDictationText - Spacing & Formatting Rules", () => {
  it("does not prepend leading space at the start of an empty document", () => {
    const res = formatDictationText("", 0, "Hello world");
    expect(res.insertText).toBe("Hello world");
    expect(res.hasLeadingBreak).toBe(false);
  });

  it("prepends a single space when preceded by non-whitespace text", () => {
    const doc = "Hello";
    const res = formatDictationText(doc, doc.length, "world");
    expect(res.insertText).toBe(" world");
    expect(res.hasLeadingBreak).toBe(false);
  });

  it("does not prepend an extra space if already preceded by a space", () => {
    const doc = "Hello ";
    const res = formatDictationText(doc, doc.length, "world");
    expect(res.insertText).toBe("world");
    expect(res.hasLeadingBreak).toBe(false);
  });

  it("does not prepend a space if preceded by a newline", () => {
    const doc = "Hello\n";
    const res = formatDictationText(doc, doc.length, "world");
    expect(res.insertText).toBe("world");
    expect(res.hasLeadingBreak).toBe(false);
  });

  it("inserts a paragraph break after a pause longer than threshold (2.5s)", () => {
    const doc = "First paragraph.";
    const res = formatDictationText(doc, doc.length, "Second paragraph.", 3.0, { pauseThresholdS: 2.5 });
    expect(res.insertText).toBe("\n\nSecond paragraph.");
    expect(res.hasLeadingBreak).toBe(true);
  });

  it("inserts only one newline after a pause if already preceded by a single newline", () => {
    const doc = "First paragraph.\n";
    const res = formatDictationText(doc, doc.length, "Second paragraph.", 3.0, { pauseThresholdS: 2.5 });
    expect(res.insertText).toBe("\nSecond paragraph.");
    expect(res.hasLeadingBreak).toBe(true);
  });

  it("inserts no extra newlines after a pause if already preceded by double newlines", () => {
    const doc = "First paragraph.\n\n";
    const res = formatDictationText(doc, doc.length, "Second paragraph.", 3.0, { pauseThresholdS: 2.5 });
    expect(res.insertText).toBe("Second paragraph.");
    expect(res.hasLeadingBreak).toBe(true);
  });

  it("does not insert a paragraph break at document start even if pause is large", () => {
    const res = formatDictationText("", 0, "Starting sentence.", 5.0, { pauseThresholdS: 2.5 });
    expect(res.insertText).toBe("Starting sentence.");
    expect(res.hasLeadingBreak).toBe(false);
  });

  it("attaches punctuation without a leading space", () => {
    const doc = "Sentence";
    const res = formatDictationText(doc, doc.length, ".");
    expect(res.insertText).toBe(".");
  });

  it("handles empty or whitespace-only chunks gracefully", () => {
    const doc = "Hello";
    const res1 = formatDictationText(doc, doc.length, "");
    expect(res1.insertText).toBe("");

    const res2 = formatDictationText(doc, doc.length, "   ");
    expect(res2.insertText).toBe("");
  });
});

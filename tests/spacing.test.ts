import { describe, it, expect } from "vitest";
import { formatDictationText, calculateAudioPauseDuration } from "../src/renderer/spacing";

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

  it("produces a paragraph break when transcripts arrive late in a backlog but audio gap is 3.0s", () => {
    // Simulated catch-up backlog: Chunk 1 ended at 10.0s, Chunk 2 starts at 13.0s (3s pause in speech)
    const lastChunkEnd = 10.0;
    const chunk2 = {
      t_start: 13.0,
      t_end: 17.5,
      text: "New paragraph after backlog.",
    };

    const pauseDurationS = calculateAudioPauseDuration(chunk2.t_start, lastChunkEnd);
    expect(pauseDurationS).toBe(3.0);

    const doc = "Previous chunk audio ended at ten seconds.";
    const res = formatDictationText(doc, doc.length, chunk2.text, pauseDurationS);
    expect(res.insertText).toBe("\n\nNew paragraph after backlog.");
    expect(res.hasLeadingBreak).toBe(true);
  });

  it("does not produce a paragraph break when audio gap is 0.1s even if arrived back-to-back", () => {
    // Continuous speech: Chunk 1 ended at 10.0s, Chunk 2 starts at 10.1s (0.1s gap)
    const lastChunkEnd = 10.0;
    const chunk2 = {
      t_start: 10.1,
      t_end: 14.5,
      text: "continuing the same thought smoothly.",
    };

    const pauseDurationS = calculateAudioPauseDuration(chunk2.t_start, lastChunkEnd);
    expect(pauseDurationS).toBeCloseTo(0.1);

    const doc = "First sentence";
    const res = formatDictationText(doc, doc.length, chunk2.text, pauseDurationS);
    expect(res.insertText).toBe(" continuing the same thought smoothly.");
    expect(res.hasLeadingBreak).toBe(false);
  });
});
